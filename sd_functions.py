import argparse, os, re
from typing import Tuple
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from stable_diffusion.ldm.util import instantiate_from_config
from stable_diffusion.optimizedSD.optimUtils import split_weighted_subprompts, logger
from transformers import logging

logging.set_verbosity_error()

# Add to system path so that other functions in ./stable_diffusion/optimizedSD can load modules using relative path
import sys

sys.path.append("./stable_diffusion/optimizedSD")


class SDOptions:
    prompt = "a painting of a virus monster playing guitar"  # the prompt to render
    outdir = "stable_diffusion/outputs/txt2img-samples"  # dir to write results to
    skip_grid = True  # do not save a grid, only individual samples. Helpful when evaluating lots of samples
    skip_save = True  # do not save individual samples. For speed measurements.
    ddim_steps = 50  # number of ddim sampling steps
    fixed_code = True  # if enabled, uses the same starting code across samples
    ddim_eta = 0.0  # dim eta (eta=0.0 corresponds to deterministic sampling
    n_iter = 1  # sample this often
    H = 512  # image height, in pixel space
    W = 512  # image width, in pixel space
    C = 4  # latent channels
    f = 8  # downsampling factor
    n_samples = (
        1  # how many samples to produce for each given prompt. A.k.a. batch size
    )
    scale = 7.5  # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    device = "cuda"  # specify GPU (cuda/cuda:0/cuda:1/...)
    seed = None  # the seed (for reproducible sampling)
    unet_bs = 1  # Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )
    turbo = True  # Reduces inference time on the expense of 1GB VRAM
    precision = "autocast"  # evaluate at this precision: choices=["full", "autocast"]
    format = "png"  # output image format ["jpg" or "png"]
    sampler = "plms"  # sampler ["ddim" or "plms"]


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


def loadModels(ckpt_filepath: str, opt: SDOptions, config_filepath: str) -> Tuple:
    sd = load_model_from_config(f"{ckpt_filepath}")
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config_filepath = OmegaConf.load(f"{config_filepath}")

    model = instantiate_from_config(config_filepath.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = opt.unet_bs
    model.cdevice = opt.device
    model.turbo = opt.turbo

    modelCS = instantiate_from_config(config_filepath.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = opt.device

    modelFS = instantiate_from_config(config_filepath.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    if opt.device != "cpu" and opt.precision == "autocast":
        model.half()
        modelCS.half()

    return model, modelCS, modelFS


def initSD():
    config_filepath = "stable_diffusion/optimizedSD/v1-inference.yaml"
    ckpt_filepath = "model.ckpt"

    opt = SDOptions()

    os.makedirs(opt.outdir, exist_ok=True)

    if opt.seed == None:
        opt.seed = randint(0, 1000000)
    seed_everything(opt.seed)

    # Load model onto the GPU
    model, modelCS, modelFS = loadModels(ckpt_filepath, opt, config_filepath)

    return opt, model, modelCS, modelFS


def txt2img(opt: SDOptions, model, modelCS, modelFS):

    
    tic = time.time()

    outpath = opt.outdir
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn(
            [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=opt.device
        )

    batch_size = opt.n_samples
    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]

    if opt.precision == "autocast" and opt.device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext

    seeds = []
    results = []
    with torch.no_grad():

        for _ in trange(opt.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):

                sample_path = os.path.join(
                    outpath, "_".join(re.split(":| ", prompts[0]))
                )[:150]
                os.makedirs(sample_path, exist_ok=True)
                base_count = len(os.listdir(sample_path))

                with precision_scope("cuda"):
                    modelCS.to(opt.device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(
                                c,
                                modelCS.get_learned_conditioning(subprompts[i]),
                                alpha=weight,
                            )
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f]

                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = model.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        seed=opt.seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=start_code,
                        sampler=opt.sampler,
                    )

                    modelFS.to(opt.device)

                    print(samples_ddim.shape)
                    print("saving images")
                    for i in range(batch_size):

                        x_samples_ddim = modelFS.decode_first_stage(
                            samples_ddim[i].unsqueeze(0)
                        )
                        x_sample = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )
                        x_sample = 255.0 * rearrange(
                            x_sample[0].cpu().numpy(), "c h w -> h w c"
                        )

                        results.append(Image.fromarray(x_sample.astype(np.uint8)))
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(
                                sample_path,
                                "seed_"
                                + str(opt.seed)
                                + "_"
                                + f"{base_count:05}.{opt.format}",
                            )
                        )
                        seeds.append(str(opt.seed))
                        opt.seed += 1
                        base_count += 1

                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    time_taken = toc - tic

    return results, time_taken, seeds


if __name__ == "__main__":
    results, time_taken, seeds = txt2img(
        "photo of a miniature bear eating a watermelon, macro lens, high definition"
    )
    pass
