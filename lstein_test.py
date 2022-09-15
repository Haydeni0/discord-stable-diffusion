#!/usr/bin/env python3
# Copyright (c) 2022 Lincoln D. Stein (https://github.com/lstein)

import argparse
import shlex
import os
import re
import sys
import copy
import warnings
import time
import ldm.dream.readline
from ldm.dream.pngwriter import PngWriter, PromptFormatter
from ldm.dream.server import DreamServer, ThreadingDreamServer
from ldm.dream.image_util import make_grid
from omegaconf import OmegaConf
from lstein_stable_diffusion.scripts.dream import create_argv_parser, create_cmd_parser



class SDOptions:
    prompt = "a painting of a virus monster playing guitar"  # the prompt to render
    full_precision = False # Use memory-intensive full precision math for calculations
    ddim_steps = 50  # number of ddim sampling steps
    g = False # Generate a grid instead of individual images
    sampler_name = "k_lms" # Set the initial sampler 
    height = 512  # image height, in pixel space
    width = 512  # image width, in pixel space
    C = 4  # latent channels
    f = 8  # downsampling factor
    n = 1 # Number of samples to generate
    scale = 7.5  # unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
    device = "cuda"  # specify GPU (cuda/cuda:0/cuda:1/...)
    seed = None  # the seed (for reproducible sampling)
    unet_bs = 1  # Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )
    turbo = True  # Reduces inference time on the expense of 1GB VRAM
    precision = "autocast"  # evaluate at this precision: choices=["full", "autocast"]
    format = "png"  # output image format ["jpg" or "png"]
    sampler = "plms"  # sampler ["ddim" or "plms"]
    weights_path = "./model.ckpt"
    

def main(prompt = "pogge"):

    # Use the argument parser defaults
    parser = create_argv_parser()
    opt = parser.parse_args()
    prompt_parser = create_cmd_parser()
    opt_prompt = prompt_parser.parse_args([prompt])

    default_width = 512
    default_height = 512
    config = "lstein_stable_diffusion/configs/stable-diffusion/v1-inference.yaml"
    weights = "./model.ckpt"


    print("Initialising...\n")
    from pytorch_lightning import logging
    from ldm.generate import Generate

    # these two lines prevent a horrible warning message from appearing
    # when the frozen CLIP tokenizer is imported
    import transformers
    transformers.logging.set_verbosity_error()

    # gets rid of annoying messages about random seed
    logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

    # creating a simple text2image object with a handful of
    # defaults passed on the command line.
    # additional parameters will be added (or overriden) during
    # the user input loop
    t2i = Generate(
        width=default_width,
        height=default_height,
        sampler_name=opt.sampler_name,
        weights=weights,
        full_precision=opt.full_precision,
        config=config,
        grid=opt.grid,
        # this is solely for recreating the prompt
        seamless=opt.seamless,
        embedding_path=opt.embedding_path,
        device_type=opt.device,
        ignore_ctrl_c=opt.infile is None,
    )

    # preload the model
    t2i.load_model()

    do_grid = opt_prompt.grid or t2i.grid

    current_outdir = opt.outdir
    if not os.path.exists(current_outdir):
        os.makedirs(current_outdir)


    results = t2i.prompt2image(image_callback=None, **vars(opt_prompt))

    pass

    

if __name__ == "__main__":
    main()
    