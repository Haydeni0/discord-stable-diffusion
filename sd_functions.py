#!/usr/bin/env python3

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




    

def txt2img(prompt = "pogge"):

    # Use the argument parser defaults
    parser = create_argv_parser()
    opt = parser.parse_args()
    prompt_parser = create_cmd_parser()
    opt_prompt = prompt_parser.parse_args([prompt, "-n2"])

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

    return results

    
    