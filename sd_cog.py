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

from typing import Optional
from discord.ext import commands
import discord
from discord import option

# Based partly on https://github.com/harubaru/discord-stable-diffusion
class StableDiffusionCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command()
    @option("pongis", str, description="asd", required=True)
    async def hello(self, ctx, *, txt: str):
        await ctx.send(f"Hello {txt}")

    @commands.slash_command(description="Generate image from text")
    @option("prompt", str, description="A text prompt for the model", required=True)
    @option("more", str, description="asd", required=False)
    async def t2i(self, ctx: discord.ApplicationContext, *, prompt: str, more: Optional[int]):
        print(
            f"Request -- {ctx.author.name}#{ctx.author.discriminator} -- Prompt: {prompt}"
        )
        await ctx.defer()
        await ctx.followup.send(f"prompt")

    async def init_t2i(self):
        # Use the argument parser defaults
        parser = create_argv_parser()
        self.opt = parser.parse_args()
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
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        # creating a simple text2image object with a handful of
        # defaults passed on the command line.
        # additional parameters will be added (or overriden) during
        # the user input loop
        t2i = Generate(
            width=default_width,
            height=default_height,
            sampler_name=self.opt.sampler_name,
            weights=weights,
            full_precision=self.opt.full_precision,
            config=config,
            grid=self.opt.grid,
            # this is solely for recreating the prompt
            seamless=self.opt.seamless,
            embedding_path=self.opt.embedding_path,
            device_type=self.opt.device,
            ignore_ctrl_c=self.opt.infile is None,
        )

        return t2i

    def txt2img(self, prompt="pogge"):

        # Use the argument parser defaults
        prompt_parser = create_cmd_parser()
        opt_prompt = prompt_parser.parse_args([prompt, "-n2"])

        # preload the model
        self.t2i.load_model()

        do_grid = opt_prompt.grid or self.t2i.grid

        current_outdir = self.opt.outdir
        if not os.path.exists(current_outdir):
            os.makedirs(current_outdir)

        results = self.t2i.prompt2image(image_callback=None, **vars(opt_prompt))

        return results


def setup(bot):
    bot.add_cog(StableDiffusionCog(bot))
