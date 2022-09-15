#!/usr/bin/env python3

import argparse
from datetime import datetime
from io import BytesIO
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

from PIL import Image

from typing import Optional
from discord.ext import commands
import discord
from discord import option
import logging as lg

# Based partly on https://github.com/harubaru/discord-stable-diffusion
class StableDiffusionCog(commands.Cog):
    def __init__(self, bot):
        self.logger = lg.getLogger(__name__)
        self.bot = bot
        self.running_sd = False
        self.t2i = self.init_t2i()

    @commands.slash_command(description="Generate image from text")
    @option("prompt", str, description="A text prompt for the model", required=True)
    @option("n", int, description="Number of images to generate", required=False)
    async def txt2img(
        self, ctx: discord.ApplicationContext, *, prompt: str, n: Optional[int] = 1
    ):

        assert n > 0
        await ctx.defer()

        author = f"{ctx.author.name}#{ctx.author.discriminator}"

        self.running_sd = True

        # Use the argument parser defaults
        prompt_parser = create_cmd_parser()
        opt_prompt = prompt_parser.parse_args([prompt, f"-n{n}"])

        # preload the model
        self.t2i.load_model()

        current_outdir = self.opt.outdir
        if not os.path.exists(current_outdir):
            os.makedirs(current_outdir)

        tic = time.time()
        results = self.t2i.prompt2image(image_callback=None, **vars(opt_prompt))
        duration = time.time() - tic
        images, seeds = tuple(zip(*results))

        discord_images = []
        for img, seed in zip(images, seeds):
            time_str = datetime.now().strftime("%Y%d%m-%H%M%S")
            file_name = f"[{time_str}]_{prompt}_{seed}_({author}).png"
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            # Reset stream position to the start (so it can be read by discord.File)
            buffer.seek(0)

            # Convert to a discord file object that can be sent to the guild
            discord_img = discord.File(buffer, filename=file_name)
            discord_images.append(discord_img)
        
        embed = discord.Embed()
        embed.color = discord.Colour.fuchsia()
        seeds_str = "|".join([str(_) for _ in seeds])
        s = "" if n == 1 else "s"
        embed.set_footer(text=f"\"{prompt}\"\nseed{s}:{seeds_str}, duration({duration:1f}))")
        await ctx.followup.send(embed=embed, files=discord_images)

    @commands.slash_command(description="Generate image from text")
    @option("echo", str, description="Text to echo back", required=False)
    async def echo(
        self, ctx: discord.ApplicationContext, *, txt: Optional[str] = "<blank>"
    ):
        await ctx.defer()

        await ctx.followup.send(f"ECHO: {txt}")

    def init_t2i(self):
        # Use the argument parser defaults
        parser = create_argv_parser()
        self.opt = parser.parse_args()
        default_width = 512
        default_height = 512
        config = "lstein_stable_diffusion/configs/stable-diffusion/v1-inference.yaml"
        weights = "./model.ckpt"

        self.logger.info("Initialising txt2img...")
        print("Initialising txt2img...")
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

        self.logger.info("Initialised txt2img")
        print("Initialised txt2img")
        return t2i


def setup(bot):
    bot.add_cog(StableDiffusionCog(bot))
