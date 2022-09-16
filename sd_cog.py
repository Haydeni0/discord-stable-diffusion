#!/usr/bin/env python3

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from io import BytesIO, StringIO
import math
import shlex
import os
import re
from statistics import quantiles
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

from utils import getImageFromUrl, discordFilename, run_in_executor

from PIL import Image

from typing import Optional
from discord.ext import commands
import discord
from discord import option
import logging as lg

# Based partly on https://github.com/harubaru/discord-stable-diffusion
class StableDiffusionCog(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.logger = lg.getLogger(__name__)
        self.bot = bot
        self.sd_query_in_progress = False
        self.t2i = self.init_t2i()

    @commands.slash_command(description="Generate image from text")
    @option("prompt", str, description="A text prompt for the model", required=True)
    @option(
        "n", int, description="Number of images to generate [default:1]", required=False
    )
    @option(
        "width",
        int,
        description="Image width, multiple of 64 [default:512]",
        required=False,
    )
    @option(
        "height",
        int,
        description="Image height, multiple of 64 [default:512]",
        required=False,
    )
    @option(
        "cfg_scale",
        int,
        description='Classifier free guidance (CFG) scale - higher numbers cause generator to "try" harder [default:7.5]',
        required=False,
    )
    @option("seed", int, description="Image seed [default:random]", required=False)
    @option(
        "steps",
        int,
        description="Number of sampling steps [default:50]",
        required=False,
    )
    async def txt2img(
        self,
        ctx: discord.ApplicationContext,
        *,
        prompt: str,
        n: Optional[int] = 1,
        width: Optional[int] = 512,
        height: Optional[int] = 512,
        cfg_scale: Optional[float] = 7.5,
        seed: Optional[int],
        steps: Optional[int] = 50,
    ):
        await ctx.defer()

        error_embed = discord.Embed(colour=discord.Colour.red())
        msg_embed = discord.Embed(colour=discord.Colour.fuchsia())
        if not (1 <= n <= 10):
            error_embed.set_footer(text="Error, n must be between 1 and 10 inclusive")
            await ctx.followup.send(embed=error_embed)
            return

        # Author of the message
        author = f"{ctx.author.name}-{ctx.author.discriminator}"

        # Create query for the query parser
        query = [
            prompt,
            f"-n{n}",
            f"-W{width}",
            f"-H{height}",
            f"-C{cfg_scale}",
            None if seed is None else f"-S{seed}",
            f"-s{steps}",
        ]
        query = [_ for _ in query if _ is not None]

        # Since the argparser doesn't do proper logging, we have to capture the stderr to log it properly
        with redirect_stderr(StringIO()) as stderr:
            try:
                query_parser = create_cmd_parser()
                query_opt = query_parser.parse_args(query)
            except:
                self.logger.error(stderr.getvalue())
                error_embed.set_footer(text="Argument error, check logs")
                await ctx.followup.send(embed=error_embed)
                return

        self.sd_query_in_progress = True

        tic = time.time()
        results = self.t2i.prompt2image(image_callback=None, **vars(query_opt))
        duration = time.time() - tic
        images, seeds = tuple(zip(*results))

        # Return txt2img results to discord, and save
        discord_images = []
        current_outdir = self.opt.outdir
        if not os.path.exists(current_outdir):
            os.makedirs(current_outdir)
        for img, seed in zip(images, seeds):
            time_str = datetime.now().strftime("%Y%d%m-%H%M%S")
            file_name = f"{time_str}_{prompt}_{seed}_{author}.png"
            file_path = os.path.join(current_outdir, file_name)

            # Save to file
            img.save(file_path, format="PNG")

            # Save to bytes buffer
            buffer = BytesIO()
            img.save(buffer, format="PNG")

            # Check if the file is larger than 8 million bytes (discord upload limit is 8MB != 8 million bytes?)
            size_limit = 8_000_000
            if buffer.__sizeof__() > size_limit:
                file_name = file_name.replace(".png", ".jpeg")
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                msg_embed.set_footer(
                    text="[File quality reduced to fit under the discord 8MB limit]"
                )

            jpeg_quality = 100
            while buffer.__sizeof__() > size_limit and jpeg_quality > 1:
                buffer = BytesIO()
                img.save(buffer, format="JPEG", optimize=True, quality=jpeg_quality)
                jpeg_quality -= 5
            if buffer.__sizeof__() > size_limit:
                self.logger.warning(
                    f"Image too large to be sent to discord ({buffer.__sizeof__()} bytes)"
                )
                error_embed.set_footer(
                    text=f"Image too large to be sent to discord ({buffer.__sizeof__()} bytes). Saved to disk."
                )
                await ctx.followup.send(embed=error_embed)
                return

            # Reset stream position to the start (so it can be read by discord.File)
            buffer.seek(0)

            # Convert to a discord file object that can be sent to the guild
            discord_img = discord.File(buffer, filename=file_name)
            discord_images.append(discord_img)

        seeds_str = "|".join([str(_) for _ in seeds])
        s = "" if n == 1 else "s"
        msg_embed.add_field(
            name=prompt, value=f"seed{s}: {seeds_str}, duration: {duration:1f}"
        )
        await ctx.followup.send(embed=msg_embed, files=discord_images)

    @commands.slash_command(description="[DEBUG] Send test message with embedding")
    async def embed(self, ctx: discord.ApplicationContext):
        await ctx.defer()
        embed = discord.Embed(colour=discord.Colour.fuchsia())
        embed.set_author(name=f"author")
        embed.add_field(name="fieldname", value="fieldvalue1", inline=False)
        embed.add_field(name="fieldname_inline", value="fieldvalue2", inline=True)
        embed.set_footer(text=f"footer")
        try:
            embed.set_thumbnail(
                url="https://is4-ssl.mzstatic.com/image/thumb/Purple71/v4/ea/c0/40/eac0409e-27bf-3f55-a14f-f5818b70523c/source/256x256bb.jpg"
            )
            embed.set_image(
                url="https://sygicwebmapstiles-d.azureedge.net/d422be30-94e3-4d8b-9724-8767e6daa1cf/16/32732/21801.png?sv=2017-07-29&sr=b&sig=QZVb%2B4XZXFi9x39RZ4Usp5AqlBQKMMcHkwWyuqQ0oq0%3D&se=2022-09-25T00%3A00%3A00Z&sp=r&rev=b"
            )
        except:
            self.logger("Embed URL no longer works")
        await ctx.followup.send(embed=embed, content="test")

    @commands.slash_command(description="Echo back a message")
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
