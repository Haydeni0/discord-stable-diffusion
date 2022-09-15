from discord.ext import commands
import logging
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

# Based partly on https://github.com/harubaru/discord-stable-diffusion
class StableDiffusionBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.logger = logging.getLogger(__name__)
        self.load_extension("sd_cog")

    async def on_ready(self):
        self.logger.info(f"Logged in as {self.user.name} ({self.user.id})")

    
