
from discord.ext import commands
import logging

# Based partly on https://github.com/harubaru/discord-stable-diffusion
class StableDiffusionBot(commands.Bot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.logger = logging.getLogger(__name__)

    async def on_ready(self):
        self.logger.info(f"Logged in as {self.user.name} ({self.user.id})")

    async def close(self):
        await self._bot.close()











