import logging
from configparser import ConfigParser

from discord import Intents
from discord.ext import commands

from bot_commands import makeBotCommands
from discord_bot import StableDiffusionBot

logger = logging.getLogger(__name__)
logging_format = "[%(asctime)s] %(name)s:%(levelname)s %(message)s"
logging.basicConfig(
    filename="bot.log", filemode="w", format=logging_format, level=logging.INFO
)


def main():
    # Load ini
    config = ConfigParser()
    config.read("config.ini")
    DISCORD_TOKEN = config.get("auth", "DISCORD_TOKEN")

    logging.info(f"Starting bot")
    print(f"Starting bot")

    # Create intents
    intents = Intents.default()
    intents.message_content = True
    # Create discord bot
    bot = StableDiffusionBot(
        command_prefix="/",
        description="Bot hosted on Hayden's PC (and GPU) \n[https://github.com/Haydeni0/discord-stable-diffusion]",
        intents=intents,
    )

    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
