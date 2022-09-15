import logging
from configparser import ConfigParser

from discord import Intents
from discord.ext import commands

from bot_commands import makeBotCommands
from discord_bot import StableDiffusionBot

logger = logging.getLogger(__name__)
logging_format = '%(name)s:%(levelname)s %(message)s'
logging.basicConfig(filename='bot.log', filemode='w',
                    format=logging_format, level=logging.INFO)

def main():
    # Load ini
    config = ConfigParser()
    config.read("config.ini")
    DISCORD_TOKEN = config.get("auth", "DISCORD_TOKEN")

    # Define the bot command prefix in the config
    if config.has_option("options", "command_prefix"):
        command_prefix = config.get("options", "command_prefix")
    else:
        command_prefix = "/"

    logging.info(f"Starting bot using command prefix {command_prefix}")
    print(f"Starting bot using command prefix {command_prefix}")

    # Create intents
    intents = Intents.default()
    intents.message_content = True
    # Create discord bot
    bot = StableDiffusionBot(
        command_prefix=command_prefix,
        description="Bot hosted on Hayden's PC (and GPU) \n[https://github.com/Haydeni0/discord-stable-diffusion]",
        intents=intents,
    )

    makeBotCommands(bot)


    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()
