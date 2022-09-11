from discord import Intents
from discord.ext import commands
from bot_commands import makeBotCommands



from configparser import ConfigParser



# load ini
config = ConfigParser()
config.read("config.ini")
DISCORD_TOKEN = config.get("auth", "DISCORD_TOKEN")

# Define the bot command prefix in the config
if config.has_option("options", "command_prefix"):
    command_prefix = config.get("options", "command_prefix")
else:
    command_prefix = "/"

print(f"Starting bot using command prefix {command_prefix}")

# Create intents
intents = Intents.default()
intents.message_content = True
# Create discord bot
bot = commands.Bot(
    command_prefix=command_prefix,
    description="Bot hosted on Hayden's PC (and GPU) \n[https://github.com/Haydeni0/discord-stable-diffusion]",
    intents=intents,
)

makeBotCommands(bot)


bot.run(DISCORD_TOKEN)
