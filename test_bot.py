from fileinput import filename
import discord
from discord import Intents
from discord.ext import commands
from dotenv import load_dotenv
from PIL import Image
import os
import io
from configparser import ConfigParser
import warnings
import random

# load ini
config = ConfigParser()
config.read("config.ini")
DISCORD_TOKEN = config.get("auth", "DISCORD_TOKEN")
DISCORD_GUILD = config.get("auth", "DISCORD_GUILD")

# Create intents
intents = Intents.default()
intents.message_content = True
# Create discord bot
bot = commands.Bot(
    command_prefix="!",
    description="aughroahoisdoijs",
    intents=intents,
)


@bot.command(name="asd")
async def asd(ctx, *, msg=""):
    await ctx.send("asd")


@bot.command(name="dice", help="Simulates rolling dice.")
async def roll(ctx, number_of_dice: int, number_of_sides: int):
    dice = [
        str(random.choice(range(1, number_of_sides + 1))) for _ in range(number_of_dice)
    ]
    await ctx.send(", ".join(dice))


bot.run(DISCORD_TOKEN)
