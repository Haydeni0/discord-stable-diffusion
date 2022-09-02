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
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    guild = discord.utils.get(client.guilds, name=DISCORD_GUILD)
    print(
        f"{client.user} is connected to the following guild:\n"
        f"{guild.name}(id: {guild.id})"
    )


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content == "raise-exception":
        raise discord.DiscordException

    await message.channel.send(message.content)


client.run(DISCORD_TOKEN)
