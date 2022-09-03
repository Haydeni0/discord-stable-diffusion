from fileinput import filename
from turtle import down
from discord import Intents
from discord.ext import commands
from PIL import Image
import os
import io
from configparser import ConfigParser
import warnings
import random
import requests
import re
import discord

# load ini
config = ConfigParser()
config.read("config.ini")
DISCORD_TOKEN = config.get("auth", "DISCORD_TOKEN")
DISCORD_GUILD = config.get("auth", "DISCORD_GUILD")
download_path = config.get("files", "download_path")

if not os.path.exists(download_path):
    os.makedirs(download_path)

# Create intents
intents = Intents.default()
intents.message_content = True
# Create discord bot
bot = commands.Bot(
    command_prefix="!",
    description="aughroahoisdoijs",
    intents=intents,
)


@bot.command(name="msg")
async def returnMessage(ctx, *, msg="<blank>"):
    await ctx.send(f"Received: {msg}")


# Take an image attachments as input and send it back
@bot.command(name="img")
async def img(ctx):
    attachment = ctx.message.attachments[0]
    filename = discordFilename(attachment)

    # save_path = saveImageFromUrl(attachment.url, filename)

    img = getImageFromUrl(attachment.url)
    img = img.rotate(180)

    # Save the image in bytes format again
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    # Reset stream position to the start (so it can be read by discord.File)
    img_bytes.seek(0)
    
    # Convert to a discord file object that can be sent to the guild
    discord_img = discord.File(img_bytes, filename=filename)

    await ctx.send(file = discord_img)


def getImageFromUrl(url: str):
    img_bytes = requests.get(url).content

    img = Image.open(io.BytesIO(img_bytes))

    return img


def discordFilename(attachment: discord.message.Attachment):
    # Gets the filename from a discord attachment
    if (attachment.filename is not None) and (
        re.fullmatch("unknown\.\w+", attachment.filename) is None
    ):
        filename = attachment.filename
    else:
        file_ext_match = re.findall("(?<=\.)\w+(?=$)", attachment.filename)
        if file_ext_match is None:
            file_ext = ""
        else:
            file_ext = file_ext_match[0]
        filename = f"img_{hash(attachment.url)}.{file_ext}"

    return filename


def saveImageFromUrl(url: str, filename: str = None) -> str:
    img_data = requests.get(url).content

    save_path = os.path.join(download_path, filename)

    # Save image (binary)
    with open(save_path, "wb") as input_img:
        input_img.write(img_data)

    return save_path


bot.run(DISCORD_TOKEN)
