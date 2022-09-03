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
import time
from sd_functions import make_txt2img

# load ini
config = ConfigParser()
config.read("config.ini")
DISCORD_TOKEN = config.get("auth", "DISCORD_TOKEN")
DISCORD_GUILD = config.get("auth", "DISCORD_GUILD")
download_path = config.get("files", "download_path")

if not os.path.exists(download_path):
    os.makedirs(download_path, exist_ok=True)

# Create intents
intents = Intents.default()
intents.message_content = True
# Create discord bot
bot = commands.Bot(
    command_prefix="!",
    description="Bot hosted on Hayden's PC (and GPU)",
    intents=intents,
)


@bot.command(name="txt2img", help="Generate an image from a prompt (local GPU stable diffusion)")
async def txt2img(ctx, *, prompt):
    msg = await ctx.send(f"“{prompt}”\n> Generating...")
    results, time_taken, seeds = make_txt2img(prompt)
    await msg.edit(content=f"“{prompt}”\n> Done in {time_taken} seconds")
    for img, seed in zip(results, seeds):
        # Save the image in bytes format again
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        # Reset stream position to the start (so it can be read by discord.File)
        img_bytes.seek(0)

        # Convert to a discord file object that can be sent to the guild
        discord_img = discord.File(img_bytes, filename=f"{seed}_{prompt}.png")
        await ctx.send(file=discord_img)


@bot.command(name="echo", help = "[DEBUG]: echo back message")
async def echoMessage(ctx, *, msg="<blank>"):
    await ctx.send(f"ECHO: {msg}")


# Take an image attachments as input and send it back
@bot.command(name="img", help = "[DEBUG] flips supplied image upside down")
async def img(ctx):
    msg = await ctx.send(f"> Downloading...")
    time.sleep(0.5)

    attachment = ctx.message.attachments[0]
    filename = discordFilename(attachment)

    # save_path = saveImageFromUrl(attachment.url, filename)
    await msg.edit(content=f"> Processing...")
    time.sleep(0.5)

    img = getImageFromUrl(attachment.url)
    img = img.rotate(180)

    # Save the image in bytes format again
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    # Reset stream position to the start (so it can be read by discord.File)
    img_bytes.seek(0)

    await msg.edit(content=f"> Done...")
    # Convert to a discord file object that can be sent to the guild
    discord_img = discord.File(img_bytes, filename=filename)

    await ctx.send(file=discord_img)


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
