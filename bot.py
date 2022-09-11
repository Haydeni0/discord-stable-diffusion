from discord import Intents
from discord.ext import commands
from utils import getImageFromUrl, discordFilename

import io
from configparser import ConfigParser

import discord
import time
from sd_functions import make_txt2img

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





bot.run(DISCORD_TOKEN)
