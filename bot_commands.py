from discord.ext import commands
from utils import getImageFromUrl, discordFilename
import discord
import time
from sd_functions import make_txt2img
import io

def makeBotCommands(bot:commands.bot):

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