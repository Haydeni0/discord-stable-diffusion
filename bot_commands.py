from discord.ext import commands
from utils import getImageFromUrl, discordFilename, run_in_executor
import discord
import time
from sd_functions import make_txt2img
import io
import asyncio

def makeBotCommands(bot:commands.bot):

    @bot.command(name="txt2img", help="Generate an image from a prompt (local GPU stable diffusion)")
    async def txt2img(ctx, *, prompt):
        msg = await ctx.send(f"“{prompt}”\n> Generating...")

        # results, time_taken, seeds = make_txt2img(prompt)

        # >>> use this block to do the txt2img asynchronously in parallel
        @run_in_executor
        def do_processing(prompt):
            return make_txt2img(prompt)
        async def do_processing_async(prompt):
            return await do_processing(prompt)
        tasks = [asyncio.create_task(do_processing_async(prompt))]
        task_results = [await t for t in asyncio.as_completed(tasks)]
        results, time_taken, seeds = task_results[0]
        # <<<
        
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
    

    @bot.command(name="wait", help = "[DEBUG] Wait for a time period")
    async def wait(ctx, t):
        # Proof of concept for running a blocking function asynchronously
        @run_in_executor
        def blocking(t = 1):
            time.sleep(t)
            return

        async def waitblock(t):
            return await blocking(t)

        t = float(t)
        tasks = [asyncio.create_task(waitblock(t))]
        [await _ for _ in asyncio.as_completed(tasks)]
        await ctx.send(f"Waited {t} seconds")





