from discord.ext import commands
from utils import getImageFromUrl, discordFilename, run_in_executor
import discord
import time
from sd_functions import txt2img
import io
import asyncio
from discord_bot import StableDiffusionBot

# Convert txt2img into a coroutine called async_txt2img
@run_in_executor
def run_txt2img(*args, **kwargs):
    return txt2img(*args, **kwargs)
async def async_txt2img(*args, **kwargs):
    return await run_txt2img(*args, **kwargs)


def makeBotCommands(bot:StableDiffusionBot):

    @bot.command(name="txt2img", help="Generate an image from a prompt (local GPU stable diffusion)")
    async def bot_txt2img(ctx, *, prompt):
        msg = await ctx.send(f"“{prompt}”\n> Waiting...")

        

        # Run txt2img with the model, if another instance is not running already
        tic = time.time()
        output = await asyncio.gather(async_txt2img(prompt))
        time_taken = tic - time.time()
        
        results, seeds = tuple(zip(*output[0]))
        # use an asyncio.queue with a maxsize of 1, where txt2img puts an element in the queue and only gets it back when processing is complete
        # This means that each time txt2img is called, it won't run until the queue is empty, and then there won't be two running at once.
        # Use await q.join() to unload the ckpt model from memory, so that multiple calls of txt2img don't load the model multiple times
        
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
    async def bot_echo(ctx, *, msg="<blank>"):
        await ctx.send(f"ECHO: {msg}")


    # Take an image attachments as input and send it back
    @bot.command(name="img", help = "[DEBUG] flips supplied image upside down")
    async def bot_img(ctx):
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
    async def bot_wait(ctx, t):
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





