from typing import Tuple
import requests
import re
import os
from PIL import Image
import discord
import io
import functools
import asyncio
import math


def getImageFromUrl(url: str) -> Image.Image:
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


def saveImageFromUrl(url: str, download_path: str, resize_num_pixels: int) -> str:
    img = getImageFromUrl(url)

    img_pixels = math.prod(img.size)
    # Resize the image
    if resize_num_pixels is not None:
        scale_factor = img_pixels / resize_num_pixels
        new_size = tuple(round(sz / math.sqrt(scale_factor)) for sz in img.size)
        img = img.resize(new_size)

    os.makedirs(download_path, exist_ok=True)

    # Get filename from url using the last segment
    filename = re.split("/", url)[-1]
    save_path = os.path.join(download_path, filename)

    img.save(save_path, "PNG")

    return save_path


def run_in_executor(f):
    # a decorator that turns a blocking function into an asynchronous function
    # from https://stackoverflow.com/questions/41063331/how-to-use-asyncio-with-existing-blocking-library
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, lambda: f(*args, **kwargs))

    return inner
