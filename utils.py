import requests
import re
import os
from PIL import Image
import discord
import io
import functools
import asyncio

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

    download_folder = "./downloads"
    if not os.path.exists(download_folder):
        os.makedirs(download_folder, exist_ok=True)

    save_path = os.path.join(download_folder, filename)

    # Save image (binary)
    with open(save_path, "wb") as input_img:
        input_img.write(img_data)

    return save_path


def run_in_executor(f):
    # from https://stackoverflow.com/questions/41063331/how-to-use-asyncio-with-existing-blocking-library
    @functools.wraps(f)
    def inner(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, lambda: f(*args, **kwargs))

    return inner