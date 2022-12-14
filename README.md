<h1 align="center">Discord Bot - Stable Diffusion</h1>

A discord bot that runs the Stable Diffusion text-to-image model on a local GPU (the computer hosting the discord bot). This uses a [fork of stable diffusion](https://github.com/lstein/stable-diffusion) repository to do the text to image generation.

---

## Requirements
- git
- Anaconda or miniconda python distribution ([link](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html))
- An Nvidia GPU with CUDA support (ideally with 8GB+ VRAM)
  - Nvidia drivers


## Installation instructions

1. Clone this repo (and its submodules) using

         git clone --recurse-submodules git@github.com:Haydeni0/discord-stable-diffusion

2. Navigate to the repository in terminal
3. Follow the instructions [here](https://github.com/lstein/stable-diffusion) to set it up
   1. Or if using linux/WSL2, run the command

            . install.sh

      to automatically set things up (requires an installation of conda).

   2. Download the weights for the model ([check the latest here](https://huggingface.co/CompVis/stable-diffusion))
      - At the time of writing, the latest one is checkpoint version 1.4 and can be downloaded under "Files and versions" from the Hugging Face page for [stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) (note that this requires a signup to the website).
   3. Move and rename the model checkpoint file to 
      > ./model.ckpt
  
4. Create a discord application and bot using the first two chapters of [this guide](https://realpython.com/how-to-make-a-discord-bot-python/) (Creating an Application, Creating a Bot)
   <details> 
   <summary> Click to view discord bot configuration </summary>

      1. Make sure to enable all the "Privileged Gateway Intents"
         ![](/readme_media/PrivilegedGatewayIntents.png)
      2. Add the bot to the discord server with these OAuth2 permissions
         ![](/readme_media//OAuth2Permissions.png)
         by opening the URL generated by the OAuth2 URL generator
      3. Generate a discord bot token
         ![](/readme_media/DiscordBotToken.png)
         
   </details>
5. Add the discord bot token to a file named ```config.ini``` in the root directory of the repository (you may have to create this yourself). The contents of the file should be:
   
   ```
   # .ini
   [auth]
   DISCORD_TOKEN=PASTE_YOUR_TOKEN_HERE
   ```
6. Run the python script ```bot.py``` using the command

         python bot.py
   
   or in Windows by using the full path to python, for example:

         C:\ProgramData\Anaconda3\envs\ldm\python.exe bot.py
         
7. Run the bot on the discord server using "```/```" commands in a discord channel, e.g:
   - ```/help```
   - ```/txt2img <your prompt here>```


---

## To-do

### Features
- img2img
- Add capability to change txt2img options from discord
- Add option that automatically prefixes prompts with stuff like "4K, 8K, high resolution, award winning, ..."
  - To automatically improve quality of image outputs without typing
- Upscaling
- Make a command to cancel the bot generation if it's taking too long

### QOL
- Capture the console output so that the loading bar can be shown in discord, the lstein fork does this somehow
- Unload the model after some time


### Development
- Add error handling to the commands

- Don't let txt2img automatically save the images, put that functionality in a separate routine
- Make sure two parallel txt2img things dont occur, give a warning if a txt2img is called again without finishing the previous one
  - Or maybe allow it by putting them in a queue (with a max length), and run them serially so the gpu isn't working on 2 at once

### Fixes
- Fix img2img mode producing garbage result because of low resolution
  - Fix the rescaling to use the volume rather than max of the width and height





## Problems
- 




