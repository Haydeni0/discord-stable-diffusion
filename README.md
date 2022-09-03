# Discord bot

## Requirements
- git
- Anaconda or miniconda python distribution ([link](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html))
- An Nvidia GPU with CUDA support (ideally with 8GB+ VRAM)
  - Nvidia drivers


## Installation instructions

1. Clone this repo (and its submodules) using

         git clone --recurse-submodules git@github.com:Haydeni0/discord-stable-diffusion

2. Navigate to the repository in terminal
3. Follow the instructions on the [original stable-diffusion repository](https://github.com/CompVis/stable-diffusion) to set it up, summarised here:
   1. Create and activate a latent diffusion conda environment (called ldm)

           conda env create -f .\stable_diffusion\environment.yaml
           conda activate ldm

   2. Download the weights for the model ([check the latest here](https://huggingface.co/CompVis/stable-diffusion))
      - At the time of writing, the latest one is checkpoint version 1.4 and can be downloaded under "Files and versions" from the Hugging Face page for [stable-diffusion-v-1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) (note that this requires a signup to the website).
   3. Move and rename the model checkpoint file to 
      > ./stable_diffusion/models/ldm/stable-diffusion-v1/model.ckpt
4. Create a discord application and bot using the first two chapters of [this guide](https://realpython.com/how-to-make-a-discord-bot-python/) (Creating an Application, Creating a Bot)
   1. Make sure to enable all the "Privileged Gateway Intents"
      ![](/readme_media/PrivilegedGatewayIntents.png)
   2. Add the bot to the discord server with these OAuth2 permissions
      ![](/readme_media//OAuth2Permissions.png)
      by opening the URL generated by the OAuth2 URL generator
   3. Generate a discord bot token
      ![](/readme_media/DiscordBotToken.png)
5. Add the discord bot token to a file named ```config.ini``` in the root directory of the repository. The contents of the file should be:
   
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
- Add error handling to the commands
- Capture the console output so that the loading bar can be shown in discord
- Add capability to change options
- Add img2img
- Add !help command



## Problems
- Stable diffusion won't work in WSL2 on my PC
  - Maybe because of my windows version not allowing Nvidia CUDA drivers on WSL2(see [stackoverflow](https://stackoverflow.com/questions/64256241/found-no-nvidia-driver-on-your-system-error-on-wsl2-conda-environment-with-pytho))




