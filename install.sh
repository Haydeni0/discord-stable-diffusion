#!/usr/bin/env bash

repo_root_dir=$(dirname "$0")

if { conda env list | grep 'ldm'; } >/dev/null 2>&1; then
    conda env update -f ${repo_root_dir}/lstein_stable_diffusion/environment.yaml --prune
else
    conda env create -f ${repo_root_dir}/lstein_stable_diffusion/environment.yaml
fi

conda init bash
/bin/bash
conda activate ldm

if {conda list | grep 'discord'; } >/dev/null 2>&1; then
    python -m pip uninstall py-cord discord.py 
fi
python -m pip install py-cord=2.1.3

python ${repo_root_dir}/lstein_stable_diffusion/scripts/preload_models.py

