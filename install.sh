#!/usr/bin/env bash

file_dir=$(dirname "$0")

if { conda env list | grep 'ldm'; } >/dev/null 2>&1; then
    conda env update -f lstein_stable_diffusion/environment.yaml
else
    conda env create -f lstein_stable_diffusion/environment.yaml
fi

conda activate ldm

python -m pip install py-cord

python lstein_stable_diffusion/scripts/preload_models.py

