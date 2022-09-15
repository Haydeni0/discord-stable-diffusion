#!/usr/bin/env bash

repo_root_dir=$(dirname "$0")

if { conda env list | grep 'ldm'; } >/dev/null 2>&1; then
    conda env update -f ${repo_root_dir}/lstein_stable_diffusion/environment.yaml
else
    conda env create -f ${repo_root_dir}/lstein_stable_diffusion/environment.yaml
fi

conda init bash
/bin/bash
conda activate ldm

python -m pip install py-cord

python ${repo_root_dir}/lstein_stable_diffusion/scripts/preload_models.py

