#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if { conda env list | grep 'ldm'; } >/dev/null 2>&1; then
    echo === Updating conda environment "ldm"
    conda env update -f ${SCRIPT_DIR}/lstein_stable_diffusion/environment.yaml --prune
else
    echo === Creating conda environment "ldm"
    conda env create -f ${SCRIPT_DIR}/lstein_stable_diffusion/environment.yaml
fi

echo === Activating conda environment "ldm"
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ldm

echo === Installing py-cord
python -m pip install py-cord==2.1.3

echo === Preloading models for stable diffusion
python ${SCRIPT_DIR}/lstein_stable_diffusion/scripts/preload_models.py

if [ -f "${SCRIPT_DIR}/config.ini" ]; then
    echo === config.ini file exists
else    
    echo Creating default config.ini
    echo \# .ini > config.ini
    echo [auth] > config.ini
    echo DISCORD_TOKEN=PASTE_YOUR_TOKEN_HERE > config.ini
fi

