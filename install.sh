#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if { conda env list | grep 'ldm'; } >/dev/null 2>&1; then
    conda env update -f ${SCRIPT_DIR}/lstein_stable_diffusion/environment.yaml --prune
else
    conda env create -f ${SCRIPT_DIR}/lstein_stable_diffusion/environment.yaml
fi

CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh
conda activate ldm

python -m pip install py-cord==2.1.3

python ${SCRIPT_DIR}/lstein_stable_diffusion/scripts/preload_models.py

