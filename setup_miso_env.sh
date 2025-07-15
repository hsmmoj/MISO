#!/bin/bash
set -e

# Quick system update and a few essentials
apt-get update
apt-get install -y wget bzip2 git mesa-utils

# Miniconda install (if it's not already there)
if ! command -v conda &> /dev/null; then
    echo "Miniconda not found. Installing it now..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/conda
    export PATH="/opt/conda/bin:$PATH"
    echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
else
    echo "Miniconda is already installed."
    export PATH="/opt/conda/bin:$PATH"
fi

# Clean out any old envs and make a fresh one
/opt/conda/bin/conda env remove -n miso || true
/opt/conda/bin/conda env create -f environment.yaml

# Fire up the new conda environment
source /opt/conda/bin/activate miso

# Install the right PyTorch, torchvision, and torchaudio (CUDA 11.8 builds)
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0 --extra-index-url https://download.pytorch.org/whl/cu118

# Grab PyTorch3D straight from GitHub
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Editable install of your project (assumes you run this from project root)
pip install -e .

echo
echo "Setup is done!"
echo "Here's the numpy and torch versions you got:"
conda list | grep numpy
conda list | grep torch

# Drop you into the environment so you can get to work
exec bash --rcfile <(echo "source /opt/conda/etc/profile.d/conda.sh; conda activate miso")
