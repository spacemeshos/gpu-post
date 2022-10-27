#!/bin/bash

set -e

if [ -z "${CUDA_LINUX}" ]; then
    echo "CUDA_LINUX variable empty"
    exit 1
fi

OS=$(lsb_release -r | cut -f 2 | tr -d .)
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS}/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS}/x86_64/ /"

sudo apt update
sudo apt install -y cuda-toolkit-${CUDA_LINUX/./-}
