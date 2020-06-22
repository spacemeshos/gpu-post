#!/bin/bash

set -e

if [ -z "${CUDA}" ]; then
    echo "CUDA variable empty"
    exit 1
fi

OS=$(lsb_release -r | cut -f 2 | tr -d .)
sudo wget -qO /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS}/x86_64/cuda-ubuntu${OS}.pin
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS}/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS}/x86_64/ /"

sudo apt update
sudo apt install -y cuda-toolkit-${CUDA/./-}
