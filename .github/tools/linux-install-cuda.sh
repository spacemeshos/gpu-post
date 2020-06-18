#!/bin/bash

set -e

if [ -z "${OS}" ]; then
    echo "OS variable empty"
    exit 1
fi
if [ -z "${CUDA}" ]; then
    echo "CUDA variable empty"
    exit 1
fi

OSVERSION=$(echo $OS | sed 's/^ubuntu-//; s/\.//')
sudo wget -qO /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OSVERSION}/x86_64/cuda-ubuntu${OSVERSION}.pin
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OSVERSION}/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OSVERSION}/x86_64/ /"

sudo apt update
sudo apt install -y cuda-toolkit-${CUDA/./-}
