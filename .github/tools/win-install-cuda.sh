#!/bin/bash

set -ex

if [ -z "${CUDA_WINDOWS}" ]; then
    echo "CUDA_WINDOWS variable empty"
    exit 1
fi

mkdir -p cache

url=https://developer.download.nvidia.com/compute/cuda/${CUDA_WINDOWS/_*/}/local_installers/cuda_${CUDA_WINDOWS}_windows.exe
filename=./cache/${url##*/}

wget -nc -q -O $filename "${url}"
md5sum -c .github/tools/win-cuda.md5

./$filename /s
