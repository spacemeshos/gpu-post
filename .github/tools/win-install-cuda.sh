#!/bin/bash

set -ex

if [ -z "${CUDA_WINDOWS}" ]; then
    echo "CUDA_WINDOWS variable empty"
    exit 1
fi

mkdir -p cache
cd cache

url=https://developer.download.nvidia.com/compute/cuda/${CUDA_WINDOWS/_*/}/local_installers/cuda_${CUDA_WINDOWS}_windows.exe
filename=${url##*/}

/C/msys64/usr/bin/wget.exe -nc -O $filename "${url}"
# md5sum -c $GITHUB_WORKSPACE/.github/tools/win-cuda.md5

./$filename /s
