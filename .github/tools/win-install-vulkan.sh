#!/bin/bash

set -ex

if [ -z "${VULKAN_WINDOWS}" ]; then
    echo "VULKAN_WINDOWS variable empty"
    exit 1
fi

mkdir -p cache

url=https://sdk.lunarg.com/sdk/download/${VULKAN_WINDOWS}/windows/VulkanSDK-${VULKAN_WINDOWS}-Installer.exe
filename=./cache/${url##*/}

wget -nc -q -O $filename "${url}"
md5sum -c .github/tools/win-cuda.md5

./$filename /S
