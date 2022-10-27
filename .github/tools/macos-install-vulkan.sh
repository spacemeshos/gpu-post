#!/bin/bash

set -e

if [ -z "${VULKAN_MAC}" ]; then
    echo "VULKAN_MAC variable empty"
    exit 1
fi

url=https://sdk.lunarg.com/sdk/download/$VULKAN_MAC/mac/vulkansdk-macos-$VULKAN_MAC.dmg

filename=${url##*/}
wget -q -O $filename "${url}"
shasum -c .github/tools/vulkan.sha256
sudo hdiutil attach $filename

cd /Volumes/vulkansdk-macos-$VULKAN_MAC
./InstallVulkan.app/Contents/MacOS/InstallVulkan --accept-licenses --default-answer --confirm-command install

cd /Users/runner/VulkanSDK/$VULKAN_MAC
python install_vulkan.py

VULKAN_ROOT_LOCATION=$PWD
VULKAN_SDK=${VULKAN_ROOT_LOCATION}/macOS
echo "VULKAN_ROOT_LOCATION=$VULKAN_ROOT_LOCATION" >> $GITHUB_ENV
echo "VULKAN_SDK_VERSION=$VULKAN_MAC" >> $GITHUB_ENV
echo "VULKAN_SDK=$VULKAN_SDK" >> $GITHUB_ENV
echo "VK_ICD_FILENAMES=${VULKAN_SDK}/etc/vulkan/icd.d/MoltenVK_icd.json" >> $GITHUB_ENV
echo "VK_LAYER_PATH=${VULKAN_SDK}/etc/vulkan/explicit_layers.d" >> $GITHUB_ENV
echo "PATH=${VULKAN_SDK}/bin:$PATH" >> $GITHUB_ENV
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${VULKAN_SDK}/lib" >> $GITHUB_ENV
