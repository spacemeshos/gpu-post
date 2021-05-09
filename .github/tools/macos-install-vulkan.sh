#!/bin/bash

set -e

VERSION=1.2.170.0
url=https://sdk.lunarg.com/sdk/download/$VERSION/mac/vulkansdk-macos-$VERSION.dmg

filename=${url##*/}
wget -q -O $filename "${url}?u="
shasum -c .github/tools/vulkan.sha256
sudo hdiutil attach $filename
cd /Volumes/vulkansdk-macos-$VERSION
python install_vulkan.py

VULKAN_ROOT_LOCATION=$PWD
VULKAN_SDK=${VULKAN_ROOT_LOCATION}/macOS
echo "VULKAN_ROOT_LOCATION=$VULKAN_ROOT_LOCATION" >> $GITHUB_ENV
echo "VULKAN_SDK_VERSION=$VERSION" >> $GITHUB_ENV
echo "VULKAN_SDK=$VULKAN_SDK" >> $GITHUB_ENV
echo "VK_ICD_FILENAMES=${VULKAN_SDK}/etc/vulkan/icd.d/MoltenVK_icd.json" >> $GITHUB_ENV
echo "VK_LAYER_PATH=${VULKAN_SDK}/etc/vulkan/explicit_layers.d" >> $GITHUB_ENV
echo "PATH=${VULKAN_SDK}/bin:$PATH" >> $GITHUB_ENV
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:${VULKAN_SDK}/lib" >> $GITHUB_ENV
