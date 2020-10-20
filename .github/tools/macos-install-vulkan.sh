#!/bin/bash

set -e

VERSION=1.2.141.0
url=https://sdk.lunarg.com/sdk/download/$VERSION/mac/vulkansdk-macos-$VERSION.dmg

filename=${url##*/}
wget -q -O $filename $url
shasum -c .github/tools/vulkan.sha256
sudo hdiutil attach $filename
cd /Volumes/vulkansdk-macos-$VERSION
python install_vulkan.py

VULKAN_ROOT_LOCATION=$PWD
VULKAN_SDK=${VULKAN_ROOT_LOCATION}/macOS
echo "::set-env name=VULKAN_ROOT_LOCATION::$VULKAN_ROOT_LOCATION"
echo "::set-env name=VULKAN_SDK_VERSION::$VERSION"
echo "::set-env name=VULKAN_SDK::$VULKAN_SDK"
echo "::set-env name=VK_ICD_FILENAMES::${VULKAN_SDK}/etc/vulkan/icd.d/MoltenVK_icd.json"
echo "::set-env name=VK_LAYER_PATH::${VULKAN_SDK}/etc/vulkan/explicit_layers.d"
echo "::set-env name=PATH::${VULKAN_SDK}/bin:$PATH"
echo "::set-env name=DYLD_LIBRARY_PATH::$DYLD_LIBRARY_PATH:${VULKAN_SDK}/lib"
