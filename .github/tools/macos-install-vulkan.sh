#!/bin/bash

set -e

url=https://sdk.lunarg.com/sdk/download/1.1.130.0/mac/vulkansdk-macos-1.1.130.0.tar.gz
checksum=d6d80ab96e3b4363be969f9d256772e9cfb8f583db130076a9a9618d2551c726

wget -q -O ${url##*/} ${url}?Human=true
shasum -c .github/tools/vulkan.sha256
tar zxf ${url##*/}

VULKAN_ROOT_LOCATION=$PWD
VULKAN_SDK_VERSION=1.1.130.0
VULKAN_SDK=${VULKAN_ROOT_LOCATION}/vulkansdk-macos-${VULKAN_SDK_VERSION}/macOS
echo "::set-env name=VULKAN_ROOT_LOCATION::$VULKAN_ROOT_LOCATION"
echo "::set-env name=VULKAN_SDK_VERSION::$VULKAN_SDK_VERSION"
echo "::set-env name=VULKAN_SDK::$VULKAN_SDK"
echo "::set-env name=VK_ICD_FILENAMES::${VULKAN_SDK}/etc/vulkan/icd.d/MoltenVK_icd.json"
echo "::set-env name=VK_LAYER_PATH::${VULKAN_SDK}/etc/vulkan/explicit_layers.d"
echo "::set-env name=PATH::${VULKAN_SDK}/bin:$PATH"
echo "::set-env name=DYLD_LIBRARY_PATH::$DYLD_LIBRARY_PATH:${VULKAN_SDK}/lib"
