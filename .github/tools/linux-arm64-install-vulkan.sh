#!/bin/bash

set -e

OS=$(lsb_release -c | cut -f 2)

sudo apt update

sudo apt install libvulkan-dev
sudo apt install glslang-dev
sudo apt install spirv-tools
sudo apt remove libllvm12

VULKAN_VERSION=$(dpkg -s vulkan-sdk | awk '/^Version:/ {split($2, a, /\./); printf "%d.%d.%d", a[1], a[2], a[3]}')
echo "VULKAN_VERSION=$VULKAN_VERSION" >> $GITHUB_ENV
