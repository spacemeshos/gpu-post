#!/bin/bash

set -e


if [ -z "${VULKAN_LINUX}" ]; then
    echo "VULKAN_LINUX variable empty"
    exit 1
fi

OS=$(lsb_release -c | cut -f 2)
wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-${OS}.list http://packages.lunarg.com/vulkan/${VULKAN_LINUX}/lunarg-vulkan-${VULKAN_LINUX}-${OS}.list

sudo apt update
sudo apt install -y vulkan-sdk

VULKAN_VERSION=$(dpkg -s vulkan-sdk | awk '/^Version:/ {split($2, a, /\./); printf "%d.%d.%d", a[1], a[2], a[3]}')
echo "VULKAN_VERSION=$VULKAN_VERSION" >> $GITHUB_ENV
