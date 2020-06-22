#!/bin/bash

set -e

OS=$(lsb_release -c | cut -f 2)
wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-${OS}.list http://packages.lunarg.com/vulkan/lunarg-vulkan-${OS}.list

sudo apt update
sudo apt install -y vulkan-sdk

VULKAN_VERSION=$(dpkg -s vulkan-sdk | awk '/^Version:/ {split($2, a, /\./); printf "%d.%d.%d", a[1], a[2], a[3]}')
echo "::set-env name=VULKAN_VERSION::$VULKAN_VERSION"
