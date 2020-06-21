#!/bin/bash -x

set -e

url=https://sdk.lunarg.com/sdk/download/1.1.130.0/mac/vulkansdk-macos-1.1.130.0.tar.gz?Human=true
checksum=d6d80ab96e3b4363be969f9d256772e9cfb8f583db130076a9a9618d2551c726

wget $url
shasum -c .github/tools/vulkan.sha256
tar zxf ${url##*/}
