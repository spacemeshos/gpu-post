# GPU Proof of Spacemesh Time Init (aka Smeshing Setup) Library

[![Build](https://github.com/spacemeshos/gpu-post/actions/workflows/build.yml/badge.svg)](https://github.com/spacemeshos/gpu-post/actions/workflows/build.yml)

## Current functionality

A C library implementing the POST API setup method for general-purpose CPUs and for CUDA and Vulkan compute processors.

## Runtime System Requirements

Windows 10/11, macOS or Linux.
One or more of the following:

- A GPU and drivers with CUDA support (minimum compute compatibility 5.0, maximum compute compatibility 9), such as a modern Nvidia GPU and Nvidia drivers version R525 or newer.
- A GPU and drivers with Vulkan 1.3 support such as a modern AMD, Apple M1 processor, and Intel GPUs.
- A x86-64 cpu such as AMD or Intel CPUs.
- A ARM 64 bit cpu such as Apple Silicon or Ampere Altra
- Both discrete and on-board GPUs are supported as long as they support the minimum CUDA or Vulkan runtime version.

We currently provide release binaries and build instructions for Windows, Mac and Ubuntu 22.04 but the library can be built on other Linux distros for usage on these systems.

## GPU Memory Requirements

### Minimum GPU RAM

- 16 KiB per CUDA core for CUDA
- 4 MiB per compute unit for Vulkan

### Recommended GPU RAM

- 2080 MiB

### Runtime linux requirements

On Linux platforms with [Hybrid](https://wiki.archlinux.org/title/hybrid_graphics) Nvidia GPU setup please use Nvidia driver R525 or newer. Older ones are known to have compatibility issues. Non hybrid cards are confirmed to be working with R520 and older versions.

---

## Build System Requirements

### All Platforms

- For building CUDA support: NVIDIA [Cuda Toolkit 11](https://developer.nvidia.com/cuda-11.0-download-archive), an NVIDIA GPU with CUDA support, and an [Nvdia driver](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/) version R525 or newer.
  - If building on Linux you should refer to the distribution preferred method installation if available
- For building Vulkan support: [Vulkan SDK 1.3](https://vulkan.lunarg.com/sdk/home) and a GPU with Vulkan 1.3 runtime support.

### Windows

- Windows 10/11.
- Microsoft Visual Studio 2022
- You may also need to install specific versions of the Windows SDK when prompted when attempting to build the library for the first time.

### Ubuntu

- Ubuntu 22.04
- Cmake, GCC 11+

### Other linux distributions

- Cmake, GCC 11+

### macOS

- Xcode
- Xcode Command Line Dev Tools
- Cmake, GCC 11+

### macOS Dev Env Setup

1. Install latest version of Xcode with the command line dev tools.
1. Download the Vulkan 1.3 sdk installer for macOS from <https://vulkan.lunarg.com/sdk/home#mac>
1. Install Vulkan SDK with the Vulkan installer.
1. Change directory to the folder where the SDK is installed (default `$ cd $HOME/VulkanSDK/1.3.xxx`) and run the install script with `$ sudo ./install_vulkan.py`
1. Add the Vulkan env vars to your `.bash_profile` file with the root location set to the sdk directory on your hard-drive. For example, if Vulkan sdk 1.2.154 is installed then the env vars should be set like this:

```bash
export VULKAN_SDK_VERSION="1.3.xxx"                   # Replace xxx with actual version
export VULKAN_ROOT_LOCATION="$HOME/VulkanSDK/1.3.xxx" # adapt to install location on your machine
export VULKAN_SDK="$VULKAN_ROOT_LOCATION/macOS"
export VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layers.d"
export PATH="/usr/local/opt/python/libexec/bin:$VULKAN_SDK/bin:$PATH"
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$VULKAN_SDK/lib/"
```

---

## Build Configuration

Default build configuration:

### Windows and Linux

```c
SPACEMESHCUDA   "Build with CUDA support"   default: ON
SPACEMESHVULKAN "Build with Vulkan support" default: ON
```

### macOS Build Configuration

```c
SPACEMESHCUDA   "Build with CUDA support"   default: OFF
SPACEMESHVULKAN "Build with Vulkan support" default: ON
```

---

## Building

To build the library with full support for both CUDA and Vulkan on Windows or on Linux use a system with an Nvidia GPU and drivers. Otherwise, turn off CUDA support and build for Vulkan only. Building on macOS only supports Vulkan.

### Building on Windows

1. Open project folder into Visual Studio 2022: `File -> Open -> Folder`.
2. Set `x64-Release` Project Settings.
3. Build: `CMake -> Rebuild All`.
4. Run test: `CMake -> Debug from Build Folder -> gpu-setup-test.exe`

### Ubuntu or macOS

If using VULKAN, make sure to clone the zlib submodule:

```bash
git submodule update --init
```

Configure your build using the default configuration:

```bash
cmake -B build
```

To disable CUDA use:

```bash
cmake -B build -DSPACEMESHCUDA=OFF
```

To disable VULKAN use:

```bash
cmake -B build -DSPACEMESHVULKAN=OFF
```

Build the project:

```bash
cmake --build build
```

Run the tests:

```bash
./build/test/gpu-setup-test -t
./build/test/gpu-setup-test -u
./build/test/gpu-setup-test -b
```

## Running the Test App

### macOS Configuration

1. Since the test app is not notarized, you may need to enable it via `spctl --add /path/to/gpu-setup-test` or by right-click-open it and click `open`.
2. Set execute permissions if not already set, e.g., `chmod a+x gpu-setup-test`
3. Add the test app's path to the dynamic lib search path, e.g., `export DYLD_LIBRARY_PATH=.`

### Linux Configuration

1. Set execute permissions if not already set, e.g., `chmod a+x gpu-setup-test`
2. Add the test app's path to the dynamic lib search path, e.g., `export LD_LIBRARY_PATH=.`

Run from the console to print usage:

```bash
$ gpu-setup-test
Usage:
--list               or -l                 print available providers
--benchmark          or -b                 run benchmark
--core               or -c                 test the core library use case
--test               or -t                 run basic test
--test-vector-check                        run a CPU test and compare with test-vector
--test-pow           or -tp                test pow computation
--test-leafs-pow     or -tlp               test pow computation while computing leafs
--unit-tests         or -u                 run unit tests
--integration-tests  or -i                 run integration tests
--label-size         or -s <1-256>         set label size [1-256]
--labels-count       or -n <1-32M>         set labels count [up to 32M]
--reference-provider or -r <id>            the result of this provider will be used as a reference [default - CPU]
--print              or -p                 print detailed data comparison report for incorrect results
--pow-diff           or -d <0-256>         count of leading zero bits in target D value [default - 16]
--srand-seed         or -ss <unsigned int> set srand seed value for POW test: 0 - use zero id/seed [default], -1 - use random value
--solution-idx       or -si <unsigned int> set solution index for POW test: index will be compared to be the found solution for Pow [default - unset]
                        -N <scrypt N>      set scrypt parameter N [default - 512]
```

## Mixing CUDA and Vulkan

By default, the library does not detect supported Vulkan GPUs if CUDA GPUs are detected. This behavior can be changed using two environment variables:

```commandline
SPACEMESH_DUAL_ENABLED
 empty or 0 - default behavior
 1 - detect Vulkan GPUs even if CUDA GPUs are detected
```

```commandline
SPACEMESH_PROVIDERS_DISABLED
 empty - default behavior
 "cuda" - do not detect CUDA GPUs
 "vulkan" - do not detect Vulkan GPUs
```

## Runtime Providers Recommendations

The library supports multiple compute providers at runtime. For best performance, use the following providers based on your OS and GPU:

| OS / GPU | Windows | Linux  | macOS   |
| -------- | ------- | -------| ------- |
| Nvidia   | CUDA    | CUDA   | Vulkan  |
| AMD      | Vulkan  | Vulkan | Vulkan  |
| Intel    | Vulkan  | Vulkan | Vulkan  |
| Apple M1 | Vulkan  | Vulkan | Vulkan  |

---

## API

Compute leaves and/or pow solution:

```c
int scryptPositions(
   uint32_t provider_id,      // POST compute provider ID
   const uint8_t *id,         // 32 bytes
   uint64_t start_position,   // e.g. 0
   uint64_t end_position,     // e.g. 49,999
   uint32_t hash_len_bits,    // (1...256) for each hash output, the number of prefix bits (not bytes) to copy into the buffer
   const uint8_t *salt,       // 32 bytes
   uint32_t options,          // compute leafs and/or compute pow
   uint8_t *out,              // memory buffer large enough to include hash_len_bits * number of requested hashes
   uint32_t N,                // scrypt N
   uint32_t R,                // scrypt r
   uint32_t P,                // scrypt p
   uint8_t *D,                // Target D for the POW computation. 256 bits.
   uint64_t *idx_solution,    // index of output where output < D if POW compute was on. MAX_UINT64 otherwise.
   uint64_t *hashes_computed, // The number of hashes computed, should be equal to the number of requested hashes.
   uint64_t *hashes_per_sec   // Performance
);
```

### Supported scrypt parameters

The api currently only supports the following N, P, R scrypt params.

- Supported N values: 1 - 28835
- Supported R values: 1
- Supported P values: 1

Gets the system's GPU capabilities. E.g. CUDA and/or NVIDIA or NONE:

```c
int stats();
```

Stops all GPU work and don’t fill the passed-in buffer with any more results:

```c
int stop(
 uint32_t ms_timeout   // timeout in milliseconds
);
```

Returns non-zero if stop in progress:

```c
SPACEMESHAPI int spacemesh_api_stop_inprogress();
```

Returns POS compute providers info:

```c
SPACEMESHAPI int spacemesh_api_get_providers(
 PostComputeProvider *providers, // out providers info buffer, if NULL - returns count of available providers
 int max_providers// buffer size
);
```

---

## Linking

1. Download release artifacts from a github release in this repo for your platform or build the artifacts from source code.
1. Copy all artifacts to your project resources directory. The files should be included in your app's runtime resources.
1. Use api.h to link the library from your code.

---

## Testing

Integration test of the basic library use case in a Spacemesh full node to generate proof of space and find a pow solution:

```bash
/build/test/.gpu-setup-test -c -n 100663296 -d 20
```

## Community Benchmarks

> Disclaimer: these are community submitted benchmarks which haven't been verified. Your milage may vary. The library is also likely to have bugs, is in alpha quality and the gpu-post algorithm is likely to change before the release of the Spacemesh 0.2 testnet.

```bash
gpu-setup-test -b -n 2000000
```

| Date        | Reporter  | Release | Compute Provider                                                               | OS & CPU                                                                           | Type   | Driver                     | mh/s |
| ----------- | --------- | ------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- | ------ | -------------------------- | ---- |
| 06/21/2021  | Obsidian  | v0.1.20 | Geforce RTX 2080ti 11GB @ stock (1350 mhz / 7000 mhz)                          | Windows 10 Pro v20H2, Build 19042.985, Intel i7-6700K @ 4.6ghz (HT enabled: 4c/8t) | CUDA   | NVIDIA 466.11              | 2.56 |
| 06/22/2021  | Scerbera  | v0.1.20 | Geforce RTX 2060 SUPER                                                         | Windows 10                                                                         | CUDA   | NVIDIA 466.11              | 1.7  |
| 06/22/2021  | Scerbera  | v0.1.20 | AMD Radeon Pro WX 7100                                                         | Windows 10                                                                         | CUDA   | NVIDIA 466.11              | 0.88 |
| 06/22/2021  | Scerbera  | v0.1.20 | RX VEGA 64  - Core Clock 1500 MHz - Memory Clock 960MHz                        | Intel i7-8700K Windows 10                                                          | Vulkan | Pro 20.Q4                  | 0.9  |
| 06/22/2021  | Scerbera  | v0.1.20 | WX7100 - Core Clock 1250MHz - Memory Clock 1700 MHz                            | Intel i7-8700K Windows 10                                                          | Vulkan | Pro 20.Q4                  | 0.87 |
| 06/28/2021  | cmoetzing | v0.1.20 | MSI GeForce RTX 2060 VENTUS GP OC - Core Clock 1365MHz - Memory Clock 1750 MHz | Ubuntu 20.04 Core i5-11600k                                                        | CUDA   | NVIDIA 465.19.01           | 1.36 |
| 06/29/2021  | avive     | v0.1.21 | GeForce RTX 3090                                                               | Ubuntu 20.04                                                                       | CUDA   | Nvidia 460.80              | 4.97 |
| 06/29/2021  | avive     | v0.1.21 | GeForce RTX 3080                                                               | Ubuntu 20.04                                                                       | CUDA   | Nvidia 460.80              | 4.08 |
| 06/30/2021  | shanyaa   | v0.1.21 | GeForce RTX 3070 @ 1.9 Ghz core, 6.8 Ghz mem                                   | Windows 10 / AMD Ryzen 5800X                                                       | CUDA   | Nvidia 466.63              | 2.7  |
| 06/30/2021  | shanyaa   | v0.1.21 | GeForce RTX 3070 @ 2 Ghz core, 8.08 Ghz mem                                    | Windows 10 / AMD Ryzen 5800X                                                       | CUDA   | Nvidia 466.63              | 3.43 |
| 07/01/2021  | avive     | v0.1.21 | [Nvdia CMP 30HX](https://www.nvidia.com/en-us/cmp/)                            | Ubuntu 20.04.2 LTS                                                                 | CUDA   | Nvidia 460.80              | 1.45 |
| 07/01/2021  | avive     | v0.1.21 | GeForce RTX 2060                                                               | Ubuntu 20.04.2 LTS                                                                 | CUDA   | Nvidia 465.27              | 1.56 |
| 07/01/2021  | shanyaa   | v0.1.21 | Intel Iris Xe (integrated graphics)                                            | Windows 10 / Intel core i7 1165G7                                                  | Vulkan | Intel 27.20.100.9565       | 0.28 |
| 07/03/2021  | neodied   | v0.1.21 | Radeon 5700XT @ 1333 MHz core, 1824 MHz mem                                    | Windows 10 / Intel core i7 9700K                                                   | Vulkan | AMD Radeon Software 21.6.1 | 1.38 |
| 07/03/2021  | neodied   | v0.1.21 | Radeon 5700XT @ 2016 MHz core, 1748 MHz mem                                    | Windows 10 / Intel core i7 9700K                                                   | Vulkan | AMD Radeon Software 21.6.1 | 1.87 |
| 12/21/2022  | lane      | v0.1.28 | Apple M1 (built-in, 8 cores, Metal 3)     | macOS 13.1                          | Vulkan | N/A | 0.15 |
| 01/27/2023  | lane      | v0.1.28 | Apple M2 Pro (built-in, 16 GPU cores, Metal 3)     | macOS 13.2                          | Vulkan | N/A | 0.56 |
| 01/27/2023  | nj      | v0.1.28 | Apple M2 Pro (built-in, 19 GPU cores, Metal 3)     | macOS 13.2                          | Vulkan | N/A | 0.57 |

## Prerelease Benchmarks

Scrypt Benchmarks (n=512, r=1, p=1) 1 byte per leaf, batch size leaves per API call.

| Date       | Reporter | impl      | cpu / gpu                        | Host OS                                              | notes                                  | kh/s  | mh/s  | x factor over 1 4ghz cpu native thread | x factor over 12 4ghz cpu native threads |
| ---------- | -------- | --------- | -------------------------------- | ---------------------------------------------------- | -------------------------------------- | ----- | ----- | -------------------------------------- | ---------------------------------------- |
| 11/19/2019 | ae       | go-scrypt | mbp + Intel i9 @ 2.9ghz - 1 core | OS X                                                 | go scrypt crypto lib (not scrypt-jane) |     7 | 0.01  | 1                                      | 1                                        |
| 11/19/2019 | ae       | sm-scrypt | Ryzen 5 2600x @ 4ghz - 1 core    | Windows 10                                           | scrypt-jane c code                     |     7 | 0.01  | 1                                      | 1                                        |
| 11/19/2019 | ae       | sm-scrypt | Nvidia Geforce RTX 2070 8GB      | Windows 10                                           | pre-optimized prototype                | 1,920 | 1.92  | 290                                    | 24.17                                    |
| 11/19/2019 | ae       | sm-scrypt | AMD Radeon RX 580                | Windows 10                                           | pre-optimized prototype                |   500 | 0.50  | 76                                     | 6.29                                     |
| 11/19/2019 | ar       | sm-scrypt | Nvidia GTX 1060 6G               | Windows 10                                           | pre-optimized prototype                |   979 | 0.98  | 148                                    | 12.32                                    |
| 11/19/2019 | ar       | sm-scrypt | AMD Radeon 570 4GB               | Windows 10                                           | pre-optimized prototype                |   355 | 0.36  | 54                                     | 4.47                                     |
| 11/12/2019 | ae       | sm-scrypt | AMD Radeon RX 580                | Windows 10                                           | optimized prototype                    |   926 | 0.93  | 140                                    | 11.65                                    |
| 11/12/2019 | ae       | sm-scrypt | AMD Radeon RX 580                | Ubuntu 18.0.4.3 LTS                                  | optimized prototype                    |   893 | 0.89  | 135                                    | 11.24                                    |
| 11/12/2019 | ae       | sm-scrypt | Nvidia Geforce RTX 2070 8GB      | Ubuntu 19.10 LTS                                     | optimized prototype                    | 1,923 | 1.92  | 292                                    | 24.37                                    |
| 01/22/2020 | seagiv   | sm-scrypt | Nvidia GTX 1060 6G               | Windows 10                                           | vulkan pre-optimized prototype         |   276 |       |                                        |                                          |
| 01/22/2020 | seagiv   | sm-scrypt | AMD Radeon 570 4GB               | Windows 10                                           | vulkan pre-optimized prototype         |   269 |       |                                        |                                          |
| 01/27/2020 | seagiv   | sm-scrypt | Nvidia GTX 1060 6G               | Windows 10                                           | vulkan optimized prototype             |   642 |       |                                        |                                          |
| 01/27/2020 | seagiv   | sm-scrypt | AMD Radeon 570 4GB               | Windows 10                                           | vulkan optimized prototype             |   966 |       |                                        |                                          |
| 01/29/2020 | seagiv   | sm-scrypt | AMD Radeon Pro 555x 4GB          | macOS 10.14.6                                        | vulkan optimized prototype             |   266 |       |                                        |                                          |
| 01/31/2020 | avive    | sm-scrypt | AMD Radeon Pro 560x 4GB          | macOS 10.14.6                                        | vulkan optimized prototype             |   406 |       |                                        |                                          |
| 01/31/2020 | avive    | sm-scrypt | Intel(R) UHD Graphics 630 1536MB | macOS 10.14.6                                        | vulkan optimized prototype             |    53 |       |                                        |                                          |
| 05/06/2020 | avive    | sm-scrypt | AMD Radeon RX 580                | Windows 10                                           | vulkan optimized prototype             | 1,074 | 1.074 |                                        |                                          |
| 09/08/2020 | avive    | sm-scrypt | Nvidia Tesla V 100 (16GB)        | Ubuntu 20.04 NVIDIA-SMI 450.51.06 CUDA Version: 11.0 | CUDA optimized prototype               | 4,166 | 4.166 |                                        |                                          |
| 09/08/2020 | avive    | sm-scrypt | Nvidia Tesla T4 (16GB)           | Ubuntu 20.04 NVIDIA-SMI 450.51.06 CUDA Version: 11.0 | CUDA optimized prototype               | 1,252 | 1.252 |                                        |                                          |
| 09/08/2020 | avive    | sm-scrypt | Nvidia Tesla P100-PCIE (32GB)    | Ubuntu 20.04 NVIDIA-SMI 450.51.06 CUDA Version: 11.0 | CUDA optimized prototype               | 2,083 | 2.083 |                                        |                                          |
| 09/08/2020 | avive    | sm-scrypt | Nvidia Tesla P4 (32GB)           | Ubuntu 20.04 NVIDIA-SMI 450.51.06 CUDA Version: 11.0 | CUDA optimized prototype               |   757 | 0.75  |                                        |                                          |
| 04/04/2020 | avive    | sm-scrypt | Apple M1                         | MacOS 11.2                                           | vulkan optimized prototype             |   214 | 0.214 |                                        |                                          |
| 04/21/2020 | avive    | sm-scrypt | Nvidia RTX 2070 Super, 8GB       | Ubuntu 20.04, Driver 460.73.01                       | CUDA optimized prototype               | 2,038 | 2.038 |                                        |                                          |

## 3rd Party Vulkan and CUDA Benchmarks

The library performance on a GPU depends on the GPU's CUDA and Vulkan performance. The following benchmarks are available from geekbench:

- [Geekbench Cuda Benchmarks](https://browser.geekbench.com/cuda-benchmarks)
- [Geekbench Vulkan Benchmarks](https://browser.geekbench.com/vulkan-benchmarks)
