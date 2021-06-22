# GPU Proof of Spacemesh Time Init (aka Smeshing Setup) Library

[![Build](https://github.com/spacemeshos/gpu-post/actions/workflows/build.yml/badge.svg)](https://github.com/spacemeshos/gpu-post/actions/workflows/build.yml)

## Current functionality
A c library implementing the POST API setup method for general-purpose CPUs and for CUDA and Vulkan compute processors.

## Runtime System Requirements
Windows 10, macOS or Ubuntu.
One or more of the following processors:
- A GPU and drivers with CUDA 11.0 support such as a modern Nvidia GPU and Nvidia drivers version R450 or later.
- A GPU and drivers with Vulkan 1.2 support such as a modern AMD, Apple M1 processor, and Intel GPUs.
- A x86-64 cpu such as AMD or Intel cpus.



- Both discrete and on-board GPUs are supported as long as they support the minimum CUDA or Vulkan runtime version.
- We currently provide release binaries and build instructions for Ubuntu 20.04 but the library can be built on other Linux distros for usage on these systems.

---

## Build System Requirements

### All Platforms
- For building CUDA support: NVIDIA Cuda Tookit 11, an NVIDIA GPU with CUDA support, and an Nvdia driver version R450 or newer.
- For building Vulkan support: Vulkan SDK 1.2 and a GPU with Vulkan 1.2 runtime support.

### Windows
- Windows 10 Pro.
- Microsoft Visual Studio 2017 (any edition). Visual Studio 2019 is NOT supported.
- You may also need to install specific versions of the Windows SDK when prompted when attempting to build the library for the first time.

### Ubuntu
- Ubuntu 20.04
- Cmake, GCC 7

### macOS
- Xcode
- Xcode Command Line Dev Tools
- Cmake, GCC 7

### macOS Dev Env Setup
1. Install latest version of Xcode with the command line dev tools.
1. Download the Vulkan 1.2 sdk installer for macOS from https://vulkan.lunarg.com/sdk/home#mac
1. Copy Vulkan SDK from the Vulkan installer volume to a directory in your hard-drive.
1. Install the SDK from your hard-drive directory and not from the installer volume by running `$ sudo ./install_vulkan.py`.
1. Add the Vulkan env vars to your `.bash_profile` file with the root location set to the sdk directory on your hard-drive. For example, if Vulkan sdk 1.2.154 is installed then the env vars should be set like this:

```bash
export VULKAN_SDK_VERSION="1.2.154.0"
export VULKAN_ROOT_LOCATION="$HOME/dev/vulkan-sdk-1.2.154"
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

### macOS
```c
SPACEMESHCUDA   "Build with CUDA support"   default: OFF
SPACEMESHVULKAN "Build with Vulkan support" default: ON
```

---

## Building

To build the library with full support for both CUDA and Vulkan on Windows or on Linux use a system with an Nvidia GPU and drivers. Otherwise, turn off CUDA support and build for Vulkan only. Building on macOS only supports Vulkan.

### Windows
1. Open project folder into Visual Studio 2017: `File -> Open -> Folder`.
2. Set `x64-Release` Project Settings.
3. Build: `CMake -> Rebuild All`.
4. Run test: `CMake -> Debug from Build Folder -> gpu-setup-test.exe`

### Ubuntu or macOS
Create a build directory:
```bash
  cd gpu-post
  mkdir build
  cd build
```

Configure your build using the default configuration:
```bash
  cmake ..
```

To disable CUDA use:
```bash
  cmake .. -DSPACEMESHCUDA=OFF
```

To disable VULKAN use:
```bash
  cmake .. -DSPACEMESHVULKAN=OFF
```


Build the project:
```bash
  make
```

Run the tests:
```bash
  ./test/gpu-setup-test -t
  ./test/gpu-setup-test -u
  ./test/gpu-setup-test -b
```

----

## Running the Test App

### macOS Configuration
1. Since the test app is not notarized, you need to enable it via `spctl --add /path/to/gpu-setup-test` or by right-click-open it and click `open`.
2. Set execute permissions. e.g. `chmod a+x gpu-setup-test`
3. Add the test app's path to the dynamic lib search path, e.g. `export DYLD_LIBRARY_PATH=.`

### Linux Configuration
1. Set execute permissions. e.g. `chmod a+x gpu-setup-test`
2. Add the test app's path to the dynamic lib search path, e.g. `export DYLD_LIBRARY_PATH=.`

Run from the console to print usage:

```bash
$ gpu-setup-test
Usage:
--list               or -l	print available providers
--benchmark          or -b	run benchmark
--core               or -c	test the core library use case
--test               or -t	run basic test
--test-vector-check		run a CPU test and compare with test-vector
--test-pow           or -tp 	test pow computation
--unit-tests         or -u 	run unit tests
--integration-tests  or -i  	run integration tests
--label-size         or -s	<1-256>	set label size [1-256]
--labels-count       or -n	<1-32M>	set labels count [up to 32M]
--reference-provider or -r	<id> the result of this provider will be used as a reference [default - CPU]
--print              or -p	print detailed data comparison report for incorrect results
--pow-diff           or -d 	<0-256> count of leading zero bits in target D value [default - 16]
--srand-seed         or -ss	<unsigned int> set srand seed value for POW test: 0 - use zero id/seed [default], -1 - use random value
```

----

## Mixing CUDA and Vulkan

By default, the library does not detect supported Vulkan GPUs if CUDA GPUs are detected. This behavior can be changed using two environment variables:
```
SPACEMESH_DUAL_ENABLED
	empty or 0 - default behavior
	1 - detect Vulkan GPUs even if CUDA GPUs are detected
```
```
SPACEMESH_PROVIDERS_DISABLED
	empty - default behavior
	"cuda" - do not detect CUDA GPUs
	"vulkan" - do not detect Vulkan GPUs
```


## Runtime Providers Recommendations

The library supports multiple compute providers at runtime. For best performance, use the following providers based on your OS and GPU:

| OS / GPU     	| Windows 	| Linux      	| macOS        	|
|------------	|----------	|-----------	|--------------	|
| Nvidia	    | CUDA      | CUDA      	| Vulkan      	|
| AMD		    | Vulkan    | Vulkan      	| Vulkan      	|
| Intel		    | Vulkan    | Vulkan      	| Vulkan      	|
| Apple M1      | Vulkan    | Vulkan        | Vulkan        |

---

## API

Compute leaves and/or pow solution:

```c
int scryptPositions(
	uint32_t provider_id,	  // POST compute provider ID
	const uint8_t *id,		 // 32 bytes
    uint64_t start_position,   // e.g. 0
    uint64_t end_position,	 // e.g. 49,999
    uint32_t hash_len_bits,	// (1...256) for each hash output, the number of prefix bits (not bytes) to copy into the buffer
    const uint8_t *salt,	   // 32 bytes
    uint32_t options,		  // compute leafs and/or compute pow
    uint8_t *out,              // memory buffer large enough to include hash_len_bits * number of requested hashes
    uint32_t N,				// scrypt N
    uint32_t R,				// scrypt r
    uint32_t P,				// scrypt p
	uint8_t *D,				// Target D for the POW computation. 256 bits.
	uint64_t *idx_solution,	// index of output where output < D if POW compute was on. MAX_UINT64 otherwise.
	uint64_t *hashes_computed, // The number of hashes computed, should be equal to the number of requested hashes.
	uint64_t *hashes_per_sec   // Performance
	);
```

Gets the system's GPU capabilities. E.g. CUDA and/or NVIDIA or NONE:
```c
int stats();
```

Stops all GPU work and don’t fill the passed-in buffer with any more results:
```c
int stop(
	uint32_t ms_timeout			// timeout in milliseconds
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

---

## Community Benchmarks

```bash
gpu-setup-test -b -n 2000000
```

| Date       	| Reporter 	| Release  	| Compute Provider | OS & CPU            	| Driver                                  	| mh/s |
|------------	|----------	|-----------	|----------------------------------	|---------------------	|----------------------------------------	|-------|
| 06/21/2021 	| Obsidian   	| v0.1.20 	| Geforce RTX 2080ti 11GB @ stock (1350mhz / 7000mhz) | Windows 10 Pro v20H2, Build 19042.985,  Intel i7-6700K @ 4.6ghz (HT enabled: 4c/8t) |   Nvidia 466.11 | 2.56  
| 06/22/2021 	| Scerbera   	| v0.1.20 	| Gergorce RTX 3090 - Gigabyte Vision - Eth mining OC settings | Windows 10 |   Nvidia 466.11 | 279  
| 06/22/2021 	| Scerbera   	| v0.1.20 	| Gergorce RTX 2060 SUPER | Windows 10 |   Nvidia 466.11 | 1.7  
| 06/22/2021 	| Scerbera   	| v0.1.20 	| AMD Radeon Pro WX 7100 | Windows 10 |   Nvidia 466.11 | 0.88 

---

## Prerelease Benchmarks

Scrypt Benchmarks (n=512, r=1, p=1) 1 byte per leaf, batch size leaves per API call.

| Date       	| Reporter 	| impl      	| cpu / gpu                        	| Host OS             	| notes                                  	| kh/s  	| mh/s 	| x factor over 1 4ghz cpu native thread 	| x factor over 12 4ghz cpu native threads 	|
|------------	|----------	|-----------	|----------------------------------	|---------------------	|----------------------------------------	|-------	|------	|----------------------------------------	|------------------------------------------	|
| 11/19/2019 	| ae       	| go-scrypt 	| mbp + Intel i9 @ 2.9ghz - 1 core 	| OS X                	| go scrypt crypto lib (not scrypt-jane) 	| 7     	| 0.01 	| 1                                      	| 1                                        	|
| 11/19/2019 	| ae       	| sm-scrypt 	| Ryzen 5 2600x @ 4ghz - 1 core    	| Windows 10          	| scrypt-jane c code                     	| 7     	| 0.01 	| 1                                      	| 1                                        	|
| 11/19/2019 	| ae       	| sm-scrypt 	| Nvidia Geforce RTX 2070 8GB      	| Windows 10          	| pre-optimized prototype                	| 1,920 	| 1.92 	| 290                                    	| 24.17                                    	|
| 11/19/2019 	| ae       	| sm-scrypt 	| AMD Radeon RX 580                	| Windows 10          	| pre-optimized prototype                	| 500   	| 0.50 	| 76                                     	| 6.29                                     	|
| 11/19/2019 	| ar       	| sm-scrypt 	| Nvidia GTX 1060 6G               	| Windows 10          	| pre-optimized prototype                	| 979   	| 0.98 	| 148                                    	| 12.32                                    	|
| 11/19/2019 	| ar       	| sm-scrypt 	| AMD Radeon 570 4GB               	| Windows 10          	| pre-optimized prototype                	| 355   	| 0.36 	| 54                                     	| 4.47                                     	|
| 11/12/2019 	| ae       	| sm-scrypt 	| AMD Radeon RX 580                	| Windows 10          	| optimized prototype                    	| 926   	| 0.93 	| 140                                    	| 11.65                                    	|
| 11/12/2019 	| ae       	| sm-scrypt 	| AMD Radeon RX 580                	| Ubuntu 18.0.4.3 LTS 	| optimized prototype                    	| 893   	| 0.89 	| 135                                    	| 11.24                                    	|
| 11/12/2019 	| ae       	| sm-scrypt 	| Nvidia Geforce RTX 2070 8GB      	| Ubuntu 19.10 LTS    	| optimized prototype                    	| 1,923 	| 1.92 	| 292                                    	| 24.37                                    	|
| 01/22/2020 	| seagiv   	| sm-scrypt 	| Nvidia GTX 1060 6G               	| Windows 10          	| vulkan pre-optimized prototype         	| 276   	|  	|                                        	|                                          	|
| 01/22/2020 	| seagiv   	| sm-scrypt 	| AMD Radeon 570 4GB               	| Windows 10          	| vulkan pre-optimized prototype         	| 269   	|  	|                                        	|                                          	|
| 01/27/2020 	| seagiv   	| sm-scrypt 	| Nvidia GTX 1060 6G               	| Windows 10          	| vulkan optimized prototype	         	| 642   	|  	|                                        	|                                          	|
| 01/27/2020 	| seagiv   	| sm-scrypt 	| AMD Radeon 570 4GB               	| Windows 10          	| vulkan optimized prototype    	     	| 966   	|  	|                                        	|                                          	|
| 01/29/2020 	| seagiv   	| sm-scrypt 	| AMD Radeon Pro 555x 4GB              	| macOS 10.14.6        	| vulkan optimized prototype    	     	| 266   	|  	|                                        	|                                          	|
| 01/31/2020 	| avive   	| sm-scrypt 	| AMD Radeon Pro 560x 4GB              	| macOS 10.14.6        	| vulkan optimized prototype    	     	| 406   	|  	|                                        	|                                          	|
| 01/31/2020 	| avive   	| sm-scrypt 	| Intel(R) UHD Graphics 630 1536MB              	| macOS 10.14.6        	| vulkan optimized prototype    	     	| 53   	|  	
| 05/06/2020 	| avive   	| sm-scrypt 	| AMD Radeon RX 580             	| Windows 10        	| vulkan optimized prototype    	     	| 1074   	|  1.074
| 09/08/2020 	| avive   	| sm-scrypt 	| Nvidia Tesla V 100 (16GB)            	| Ubuntu 20.04 NVIDIA-SMI 450.51.06 CUDA Version: 11.0 | CUDA optimized prototype    	     	| 4,166   	|  4.166
| 09/08/2020 	| avive   	| sm-scrypt 	| Nvidia Tesla T4 (16GB)            	| Ubuntu 20.04 NVIDIA-SMI 450.51.06 CUDA Version: 11.0 | CUDA optimized prototype   	     	|   	1,252 |  1.252
| 09/08/2020 	| avive   	| sm-scrypt 	| Nvidia Tesla P100-PCIE (32GB)         | Ubuntu 20.04 NVIDIA-SMI 450.51.06 CUDA Version: 11.0 | CUDA optimized prototype   	     	|   	2083 | 2.083
| 09/08/2020 	| avive   	| sm-scrypt 	| Nvidia Tesla P4 (32GB)         | Ubuntu 20.04 NVIDIA-SMI 450.51.06 CUDA Version: 11.0 | CUDA optimized prototype   	     	|   	757.57 | 0.75
| 04/04/2020 	| avive   	| sm-scrypt 	| Apple M1         | MacOS 11.2 | vulkan optimized prototype   	     	|   	214 | 0.214
| 04/21/2020 	| avive   	| sm-scrypt 	| Nvidia RTX 2070 Super, 8GB    | Ubuntu 20.04, Driver 460.73.01  | CUDA optimized prototype   	     	|  2038 | 2.038

---


## 3rd Party Vulkan and CUDA Benchmarks
The library performance on a GPU depends on the GPU's CUDA and Vulkan performance. The following benchmarks are available from geekbench:

- [Geekbench Cuda Benchmarks](https://browser.geekbench.com/cuda-benchmarks)
- [Geekbench Vulkan Benchmarks](https://browser.geekbench.com/vulkan-benchmarks)
