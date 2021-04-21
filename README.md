# GPU Proof of Spacemesh Time Init (aka Smeshing Setup) Library

## Current functionality
A c libraray implementing the POST API setup method for cpu, cuda and vulkan compute platforms.

## Runtime Requirements
OS: Windows 10, macOS or Linux.
GPI: One of the following:
- A GPU and drivers supporting CUDA 10.0 (or later) runtime such as a modern Nvidia GPU.
- A GPU and drivers supporting Vulkan 1.2 (or later) runtime such as a modern AMD and Intel GPUs.

Both discrete and on-board GPUs are supported as long as they support the minimum CUDA or Vulkan runtimes. 

## Build System Requirements

### Windows
- Windows 10 Pro.
- Microsoft Visual Studio 2017 (any edition such as community is okay). Visual Studio 2019 is NOT supported. You may also need to install specific versions of the Windows SDK when prompted when attempting to build for the first time.
- NVIDIA GPU Computing Toolkit 10.0 (but not later versions), and an NVIDIA GPU supporting CUDA 10.0 computation for CUDA testing.
- Vulkan SDK 1.2, and an AMD GPU supporting Vulkan 1.2 for Vulkan testing.

### Linux
- Modern 64-bit Linux, such as Ubuntu, Debian.
- NVIDIA GPU Computing Toolkit 9 or 10, and an NVIDIA GPU supporting CUDA 9 or 10 computation for CUDA testing.
- Vulkan SDK 1.2.
- Cmake
- GCC 6 or 7

### OS X
- Cmake
- Vulkan SDK 1.2.

### OS X Dev Env Setup
1. Install latest version of Xcode with the command line dev tools.
1. Download the Vulkan 1.2 sdk installer for OS X from https://vulkan.lunarg.com/sdk/home#mac
1. Copy Vulkan SDK from the Vulkan installer volume to a directory in your hard-drive.
1. Install the SDK from your hard-drive directory and not from the installer volume by running $ sudo ./install_vulkan.py.
1. Add the Vulkan env vars to your .bash_profile file with the root location set to the sdk directory on your hard-drive. For example, if Vulkan sdk 1.2.154 is installed then the env vars should be set like this:

```
export VULKAN_SDK_VERSION="1.2.154.0"
export VULKAN_ROOT_LOCATION="$HOME/dev/vulkan-sdk-1.2.154"
export VULKAN_SDK="$VULKAN_ROOT_LOCATION/macOS"
export VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layers.d"
export PATH="/usr/local/opt/python/libexec/bin:$VULKAN_SDK/bin:$PATH"
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$VULKAN_SDK/lib/"
```

## Building

Build options:
### Windows and Linux
```
SPACEMESHCUDA   "Build with CUDA support"   default: ON
SPACEMESHVULKAN "Build with Vulkan support" default: ON
```
### macOS
```
SPACEMESHCUDA   "Build with CUDA support"   default: OFF
SPACEMESHVULKAN "Build with Vulkan support" default: ON
```

### Windows
1. Open project folder into Visual Studio 2017: `File -> Open -> Folder`.
2. Set "x64-Release" Project Settings.
3. Build: `CMake -> Rebuild All`.
4. Run test: `CMake -> Debug from Build Folder -> gpu-setup-test.exe`

### Linux or macOS
1. Create build directory:
```
  cd gpu-post
  mkdir build
  cd build
```
2. Configure:
```
  cmake ..
```
Disable OpenCL:
```
  cmake .. -DSPACEMESHCL=OFF
```
Disable CUDA:
```
  cmake .. -DSPACEMESHCUDA=OFF
```
3. Build:
```
  make
```
4. Run test:
```  
  ./test/gpu-setup-test
```

CUDA 9 Configuration:
```
  cmake .. -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6
```
You may need to set CUDA_TOOLKIT_ROOT_DIR:
```
  cmake .. -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0
```

## Recommendations

Recommendations for choosing an implementation:

| OS / GPU     	| Windows 	| Linux      	| macOS        	|
|------------	|----------	|-----------	|--------------	|
| Nvidia	| CUDA      	| CUDA      	| Vulkan      	|
| AMD		| Vulkan      	| Vulkan      	| Vulkan      	|
| Intel		| Vulkan      	| Vulkan      	| Vulkan      	|

## API

```
int scryptPositions(
    const uint8_t *id,			// 32 bytes
    uint64_t start_position,	// e.g. 0
    uint64_t end_position,		// e.g. 49,999
    uint32_t hash_len_bits,		// (1...256) for each hash output, the number of prefix bits (not bytes) to copy into the buffer
    const uint8_t *salt,		// 32 bytes
    uint32_t options,			// throttle etc.
    uint8_t *out,				// memory buffer large enough to include hash_len_bits * number of requested hashes
    uint32_t N,					// scrypt N
    uint32_t R,					// scrypt r
    uint32_t P					// scrypt p
);
```

return to the client the system GPU capabilities. E.g. OPENCL, CUDA/NVIDIA or NONE
```
int stats();
```

stop all GPU work and don’t fill the passed-in buffer with any more results.
```
int stop(
	uint32_t ms_timeout			// timeout in milliseconds
);
```

return non-zero if stop in progress
```
SPACEMESHAPI int spacemesh_api_stop_inprogress();
```

return POST compute providers info
```
SPACEMESHAPI int spacemesh_api_get_providers(
	PostComputeProvider *providers, // out providers info buffer, if NULL - return count of available providers
	int max_providers			    // buffer size
);
```

## Initial Benchmarks

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


