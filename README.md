# GPU Proof of Spacemesh Time Init (aka Smeshing Settup) Prototype

## Current functionality
- A c libraray implementing the POST API setup method for cpu, cuda and openCL compute platforms.

## System Requirements
- Windows 10 Pro.
- Microsoft Visual Studio 2017 (any edition should be okay. Visual Studio 2019 is not supported. You may also need to install specific versions of the Windows SDK when prompted when attempting to build for the first time.
- NVIDIA GPU Computing Toolkit 10.0 (but not later versions), and an NVIDIA GPU supporting CUDA 10.0 computation for CUDA testing.
- An AMD GPU supporting OpenCL 2.0 or newer for OpenCL testing.

## Biulding
1. Load solution spacemesh.sln info Visual Studio 2017.
2. Set "test" as startup project. In the Solution Explorer right click on "test", select "Set as StartUp Project".
3. Set "Release" configuration and "x64" platform.
4. Build: Build menu -> Rebuild Solution.
5. Run test: Debug menu -> Start Without Debugging.

## Initial Benchmarks
Will be added soone here!
