# c:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvars64.bat
#    - run: cmake -G "Ninja" -DCMAKE_IGNORE_PATH="C:/Strawberry/c/bin;C:/ProgramData/chocolatey/bin" -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_MAKE_PROGRAM="c:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" .
#
#    - uses: actions/upload-artifact@v2
#      with:
#        path: "D:/a/gpu-post/gpu-post/CMakeFiles/CMakeOutput.log"
#
#    - run: cmake --build .

Invoke-WebRequest http://developer.download.nvidia.com/compute/cuda/10.2/Prod/network_installers/cuda_10.2.89_win10_network.exe -OutFile ./cuda_10.2.89_win10_network.exe
Start-Process -Wait -FilePath ./cuda_10.2.89_win10_network.exe -ArgumentList "-s nvcc,visual_studio_integration"
