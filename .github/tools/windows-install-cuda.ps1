Invoke-WebRequest http://developer.download.nvidia.com/compute/cuda/10.2/Prod/network_installers/cuda_10.2.89_win10_network.exe -OutFile ./cuda_10.2.89_win10_network.exe
Start-Process -Wait -FilePath ./cuda_10.2.89_win10_network.exe -ArgumentList "-s nvcc,visual_studio_integration"
