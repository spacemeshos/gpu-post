set(SOURCES
    test.cpp
)

set(CUDA_LIB_PATH ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/)

add_executable(gpu-setup-test ${SOURCES})

target_include_directories(gpu-setup-test PRIVATE ..)
target_link_libraries(gpu-setup-test PRIVATE gpu-setup -ldl)

if (SPACEMESHCUDA)
    if(MSVC)
        target_link_libraries(gpu-setup-test PRIVATE "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib")
    else()
        target_link_libraries(gpu-setup-test PRIVATE -L${CUDA_LIB_PATH} -lcuda)
    endif()
endif()

if (SPACEMESHVULKAN)
    # Add Include/Lib (Windows) and include/lib (Linux)
    if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
        target_link_libraries(gpu-setup-test PRIVATE "$ENV{VULKAN_SDK}/Lib/vulkan-1.lib")
    endif()

    if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        target_link_libraries(gpu-setup-test PRIVATE -L$ENV{VULKAN_SDK}/lib -lvulkan)
    endif()
endif()