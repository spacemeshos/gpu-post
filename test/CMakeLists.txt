set(SOURCES
    test.cpp
    test.hpp
    unit-tests.cpp
    integration-tests.cpp
)

add_executable(gpu-setup-test ${SOURCES})

target_include_directories(gpu-setup-test PRIVATE ..)
target_link_libraries(gpu-setup-test gpu-setup)
if(APPLE OR UNIX)
    target_link_libraries(gpu-setup-test -ldl)
endif()

if (NOT SPACEMESH_LIB_SHARED)
    if (SPACEMESHCUDA)
        set(CUDA_LIB_PATH ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/)
        if(MSVC)
            target_link_libraries(gpu-setup-test "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib")
        else()
            target_link_libraries(gpu-setup-test -L${CUDA_LIB_PATH} -lcuda)
        endif()
    endif()

    if (SPACEMESHVULKAN)
        # Add Include/Lib (Windows) and include/lib (Linux)
        if(WIN32)
            target_link_libraries(gpu-setup-test "$ENV{VULKAN_SDK}/Lib/vulkan-1.lib" "$ENV{VULKAN_SDK}/Lib/shaderc_combined.lib")
        elseif(APPLE)
            target_link_libraries(gpu-setup-test -L$ENV{VULKAN_SDK}/lib -lvulkan shaderc_combined)
        elseif(UNIX)
            target_link_libraries(gpu-setup-test -L$ENV{VULKAN_SDK}/lib -lvulkan shaderc_combined)
        endif()
    endif()
endif()