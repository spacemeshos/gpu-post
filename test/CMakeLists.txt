set(SOURCES
    test.cpp
)

set(CUDA_LIB_PATH ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/)

add_executable(gpu-setup-test ${SOURCES})

target_include_directories(gpu-setup-test PRIVATE ..)
target_link_libraries(gpu-setup-test PRIVATE gpu-setup)

if (SPACEMESHCUDA)
    if(MSVC)
        target_link_libraries(gpu-setup-test PRIVATE "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib")
    else()
        target_link_libraries(gpu-setup-test PRIVATE -L${CUDA_LIB_PATH} -lcuda)
    endif()
endif()
