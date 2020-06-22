add_definitions(-DSPACEMESHAPI_BUILD=1)

set(SOURCES
    api.c
    ../include/api.h
    api_internal.cpp
    api_internal.h

    scrypt-jane/scrypt-jane-chacha.h
    scrypt-jane/scrypt-jane-mix_chacha.h
    scrypt-jane/scrypt-jane-portable-x86.h
    scrypt-jane/scrypt-jane-portable.h
    scrypt-jane/scrypt-jane-romix-basic.h
    scrypt-jane/scrypt-jane-romix-template.h
    scrypt-jane/scrypt-jane.cpp
    scrypt-jane/scrypt-jane.h
)

if (MSVC)
    set(SOURCES
        ${SOURCES}
        compat/compat.h
        compat/gettimeofday.c
        compat/unistd.h
        compat/sys/time.h
    )
    include_directories(compat)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# A custom command and target to turn the OpenCL kernel into a byte array header

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scrypt-chacha-cl.inl
    COMMAND ${CMAKE_COMMAND} ARGS
    -DBIN2H_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/opencl/scrypt-chacha.cl"
    -DBIN2H_VARIABLE_NAME=scrypt_chacha_cl
    -DBIN2H_HEADER_FILE="${CMAKE_CURRENT_BINARY_DIR}/scrypt-chacha-cl.inl"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
    COMMENT "Generating OpenCL Kernel Byte Array"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/opencl/scrypt-chacha.cl
)

# A custom command and target to turn the Vulkan kernel into a byte array header

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scrypt-chacha-vulkan.inl
    COMMAND ${CMAKE_COMMAND} ARGS
    -DBIN2H_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/vulkan/scrypt-chacha.comp"
    -DBIN2H_VARIABLE_NAME=scrypt_chacha_comp
    -DBIN2H_HEADER_FILE="${CMAKE_CURRENT_BINARY_DIR}/scrypt-chacha-vulkan.inl"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
    COMMENT "Generating Vulkan Kernel Byte Array"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/scrypt-chacha.comp
)

include_directories(.)
include_directories(../include)

if (SPACEMESHCL)
    set(SOURCES
        ${SOURCES}
        opencl/driver-opencl.c
        opencl/driver-opencl.h
        opencl/ocl.c
        opencl/ocl.h

        opencl/CL/cl.h
        opencl/CL/cl_egl.h
        opencl/CL/cl_ext.h
        opencl/CL/cl_gl.h
        opencl/CL/cl_gl_ext.h
        opencl/CL/cl_platform.h
        opencl/CL/opencl.h

        opencl/ICD/icd.c
        opencl/ICD/icd.h
        opencl/ICD/icd_dispatch.c
        opencl/ICD/icd_dispatch.h
        opencl/ICD/icd_envvars.h
        opencl/ICD/icd_platform.h

        ${CMAKE_CURRENT_BINARY_DIR}/scrypt-chacha-cl.inl
    )
    if (MSVC)
        set(SOURCES
            ${SOURCES}
            opencl/ICD/windows/icd_windows.c
            opencl/ICD/windows/icd_windows.h
            opencl/ICD/windows/icd_windows_dxgk.c
            opencl/ICD/windows/icd_windows_dxgk.h
            opencl/ICD/windows/icd_windows_envvars.c
        )
    else()
        set(SOURCES
            ${SOURCES}
            opencl/ICD/linux/icd_linux.c
            opencl/ICD/linux/icd_linux_envvars.c
        )
    endif()
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
    include_directories(opencl)
endif()

if (SPACEMESHCUDA)
    if(MSVC)
        find_package(CUDA 10.0 REQUIRED)
    else()
        find_package(CUDA REQUIRED)
    endif()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v;-use_fast_math;-maxrregcount=80)
    list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")

    list(APPEND CUDA_NVCC_FLAGS_RELEASE -O3)
    list(APPEND CUDA_NVCC_FLAGS_DEBUG -G)

    if(COMPUTE AND (COMPUTE GREATER 0))
        list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_${COMPUTE},code=sm_${COMPUTE}")
    else()
        set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS}
#            "-gencode arch=compute_30,code=sm_30"
#            "-gencode arch=compute_35,code=sm_35"
            "-gencode arch=compute_50,code=sm_50"
#            "-gencode arch=compute_52,code=sm_52"
#            "-gencode arch=compute_53,code=sm_53"
            "-gencode arch=compute_60,code=sm_60"
#            "-gencode arch=compute_61,code=sm_61"
#            "-gencode arch=compute_62,code=sm_62"
#            "-gencode arch=compute_70,code=sm_70"
        )
        if(CUDA_VERSION_MAJOR GREATER 9)
            set(
                CUDA_NVCC_FLAGS
                ${CUDA_NVCC_FLAGS}
                "-gencode arch=compute_75,code=sm_75"
            )
        endif()
    endif()

    include_directories(${CUDA_INCLUDE_DIRS})

    message("------------------------------------------------------------------------")
    message("-- CUDA nvcc flags ${CUDA_NVCC_FLAGS}")
    message("-- CUDA host compiler ${CUDA_HOST_COMPILER}")
    message("-- Compute ${COMPUTE}")
    message("-- CUDA include dirs ${CUDA_INCLUDE_DIRS}")
    message("------------------------------------------------------------------------")

    set(SOURCES
        ${SOURCES}
        cuda/cuda_helper.h
        cuda/driver-cuda.cpp
        cuda/driver-cuda.h
        cuda/keccak.cu
        cuda/keccak.h
        cuda/salsa_kernel.cu
        cuda/salsa_kernel.h
        cuda/titan_kernel.cu
        cuda/titan_kernel.h
    )
endif()

if (SPACEMESHVULKAN)
    # Look for vulkan SDK
    # For a Windows build on Linux, install the Vulkan SDK with wine, and set VULKAN_SDK to ..../.wine/drive_c/VulkanSDK/1.x.y
    # For Linux, install libvulkan-dev
    find_path(VulkanIncludes vulkan/vulkan.h PATHS $ENV{VULKAN_SDK}/Include $ENV{VULKAN_SDK}/include /usr )
    if (NOT VulkanIncludes)
        message(FATAL_ERROR "Can't find folder containing vulkan.h")
    endif()

    find_path(GLSLANG_SPIRV_INCLUDE_DIR SPIRV/spirv.hpp HINTS "${VulkanIncludes}" "${VulkanIncludes}/../source/glslang/" "${VulkanIncludes}/../glslang" DOC "Path to SPIRV/spirv.hpp")
    if (NOT GLSLANG_SPIRV_INCLUDE_DIR)
        message(FATAL_ERROR "Can't find folder containing spirv.hpp")
    endif()

    include_directories(${VulkanIncludes})
    include_directories(${GLSLANG_SPIRV_INCLUDE_DIR})
#    include_directories(../third_party/glslang)

    message("------------------------------------------------------------------------")
    message("-- Vulkan API Includes ${VulkanIncludes}")
    message("------------------------------------------------------------------------")

    set(SOURCES
        ${SOURCES}
        vulkan/driver-vulkan.c
        vulkan/driver-vulkan.h
        vulkan/vulkan-helpers.c
        vulkan/vulkan-helpers.h
        vulkan/glsl_compiler.cpp

        ${CMAKE_CURRENT_BINARY_DIR}/scrypt-chacha-vulkan.inl
    )

    include_directories(${CMAKE_CURRENT_BINARY_DIR})
endif()

if (SPACEMESH_LIB_SHARED)
    if (SPACEMESHCUDA)
        cuda_add_library(gpu-setup SHARED ${SOURCES})
    else()
        add_library(gpu-setup SHARED ${SOURCES})
    endif()

    if (SPACEMESHCUDA)
        set(CUDA_LIB_PATH ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/)
        if(MSVC)
            target_link_libraries(gpu-setup "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib")
        else()
            target_link_libraries(gpu-setup -L${CUDA_LIB_PATH} -lcuda)
        endif()
    endif()

    if (SPACEMESHVULKAN)
        # Add Include/Lib (Windows) and include/lib (Linux)
        if(WIN32)
            target_link_libraries(gpu-setup "$ENV{VULKAN_SDK}/Lib/shaderc_combined.lib")
        elseif(APPLE)
            target_link_libraries(gpu-setup -L$ENV{VULKAN_SDK}/lib shaderc_combined)
        elseif(UNIX)
            target_link_libraries(gpu-setup -L$ENV{VULKAN_SDK}/lib shaderc_combined)
        endif()
    endif()
else()
    if (SPACEMESHCUDA)
        cuda_add_library(gpu-setup ${SOURCES})
    else()
        add_library(gpu-setup ${SOURCES})
    endif()
endif()

