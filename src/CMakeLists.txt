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
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

if (APPLE)
    # we should check here for ARM / Intel too
    add_compile_definitions(SPACEMESH_VULKAN_COMPATIBILITY_NEEDED=1)
endif()

include_directories(.)
include_directories(../include)

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
        if(CUDA_VERSION_MAJOR GREATER 10)
            set(
                CUDA_NVCC_FLAGS
                ${CUDA_NVCC_FLAGS}
                "-gencode arch=compute_86,code=sm_86"
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

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders-vault.inl
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vulkan/gen/shaders-gen
    COMMENT "Generating Vulkan Kernels Byte Array"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/gen/scrypt-chacha.comp ${CMAKE_CURRENT_BINARY_DIR}/vulkan/gen/shaders-gen
)

if (SPACEMESHVULKAN)
    # Look for vulkan SDK
    # For a Windows build on Linux, install the Vulkan SDK with wine, and set VULKAN_SDK to ..../.wine/drive_c/VulkanSDK/1.x.y
    # For Linux, install libvulkan-dev
    find_path(VulkanIncludes vulkan/vulkan.h PATHS $ENV{VULKAN_SDK}/Include $ENV{VULKAN_SDK}/include /usr )
    if (NOT VulkanIncludes)
        message(FATAL_ERROR "Can't find folder containing vulkan.h")
    endif()

    find_path(GLSLANG_SPIRV_INCLUDE_DIR SPIRV/GlslangToSpv.h HINTS "${VulkanIncludes}" "${VulkanIncludes}/../source/glslang/" "${VulkanIncludes}/../glslang" "${VulkanIncludes}/glslang" DOC "Path to SPIRV/GlslangToSpv.h")
    if (NOT GLSLANG_SPIRV_INCLUDE_DIR)
        message(FATAL_ERROR "Can't find folder containing spirv.hpp")
    endif()

    include_directories(${VulkanIncludes})
    include_directories(${GLSLANG_SPIRV_INCLUDE_DIR})

    message("------------------------------------------------------------------------")
    message("-- Vulkan API Includes ${VulkanIncludes}")
    message("------------------------------------------------------------------------")

    set(SOURCES
        ${SOURCES}
        vulkan/driver-vulkan.cpp
        vulkan/driver-vulkan.h
        vulkan/vulkan-helpers.c
        vulkan/vulkan-helpers.h

		${CMAKE_CURRENT_BINARY_DIR}/vulkan-shaders-vault.inl
    )

    include_directories(vulkan/zlib)
	include_directories(${CMAKE_CURRENT_BINARY_DIR})
	include_directories(${CMAKE_CURRENT_BINARY_DIR}/vulkan/zlib)

    add_subdirectory(vulkan/zlib)
    add_subdirectory(vulkan/gen)
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
	    target_link_libraries(gpu-setup zlibstatic)
    endif()
else()
    if (SPACEMESHCUDA)
        cuda_add_library(gpu-setup ${SOURCES})
    else()
        add_library(gpu-setup ${SOURCES})
    endif()
endif()

