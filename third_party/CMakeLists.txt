cmake_minimum_required(VERSION 3.1)

project(third_party LANGUAGES C CXX)

if (SPACEMESHVULKAN)
    # Look for vulkan SDK
    # For a Windows build on Linux, install the Vulkan SDK with wine, and set VULKAN_SDK to ..../.wine/drive_c/VulkanSDK/1.x.y
    # For Linux, install libvulkan-dev
    find_path(VulkanIncludes vulkan/vulkan.h PATHS $ENV{VULKAN_SDK}/Include $ENV{VULKAN_SDK}/include /usr )
    if (NOT VulkanIncludes)
        message(FATAL_ERROR "Can't find folder containing vulkan.h")
    endif()

    include_directories(${VulkanIncludes})
    include_directories(glslang)

    # glslang
    option(ENABLE_SPVREMAPPER OFF)
    option(ENABLE_GLSLANG_BINARIES OFF)
    option(ENABLE_HLSL OFF)
    option(ENABLE_OPT OFF)
    option(BUILD_TESTING OFF)

    add_subdirectory(glslang)

    if (NOT TARGET glslang-default-resource-limits)
        add_library(glslang-default-resource-limits
                    glslang/StandAlone/ResourceLimits.cpp)

        set_property(TARGET glslang-default-resource-limits PROPERTY FOLDER "ThirdParty")

        target_include_directories(glslang-default-resource-limits
                                    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/glslang/StandAlone)
    endif()

    if(NOT MSVC)
        target_compile_options(glslang PRIVATE
            "-Wno-logical-op-parentheses"
            "-Wno-unused-parameter")

        target_compile_options(SPIRV PRIVATE
            "-Wno-logical-op-parentheses"
            "-Wno-unused-parameter")
    endif()

    set_property(TARGET glslang PROPERTY FOLDER "ThirdParty")
    set_property(TARGET OGLCompiler PROPERTY FOLDER "ThirdParty")
    set_property(TARGET OSDependent PROPERTY FOLDER "ThirdParty")
    set_property(TARGET SPIRV PROPERTY FOLDER "ThirdParty")
endif()
