cmake_minimum_required(VERSION 3.1)

project(spacemesh)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif ()

if (APPLE)
    option(SPACEMESHCUDA   "Build with CUDA support"   OFF)

    # we should check here for ARM / Intel too
    add_compile_definitions(SPACEMESH_VULKAN_COMPATIBILITY_NEEDED=1)
else()
    option(SPACEMESHCUDA   "Build with CUDA support"   ON)
endif()
option(SPACEMESHVULKAN "Build with Vulkan support" ON)
option(SPACEMESH_LIB_SHARED "Build as shared library" ON)

# propagates CMake configuration options to the compiler
if (SPACEMESHCUDA)
    add_definitions(-DHAVE_CUDA=1)
endif()
if (SPACEMESHVULKAN)
    add_definitions(-DHAVE_VULKAN=1)
    add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
endif()

if (MSVC)
    string(REGEX REPLACE "/W[3|4]" "/w" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FS")
#    if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
#        if (SPACEMESHVULKAN)
#            # use release static runtime
#            foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
#                CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
#                flag_var CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
#                CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
#
#                if (${flag_var} MATCHES "/MDd")
#                    string(REGEX REPLACE "/MDd" "/MT" ${flag_var} "${${flag_var}}")
#                endif(${flag_var} MATCHES "/MDd")
#            endforeach(flag_var)
#        endif()
#    else()
#        # use static runtime
#        foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
#            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
#            flag_var CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
#            CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
#
#            if (${flag_var} MATCHES "/MD")
#                string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
#            endif(${flag_var} MATCHES "/MD")
#        endforeach(flag_var)
#    endif()
endif()

message("------------------------------------------------------------------------")
message("-- CMake ${CMAKE_VERSION}")
message("-- Build ${CMAKE_BUILD_TYPE} / ${CMAKE_SYSTEM_NAME}")
message("------------------------------------------------------------- components")
message("-- SPACEMESHCUDA       Build CUDA components                    ${SPACEMESHCUDA}")
message("-- SPACEMESHVULKAN     Build Vulkan components                  ${SPACEMESHVULKAN}")
message("------------------------------------------------------------------------")
message("")

if(UNIX AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc -pthread")
endif()

add_subdirectory(src)
add_subdirectory(test)
