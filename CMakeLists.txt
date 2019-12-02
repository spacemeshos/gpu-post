# cmake .. -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0

cmake_minimum_required(VERSION 3.1)

project(spacemesh)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif ()

option(SPACEMESHCL   "Build with OpenCL support" ON)
option(SPACEMESHCUDA "Build with CUDA support"   ON)

# propagates CMake configuration options to the compiler
function(configureProject)
    if (SPACEMESHCL)
        add_definitions(-DHAVE_OPENCL=1)
    endif()
    if (SPACEMESHCUDA)
        add_definitions(-DHAVE_CUDA=1)
    endif()
endfunction()

configureProject()

if (MSVC)
    string(REGEX REPLACE "/W[3|4]" "/w" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FS")
    # use static runtime
    foreach(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        flag_var CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)

        if (${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endif()

message("------------------------------------------------------------------------")
message("-- CMake ${CMAKE_VERSION}")
message("-- Build ${CMAKE_BUILD_TYPE} / ${CMAKE_SYSTEM_NAME}")
message("------------------------------------------------------------- components")
message("-- SPACEMESHCL         Build OpenCL components                  ${SPACEMESHCL}")
message("-- SPACEMESHCUDA       Build CUDA components                    ${SPACEMESHCUDA}")
message("------------------------------------------------------------------------")
message("")

if(UNIX AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc -pthread")
endif()

add_subdirectory(src)
add_subdirectory(test)
