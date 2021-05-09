include_directories(${CMAKE_CURRENT_BINARY_DIR})

# A custom command and target to turn the Vulkan kernel into a byte array header

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scrypt-chacha-vulkan.inl
    COMMAND ${CMAKE_COMMAND} ARGS
    -DBIN2H_SOURCE_FILE="${CMAKE_CURRENT_SOURCE_DIR}/scrypt-chacha.comp"
    -DBIN2H_VARIABLE_NAME=scrypt_chacha_comp
    -DBIN2H_HEADER_FILE="${CMAKE_CURRENT_BINARY_DIR}/scrypt-chacha-vulkan.inl"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/../../bin2h.cmake"
    COMMENT "Generating Vulkan Kernel Byte Array"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scrypt-chacha.comp
)

set(SOURCES
    main.cpp

    ${CMAKE_CURRENT_BINARY_DIR}/scrypt-chacha-vulkan.inl
)

add_executable(shaders-gen ${SOURCES})

# Add Include/Lib (Windows) and include/lib (Linux)
if(WIN32)
    target_link_libraries(shaders-gen "$ENV{VULKAN_SDK}/Lib/vulkan-1.lib" "$ENV{VULKAN_SDK}/Lib/shaderc_combined.lib")
elseif(APPLE)
    target_link_libraries(shaders-gen -L$ENV{VULKAN_SDK}/lib -lvulkan shaderc_combined)
elseif(UNIX)
    target_link_libraries(shaders-gen -L$ENV{VULKAN_SDK}/lib -lvulkan shaderc_combined)
endif()
