cmake_minimum_required(VERSION 3.20)

project(mlinalg LANGUAGES CXX VERSION 0.8)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)

# Check if the compiler supports -mavx
check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_MAVX)
if(COMPILER_SUPPORTS_MAVX)
    message(STATUS "Compiler supports -mavx")
    add_compile_options("-mavx")
endif()

# Check if the compiler supports -mavx2
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_MAVX2)
if(COMPILER_SUPPORTS_MAVX2)
    message(STATUS "Compiler supports -mavx2")
    add_compile_options("-mavx2")
endif()

# Check if the compiler supports -mfma
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_FMA)
    message(STATUS "Compiler supports -mfma")
    add_compile_options("-mfma")
else()
    message(
        WARNING
        "Compiler does not support -mfma. FMA intrinsics may not be available!"
    )
endif()

# For MSVC, add the appropriate flag
if(MSVC)
    add_compile_options("/arch:AVX")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
add_compile_options(-Wall -Wextra -Wfloat-equal)
# add_compile_options(-Werror=return-type)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -fsanitize=address")
# set(CMAKE_CXX_FLAGS_DEBUG "-ggdb")

set(CMAKE_CXX_FLAGS_RELEASE "-Os -fno-ident")
set(CMAKE_CXX_FLAGS_PROFILE "-g -pg")
set(CMAKE_CXX_FLAGS_ASM "-S -fverbose-asm -masm=intel")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#################
# Catch2        #
#################
find_package(Catch2 3 REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
