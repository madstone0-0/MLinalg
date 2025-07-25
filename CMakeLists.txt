cmake_minimum_required(VERSION 3.20)

project(mlinalg LANGUAGES CXX VERSION 0.8)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)

# Unity Build
set(CMAKE_UNITY_BUILD ON)

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

option(STACKTRACE "Enable stacktrace support" OFF)

if(NOT DEFINED CACHE{STACKTRACE})
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(STACKTRACE ON CACHE BOOL "Enable stacktrace support" FORCE)
    endif()
endif()

message(STATUS "STACKTRACE option is set to: ${STACKTRACE}")

if(STACKTRACE)
    message(STATUS "Stacktrace support enabled")
    add_compile_definitions(STACKTRACE)
else()
    message(STATUS "Stacktrace support disabled")
endif()

option(STRASSEN "Enable Strassen algorithm" OFF)
if(STRASSEN)
    add_definitions(-DSTRASSEN)
endif()

option(
    BY_DEF
    "Use algorithms derived from the mathetical defintion of operations"
    OFF
)
if(BY_DEF)
    add_definitions(-DBY_DEF)
    message(
        STATUS
        "Using algorithms derived from the mathematical definition of operations"
    )
endif()

option(EXAMPLES "Compile usage examples" OFF)
if(EXAMPLES)
    message(STATUS "Compiling examples")
    add_subdirectory(src/examples)
else()
    message(STATUS "Skipping examples")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_link_options(-rdynamic -Wl,--export-dynamic)
    add_compile_options(-Wcomma -ferror-limit=100)
    # set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
endif()

add_compile_options(-Wall -Wextra -Wfloat-equal -Werror=return-type)

set(CMAKE_CXX_FLAGS_RELEASE
    "-O2 -fno-ident -fno-rtti -fno-fast-math -fomit-frame-pointer"
)
set(CMAKE_CXX_FLAGS_PROFILE "-g -pg")
set(CMAKE_CXX_FLAGS_ASM "-S -fverbose-asm -masm=intel")

set(CMAKE_VERIFY_INTERFACE_HEADER_SETS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#################
# Catch2        #
#################
find_package(Catch2 3 REQUIRED)

add_subdirectory(src)

#################
# Boost         #
#################
# For debug builds, we need to link against the boost stacktrace library
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    find_package(Boost CONFIG REQUIRED)
    if(Boost_FOUND)
        message(STATUS "Boost stacktrace found")
        if(STACKTRACE)
            target_link_libraries(
                mlinalg
                INTERFACE boost_stacktrace_addr2line dl
            )
        endif()
    else()
        message(WARNING "Boost stacktrace not found")
    endif()
endif()
