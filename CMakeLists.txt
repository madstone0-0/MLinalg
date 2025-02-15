cmake_minimum_required(VERSION 3.20)

project(mlinalg LANGUAGES CXX VERSION 0.7)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#################
# Catch2        #
#################
find_package(Catch2 3 REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
