cmake_minimum_required(VERSION 3.20)

project(mlinalg LANGUAGES CXX VERSION 0.5)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# add_compile_options(-Wall -Wextra -Werror=return-type)
add_compile_options(-Werror=return-type)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -fsanitize=address")

set(CMAKE_CXX_FLAGS_RELEASE "-Os -fno-ident")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#################
# Catch2        #
#################
find_package(Catch2 3 REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
