cmake_minimum_required(VERSION 3.20)

project(test LANGUAGES CXX VERSION 1.0)

add_executable(fundamental fundamental.cpp)
add_executable(operations operations.cpp)
add_executable(examples examples.cpp)
target_link_libraries(
    fundamental
    PRIVATE Catch2::Catch2WithMain mlinalg::mlinalg
)
target_link_libraries(
    operations
    PRIVATE Catch2::Catch2WithMain mlinalg::mlinalg
)
target_link_libraries(examples PRIVATE mlinalg::mlinalg)

include(CTest)
include(Catch)
enable_testing()
catch_discover_tests(fundamental)
catch_discover_tests(operations)
