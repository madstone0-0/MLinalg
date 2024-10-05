cmake_minimum_required(VERSION 3.5)

project(test LANGUAGES CXX VERSION 1.0)

add_executable(fundamental fundamental.cpp)
add_executable(fundamental_main fundamental_main.cpp)
target_link_libraries(
    fundamental
    PRIVATE Catch2::Catch2WithMain mlinalg::mlinalg
)
target_link_libraries(fundamental_main PRIVATE mlinalg::mlinalg)

include(CTest)
include(Catch)
enable_testing()
catch_discover_tests(fundamental)
