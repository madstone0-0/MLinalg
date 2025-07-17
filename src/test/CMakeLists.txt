cmake_minimum_required(VERSION 3.20)

project(test LANGUAGES CXX VERSION 1.0)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Asm")
    include(CTest)
    include(Catch)
    enable_testing()

    add_executable(operations operations.cpp)
    target_link_libraries(
        operations
        PRIVATE Catch2::Catch2WithMain mlinalg::mlinalg
    )
    catch_discover_tests(operations)

    add_executable(bench bench.cpp)
    target_link_libraries(bench PRIVATE Catch2::Catch2WithMain mlinalg::mlinalg)

    add_subdirectory(fundamental)
endif()

# If dummy translation unit exists then compile it
if(
    EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/dummy.cpp
    AND CMAKE_BUILD_TYPE STREQUAL "Debug"
)
    add_executable(dummy dummy.cpp)
    target_link_libraries(dummy PRIVATE mlinalg::mlinalg)
endif()

if(
    EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/profile.cpp
    AND CMAKE_BUILD_TYPE STREQUAL "Debug"
)
    add_executable(profile profile.cpp)
    target_link_libraries(profile PRIVATE mlinalg::mlinalg)
endif()
