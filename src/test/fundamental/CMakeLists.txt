if(NOT CMAKE_BUILD_TYPE STREQUAL "Asm")
    add_executable(vector vector.cpp)
    target_link_libraries(
        vector
        PRIVATE Catch2::Catch2WithMain mlinalg::mlinalg
    )
    catch_discover_tests(vector)

    add_executable(matrix matrix.cpp)
    target_link_libraries(
        matrix
        PRIVATE Catch2::Catch2WithMain mlinalg::mlinalg
    )
    catch_discover_tests(matrix)
endif()
