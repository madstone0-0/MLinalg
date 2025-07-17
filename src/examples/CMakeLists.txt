project(examples LANGUAGES CXX VERSION 1.0)

add_executable(lin_reg lin_reg.cpp)
target_link_libraries(lin_reg PRIVATE mlinalg::mlinalg)

add_executable(general general.cpp)
target_link_libraries(general PRIVATE mlinalg::mlinalg)
