cmake_minimum_required(VERSION 3.20)

project(mlinalg LANGUAGES CXX VERSION 1.0)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

file(
    GLOB HPP
    CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/pub/structures/*.hpp
    ${CMAKE_SOURCE_DIR}/src/pub/*.hpp
    ${CMAKE_SOURCE_DIR}/src/*.hpp
)

add_library(mlinalg INTERFACE)
add_library(mlinalg::mlinalg ALIAS mlinalg)

target_include_directories(
    mlinalg
    INTERFACE
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/pub>
)

include(GNUInstallDirs)
install(
    TARGETS mlinalg
    EXPORT mlinalgTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties(mlinalg PROPERTIES EXPORT_NAME mlinalg)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/src/pub/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mlinalg
    FILES_MATCHING
    PATTERN "*.hpp"
)
install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/src/pub/structures/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mlinalg/structures
    FILES_MATCHING
    PATTERN "*.hpp"
)
install(
    EXPORT mlinalgTargets
    FILE mlinalgTargets.cmake
    NAMESPACE mlinalg::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlinalg
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/mlinalgConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/mlinalgConfig.cmake.in
    ${CMAKE_BINARY_DIR}/mlinalgConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlinalg
)

install(
    FILES
        ${CMAKE_BINARY_DIR}/mlinalgConfig.cmake
        ${CMAKE_BINARY_DIR}/mlinalgConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mlinalg
)

export(
    EXPORT mlinalgTargets
    FILE ${CMAKE_BINARY_DIR}/mlinalgTargets.cmake
    NAMESPACE mlinalg::
)

export(PACKAGE mlinalg)
