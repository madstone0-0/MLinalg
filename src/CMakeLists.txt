cmake_minimum_required(VERSION 3.5)

project(mlinalg LANGUAGES CXX VERSION 1.0)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

# file(
#     GLOB CPP
#     CONFIGURE_DEPENDS
#     ${PROJECT_SOURCE_DIR}/pub/*.cpp
#     ${PROJECT_SOURCE_DIR}/pub/*.hpp
# )
file(
    GLOB HPP
    CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/pub/structures/*.hpp
    ${CMAKE_SOURCE_DIR}/src/pub/*.hpp
    ${CMAKE_SOURCE_DIR}/src/*.hpp
)

add_library(mlinalg STATIC ${HPP})
add_library(mlinalg::mlinalg ALIAS mlinalg)

target_include_directories(
    mlinalg
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/pub>
    PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/pub
)
set_target_properties(mlinalg PROPERTIES LINKER_LANGUAGE CXX)

include(GNUInstallDirs)
install(
    TARGETS mlinalg
    EXPORT mlinalgTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

set_target_properties(mlinalg PROPERTIES EXPORT_NAME mlinalg)
install(FILES ${HPP} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mlinalg)
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
