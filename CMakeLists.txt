cmake_minimum_required(VERSION 3.20)
project(liblloyal VERSION 0.1.0 LANGUAGES CXX)

# =============================================================================
# liblloyal - Header-only C++20 library for llama.cpp inference wrappers
# =============================================================================
#
# This library provides type-safe, ergonomic wrappers around llama.cpp for
# multiple language bindings. All implementations are header-only with inline
# specifiers.
#
# Dependencies:
#   - llama.cpp (b6870 or compatible)
#   - C++20 compiler
#
# Usage:
#   add_subdirectory(packages/liblloyal)
#   target_link_libraries(your_target PRIVATE liblloyal::liblloyal)
#

# =============================================================================
# Configuration
# =============================================================================

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Define INTERFACE Library Target
# =============================================================================

add_library(liblloyal INTERFACE)
add_library(liblloyal::liblloyal ALIAS liblloyal)

# =============================================================================
# Include Directories
# =============================================================================

target_include_directories(liblloyal INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# =============================================================================
# Dependencies
# =============================================================================

# llama.cpp dependency
# Consumer must provide llama target or locate llama.cpp
if(TARGET llama)
    target_link_libraries(liblloyal INTERFACE llama)
    message(STATUS "liblloyal: Found llama.cpp target")
else()
    message(WARNING "liblloyal: llama.cpp target not found. Consumer must provide llama target.")
endif()

# =============================================================================
# Version Check for llama.cpp
# =============================================================================

# liblloyal is tested against llama.cpp b6870
# This is a soft check - we warn but don't fail the build
if(DEFINED LLAMA_CPP_VERSION)
    if(NOT LLAMA_CPP_VERSION STREQUAL "b6870")
        message(WARNING
            "liblloyal: Designed for llama.cpp b6870, found ${LLAMA_CPP_VERSION}. "
            "API compatibility not guaranteed.")
    else()
        message(STATUS "liblloyal: Using recommended llama.cpp version b6870")
    endif()
else()
    message(STATUS
        "liblloyal: LLAMA_CPP_VERSION not set. Recommended: b6870. "
        "Define LLAMA_CPP_VERSION in your llama.cpp CMakeLists to enable version check.")
endif()

# =============================================================================
# Compiler Requirements
# =============================================================================

target_compile_features(liblloyal INTERFACE cxx_std_20)

# =============================================================================
# Installation (Optional - for package managers)
# =============================================================================

include(GNUInstallDirs)

install(TARGETS liblloyal
    EXPORT liblloyal-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/lloyal
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT liblloyal-targets
    FILE liblloyal-targets.cmake
    NAMESPACE liblloyal::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/liblloyal
)

# Generate and install package config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/liblloyal-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/liblloyal-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/liblloyal
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/liblloyal-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/liblloyal-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/liblloyal-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/liblloyal
)

# =============================================================================
# Testing (Enable with -DLIBLLOYAL_BUILD_TESTS=ON)
# =============================================================================

option(LIBLLOYAL_BUILD_TESTS "Build liblloyal smoke tests" OFF)

if(LIBLLOYAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# =============================================================================
# Documentation
# =============================================================================

message(STATUS "liblloyal configured:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: C++20")
message(STATUS "  Include Dir: ${CMAKE_CURRENT_SOURCE_DIR}/include")
message(STATUS "  Build Tests: ${LIBLLOYAL_BUILD_TESTS}")
