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
#   - llama.cpp (b8119 or compatible)
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
# llama.cpp Common Library Include Path
# =============================================================================
# When building with LLAMA_BUILD_COMMON=ON, add paths for:
#   - common/*.h headers (chat.h, common.h, json-schema-to-grammar.h)
#   - vendor/nlohmann/* for JSON support

if(TARGET common)
    if(TARGET llama)
        # Get llama.cpp source directory from the llama target
        get_target_property(LLAMA_INCLUDE_DIRS llama INTERFACE_INCLUDE_DIRECTORIES)
        foreach(dir ${LLAMA_INCLUDE_DIRS})
            if(EXISTS "${dir}/llama.h")
                get_filename_component(LLAMA_CPP_ROOT "${dir}" DIRECTORY)
                target_include_directories(liblloyal INTERFACE
                    $<BUILD_INTERFACE:${LLAMA_CPP_ROOT}/common>
                    $<BUILD_INTERFACE:${LLAMA_CPP_ROOT}/vendor>
                )
                message(STATUS "liblloyal: Added common library includes from ${LLAMA_CPP_ROOT}")
                break()
            endif()
        endforeach()
    endif()
    # Link common library so consumers get symbols for common_chat_*, json_schema_to_grammar, etc.
    target_link_libraries(liblloyal INTERFACE common)
endif()

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
# Include Path Setup for Consumers
# =============================================================================
# liblloyal headers use #include <llama/llama.h> but llama.cpp provides
# include/llama.h (flat structure). We create the expected namespace structure
# in the build directory at configure time using wrapper headers.
#
# Why wrapper headers? Cross-platform compatibility without symbol conflicts.
# - Symlinks fail on Windows (require admin/Developer Mode)
# - Copying creates duplicate definitions when llama target exports its paths
# - Wrapper headers forward-include, creating the namespace without duplication
#
# Template files: cmake/templates/{llama,ggml}.h.in
# Generated at configure time via configure_file() with @ONLY

if(TARGET llama)
    # Get llama.cpp include directory from llama target
    get_target_property(LLAMA_INCLUDE_DIRS llama INTERFACE_INCLUDE_DIRECTORIES)

    foreach(dir ${LLAMA_INCLUDE_DIRS})
        if(EXISTS "${dir}/llama.h")
            set(LLAMA_CPP_INCLUDE_DIR "${dir}")
            break()
        endif()
    endforeach()

    # Get ggml include directory from ggml or ggml-base target (if exists)
    # Also check relative to llama.cpp source dir as fallback
    if(TARGET ggml)
        get_target_property(GGML_INCLUDE_DIRS ggml INTERFACE_INCLUDE_DIRECTORIES)
        foreach(dir ${GGML_INCLUDE_DIRS})
            if(EXISTS "${dir}/ggml.h")
                set(GGML_INCLUDE_DIR "${dir}")
                break()
            endif()
        endforeach()
    endif()
    
    if(NOT GGML_INCLUDE_DIR AND TARGET ggml-base)
        get_target_property(GGML_INCLUDE_DIRS ggml-base INTERFACE_INCLUDE_DIRECTORIES)
        foreach(dir ${GGML_INCLUDE_DIRS})
            if(EXISTS "${dir}/ggml.h")
                set(GGML_INCLUDE_DIR "${dir}")
                break()
            endif()
        endforeach()
    endif()
    
    # Fallback: look relative to llama.cpp include dir (ggml is sibling)
    if(NOT GGML_INCLUDE_DIR AND LLAMA_CPP_INCLUDE_DIR)
        # llama.h is in llama.cpp/include, ggml.h is in llama.cpp/ggml/include
        get_filename_component(LLAMA_CPP_ROOT "${LLAMA_CPP_INCLUDE_DIR}" DIRECTORY)
        set(GGML_FALLBACK_DIR "${LLAMA_CPP_ROOT}/ggml/include")
        if(EXISTS "${GGML_FALLBACK_DIR}/ggml.h")
            set(GGML_INCLUDE_DIR "${GGML_FALLBACK_DIR}")
        endif()
    endif()

    # Create llama/ namespace with wrapper headers at configure time
    if(LLAMA_CPP_INCLUDE_DIR)
        set(PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include/llama)
        file(MAKE_DIRECTORY ${PUBLIC_INCLUDE_DIR})

        # Generate wrapper headers from templates
        # These forward-include the actual headers, creating the llama/ namespace
        # without duplicating definitions
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/llama.h.in
            ${PUBLIC_INCLUDE_DIR}/llama.h
            @ONLY
        )

        if(GGML_INCLUDE_DIR)
            configure_file(
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/ggml.h.in
                ${PUBLIC_INCLUDE_DIR}/ggml.h
                @ONLY
            )
        endif()

        # Export build/include BEFORE llama.cpp's paths so our wrappers are found first
        target_include_directories(liblloyal BEFORE INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        )

        message(STATUS "liblloyal: Created llama/ include namespace at ${PUBLIC_INCLUDE_DIR}")
    else()
        message(WARNING "liblloyal: Could not find llama.cpp include directory. Consumers may need manual include path setup.")
    endif()
endif()

# =============================================================================
# Version Check for llama.cpp
# =============================================================================

# liblloyal is tested against llama.cpp b8119
# This is a soft check - we warn but don't fail the build
if(DEFINED LLAMA_CPP_VERSION)
    if(NOT LLAMA_CPP_VERSION STREQUAL "b8119")
        message(WARNING
            "liblloyal: Designed for llama.cpp b8119, found ${LLAMA_CPP_VERSION}. "
            "API compatibility not guaranteed.")
    else()
        message(STATUS "liblloyal: Using recommended llama.cpp version b8119")
    endif()
else()
    message(STATUS
        "liblloyal: LLAMA_CPP_VERSION not set. Recommended: b8119. "
        "Define LLAMA_CPP_VERSION in your llama.cpp CMakeLists to enable version check.")
endif()

# =============================================================================
# Compiler Requirements
# =============================================================================

target_compile_features(liblloyal INTERFACE cxx_std_20)

# =============================================================================
# Installation (Optional - for package managers)
# =============================================================================
# Only configure install rules when liblloyal is the top-level project.
# When used via add_subdirectory(), the parent project handles installation.

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
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
endif()

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
