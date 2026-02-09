cmake_minimum_required(VERSION 3.14.0)
project(lloyal_tests)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Fetch doctest using CMake's FetchContent
include(FetchContent)

# Suppress CMake compatibility warnings from doctest
set(CMAKE_POLICY_DEFAULT_CMP0169 OLD CACHE STRING "Allow doctest compatibility" FORCE)

# Suppress doctest's minimum CMake version requirements
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum CMake version policy" FORCE)

FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest
  GIT_TAG        v2.4.11
)

FetchContent_MakeAvailable(doctest)

# Test executable (unit tests with stubs)
add_executable(TestRunner
  main.cpp
  model_registry_test.cpp
  kv_test.cpp
  decoder_test.cpp
  decoder_seq_test.cpp
  tokenizer_test.cpp
  sampler_test.cpp
  chat_template_test.cpp
  embedding_test.cpp
  metrics_test.cpp
  branch_test.cpp
  stubs/llama_stubs.cpp
)

# Find nlohmann/json.hpp from llama.cpp vendor (needed by liblloyal headers)
# Can be overridden via -DNLOHMANN_JSON_DIR=/path/to/nlohmann
if(NOT DEFINED NLOHMANN_JSON_DIR)
  # Default: lloyal.node structure
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp/vendor")
    set(NLOHMANN_JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp/vendor")
  endif()
endif()

# Validate nlohmann/json.hpp is available
if(NOT NLOHMANN_JSON_DIR OR NOT EXISTS "${NLOHMANN_JSON_DIR}/nlohmann/json.hpp")
  message(FATAL_ERROR
    "nlohmann/json.hpp not found. Either:\n"
    "  1. Set -DNLOHMANN_JSON_DIR=/path/to/dir containing nlohmann/\n"
    "  2. Install system package: apt install nlohmann-json3-dev\n"
    "  3. Ensure llama.cpp vendor directory exists at ../../llama.cpp/vendor")
endif()

# Include directories
# IMPORTANT: stubs/ must come BEFORE llama headers so our mock llama/llama.h is found first
target_include_directories(TestRunner PRIVATE
  ./stubs                       # Mock llama.cpp (must be first!)
  ../include                    # liblloyal headers (<lloyal/*.hpp>)
  ${NLOHMANN_JSON_DIR}          # nlohmann/json.hpp from llama.cpp vendor
)

# Link doctest
target_link_libraries(TestRunner PRIVATE doctest::doctest)

# Compiler warnings
target_compile_options(TestRunner PRIVATE -Wall -Wextra)

# Optional: Enable sanitizers (pass -DLLOYAL_ENABLE_SANITIZERS=ON to cmake)
option(LLOYAL_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)
if(LLOYAL_ENABLE_SANITIZERS)
  # Detect compiler for sanitizer flag compatibility
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Enabling sanitizers (Clang: ASan + UBSan + Integer + Float)")
    target_compile_options(TestRunner PRIVATE
      -fsanitize=address
      -fsanitize=undefined
      -fsanitize=integer              # Clang-only: catches unsigned overflow & implicit truncation
      -fsanitize=float-divide-by-zero # Clang-only: catches float division by zero
      -fno-sanitize=unsigned-shift-base  # Clang-only: allow shifting into sign bit
      -fno-omit-frame-pointer
      -g
    )
    target_link_options(TestRunner PRIVATE
      -fsanitize=address
      -fsanitize=undefined
      -fsanitize=integer
      -fsanitize=float-divide-by-zero
    )
  else()
    # GCC: only supports address and undefined sanitizers
    message(STATUS "Enabling sanitizers (GCC: ASan + UBSan)")
    target_compile_options(TestRunner PRIVATE
      -fsanitize=address
      -fsanitize=undefined
      -fno-omit-frame-pointer
      -g
    )
    target_link_options(TestRunner PRIVATE
      -fsanitize=address
      -fsanitize=undefined
    )
  endif()
endif()

# Optional: Custom target for running tests
add_custom_target(run_tests
  COMMAND TestRunner
  DEPENDS TestRunner
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running liblloyal tests..."
)

# ===== INTEGRATION TESTS =====
# Optional: Build integration tests that link against real llama.cpp
# These tests verify llama.cpp behavior hasn't changed in breaking ways
option(LLOYAL_BUILD_INTEGRATION_TESTS "Build integration tests with real llama.cpp" OFF)

if(LLOYAL_BUILD_INTEGRATION_TESTS)
  message(STATUS "Building integration tests with real llama.cpp")

  # Configure llama.cpp paths
  # Can be overridden via -DLLAMA_CPP_DIR=/path/to/llama.cpp
  if(NOT DEFINED LLAMA_CPP_DIR)
    # Default: lloyal.node structure (../../llama.cpp)
    set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../llama.cpp")
  endif()

  # Determine build directory
  # Can be overridden via -DLLAMA_CPP_BUILD_DIR=/path/to/build
  # (useful when lloyal.node build outputs to build/Release/ instead of llama.cpp/build/)
  if(NOT DEFINED LLAMA_CPP_BUILD_DIR)
    # Auto-detect based on platform (for backwards compat with lloyal.node)
    if(EXISTS "${LLAMA_CPP_DIR}/build-apple")
      set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_DIR}/build-apple")
    elseif(EXISTS "${LLAMA_CPP_DIR}/build-linux")
      set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_DIR}/build-linux")
    elseif(EXISTS "${LLAMA_CPP_DIR}/build-windows")
      set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_DIR}/build-windows")
    else()
      # Default to build/ (CI uses this)
      set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_DIR}/build")
    endif()
  endif()

  set(LLAMA_CPP_DYLIB_DIR "${LLAMA_CPP_BUILD_DIR}")
  set(LLAMA_CPP_INCLUDE_DIR "${LLAMA_CPP_DIR}/include")
  set(GGML_INCLUDE_DIR "${LLAMA_CPP_DIR}/ggml/include")

  # Convert to absolute paths (symlinks need absolute paths to resolve correctly)
  get_filename_component(LLAMA_CPP_INCLUDE_DIR "${LLAMA_CPP_INCLUDE_DIR}" REALPATH)
  get_filename_component(GGML_INCLUDE_DIR "${GGML_INCLUDE_DIR}" REALPATH)
  get_filename_component(LLAMA_CPP_DYLIB_DIR "${LLAMA_CPP_DYLIB_DIR}" REALPATH)

  message(STATUS "Using llama.cpp from: ${LLAMA_CPP_DIR}")
  message(STATUS "Using llama.cpp dylib from: ${LLAMA_CPP_DYLIB_DIR}")

  # Integration test executable
  add_executable(IntegrationRunner
    integration/main.cpp
    integration/behavioral_contract_test.cpp
    integration/init_context_test.cpp
    integration/e2e_parameter_flow_test.cpp
    integration/sampler_integration_test.cpp
    integration/clear_and_reseed_test.cpp
    integration/chat_template_integration_test.cpp
    integration/kv_file_persistence_test.cpp
    integration/embedding_integration_test.cpp
    integration/multi_sequence_integration_test.cpp
    integration/branch_integration_test.cpp
  )

  # Create llama/ subdirectory and symlink headers so #include <llama/llama.h> works
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/llama)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${LLAMA_CPP_INCLUDE_DIR}/llama.h
      ${CMAKE_CURRENT_BINARY_DIR}/include/llama/llama.h
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${GGML_INCLUDE_DIR}/ggml.h
      ${CMAKE_CURRENT_BINARY_DIR}/include/llama/ggml.h
  )

  # Common library paths
  set(LLAMA_CPP_COMMON_DIR "${LLAMA_CPP_DIR}/common")
  set(LLAMA_CPP_VENDOR_DIR "${LLAMA_CPP_DIR}/vendor")

  target_include_directories(IntegrationRunner PRIVATE
    ../include                              # liblloyal headers
    ${CMAKE_CURRENT_BINARY_DIR}/include     # For llama/llama.h
    ${LLAMA_CPP_INCLUDE_DIR}                # llama.cpp headers
    ${GGML_INCLUDE_DIR}                     # ggml headers
    ${LLAMA_CPP_COMMON_DIR}                 # common library headers (common.h, chat.h, etc.)
    ${LLAMA_CPP_VENDOR_DIR}                 # vendor headers (nlohmann/json.hpp)
  )

  # Find common library (static)
  # lloyal-node builds to: build/llama.cpp/common/libcommon.a (parent of Release/)
  # Standalone llama.cpp builds to: build/common/libcommon.a
  get_filename_component(LLAMA_BUILD_PARENT "${LLAMA_CPP_DYLIB_DIR}" DIRECTORY)

  find_library(LLAMA_COMMON_LIB
    NAMES common libcommon
    PATHS
      ${LLAMA_BUILD_PARENT}/llama.cpp/common   # lloyal-node: build/llama.cpp/common
      ${LLAMA_CPP_DYLIB_DIR}/llama.cpp/common  # alternative
      ${LLAMA_CPP_DYLIB_DIR}/common            # standalone llama.cpp
      ${LLAMA_CPP_DYLIB_DIR}
    NO_DEFAULT_PATH
  )

  if(NOT LLAMA_COMMON_LIB)
    # Fallback to expected paths (cross-platform static lib names)
    set(_COMMON_PATHS
      "${LLAMA_BUILD_PARENT}/llama.cpp/common"
      "${LLAMA_CPP_DYLIB_DIR}/llama.cpp/common"
      "${LLAMA_CPP_DYLIB_DIR}/common"
      "${LLAMA_CPP_DYLIB_DIR}"
    )
    foreach(_path ${_COMMON_PATHS})
      if(EXISTS "${_path}/libcommon.a")
        set(LLAMA_COMMON_LIB "${_path}/libcommon.a")
        break()
      elseif(EXISTS "${_path}/common.lib")
        set(LLAMA_COMMON_LIB "${_path}/common.lib")
        break()
      endif()
    endforeach()
  endif()

  # Validate common library was found
  if(NOT LLAMA_COMMON_LIB)
    message(FATAL_ERROR
      "libcommon not found. Ensure llama.cpp was built with LLAMA_BUILD_COMMON=ON.\n"
      "Searched paths:\n"
      "  - ${LLAMA_BUILD_PARENT}/llama.cpp/common\n"
      "  - ${LLAMA_CPP_DYLIB_DIR}/common\n"
      "  - ${LLAMA_CPP_DYLIB_DIR}")
  endif()

  message(STATUS "Common library: ${LLAMA_COMMON_LIB}")

  # Find llama library (cross-platform)
  find_library(LLAMA_LIB
    NAMES llama
    PATHS ${LLAMA_CPP_DYLIB_DIR}
    NO_DEFAULT_PATH
  )
  if(NOT LLAMA_LIB)
    # Fallback: construct expected path
    set(LLAMA_LIB "${LLAMA_CPP_DYLIB_DIR}/libllama${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()

  # Find ggml libraries if separate (lloyal-node structure has separate dylibs)
  find_library(GGML_LIB NAMES ggml PATHS ${LLAMA_CPP_DYLIB_DIR} NO_DEFAULT_PATH)
  find_library(GGML_BASE_LIB NAMES ggml-base PATHS ${LLAMA_CPP_DYLIB_DIR} NO_DEFAULT_PATH)

  set(LLAMA_LIBS ${LLAMA_LIB})
  if(GGML_LIB)
    list(APPEND LLAMA_LIBS ${GGML_LIB})
  endif()
  if(GGML_BASE_LIB)
    list(APPEND LLAMA_LIBS ${GGML_BASE_LIB})
  endif()

  target_link_libraries(IntegrationRunner PRIVATE
    ${LLAMA_COMMON_LIB}
    ${LLAMA_LIBS}
    doctest::doctest
  )

  # Set RPATH so dylib can be found at runtime
  set_target_properties(IntegrationRunner PROPERTIES
    BUILD_RPATH "${LLAMA_CPP_DYLIB_DIR}"
    INSTALL_RPATH "${LLAMA_CPP_DYLIB_DIR}"
  )

  # Optional: Add test to CTest
  add_test(NAME integration_tests COMMAND IntegrationRunner)
  set_tests_properties(integration_tests PROPERTIES
    LABELS "integration;llama"
    TIMEOUT 180
  )
endif()
