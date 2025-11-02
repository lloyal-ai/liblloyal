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
  tokenizer_test.cpp
  sampler_test.cpp
  stubs/llama_stubs.cpp
)

# Include directories
# IMPORTANT: stubs/ must come BEFORE llama headers so our mock llama/llama.h is found first
target_include_directories(TestRunner PRIVATE
  ./stubs                       # Mock llama.cpp (must be first!)
  ../include                    # liblloyal headers (<lloyal/*.hpp>)
)

# Link doctest
target_link_libraries(TestRunner PRIVATE doctest::doctest)

# Compiler warnings
target_compile_options(TestRunner PRIVATE -Wall -Wextra)

# Optional: Enable sanitizers (pass -DLLOYAL_ENABLE_SANITIZERS=ON to cmake)
option(LLOYAL_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan" OFF)
if(LLOYAL_ENABLE_SANITIZERS)
  message(STATUS "Enabling sanitizers (AddressSanitizer + UBSan)")
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

  # Default to liblloyal's vendored pre-built xcframework (can be overridden)
  if(NOT DEFINED LLAMA_CPP_FRAMEWORK_PATH)
    # Pin to specific llama.cpp version (tag b6870, commit 338074c3)
    set(LLAMA_CPP_VERSION "b6870")
    set(LLAMA_CPP_FRAMEWORK_PATH "lib/${LLAMA_CPP_VERSION}/llama.xcframework/macos-arm64_x86_64/llama.framework")
    message(STATUS "Using vendored llama.cpp framework: ${LLAMA_CPP_VERSION} (${LLAMA_CPP_FRAMEWORK_PATH})")
  endif()

  # Integration test executable
  add_executable(IntegrationRunner
    integration/main.cpp
    integration/behavioral_contract_test.cpp
    integration/init_context_test.cpp
    integration/e2e_parameter_flow_test.cpp
    integration/sampler_integration_test.cpp
    integration/clear_and_reseed_test.cpp
    integration/rope_position_invariant_test.cpp
  )

  # Create llama/ subdirectory and symlink headers so #include <llama/llama.h> works
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/llama)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_SOURCE_DIR}/${LLAMA_CPP_FRAMEWORK_PATH}/Headers/llama.h
      ${CMAKE_CURRENT_BINARY_DIR}/include/llama/llama.h
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_SOURCE_DIR}/${LLAMA_CPP_FRAMEWORK_PATH}/Headers/ggml.h
      ${CMAKE_CURRENT_BINARY_DIR}/include/llama/ggml.h
  )

  target_include_directories(IntegrationRunner PRIVATE
    ../include                              # liblloyal headers
    ${CMAKE_CURRENT_BINARY_DIR}/include     # For llama/llama.h
    ${LLAMA_CPP_FRAMEWORK_PATH}/Headers     # For other headers
  )

  target_link_libraries(IntegrationRunner PRIVATE
    "-framework Foundation"
    "-F${CMAKE_CURRENT_SOURCE_DIR}/${LLAMA_CPP_FRAMEWORK_PATH}/.."
    "-framework llama"
    doctest::doctest
  )

  # Set RPATH so framework can be found at runtime
  set_target_properties(IntegrationRunner PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/${LLAMA_CPP_FRAMEWORK_PATH}/.."
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/${LLAMA_CPP_FRAMEWORK_PATH}/.."
  )

  # Optional: Add test to CTest
  add_test(NAME integration_tests COMMAND IntegrationRunner)
  set_tests_properties(integration_tests PROPERTIES
    LABELS "integration;llama"
    TIMEOUT 180
  )
endif()
