#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <llama/llama.h>
#include <lloyal/model_registry.hpp>

/**
 * Test configuration from environment variables.
 * Allows runtime configuration of test behavior without recompilation.
 */
namespace TestConfig {

// Get n_gpu_layers from env (default 0 for CPU, use -1 for all layers on Metal)
inline int32_t n_gpu_layers() {
  const char* env = std::getenv("LLAMA_N_GPU_LAYERS");
  return env ? std::atoi(env) : 0;
}

// Shared model cache — keeps ModelRegistry's weak_ptr alive across TEST_CASEs
// so weights are loaded from disk only once for the entire test suite.
// Uses inline + static local to guarantee a single instance across all TUs (C++17).
inline std::shared_ptr<llama_model> acquire_test_model() {
  static std::shared_ptr<llama_model> cached;
  if (!cached) {
    const char* path = std::getenv("LLAMA_TEST_MODEL");
    if (!path || !*path) return nullptr;
    auto params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers();
    cached = lloyal::ModelRegistry::acquire(path, params);
  }
  return cached;
}

} // namespace TestConfig
