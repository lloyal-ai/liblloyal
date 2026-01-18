#pragma once

#include <cstdint>
#include <cstdlib>

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

} // namespace TestConfig
