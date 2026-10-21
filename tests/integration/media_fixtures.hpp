#pragma once

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <llama/llama.h>
#include <mtmd.h>
#include "test_config.hpp"

/**
 * What the media cases share: fixture bytes, and one projector per process.
 */
namespace TestConfig {

/** A fixture's bytes, or empty when it cannot be read. */
inline std::vector<uint8_t> read_fixture(const char* name) {
  const std::string path = std::string(LLOYAL_TEST_FIXTURES_DIR) + "/" + name;
  std::vector<uint8_t> bytes;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return bytes;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n > 0) {
    bytes.resize(static_cast<size_t>(n));
    if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
  }
  std::fclose(f);
  return bytes;
}

struct MtmdDeleter {
  void operator()(mtmd_context* c) const { if (c) mtmd_free(c); }
};

/**
 * The projector named by LLAMA_MMPROJ_MODEL, loaded once for the process. A
 * projector is hundreds of megabytes and init runs a warmup encode. Mirrors
 * acquire_test_model(); constructed after the model, so it is destroyed before
 * it (mtmd holds a borrowed llama_model*).
 */
inline mtmd_context* acquire_mtmd(const llama_model* model) {
  static std::unique_ptr<mtmd_context, MtmdDeleter> cached;
  if (!cached) {
    const char* path = std::getenv("LLAMA_MMPROJ_MODEL");
    if (!path || !*path) return nullptr;
    auto p = mtmd_context_params_default();
    p.use_gpu        = n_gpu_layers() != 0;
    p.print_timings  = false;
    p.n_threads      = 4;
    p.warmup         = false;  // each media case executes the encoder
    cached.reset(mtmd_init_from_file(path, model, p));
  }
  return cached.get();
}

}  // namespace TestConfig
