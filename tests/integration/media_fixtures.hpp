#pragma once

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <doctest/doctest.h>
#include <llama/llama.h>
#include <lloyal/mtmd.hpp>
#include <mtmd.h>
#include "test_config.hpp"

/**
 * What the media cases share: fixture bytes, the committed clip, and one
 * projector per process.
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
 * The projector named by LLAMA_MMPROJ_MODEL, loaded once for the process per
 * image budget. A projector is hundreds of megabytes and init runs a warmup
 * encode. Mirrors acquire_test_model(); constructed after the model, so it is
 * destroyed before it (mtmd holds a borrowed llama_model*).
 *
 * `image_tokens` pins both ends of the projector's image budget, as llama.cpp's
 * --image-min-tokens and --image-max-tokens do; 0 keeps the projector's own.
 */
inline mtmd_context* acquire_mtmd(const llama_model* model, int32_t image_tokens = 0) {
  static std::map<int32_t, std::unique_ptr<mtmd_context, MtmdDeleter>> cached;
  auto& slot = cached[image_tokens];
  if (!slot) {
    const char* path = std::getenv("LLAMA_MMPROJ_MODEL");
    if (!path || !*path) return nullptr;
    auto p = mtmd_context_params_default();
    p.use_gpu          = n_gpu_layers() != 0;
    p.print_timings    = false;
    p.n_threads        = 4;
    p.warmup           = false;  // each media case executes the encoder
    p.image_min_tokens = image_tokens > 0 ? image_tokens : p.image_min_tokens;
    p.image_max_tokens = image_tokens > 0 ? image_tokens : p.image_max_tokens;
    slot.reset(mtmd_init_from_file(path, model, p));
  }
  return slot.get();
}

/** The committed clip's 16 frames, sampled at 4 per second (fixtures/video). */
inline std::vector<std::vector<uint8_t>> video_frames() {
  std::vector<std::vector<uint8_t>> frames;
  for (int i = 0; i < 16; ++i) {
    char name[32];
    std::snprintf(name, sizeof name, "video/frame-%02d.png", i);
    frames.push_back(read_fixture(name));
    REQUIRE_MESSAGE(!frames.back().empty(), "fixtures/video frames missing");
  }
  return frames;
}

/// Borrowed views over sampled frames, in the shape MediaInput::Kind::Video takes.
struct Clip {
  std::vector<std::vector<uint8_t>> bytes = video_frames();
  std::vector<std::span<const uint8_t>> frames;
  std::vector<int64_t> timestamps_ms;
  std::vector<lloyal::VideoNote> notes;

  explicit Clip(int64_t frame_ms = 250) {
    for (size_t i = 0; i < bytes.size(); ++i) {
      frames.emplace_back(bytes[i]);
      timestamps_ms.push_back(static_cast<int64_t>(i) * frame_ms);
    }
  }

  lloyal::MediaInput input(int64_t interval_ms = 5000) const {
    return lloyal::MediaInput{lloyal::MediaInput::Kind::Video, {},
                              lloyal::VideoFrames{frames, timestamps_ms, interval_ms, notes}};
  }
};

}  // namespace TestConfig
