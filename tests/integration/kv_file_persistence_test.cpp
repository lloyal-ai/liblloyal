#include <doctest/doctest.h>
#include "test_config.hpp"
#include <lloyal/decoder.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <llama/llama.h>

#include <filesystem>
#include <fstream>

using namespace lloyal;

/**
 * Integration Tests: KV File Persistence
 *
 * Validates write_file/read_file with real llama.cpp models
 * Tests full round-trip: save → load → verify state
 *
 * REQUIRES: LLAMA_TEST_MODEL env var (any valid model)
 */

static const char *MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

#define REQUIRE_MODEL()                                                        \
  if (!MODEL_PATH || !*MODEL_PATH) {                                           \
    MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set");                              \
    return;                                                                    \
  }

struct LlamaBackendGuard {
  LlamaBackendGuard() { llama_backend_init(); }
  ~LlamaBackendGuard() { llama_backend_free(); }
};

TEST_CASE("Integration: write_file/read_file round-trip") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // 1. Populate KV cache with known tokens
  std::string test_text = "The quick brown fox jumps over the lazy dog.";
  auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
  REQUIRE_FALSE(tokens.empty());

  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch);

  // 2. Write state to file
  const std::string filepath = "test_session.llama";
  size_t bytes_written = kv::write_file(ctx, 0, filepath, tokens);

  REQUIRE(bytes_written > 0);
  REQUIRE(std::filesystem::exists(filepath));

  INFO("File written: " << bytes_written << " bytes");

  // 3. Clear KV cache
  kv::clear_all(ctx);
  CHECK(kv::pos_max(ctx, 0) == -1);

  // 4. Read state from file
  auto data = kv::read_file(ctx, 0, filepath);

  REQUIRE(data.bytes_read == bytes_written);
  REQUIRE(data.tokens.size() == tokens.size());

  // Verify tokens match
  for (size_t i = 0; i < tokens.size(); ++i) {
    CHECK(data.tokens[i] == tokens[i]);
  }

  // 5. Verify KV state restored
  llama_pos max_pos_after = kv::pos_max(ctx, 0);
  CHECK(max_pos_after == static_cast<llama_pos>(tokens.size() - 1));

  // Cleanup
  std::filesystem::remove(filepath);
  llama_free(ctx);
}

TEST_CASE("Integration: write_file creates valid session format") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 256;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  // Populate cache
  std::vector<llama_token> tokens = {1, 100, 200, 300, 400};
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch);

  // Write file
  const std::string filepath = "validation_test.llama";
  size_t bytes = kv::write_file(ctx, 0, filepath, tokens);
  REQUIRE(bytes > 0);

  // Verify file format manually
  std::ifstream file(filepath, std::ios::binary);
  REQUIRE(file.is_open());

  // Read magic and version
  uint32_t magic = 0, version = 0;
  file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  file.read(reinterpret_cast<char *>(&version), sizeof(version));

  INFO("Magic: " << std::hex << magic);
  INFO("Version: " << version);

  // Verify magic matches llama.cpp format
  CHECK(magic != 0);   // Should have valid magic
  CHECK(version != 0); // Should have valid version

  // Cleanup
  file.close();
  std::filesystem::remove(filepath);
  llama_free(ctx);
}

TEST_CASE("Integration: read_file rejects invalid files") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 256;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  SUBCASE("Non-existent file throws") {
    CHECK_THROWS_AS(kv::read_file(ctx, 0, "nonexistent.llama"),
                    std::runtime_error);
  }

  SUBCASE("Invalid magic/version throws") {
    // Create file with bad magic
    const std::string bad_file = "bad_magic.llama";
    std::ofstream out(bad_file, std::ios::binary);
    uint32_t bad_magic = 0xDEADBEEF;
    out.write(reinterpret_cast<const char *>(&bad_magic), sizeof(bad_magic));
    out.close();

    CHECK_THROWS_AS(kv::read_file(ctx, 0, bad_file), std::runtime_error);

    std::filesystem::remove(bad_file);
  }

  llama_free(ctx);
}

TEST_CASE("Integration: write_file/read_file preserves generation capability") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Setup: decode prompt
  std::string prompt = "Hello world";
  auto tokens = tokenizer::tokenize(vocab, prompt, false, false);
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch);

  // Sample one token BEFORE save
  llama_token token_before = sampler::greedy(ctx, vocab);

  // Save state
  const std::string filepath = "gen_test.llama";
  kv::write_file(ctx, 0, filepath, tokens);

  // Clear and restore
  kv::clear_all(ctx);
  auto data = kv::read_file(ctx, 0, filepath);

  // Sample one token AFTER restore
  llama_token token_after = sampler::greedy(ctx, vocab);

  // Should produce identical token (deterministic)
  CHECK(token_after == token_before);

  // Cleanup
  std::filesystem::remove(filepath);
  llama_free(ctx);
}

TEST_CASE("Integration: write_file/read_file with larger context") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 1024; // Larger context

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Generate a longer token sequence
  std::string long_text =
      "The quick brown fox jumps over the lazy dog. "
      "This is a longer text to test file persistence with larger contexts. "
      "We want to ensure that the file format can handle multiple sentences "
      "and a reasonable number of tokens without issues.";

  auto tokens = tokenizer::tokenize(vocab, long_text, false, false);
  REQUIRE(tokens.size() > 20); // Ensure we have a reasonable token count

  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch);

  // Write and read
  const std::string filepath = "large_context_test.llama";
  size_t bytes_written = kv::write_file(ctx, 0, filepath, tokens);
  REQUIRE(bytes_written > 0);

  INFO("Large context file size: " << bytes_written << " bytes for "
                                    << tokens.size() << " tokens");

  // Clear and restore
  kv::clear_all(ctx);
  auto data = kv::read_file(ctx, 0, filepath);

  // Verify all tokens match
  REQUIRE(data.tokens.size() == tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    CHECK(data.tokens[i] == tokens[i]);
  }

  // Verify KV state
  llama_pos max_pos = kv::pos_max(ctx, 0);
  CHECK(max_pos == static_cast<llama_pos>(tokens.size() - 1));

  // Cleanup
  std::filesystem::remove(filepath);
  llama_free(ctx);
}
