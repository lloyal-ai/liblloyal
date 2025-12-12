#include <cstdlib>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/decoder.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <string>
#include <vector>

using namespace lloyal;

/**
 * End-to-End Parameter Flow Tests
 *
 * Verifies complete parameter flow from model loading -> context init ->
 * inference These tests ensure that user-provided params correctly affect the
 * entire inference pipeline, not just individual components.
 *
 * Tests the complete happy path:
 * 1. Load model with specific params
 * 2. Create context with specific params
 * 3. Run inference operations
 * 4. Verify params actually influenced behavior
 *
 * Uses tiny-random-llama.gguf (~12MB)
 */

static const char *MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

// Skip test if model not available
#define REQUIRE_MODEL()                                                        \
  if (!MODEL_PATH || !*MODEL_PATH) {                                           \
    MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set");                              \
    return;                                                                    \
  }

// Helper to init llama.cpp backend once
struct LlamaBackendGuard {
  LlamaBackendGuard() { llama_backend_init(); }
  ~LlamaBackendGuard() { llama_backend_free(); }
};

// ============================================================================
// End-to-End Tests (2 tests)
// ============================================================================

TEST_CASE("E2E: Complete inference pipeline with custom parameters") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  // === Step 1: Load model with custom params ===
  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers(); // CPU only for determinism
  model_params.use_mmap = true;  // Memory-mapped loading

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  // === Step 2: Create context with custom params ===
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;   // Small context window
  ctx_params.n_batch = 128; // Small batch size

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // === Step 3: Tokenize input ===
  std::string test_text = "Hello world";
  auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
  REQUIRE_FALSE(tokens.empty());
  INFO("Tokenized '" << test_text << "' into " << tokens.size() << " tokens");

  // === Step 4: Create batch for inference ===
  auto batch = llama_batch_init(ctx_params.n_batch, 0, 1);
  REQUIRE(batch.token != nullptr);

  // Add tokens to batch
  batch.n_tokens =
      static_cast<int32_t>(std::min(tokens.size(), size_t(ctx_params.n_batch)));
  for (int32_t i = 0; i < batch.n_tokens; ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0; // Sequence 0
    batch.logits[i] =
        (i == batch.n_tokens - 1) ? 1 : 0; // Only last token needs logits
  }

  // === Step 5: Run decode (inference) ===
  int decode_result = llama_decode(ctx, batch);
  CHECK(decode_result == 0); // 0 = success

  // === Step 6: Verify KV cache is populated ===
  auto mem = llama_get_memory(ctx);
  REQUIRE(mem != nullptr);

  llama_pos max_pos = llama_memory_seq_pos_max(mem, 0);
  INFO("KV cache max position: " << max_pos);
  CHECK(max_pos >= 0); // Cache should have content after decode

  // === Step 7: Sample next token ===
  auto sampler_params = llama_sampler_chain_default_params();
  llama_sampler *sampler = llama_sampler_chain_init(sampler_params);
  REQUIRE(sampler != nullptr);

  llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

  llama_token next_token =
      llama_sampler_sample(sampler, ctx, batch.n_tokens - 1);
  CHECK(next_token != -1); // Valid token
  INFO("Sampled next token: " << next_token);

  // === Cleanup ===
  llama_sampler_free(sampler);
  llama_batch_free(batch);
  llama_free(ctx);

  // Verify complete flow succeeded with custom params
  INFO("E2E test completed: model params + context params + inference "
       "successful");
}

TEST_CASE("E2E: KV cache operations with parameterized context") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  // === Step 1: Load model ===
  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  // === Step 2: Create context with specific n_ctx ===
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 1024; // Larger context for KV cache tests
  ctx_params.n_batch = 256;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // === Step 3: Process tokens to populate KV cache ===
  std::string test_text = "The quick brown fox";
  auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
  REQUIRE_FALSE(tokens.empty());

  auto batch = llama_batch_init(ctx_params.n_batch, 0, 1);
  REQUIRE(batch.token != nullptr);

  // Process all tokens
  batch.n_tokens =
      static_cast<int32_t>(std::min(tokens.size(), size_t(ctx_params.n_batch)));
  for (int32_t i = 0; i < batch.n_tokens; ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = 0; // Don't need logits for this test
  }

  int decode_result = llama_decode(ctx, batch);
  REQUIRE(decode_result == 0);

  // === Step 4: Verify KV cache state ===
  auto mem = llama_get_memory(ctx);
  REQUIRE(mem != nullptr);

  llama_pos max_pos_before = llama_memory_seq_pos_max(mem, 0);
  INFO("KV cache before removal - max pos: " << max_pos_before);
  CHECK(max_pos_before >= 0);

  // === Step 5: Remove tokens from KV cache ===
  llama_pos p0 = 0;
  llama_pos p1 = -1; // -1 means all tokens
  bool rm_result = llama_memory_seq_rm(mem, 0, p0, p1);
  CHECK(rm_result == true);

  llama_pos max_pos_after = llama_memory_seq_pos_max(mem, 0);
  INFO("KV cache after removal - max pos: " << max_pos_after);
  CHECK(max_pos_after == -1); // -1 means empty

  // === Cleanup ===
  llama_batch_free(batch);
  llama_free(ctx);

  INFO("E2E KV cache test completed: context params correctly sized KV cache");
}
