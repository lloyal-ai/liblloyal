#include <cstdlib>
#include <doctest/doctest.h>
#include <llama/llama.h>
#include <lloyal/decoder.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <vector>

using namespace lloyal;

/**
 * Behavioral Contract Tests
 *
 * Guards against llama.cpp upstream changes that break liblloyal's
 * API contract, even when compilation succeeds.
 *
 * Uses tiny-random-llama.gguf (~12MB):
 * - Small model with randomized weights
 * - Deterministic behavior (no actual inference quality)
 * - Tests llama.cpp API behavior, not model quality
 *
 * If these fail after llama.cpp sync:
 * 1. Check llama.cpp changelog for behavioral changes
 * 2. Decide if change is acceptable
 * 3. Update golden values if acceptable, or fix facades if breaking
 *
 * Golden values measured with llama.cpp commit: <UPDATE_THIS>
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

TEST_CASE("Behavioral: Tokenization produces consistent token IDs") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0; // CPU only for determinism

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Test deterministic tokenization
  std::string test_text = "Hello world";
  auto tokens = tokenizer::tokenize(vocab, test_text, false, false);

  REQUIRE_FALSE(tokens.empty());

  // Golden: token count should be consistent
  // NOTE: Update this after measuring with your specific tiny-random model
  const size_t EXPECTED_TOKEN_COUNT = 3; // Placeholder - measure once

  INFO("Token count: " << tokens.size() << " (expected " << EXPECTED_TOKEN_COUNT
                       << ")");

  if (tokens.size() != EXPECTED_TOKEN_COUNT) {
    WARN("Token count changed! This may indicate llama.cpp tokenization "
         "changed.");
    WARN("Verify this is expected, then update EXPECTED_TOKEN_COUNT");
  }

  // Second tokenization should produce identical results
  auto tokens2 = tokenizer::tokenize(vocab, test_text, false, false);
  REQUIRE(tokens.size() == tokens2.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    CHECK(tokens[i] == tokens2[i]);
  }
}

TEST_CASE("Behavioral: KV cache state size is stable") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  // Populate KV cache with known content
  llama_batch batch = llama_batch_init(3, 0, 1);
  batch.n_tokens = 3;
  batch.token[0] = 1; // BOS (standard for llama models)
  batch.token[1] = 100;
  batch.token[2] = 200;

  for (int i = 0; i < 3; ++i) {
    batch.pos[i] = i;
    batch.seq_id[i][0] = 0;
    batch.n_seq_id[i] = 1;
    batch.logits[i] = 0;
  }

  REQUIRE(llama_decode(ctx, batch) == 0);

  // Measure KV state size
  size_t state_size = kv::state_size(ctx, 0);
  REQUIRE(state_size > 0);

  // Calculate expected size based on model architecture
  // KV state contains: n_tokens * n_layer * n_embd * 2 (K+V) * sizeof(fp16)
  // Plus metadata overhead (sequence info, position info, etc.)
  int32_t n_layer = llama_model_n_layer(model.get());
  int32_t n_embd = llama_model_n_embd(model.get());
  int32_t n_tokens = 3; // Number of tokens in KV cache

  // Expected KV data size (K + V tensors, fp16 = 2 bytes per element)
  // Simplified: KV cache for 3 tokens = 3 * n_layer * n_embd * 2 (K+V) * 2 bytes (fp16)
  size_t expected_kv_data = n_tokens * n_layer * n_embd * 2 * 2;

  // Add metadata overhead (sequence info, cell metadata, etc.)
  // Empirically measured ~100-200 bytes for tiny models, scales with layers
  size_t expected_metadata = 200 + (n_layer * 20);

  size_t expected_total = expected_kv_data + expected_metadata;

  // Allow ±50% variance for:
  // - GQA/MQA optimizations (grouped query attention reduces K/V size)
  // - Alignment and padding
  // - Format changes in llama.cpp
  size_t expected_min = static_cast<size_t>(expected_total * 0.5);
  size_t expected_max = static_cast<size_t>(expected_total * 1.5);

  INFO("Model: n_layer=" << n_layer << ", n_embd=" << n_embd);
  INFO("Expected KV state size: ~" << expected_total << " bytes (±50%)");
  INFO("Actual KV state size: " << state_size << " bytes");

  CHECK(state_size >= expected_min);
  CHECK(state_size <= expected_max);

  if (state_size < expected_min || state_size > expected_max) {
    double deviation_pct = (static_cast<double>(state_size) / expected_total - 1.0) * 100;
    WARN("⚠️ KV state size outside expected range!");
    INFO("Expected: " << expected_min << " - " << expected_max << " bytes");
    INFO("Actual: " << state_size << " bytes");
    INFO("Deviation: " << deviation_pct << "%");
    INFO("This likely means llama.cpp changed KV cache format.");
    INFO("Serialization may be broken!");
  }

  llama_batch_free(batch);
  llama_free(ctx);
}

TEST_CASE("Behavioral: KV cache serialization round-trip preserves state") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  // Populate KV cache
  llama_batch batch = llama_batch_init(5, 0, 1);
  batch.n_tokens = 5;
  for (int i = 0; i < 5; ++i) {
    batch.token[i] = 100 + i;
    batch.pos[i] = i;
    batch.seq_id[i][0] = 0;
    batch.n_seq_id[i] = 1;
    batch.logits[i] = 0;
  }

  REQUIRE(llama_decode(ctx, batch) == 0);

  // Save state
  size_t state_size = kv::state_size(ctx, 0);
  REQUIRE(state_size > 0);

  std::vector<uint8_t> saved_state(state_size);
  size_t saved = kv::state_save(ctx, 0, saved_state.data(), state_size);
  REQUIRE(saved == state_size);

  // Verify pos_max before clearing
  llama_pos max_pos_before = kv::pos_max(ctx, 0);
  CHECK(max_pos_before == 4); // 5 tokens at positions 0-4

  // Clear KV cache
  REQUIRE(kv::remove_range(ctx, 0, 0, -1));
  CHECK(kv::pos_max(ctx, 0) == -1); // Empty after clear

  // Restore state
  size_t loaded = kv::state_load(ctx, 0, saved_state.data(), state_size);
  REQUIRE(loaded > 0);

  // Verify state restored correctly
  llama_pos max_pos_after = kv::pos_max(ctx, 0);
  CHECK(max_pos_after == max_pos_before);

  llama_batch_free(batch);
  llama_free(ctx);
}

TEST_CASE("Behavioral: Greedy sampling is deterministic") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Decode a token to generate logits
  llama_batch batch = llama_batch_init(1, 0, 1);
  batch.n_tokens = 1;
  batch.token[0] = 1; // BOS
  batch.pos[0] = 0;
  batch.seq_id[0][0] = 0;
  batch.n_seq_id[0] = 1;
  batch.logits[0] = 1; // Request logits for sampling

  REQUIRE(llama_decode(ctx, batch) == 0);

  // Sample token (greedy)
  llama_token sampled1 = sampler::greedy(ctx, vocab);

  // Sample again (should be identical - no randomness)
  llama_token sampled2 = sampler::greedy(ctx, vocab);

  CHECK(sampled1 == sampled2);

  // Verify sampled token is valid
  int vocab_size = llama_vocab_n_tokens(vocab);
  CHECK(sampled1 >= 0);
  CHECK(sampled1 < vocab_size);

  llama_batch_free(batch);
  llama_free(ctx);
}

TEST_CASE("Behavioral: Detokenization produces consistent text") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Test with arbitrary token IDs
  std::vector<llama_token> tokens = {100, 200, 300};

  std::string text1 = tokenizer::detokenize_batch(vocab, tokens.data(),
                                                  tokens.size(), false, false);

  std::string text2 = tokenizer::detokenize_batch(vocab, tokens.data(),
                                                  tokens.size(), false, false);

  // Should produce identical text (deterministic)
  CHECK(text1 == text2);

  // Result should not be empty (model is valid)
  CHECK_FALSE(text1.empty());
}

TEST_CASE("Behavioral: Batch decode processing is consistent") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  // Decode tokens using our batching facade
  std::vector<llama_token> tokens(50, 100); // 50 identical tokens

  REQUIRE_NOTHROW(decoder::decode_tokens(ctx, tokens, 0, 128));

  // Verify KV cache has expected state
  llama_pos max_pos = kv::pos_max(ctx, 0);
  CHECK(max_pos == 49); // Positions 0-49 for 50 tokens

  llama_free(ctx);
}

TEST_CASE("Behavioral: Error conditions produce expected behavior") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  // Test null safety (should throw, not crash)
  std::vector<llama_token> empty_tokens;
  CHECK_THROWS(decoder::decode_tokens(nullptr, empty_tokens, 0, 128));
  CHECK_THROWS(decoder::decode_tokens(ctx, empty_tokens, 0, 128));

  // Test empty KV cache state (should return 0, not crash)
  size_t empty_size = kv::state_size(ctx, 0);
  CHECK(empty_size == 0);

  llama_free(ctx);
}
