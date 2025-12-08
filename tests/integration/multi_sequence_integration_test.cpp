#include <cstdlib>
#include <doctest/doctest.h>
#include <llama/llama.h>
#include <lloyal/decoder.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <string>
#include <vector>

using namespace lloyal;

/**
 * Multi-Sequence Integration Tests
 *
 * Validates that the seq_id parameter in decode_tokens() works correctly
 * with real llama.cpp. Tests:
 * - Decoding to different sequences populates different KV regions
 * - Sequences are isolated (clearing one doesn't affect others)
 * - Backward compatibility (default seq_id=0 works as before)
 *
 * Uses tiny-random-llama.gguf (~12MB)
 *
 * NOTE: These tests require a context initialized with n_seq_max > 1
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

// ============================================================================
// Multi-Sequence Tests
// ============================================================================

TEST_CASE("Integration: multi-sequence decode populates different KV regions") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  // Load model
  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  // Create context with n_seq_max = 4 (support 4 parallel sequences)
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;  // Enable multi-sequence

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Tokenize test prompt
  std::string test_text = "Hello world";
  auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
  REQUIRE_FALSE(tokens.empty());
  INFO("Tokenized '" << test_text << "' into " << tokens.size() << " tokens");

  // Decode to sequence 0
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 0);
  llama_pos pos_seq0 = kv::pos_max(ctx, 0);
  INFO("Seq 0 pos_max after decode: " << pos_seq0);
  CHECK(pos_seq0 >= 0);  // Should have tokens

  // Decode to sequence 1 (independent)
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 1);
  llama_pos pos_seq1 = kv::pos_max(ctx, 1);
  INFO("Seq 1 pos_max after decode: " << pos_seq1);
  CHECK(pos_seq1 >= 0);  // Should have tokens

  // Both sequences should have the same number of tokens
  CHECK(pos_seq0 == pos_seq1);

  // Seq 0 should be unchanged after decoding to seq 1
  llama_pos pos_seq0_after = kv::pos_max(ctx, 0);
  CHECK(pos_seq0_after == pos_seq0);

  llama_free(ctx);
}

TEST_CASE("Integration: clearing one sequence doesn't affect others") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Test input", false, false);
  REQUIRE_FALSE(tokens.empty());

  // Decode to both sequences
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 0);
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 1);

  llama_pos pos_seq0_before = kv::pos_max(ctx, 0);
  llama_pos pos_seq1_before = kv::pos_max(ctx, 1);
  CHECK(pos_seq0_before >= 0);
  CHECK(pos_seq1_before >= 0);

  // Clear sequence 0 only (remove all tokens from seq 0)
  kv::remove_range(ctx, 0, 0, -1);

  // Seq 0 should be empty, seq 1 should be intact
  llama_pos pos_seq0_after = kv::pos_max(ctx, 0);
  llama_pos pos_seq1_after = kv::pos_max(ctx, 1);

  CHECK(pos_seq0_after == -1);  // Empty (no tokens)
  CHECK(pos_seq1_after == pos_seq1_before);  // Unchanged

  llama_free(ctx);
}

TEST_CASE("Integration: backward compatibility - default seq_id=0") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  // Standard single-sequence context (n_seq_max defaults to 1)
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 256;
  ctx_params.n_batch = 64;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello", false, false);
  REQUIRE_FALSE(tokens.empty());

  // Old API (no seq_id parameter) should still work
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch);

  // Check seq 0 has tokens
  llama_pos pos = kv::pos_max(ctx, 0);
  CHECK(pos >= 0);

  llama_free(ctx);
}

TEST_CASE("Integration: decode with explicit seq_id=0 matches default") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 256;
  ctx_params.n_batch = 64;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Test", false, false);
  REQUIRE_FALSE(tokens.empty());

  // Explicit seq_id=0 should work same as default
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 0);

  llama_pos pos = kv::pos_max(ctx, 0);
  CHECK(pos >= 0);

  llama_free(ctx);
}

// ============================================================================
// System 2 Branching Tests (seq_cp, seq_keep)
// ============================================================================

TEST_CASE("Integration: seq_cp copies KV cache to new sequence") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "The quick brown fox", false, false);
  REQUIRE_FALSE(tokens.empty());

  // Decode to sequence 0
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 0);
  llama_pos pos_seq0 = kv::pos_max(ctx, 0);
  CHECK(pos_seq0 >= 0);
  INFO("Seq 0 pos_max before copy: " << pos_seq0);

  // Sequence 2 should be empty before copy
  llama_pos pos_seq2_before = kv::pos_max(ctx, 2);
  CHECK(pos_seq2_before == -1);

  // Copy seq 0 -> seq 2 (System 2 fork)
  kv::seq_cp(ctx, 0, 2);

  // Both sequences should now have same position
  llama_pos pos_seq0_after = kv::pos_max(ctx, 0);
  llama_pos pos_seq2_after = kv::pos_max(ctx, 2);
  INFO("Seq 0 pos_max after copy: " << pos_seq0_after);
  INFO("Seq 2 pos_max after copy: " << pos_seq2_after);

  CHECK(pos_seq0_after == pos_seq0);  // Source unchanged
  CHECK(pos_seq2_after == pos_seq0);  // Dest matches source

  llama_free(ctx);
}

// NOTE: Partial range seq_cp is NOT supported by llama.cpp
// The assertion "seq_cp() is only supported for full KV buffers" fires
// when p0/p1 are not the full range. This is a llama.cpp limitation.
// For System 2 branching, we must copy entire sequences.
//
// TEST_CASE("Integration: seq_cp with range copies partial sequence") - REMOVED

// NOTE: seq_keep behavior differs from expectations
// After seq_cp, cells are shared across sequences. seq_keep removes the sequence
// from cells but pos_max still returns positions for all sequences because the
// cells still exist. For System 2, we should use remove_range per-sequence instead.
//
// For now, the critical operation is seq_cp (branching), which works correctly.
// Pruning can be done via kv::remove_range(ctx, seq_id, 0, -1) for each unwanted sequence.
//
// TEST_CASE("Integration: seq_keep after branch") - SKIPPED pending llama.cpp investigation
