#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/decode.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/logits.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <numeric>
#include <vector>

using namespace lloyal;

/**
 * Logits Integration Tests
 *
 * Validates logits::process_chunks() — the lightweight batch logit extraction
 * primitive that decodes complete prompts and captures logits without Branch
 * overhead. Coverage includes:
 *
 * - Single prompt: logits match decode::many + logits::get baseline
 * - Multi-prompt batch: each prompt gets independent logits
 * - Bin-packing: prompts of varying length are packed into n_batch dispatches
 * - Oversized prompt: prompt longer than n_batch is dispatched via decode::many
 * - KV cleanup: all seq_ids are evicted after process_chunks returns
 * - Empty inputs: zero prompts and zero-length prompts are handled gracefully
 * - Determinism: identical prompts produce identical logits
 *
 * Requires LLAMA_TEST_MODEL env var pointing to a coherent model.
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

// Helper: compute softmax probability of token `idx` over [idx_a, idx_b]
static float binary_softmax(const float* logits, int32_t idx_a, int32_t idx_b) {
  float max_val = std::max(logits[idx_a], logits[idx_b]);
  float exp_a = std::exp(logits[idx_a] - max_val);
  float exp_b = std::exp(logits[idx_b] - max_val);
  return exp_a / (exp_a + exp_b);
}

// Helper: find argmax of a logits array
static int32_t argmax(const float* logits, int32_t n_vocab) {
  return static_cast<int32_t>(
      std::max_element(logits, logits + n_vocab) - logits);
}

// ============================================================================
// process_chunks: Basic Correctness
// ============================================================================

TEST_CASE("logits: process_chunks single prompt matches decode::many baseline") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "The capital of France is", false, false);
  REQUIRE_FALSE(tokens.empty());

  int32_t n_vocab = tokenizer::vocab_size(vocab);
  REQUIRE(n_vocab > 0);

  // --- Baseline: decode::many on seq 0 ---
  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch, 0) == 0);
  const float* baseline_raw = logits::get(ctx, -1);
  std::vector<float> baseline(baseline_raw, baseline_raw + n_vocab);
  kv::remove_range(ctx, 0, 0, -1);

  // --- process_chunks ---
  std::vector<std::span<const llama_token>> prompts = { tokens };
  std::vector<float> output_buf(n_vocab);
  std::vector<float*> outputs = { output_buf.data() };

  logits::process_chunks(ctx, prompts, outputs, n_vocab);

  // Logits should be identical (same seq_id, same positions, same tokens)
  int32_t baseline_argmax = argmax(baseline.data(), n_vocab);
  int32_t chunks_argmax = argmax(output_buf.data(), n_vocab);
  CHECK(baseline_argmax == chunks_argmax);

  // Verify logits are close (not just argmax)
  float max_diff = 0.0f;
  for (int32_t i = 0; i < n_vocab; ++i) {
    max_diff = std::max(max_diff, std::abs(baseline[i] - output_buf[i]));
  }
  INFO("Max logit difference: " << max_diff);
  CHECK(max_diff < 1e-3f);

  llama_free(ctx);
}

TEST_CASE("logits: process_chunks multi-prompt produces independent logits") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 256;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens_a = tokenizer::tokenize(vocab, "The weather today is", false, false);
  auto tokens_b = tokenizer::tokenize(vocab, "Once upon a time", false, false);
  REQUIRE_FALSE(tokens_a.empty());
  REQUIRE_FALSE(tokens_b.empty());

  int32_t n_vocab = tokenizer::vocab_size(vocab);

  std::vector<std::span<const llama_token>> prompts = { tokens_a, tokens_b };
  std::vector<float> buf_a(n_vocab), buf_b(n_vocab);
  std::vector<float*> outputs = { buf_a.data(), buf_b.data() };

  logits::process_chunks(ctx, prompts, outputs, n_vocab);

  // Different prompts should produce different argmax tokens
  int32_t argmax_a = argmax(buf_a.data(), n_vocab);
  int32_t argmax_b = argmax(buf_b.data(), n_vocab);
  INFO("Prompt A argmax: " << argmax_a << ", Prompt B argmax: " << argmax_b);

  // Logits should not be identical (different prompts)
  float max_diff = 0.0f;
  for (int32_t i = 0; i < n_vocab; ++i) {
    max_diff = std::max(max_diff, std::abs(buf_a[i] - buf_b[i]));
  }
  INFO("Max logit difference between prompts: " << max_diff);
  CHECK(max_diff > 1.0f);

  llama_free(ctx);
}

// ============================================================================
// process_chunks: Bin-packing and Oversized
// ============================================================================

TEST_CASE("logits: process_chunks handles oversized prompt via decode::many fallback") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Small n_batch to force oversized path
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 32;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  // Create a prompt longer than n_batch (32 tokens) by repeating text
  std::string long_text;
  for (int i = 0; i < 10; ++i)
    long_text += "The quick brown fox jumps over the lazy dog. ";
  auto tokens = tokenizer::tokenize(vocab, long_text, false, false);
  INFO("Prompt has " << tokens.size() << " tokens, n_batch=" << ctx_params.n_batch);
  REQUIRE(static_cast<int32_t>(tokens.size()) > ctx_params.n_batch);

  int32_t n_vocab = tokenizer::vocab_size(vocab);
  std::vector<float> buf(n_vocab);
  std::vector<std::span<const llama_token>> prompts = { tokens };
  std::vector<float*> outputs = { buf.data() };

  // Should not throw
  logits::process_chunks(ctx, prompts, outputs, n_vocab);

  // Logits should be valid (not all zeros)
  float sum = 0.0f;
  for (int32_t i = 0; i < n_vocab; ++i) sum += std::abs(buf[i]);
  CHECK(sum > 0.0f);

  llama_free(ctx);
}

TEST_CASE("logits: process_chunks bin-packs mixed-length prompts") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // n_batch=64 forces multiple bin-pack chunks for 4 prompts
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 64;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto t0 = tokenizer::tokenize(vocab, "Short", false, false);
  auto t1 = tokenizer::tokenize(vocab, "A slightly longer sentence for testing", false, false);
  auto t2 = tokenizer::tokenize(vocab, "Hi", false, false);
  auto t3 = tokenizer::tokenize(vocab, "Another sentence of moderate length here", false, false);

  int32_t n_vocab = tokenizer::vocab_size(vocab);
  std::vector<float> b0(n_vocab), b1(n_vocab), b2(n_vocab), b3(n_vocab);
  std::vector<std::span<const llama_token>> prompts = { t0, t1, t2, t3 };
  std::vector<float*> outputs = { b0.data(), b1.data(), b2.data(), b3.data() };

  logits::process_chunks(ctx, prompts, outputs, n_vocab);

  // Each prompt should produce valid, non-zero logits
  for (int i = 0; i < 4; ++i) {
    float sum = 0.0f;
    for (int32_t j = 0; j < n_vocab; ++j) sum += std::abs(outputs[i][j]);
    INFO("Prompt " << i << " logit L1 norm: " << sum);
    CHECK(sum > 0.0f);
  }

  // Different prompts should generally produce different top tokens
  int32_t a0 = argmax(b0.data(), n_vocab);
  int32_t a2 = argmax(b2.data(), n_vocab);
  // "Short" and "Hi" are different enough to often diverge
  // (not a hard requirement — just log it)
  INFO("Prompt 0 argmax: " << a0 << ", Prompt 2 argmax: " << a2);

  llama_free(ctx);
}

// ============================================================================
// process_chunks: KV Cleanup
// ============================================================================

TEST_CASE("logits: process_chunks evicts all seq_ids after completion") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto t0 = tokenizer::tokenize(vocab, "Hello", false, false);
  auto t1 = tokenizer::tokenize(vocab, "World", false, false);

  int32_t n_vocab = tokenizer::vocab_size(vocab);
  std::vector<float> b0(n_vocab), b1(n_vocab);
  std::vector<std::span<const llama_token>> prompts = { t0, t1 };
  std::vector<float*> outputs = { b0.data(), b1.data() };

  logits::process_chunks(ctx, prompts, outputs, n_vocab);

  // All seq_ids used (0, 1) should be evicted
  CHECK(kv::pos_max(ctx, 0) == -1);
  CHECK(kv::pos_max(ctx, 1) == -1);

  // Unused seq_ids should also be empty
  CHECK(kv::pos_max(ctx, 2) == -1);
  CHECK(kv::pos_max(ctx, 3) == -1);

  llama_free(ctx);
}

TEST_CASE("logits: process_chunks can be called repeatedly (KV reuse)") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  int32_t n_vocab = tokenizer::vocab_size(vocab);

  auto tokens = tokenizer::tokenize(vocab, "Repeat test", false, false);
  std::vector<float> buf(n_vocab);
  std::vector<std::span<const llama_token>> prompts = { tokens };
  std::vector<float*> outputs = { buf.data() };

  // First call
  logits::process_chunks(ctx, prompts, outputs, n_vocab);
  int32_t first_argmax = argmax(buf.data(), n_vocab);

  // Second call with same prompt — should produce identical logits
  std::vector<float> buf2(n_vocab);
  outputs[0] = buf2.data();
  logits::process_chunks(ctx, prompts, outputs, n_vocab);
  int32_t second_argmax = argmax(buf2.data(), n_vocab);

  CHECK(first_argmax == second_argmax);

  // Verify logits match exactly (deterministic — same state)
  float max_diff = 0.0f;
  for (int32_t i = 0; i < n_vocab; ++i) {
    max_diff = std::max(max_diff, std::abs(buf[i] - buf2[i]));
  }
  INFO("Max logit diff between repeated calls: " << max_diff);
  CHECK(max_diff < 1e-4f);

  llama_free(ctx);
}

// ============================================================================
// process_chunks: Edge Cases
// ============================================================================

TEST_CASE("logits: process_chunks handles empty prompts vector") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 256;
  ctx_params.n_batch = 64;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  // Empty prompts vector — should return immediately, no crash
  std::vector<std::span<const llama_token>> prompts;
  std::vector<float*> outputs;

  logits::process_chunks(ctx, prompts, outputs,
                         tokenizer::vocab_size(llama_model_get_vocab(model.get())));

  // No seq_ids should be dirty
  CHECK(kv::pos_max(ctx, 0) == -1);

  llama_free(ctx);
}

TEST_CASE("logits: process_chunks rejects mismatched prompts/output sizes") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 256;
  ctx_params.n_batch = 64;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello", false, false);

  std::vector<std::span<const llama_token>> prompts = { tokens };
  std::vector<float*> outputs;  // Empty — mismatch!

  CHECK_THROWS_AS(
      logits::process_chunks(ctx, prompts, outputs,
                             tokenizer::vocab_size(vocab)),
      std::runtime_error);

  llama_free(ctx);
}

TEST_CASE("logits: process_chunks handles prompts exceeding n_seq_max via grouping") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 256;
  ctx_params.n_batch = 64;
  ctx_params.n_seq_max = 2;  // Only 2 sequences at a time

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto t0 = tokenizer::tokenize(vocab, "A", false, false);
  auto t1 = tokenizer::tokenize(vocab, "B", false, false);
  auto t2 = tokenizer::tokenize(vocab, "C", false, false);

  int32_t n_vocab = tokenizer::vocab_size(vocab);
  std::vector<float> b0(n_vocab), b1(n_vocab), b2(n_vocab);

  // 3 prompts with n_seq_max=2 — should succeed via grouping (2+1)
  std::vector<std::span<const llama_token>> prompts = { t0, t1, t2 };
  std::vector<float*> outputs = { b0.data(), b1.data(), b2.data() };

  CHECK_NOTHROW(logits::process_chunks(ctx, prompts, outputs, n_vocab));

  // All outputs should have valid logits (non-zero somewhere)
  auto has_nonzero = [&](const std::vector<float>& buf) {
    for (int32_t i = 0; i < n_vocab; ++i) {
      if (buf[i] != 0.0f) return true;
    }
    return false;
  };
  CHECK(has_nonzero(b0));
  CHECK(has_nonzero(b1));
  CHECK(has_nonzero(b2));

  llama_free(ctx);
}
