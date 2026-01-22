// File: packages/liblloyal/tests/integration/clear_and_reseed_test.cpp

#include <algorithm>
#include <cmath>
#include <doctest/doctest.h>
#include <iomanip>
#include <iostream>
#include <llama/llama.h>
#include <lloyal/decoder.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <vector>

using namespace lloyal;

/**
 * clear_and_reseed API Test Suite
 *
 * Implementation validation for the clear_and_reseed KV cache primitive.
 * These are engineering tests (CI gates), not experimental evidence.
 *
 * For proper experimental evaluation with teacher-forced corpus perplexity,
 * see tests/streaming_llm.mjs with --dataset flag pointing to a corpus
 * (e.g., pg19_first_book.txt).
 *
 * ============================================================================
 * TEST STRUCTURE
 * ============================================================================
 *
 * TEST 1: Position Contiguity [FOUNDATIONAL]
 *   Validates that clear_and_reseed produces contiguous positions [0,1,2,...].
 *   This is the foundation of Blink KV's correctness: bounded positions via
 *   cache reconstruction rather than unbounded gaps from naive eviction.
 *
 *   If this test fails, nothing else matters.
 *
 * TEST 2: Smoke Test
 *   Validates that context compression doesn't cause catastrophic degradation.
 *   Uses self-generated tokens as targets (fast, CI-friendly).
 *
 *   NOTE: This is a CI gate, not experimental evidence. For proper corpus-based
 *   evaluation, use: node streaming_llm.mjs --dataset=pg19_first_book.txt
 *
 * ============================================================================
 * DIVERGENCE METRIC
 * ============================================================================
 *
 * We use ½·Jeffreys divergence (symmetrized KL) for distribution comparison:
 *   ½[KL(P||Q) + KL(Q||P)]
 *
 * This is the α=1 endpoint of Nielsen's unified K-J divergence family.
 *
 * References:
 *   Nielsen, F. "A family of statistical symmetric divergences based on
 *   Jensen's inequality." arXiv:1009.4004, 2010. Equation 19.
 *
 * ============================================================================
 * REQUIREMENTS
 * ============================================================================
 *
 * Requires coherent model set via LLAMA_TEST_MODEL env var.
 * (NOT the gibberish tiny-random-llama.gguf used in unit tests)
 */

static const char *MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

#define REQUIRE_COHERENT_MODEL()                                               \
  if (!MODEL_PATH || !*MODEL_PATH) {                                           \
    MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set");                              \
    MESSAGE("Set to a COHERENT model (not tiny-random-llama.gguf)");           \
    return;                                                                    \
  }

struct LlamaBackendGuard {
  LlamaBackendGuard() { llama_backend_init(); }
  ~LlamaBackendGuard() { llama_backend_free(); }
};

// ============================================================================
// HELPERS
// ============================================================================

static int argmax(const float *logits, int n) {
  int idx = 0;
  float best = logits[0];
  for (int i = 1; i < n; ++i) {
    if (logits[i] > best) {
      best = logits[i];
      idx = i;
    }
  }
  return idx;
}

static float max_abs_diff(const float *a, const float *b, int n) {
  float max_diff = 0.0f;
  for (int i = 0; i < n; ++i) {
    float diff = std::abs(a[i] - b[i]);
    if (diff > max_diff) max_diff = diff;
  }
  return max_diff;
}

static double cosine_similarity(const float *a, const float *b, int n) {
  double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
  for (int i = 0; i < n; ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12);
}

static void softmax_inplace(std::vector<double> &p) {
  double mx = *std::max_element(p.begin(), p.end());
  double sum = 0.0;
  for (double &x : p) {
    x = std::exp(x - mx);
    sum += x;
  }
  for (double &x : p)
    x /= sum;
}

/**
 * Kullback-Leibler divergence: KL(P || Q) = Σ p_i log(p_i / q_i)
 */
static double kl_div(const std::vector<double> &p, const std::vector<double> &q) {
  double d = 0.0;
  for (size_t i = 0; i < p.size(); ++i) {
    d += p[i] * std::log(std::max(1e-12, p[i] / std::max(1e-12, q[i])));
  }
  return d;
}

/**
 * ½·Jeffreys divergence (symmetrized KL)
 *
 * Computes: ½ [KL(P||Q) + KL(Q||P)] = ½ J(P,Q)
 *
 * This is the α=1 case of Nielsen's unified K-J divergence family:
 *   D^J_{K,α}[p:q] := (D_{K,α}[p:q] + D_{K,α}[q:p]) / 2
 *
 * References:
 *   Nielsen, F. "A family of statistical symmetric divergences based on
 *   Jensen's inequality." arXiv:1009.4004, 2010. Equation 19.
 */
static double half_jeffreys_div(const std::vector<double> &p,
                                const std::vector<double> &q) {
  return 0.5 * (kl_div(p, q) + kl_div(q, p));
}

static std::vector<int> topk(const float *logits, int n, int k) {
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i)
    idx[i] = i;

  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  idx.resize(k);
  return idx;
}

// ============================================================================
// TEST 1: POSITION CONTIGUITY [FOUNDATIONAL]
// Validates clear_and_reseed produces bounded contiguous positions.
// This underpins Blink KV's correctness: [0,1,2,...] not [0,1,2,3,...,19997]
// ============================================================================

TEST_CASE("clear_and_reseed: Position contiguity") {
  REQUIRE_COHERENT_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 256;
  ctx_params.n_threads = 1;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Generate tokens
  std::string prompt = "Testing position contiguity after cache reconstruction. "
                       "This validates the foundational property of Blink KV.";
  auto tokens = tokenizer::tokenize(vocab, prompt, false, false);
  
  REQUIRE(tokens.size() >= 8);

  const int SINK_COUNT = 4;
  const int TAIL_COUNT = static_cast<int>(tokens.size()) - SINK_COUNT;
  
  REQUIRE(TAIL_COUNT > 0);

  std::vector<llama_token> sinks(tokens.begin(), tokens.begin() + SINK_COUNT);
  std::vector<llama_token> tail(tokens.begin() + SINK_COUNT, tokens.end());

  INFO("=== POSITION CONTIGUITY TEST ===");
  INFO("This is the foundational test for Blink KV.");
  INFO("");
  INFO("Configuration:");
  INFO("  Total tokens: " << tokens.size());
  INFO("  Sinks: " << SINK_COUNT << " tokens → positions [0-" << (SINK_COUNT-1) << "]");
  INFO("  Tail:  " << TAIL_COUNT << " tokens → positions [" << SINK_COUNT << "-" << (tokens.size()-1) << "]");

  // Execute clear_and_reseed
  kv::clear_and_reseed(ctx, sinks, tail, ctx_params.n_batch);

  auto mem = llama_get_memory(ctx);
  llama_pos max_pos = llama_memory_seq_pos_max(mem, 0);

  // Expected: positions [0, 1, ..., total-1] with NO GAPS
  llama_pos expected_max = static_cast<llama_pos>(tokens.size() - 1);

  INFO("");
  INFO("Result:");
  INFO("  Expected max_pos: " << expected_max);
  INFO("  Actual max_pos:   " << max_pos);

  CHECK(max_pos == expected_max);

  if (max_pos == expected_max) {
    INFO("");
    INFO("========================================");
    INFO("✅ POSITION CONTIGUITY VALIDATED");
    INFO("========================================");
    INFO("Cache positions are [0-" << max_pos << "] with no gaps.");
    INFO("========================================");
  } else {
    INFO("");
    INFO("========================================");
    INFO("❌ POSITION CONTIGUITY FAILED");
    INFO("========================================");
    FAIL("Position contiguity check failed");
  }

  // === JSON OUTPUT ===
  std::cout << "\n=== JSON_RESULT ===" << std::endl;
  std::cout << "{";
  std::cout << "\"test\":\"position_contiguity\",";
  std::cout << "\"total_tokens\":" << tokens.size() << ",";
  std::cout << "\"sink_count\":" << SINK_COUNT << ",";
  std::cout << "\"tail_count\":" << TAIL_COUNT << ",";
  std::cout << "\"expected_max_pos\":" << expected_max << ",";
  std::cout << "\"actual_max_pos\":" << max_pos << ",";
  std::cout << "\"pass\":" << (max_pos == expected_max ? "true" : "false");
  std::cout << "}" << std::endl;
  std::cout << "=== END_JSON_RESULT ===" << std::endl;

  llama_free(ctx);
}

// ============================================================================
// TEST 2: SMOKE TEST
// Validates context compression doesn't cause catastrophic degradation.
// Uses self-generated targets (fast CI gate, not experimental evidence).
// ============================================================================

TEST_CASE("clear_and_reseed: Smoke test (non-catastrophic compression)") {
  REQUIRE_COHERENT_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 256;
  ctx_params.n_threads = 1;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  int n_vocab = llama_vocab_n_tokens(vocab);

  INFO("=== SMOKE TEST ===");
  INFO("CI gate: validates compression isn't catastrophic.");
  INFO("NOTE: Uses self-generated tokens. For corpus evaluation:");
  INFO("      node streaming_llm.mjs --dataset=pg19_first_book.txt");
  INFO("");

  // === PHASE 1: Generate sequence ===
  std::string prompt = "The quick brown fox jumps over the lazy dog.";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
  
  std::vector<llama_token> all_tokens = prompt_tokens;
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch);
  int n_past = static_cast<int>(prompt_tokens.size());

  const int TARGET_LENGTH = 1500;
  INFO("Generating " << TARGET_LENGTH << " tokens...");
  
  while (all_tokens.size() < TARGET_LENGTH) {
    llama_token next = sampler::greedy(ctx, vocab);
    all_tokens.push_back(next);
    decoder::decode_tokens(ctx, {next}, n_past++, ctx_params.n_batch);
  }

  // === PHASE 2: Capture BEFORE ===
  const float *logits_before = llama_get_logits_ith(ctx, -1);
  REQUIRE(logits_before != nullptr);
  
  std::vector<float> logits_before_copy(logits_before, logits_before + n_vocab);
  
  std::vector<double> probs_before(n_vocab);
  for (int i = 0; i < n_vocab; ++i) probs_before[i] = logits_before[i];
  softmax_inplace(probs_before);
  
  int top1_before = argmax(logits_before, n_vocab);
  auto top10_before = topk(logits_before, n_vocab, 10);

  // === PHASE 3: Compress ===
  const int SINK_COUNT = 4;
  const int TAIL_COUNT = 1020;
  const int EVICTED = static_cast<int>(all_tokens.size()) - SINK_COUNT - TAIL_COUNT;
  
  INFO("Compressing: " << all_tokens.size() << " → " << (SINK_COUNT + TAIL_COUNT) 
       << " tokens (evicting " << EVICTED << ")");
  
  std::vector<llama_token> sinks(all_tokens.begin(), all_tokens.begin() + SINK_COUNT);
  std::vector<llama_token> tail(all_tokens.end() - TAIL_COUNT, all_tokens.end());
  
  kv::clear_and_reseed(ctx, sinks, tail, ctx_params.n_batch);

  // === PHASE 4: Capture AFTER ===
  const float *logits_after = llama_get_logits_ith(ctx, -1);
  REQUIRE(logits_after != nullptr);
  
  std::vector<double> probs_after(n_vocab);
  for (int i = 0; i < n_vocab; ++i) probs_after[i] = logits_after[i];
  softmax_inplace(probs_after);
  
  int top1_after = argmax(logits_after, n_vocab);
  auto top10_after = topk(logits_after, n_vocab, 10);

  // === PHASE 5: Metrics ===
  INFO("");
  INFO("Metrics:");
  
  bool top1_match = (top1_before == top1_after);
  INFO("  Top-1 match: " << (top1_match ? "YES" : "NO"));
  
  int overlap = 0;
  for (int a : top10_after) {
    overlap += std::count(top10_before.begin(), top10_before.end(), a);
  }
  INFO("  Top-10 overlap: " << overlap << "/10");
  CHECK(overlap >= 7);
  
  double divergence = half_jeffreys_div(probs_before, probs_after);
  INFO("  ½·Jeffreys divergence: " << std::scientific << divergence);
  CHECK(divergence < 0.1);
  
  double cos_sim = cosine_similarity(logits_before_copy.data(), logits_after, n_vocab);
  INFO("  Cosine similarity: " << std::fixed << std::setprecision(6) << cos_sim);

  bool acceptable = (overlap >= 7) && (divergence < 0.1);
  
  if (acceptable) {
    INFO("");
    INFO("========================================");
    INFO("✅ SMOKE TEST PASSED");
    INFO("========================================");
  } else {
    INFO("");
    INFO("========================================");
    INFO("⚠️  SMOKE TEST WARNING");
    INFO("========================================");
  }

  // === JSON OUTPUT ===
  std::cout << "\n=== JSON_RESULT ===" << std::endl;
  std::cout << "{";
  std::cout << "\"test\":\"smoke_test\",";
  std::cout << "\"total_tokens\":" << all_tokens.size() << ",";
  std::cout << "\"compressed_tokens\":" << (SINK_COUNT + TAIL_COUNT) << ",";
  std::cout << "\"evicted_tokens\":" << EVICTED << ",";
  std::cout << "\"top1_match\":" << (top1_match ? "true" : "false") << ",";
  std::cout << "\"top10_overlap\":" << overlap << ",";
  std::cout << "\"half_jeffreys_divergence\":" << std::scientific << divergence << ",";
  std::cout << "\"cosine_similarity\":" << std::fixed << std::setprecision(6) << cos_sim << ",";
  std::cout << "\"acceptable\":" << (acceptable ? "true" : "false");
  std::cout << "}" << std::endl;
  std::cout << "=== END_JSON_RESULT ===" << std::endl;

  llama_free(ctx);
}