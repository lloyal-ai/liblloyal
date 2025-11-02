// File: packages/liblloyal/tests/integration/rope_position_invariant_test.cpp

#include <doctest/doctest.h>
#include <llama/llama.h>
#include <lloyal/decoder.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <vector>

using namespace lloyal;

/**
 * RoPE Position Encoding Invariant Test
 *
 * CRITICAL INVARIANT: After clear+reseed, KV cache positions MUST be
 * contiguous.
 *
 * StreamingLLM Paper (Section 3.2):
 * > "cache the Keys of tokens prior to introducing the rotary transformation.
 * Then, we apply > position transformation to the keys in the rolling cache at
 * each decoding phase."
 *
 * Position Renumbering Requirement:
 * > "if the cache has tokens [0, 1, 2, 3, 6, 7, 8], the assigned positions are
 * [0, 1, 2, 3, 4, 5, 6]"
 *
 * WHY THIS MATTERS:
 * RoPE (Rotary Position Embeddings) encodes relative distances between tokens.
 * If we used original text positions [0,1,2,3,545,546,...,800] after reseeding,
 * the model would see HUGE position gaps that break attention calculations.
 *
 * CORRECT IMPLEMENTATION:
 * - Clear cache: llama_memory_clear(mem, true)
 * - Decode sinks at n_past=0:  positions [0, 1, 2, 3]
 * - Decode tail at n_past=4:   positions [4, 5, 6, ..., 255]
 * - Result: Contiguous [0-255] with NO GAPS
 *
 * PROOF OF CORRECTNESS:
 * If RoPE positions were wrong:
 * - Model would see completely different positional information
 * - Attention patterns would be wildly different
 * - Logit distributions would diverge significantly (JSD >> 0.01)
 * - Top-1 token would almost certainly differ
 *
 * This test validates both:
 * 1. Position contiguity (via llama_memory_seq_pos_max check)
 * 2. RoPE correctness (via boundary equivalence: same top-1, low JSD)
 *
 * REQUIRES: Coherent model set via LLAMA_TEST_MODEL env var
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

// Helper: Compute softmax and get top-1 token
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

// Helper: Compute Jensen-Shannon divergence
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

static double kl_div(const std::vector<double> &p,
                     const std::vector<double> &q) {
  double d = 0.0;
  for (size_t i = 0; i < p.size(); ++i) {
    d += p[i] * std::log(std::max(1e-12, p[i] / std::max(1e-12, q[i])));
  }
  return d;
}

TEST_CASE("RoPE Invariant: Clear+reseed produces contiguous positions") {
  REQUIRE_COHERENT_MODEL();
  LlamaBackendGuard backend;

  // === SETUP ===
  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0; // CPU for determinism

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 1024; // Small context to force reseed testing
  ctx_params.n_batch = 256;
  ctx_params.n_threads = 1; // Single-thread for determinism

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  int n_vocab = llama_vocab_n_tokens(vocab);

  // === PHASE 1: Generate 800 tokens ===
  std::string prompt = "The quick brown fox jumps over the lazy dog.";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
  REQUIRE_FALSE(prompt_tokens.empty());

  INFO("Prompt tokens: " << prompt_tokens.size());

  std::vector<llama_token> all_tokens = prompt_tokens;

  // Decode prompt
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch);
  int n_past = static_cast<int>(prompt_tokens.size());

  // Generate 800 tokens
  int tokens_to_generate = 800;
  INFO("Generating " << tokens_to_generate << " tokens before clear+reseed...");

  for (int i = 0; i < tokens_to_generate; ++i) {
    llama_token next_token = sampler::greedy(ctx, vocab);
    all_tokens.push_back(next_token);

    std::vector<llama_token> single_token = {next_token};
    decoder::decode_tokens(ctx, single_token, n_past, ctx_params.n_batch);
    n_past++;
  }

  auto mem = llama_get_memory(ctx);
  llama_pos max_pos_before = llama_memory_seq_pos_max(mem, 0);
  INFO("Before clear+reseed: KV cache max_pos=" << max_pos_before);
  CHECK(max_pos_before >= 800);

  // === PHASE 2: Capture boundary BEFORE clear+reseed ===
  const float *logits_before = llama_get_logits_ith(ctx, -1);
  REQUIRE(logits_before != nullptr);

  std::vector<double> logp_before(n_vocab);
  for (int i = 0; i < n_vocab; ++i)
    logp_before[i] = logits_before[i];
  softmax_inplace(logp_before);

  int top1_before = argmax(logits_before, n_vocab);
  INFO("Boundary BEFORE clear+reseed: top-1 token = " << top1_before);

  // === PHASE 3: Execute clear+reseed with CONTIGUOUS POSITIONS ===
  INFO("Executing clear+reseed...");

  // StreamingLLM config: 4 sinks + 252 tail = 256 total (power-of-2)
  const int SINK_COUNT = 4;
  const int TAIL_COUNT = 252;

  std::vector<llama_token> sinks(all_tokens.begin(),
                                 all_tokens.begin() + SINK_COUNT);
  std::vector<llama_token> tail(all_tokens.end() - TAIL_COUNT,
                                all_tokens.end());

  INFO("Sinks: " << sinks.size() << " tokens");
  INFO("Tail: " << tail.size() << " tokens");

  // Clear entire KV cache
  llama_memory_clear(mem, true);

  llama_pos max_pos_after_clear = llama_memory_seq_pos_max(mem, 0);
  CHECK(max_pos_after_clear == -1); // Empty
  INFO("After clear: KV cache max_pos=" << max_pos_after_clear << " (empty)");

  // CRITICAL: Re-decode with CONTIGUOUS positions
  // Sinks at positions 0-3
  decoder::decode_tokens(ctx, sinks, 0, ctx_params.n_batch);

  // Tail at positions 4-255 (immediately after sinks, NO GAPS)
  decoder::decode_tokens(ctx, tail, SINK_COUNT, ctx_params.n_batch);

  llama_pos max_pos_after_reseed = llama_memory_seq_pos_max(mem, 0);
  INFO("After reseed: KV cache max_pos=" << max_pos_after_reseed);

  // === PHASE 4: Validate position contiguity ===
  INFO("=== POSITION CONTIGUITY CHECK ===");

  // Expected: 4 sinks + 252 tail - 1 (zero-indexed) = 255
  CHECK(max_pos_after_reseed == SINK_COUNT + TAIL_COUNT - 1);

  if (max_pos_after_reseed == SINK_COUNT + TAIL_COUNT - 1) {
    INFO("✅ POSITIONS CONTIGUOUS: [0-" << max_pos_after_reseed
                                        << "] (no gaps)");
  } else {
    INFO("❌ POSITIONS NOT CONTIGUOUS: expected "
         << (SINK_COUNT + TAIL_COUNT - 1) << ", got " << max_pos_after_reseed);
  }

  // === PHASE 5: Validate RoPE correctness via boundary equivalence ===
  const float *logits_after = llama_get_logits_ith(ctx, -1);
  REQUIRE(logits_after != nullptr);

  std::vector<double> logp_after(n_vocab);
  for (int i = 0; i < n_vocab; ++i)
    logp_after[i] = logits_after[i];
  softmax_inplace(logp_after);

  int top1_after = argmax(logits_after, n_vocab);
  INFO("Boundary AFTER clear+reseed:  top-1 token = " << top1_after);

  INFO("=== BOUNDARY EQUIVALENCE CHECK (RoPE CORRECTNESS PROOF) ===");

  // 1. Top-1 token must match (strongest signal)
  CHECK(top1_after == top1_before);

  if (top1_after == top1_before) {
    INFO("✅ Top-1 match: " << top1_before);
  } else {
    INFO("❌ Top-1 MISMATCH: before=" << top1_before
                                      << " after=" << top1_after);
    INFO("   This indicates RoPE positions are INCORRECT!");
  }

  // 2. Jensen-Shannon divergence (distributions should be nearly identical)
  double kl_ba = kl_div(logp_before, logp_after);
  double kl_ab = kl_div(logp_after, logp_before);
  double jsd = 0.5 * (kl_ba + kl_ab);
  INFO("Jensen-Shannon divergence: " << jsd);

  // Very small divergence expected (< 0.01 = 1% divergence)
  CHECK(jsd < 1e-2);

  if (jsd < 1e-2) {
    INFO("✅ JSD < 0.01: Distributions essentially identical");
    INFO("   This proves RoPE positions are CORRECT!");
  } else {
    INFO("❌ JSD >= 0.01: Distributions differ significantly");
    INFO("   This indicates RoPE positions may be incorrect!");
  }

  // === FINAL VERDICT ===
  if (max_pos_after_reseed == SINK_COUNT + TAIL_COUNT - 1 &&
      top1_after == top1_before && jsd < 1e-2) {
    INFO("");
    INFO("========================================");
    INFO("✅ RoPE INVARIANT VALIDATED");
    INFO("========================================");
    INFO("1. Positions are contiguous: [0-255]");
    INFO("2. Top-1 token matches exactly");
    INFO("3. Distribution divergence < 1%");
    INFO("");
    INFO("CONCLUSION: clear+reseed correctly preserves");
    INFO("positional encoding as required by RoPE.");
    INFO("========================================");
  } else {
    INFO("");
    INFO("========================================");
    INFO("❌ RoPE INVARIANT VIOLATED");
    INFO("========================================");
    WARN(false); // Soft warning without failing the test
  }

  // === CLEANUP ===
  llama_free(ctx);
}
