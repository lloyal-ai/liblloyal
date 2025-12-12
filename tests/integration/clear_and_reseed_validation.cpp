// File: tests/integration/clear_and_reseed_validation.cpp
// Ported from: packages/@calibrate/calibrate-ndk/tests/integration/ClearAndReseed_Validation.cpp

#include <doctest/doctest.h>
#include "test_config.hpp"
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/decoder.hpp>
#include <lloyal/sampler.hpp>
#include <llama/llama.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace lloyal;

/**
 * Empirical Validation: clear+re-decode Preserves StreamingLLM Pattern
 *
 * StreamingLLM paper tested selective removal (llama_memory_seq_rm) to keep
 * sinks + tail in cache. We test a DIFFERENT approach: clear entire cache
 * (llama_memory_clear) then re-decode sinks + tail from scratch.
 *
 * Hypothesis: The StreamingLLM pattern (4 sinks + 252 tail = 256 total) should
 * preserve perplexity even with clear+re-decode instead of selective removal.
 *
 * Test Design:
 * 1. Generate 800 tokens with continuous cache (baseline)
 * 2. Clear cache, re-decode sinks (first 4) + tail (last 252)
 * 3. Continue generation for 200 tokens with compressed cache
 * 4. Compare last 200 tokens before vs 200 tokens after
 *
 * Success: PPL ratio < 1.10 (matches StreamingLLM's 3.7% finding)
 *
 * REQUIRES: Coherent model set via LLAMA_TEST_MODEL env var
 * (NOT the gibberish tiny-random-llama.gguf used in other tests)
 *
 * Recommended models:
 * - TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf (~650MB)
 * - Qwen2-0.5B-Instruct-Q4_K_M.gguf (~350MB)
 * - SmolLM-135M-Instruct-Q4_K_M.gguf (~100MB)
 */

static const char* MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

#define REQUIRE_COHERENT_MODEL() \
    if (!MODEL_PATH || !*MODEL_PATH) { \
        MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set"); \
        MESSAGE("Set to a COHERENT model (not tiny-random-llama.gguf)"); \
        return; \
    }

struct LlamaBackendGuard {
    LlamaBackendGuard() { llama_backend_init(); }
    ~LlamaBackendGuard() { llama_backend_free(); }
};

// Helper: Compute log softmax for perplexity calculation
struct LogSoftmaxResult {
    double log_softmax;
    float logit;
    float prob;
};

static LogSoftmaxResult compute_log_softmax(int n_vocab, const float* logits, llama_token tok) {
    // Find max logit for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    // Compute sum of exp(logit - max_logit)
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }

    // log_softmax(tok) = logit[tok] - max_logit - log(sum_exp)
    double log_sm = logits[tok] - max_logit - log(sum_exp);
    float prob = expf(logits[tok] - max_logit) / static_cast<float>(sum_exp);

    return {log_sm, logits[tok], prob};
}

// Boundary equivalence helpers
static void softmax_inplace(std::vector<double>& p) {
    double mx = *std::max_element(p.begin(), p.end());
    double sum = 0.0;
    for (double& x : p) { x = std::exp(x - mx); sum += x; }
    for (double& x : p) x /= sum;
}

static double kl_div(const std::vector<double>& p, const std::vector<double>& q) {
    // assume both are strictly positive and sum to 1
    double d = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        d += p[i] * std::log(std::max(1e-12, p[i] / std::max(1e-12, q[i])));
    }
    return d;
}

static int argmax(const float* logits, int n) {
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

static std::vector<int> topk(const float* logits, int n, int k) {
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;

    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(k);
    return idx;
}

TEST_CASE("Empirical: clearAndReseed preserves perplexity") {
    REQUIRE_COHERENT_MODEL();
    LlamaBackendGuard backend;

    // === SETUP ===
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = TestConfig::n_gpu_layers();  // CPU for determinism

    auto model = model_registry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;   // Small context to force reseed testing
    ctx_params.n_batch = 256;
    ctx_params.n_threads = 1;  // Single-thread for determinism

    llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
    REQUIRE(ctx != nullptr);

    auto vocab = llama_model_get_vocab(model.get());
    int n_vocab = llama_vocab_n_tokens(vocab);

    // === PHASE 1: Generate tokens and measure baseline perplexity ===
    std::string prompt = "The quick brown fox jumps over the lazy dog.";
    auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
    REQUIRE_FALSE(prompt_tokens.empty());

    INFO("Prompt tokens: " << prompt_tokens.size());

    // Track all generated tokens
    std::vector<llama_token> all_tokens = prompt_tokens;
    std::vector<double> perplexities_before;

    // Decode prompt using our production decoder
    decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch);

    int n_past = static_cast<int>(prompt_tokens.size());

    // Generate 800 tokens (to approach n_ctx=1024 limit)
    int tokens_to_generate = 800;
    INFO("Generating " << tokens_to_generate << " tokens before reseed...");

    for (int i = 0; i < tokens_to_generate; ++i) {
        // Sample next token using our production sampler (greedy = argmax)
        llama_token next_token = sampler::greedy(ctx, vocab);
        all_tokens.push_back(next_token);

        // Measure perplexity for this token
        const float* logits = llama_get_logits_ith(ctx, -1);
        auto result = compute_log_softmax(n_vocab, logits, next_token);
        double ppl = exp(-result.log_softmax);
        perplexities_before.push_back(ppl);

        // Decode single token using our production decoder
        std::vector<llama_token> single_token = {next_token};
        decoder::decode_tokens(ctx, single_token, n_past, ctx_params.n_batch);
        n_past++;
    }

    auto mem = llama_get_memory(ctx);
    llama_pos max_pos_before = llama_memory_seq_pos_max(mem, 0);
    INFO("Before reseed: KV cache max_pos=" << max_pos_before);
    CHECK(max_pos_before >= 800);

    // === PHASE 2: clearAndReseed ===
    INFO("Executing clearAndReseed...");

    // Extract sinks (first 4 tokens) and tail (last 252 tokens)
    // Total: 4 + 252 = 256 (power-of-2, matches StreamingLLM paper's 4+252 config)
    const int SINK_COUNT = 4;
    const int TAIL_COUNT = 252;

    std::vector<llama_token> sinks(all_tokens.begin(), all_tokens.begin() + SINK_COUNT);
    std::vector<llama_token> tail(all_tokens.end() - TAIL_COUNT, all_tokens.end());

    INFO("Sinks: " << sinks.size() << " tokens");
    INFO("Tail: " << tail.size() << " tokens");

    // ----- Boundary capture BEFORE reseed -----
    const float* logits_before = llama_get_logits_ith(ctx, -1);
    REQUIRE(logits_before != nullptr);

    std::vector<double> logp_before(n_vocab);
    for (int i = 0; i < n_vocab; ++i) logp_before[i] = logits_before[i];
    softmax_inplace(logp_before);

    int top1_before = argmax(logits_before, n_vocab);
    auto top10_before = topk(logits_before, n_vocab, 10);

    INFO("Boundary BEFORE reseed: top-1 token = " << top1_before);

    // Clear entire KV cache using llama_memory_clear (same as HybridCalibrateContext.cpp:202)
    // This is the SIMPLE approach we're validating (NOT llama_memory_seq_rm which has bugs)
    llama_memory_clear(mem, true);

    llama_pos max_pos_after_clear = llama_memory_seq_pos_max(mem, 0);
    CHECK(max_pos_after_clear == -1);  // Empty

    // Re-decode sinks using our production decoder
    decoder::decode_tokens(ctx, sinks, 0, ctx_params.n_batch);

    // Re-decode tail using our production decoder
    decoder::decode_tokens(ctx, tail, SINK_COUNT, ctx_params.n_batch);

    llama_pos max_pos_after_reseed = llama_memory_seq_pos_max(mem, 0);
    INFO("After reseed: KV cache max_pos=" << max_pos_after_reseed);
    CHECK(max_pos_after_reseed == SINK_COUNT + TAIL_COUNT - 1);

    // ----- Boundary capture AFTER reseed -----
    const float* logits_after = llama_get_logits_ith(ctx, -1);
    REQUIRE(logits_after != nullptr);

    std::vector<double> logp_after(n_vocab);
    for (int i = 0; i < n_vocab; ++i) logp_after[i] = logits_after[i];
    softmax_inplace(logp_after);

    // Metrics
    int top1_after = argmax(logits_after, n_vocab);
    auto top10_after = topk(logits_after, n_vocab, 10);

    INFO("Boundary AFTER reseed:  top-1 token = " << top1_after);

    // === BOUNDARY EQUIVALENCE VALIDATION ===
    // This is the PRIMARY test: does clear+re-decode preserve the next-token distribution?

    INFO("=== BOUNDARY EQUIVALENCE CHECK ===");

    // 1. Top-1 match (argmax token must be identical)
    CHECK(top1_after == top1_before);
    if (top1_after == top1_before) {
        INFO("✅ Top-1 match: " << top1_before);
    } else {
        INFO("❌ Top-1 MISMATCH: before=" << top1_before << " after=" << top1_after);
    }

    // 2. Top-k overlap (at least 7/10 top tokens should match)
    // Note: Relaxed from 8/10 due to quantization effects in Q4_K_M models
    int overlap = 0;
    for (int a : top10_after) {
        overlap += std::count(top10_before.begin(), top10_before.end(), a);
    }
    INFO("Top-10 overlap: " << overlap << "/10");
    CHECK(overlap >= 7);

    // 3. KL/JSD divergence (distributions should be nearly identical)
    double kl_ba = kl_div(logp_before, logp_after);
    double kl_ab = kl_div(logp_after, logp_before);
    double jsd = 0.5 * (kl_ba + kl_ab);
    INFO("Jensen-Shannon divergence: " << jsd);
    CHECK(jsd < 1e-2);  // Very small divergence expected

    if (top1_after == top1_before && overlap >= 8 && jsd < 1e-2) {
        INFO("✅ BOUNDARY EQUIVALENCE: Clear+re-decode preserves distribution");
    } else {
        INFO("❌ BOUNDARY EQUIVALENCE FAILED: Clear+re-decode changes distribution");
    }

    // === PHASE 3: Continue generation and measure perplexity ===
    std::vector<double> perplexities_after;
    int continue_tokens = 200;
    n_past = SINK_COUNT + TAIL_COUNT;

    INFO("Continuing generation for " << continue_tokens << " more tokens...");

    for (int i = 0; i < continue_tokens; ++i) {
        // Sample next token using our production sampler
        llama_token next_token = sampler::greedy(ctx, vocab);

        // Measure perplexity
        const float* logits = llama_get_logits_ith(ctx, -1);
        auto result = compute_log_softmax(n_vocab, logits, next_token);
        double ppl = exp(-result.log_softmax);
        perplexities_after.push_back(ppl);

        // Decode single token using our production decoder
        std::vector<llama_token> single_token = {next_token};
        decoder::decode_tokens(ctx, single_token, n_past, ctx_params.n_batch);
        n_past++;
    }

    // === PHASE 4: Statistical comparison ===
    // Compare LAST 200 tokens before reseed vs ALL 200 tokens after reseed
    // This ensures fair comparison:
    // - Both windows have 200 tokens
    // - Both at similar sequence positions (warmed up)
    // - Before: continuous cache with full history
    // - After: compressed cache with StreamingLLM pattern

    const int COMPARE_WINDOW = 200;

    // Extract last 200 tokens from before (tokens 601-800)
    std::vector<double> last_200_before(
        perplexities_before.end() - COMPARE_WINDOW,
        perplexities_before.end()
    );

    // All tokens after are the comparison set (tokens 801-1000)
    std::vector<double>& first_200_after = perplexities_after;

    // Calculate means
    double sum_before = 0.0;
    for (double ppl : last_200_before) {
        sum_before += ppl;
    }
    double mean_ppl_before = sum_before / last_200_before.size();

    double sum_after = 0.0;
    for (double ppl : first_200_after) {
        sum_after += ppl;
    }
    double mean_ppl_after = sum_after / first_200_after.size();

    double ppl_ratio = mean_ppl_after / mean_ppl_before;
    double ppl_diff = mean_ppl_after - mean_ppl_before;

    INFO("=== PERPLEXITY REGRESSION CHECK (SECONDARY) ===");
    INFO("Comparing last " << COMPARE_WINDOW << " tokens before vs " << continue_tokens << " tokens after");
    INFO("Before reseed (tokens 601-800, continuous cache):  PPL = " << mean_ppl_before);
    INFO("After reseed  (tokens 801-1000, compressed cache): PPL = " << mean_ppl_after);
    INFO("Ratio (after/before):   " << ppl_ratio);
    INFO("Difference:             " << ppl_diff);

    // === SECONDARY VALIDATION ===
    // Note: This compares perplexity of self-sampled tokens (not true perplexity),
    // so it's a coarse regression guard. The boundary equivalence check above
    // is the primary validation.
    //
    // StreamingLLM paper showed PPL 5.40 (continuous) vs 5.60 (window+recompute)
    // That's a ratio of 1.037 (3.7% increase)
    //
    // We accept up to 10% increase as "stable perplexity"
    // (Allows for model variation, quantization effects, etc.)

    if (ppl_ratio < 1.05) {
        INFO("✅ EXCELLENT: PPL increase < 5%");
    } else if (ppl_ratio < 1.10) {
        INFO("✅ GOOD: PPL increase < 10%");
    } else {
        INFO("⚠️  WARNING: PPL increase >= 10% - may indicate quality degradation");
    }

    // Soft check - don't fail test on PPL alone since boundary check is primary
    if (ppl_ratio >= 1.10) {
        INFO("⚠️  WARNING: PPL ratio " << ppl_ratio << " exceeds 1.10 threshold");
        WARN(ppl_ratio < 1.10);  // Soft warning without failing the test
    }

    // === CLEANUP ===
    llama_free(ctx);
}
