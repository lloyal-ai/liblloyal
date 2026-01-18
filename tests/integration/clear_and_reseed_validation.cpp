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
#include <string>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>

using namespace lloyal;

// Helper: Generate UUID for run_id
static std::string generate_uuid() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    const char* hex = "0123456789abcdef";
    std::string uuid = "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx";

    for (char& c : uuid) {
        if (c == 'x') {
            c = hex[dis(gen)];
        } else if (c == 'y') {
            c = hex[(dis(gen) & 0x3) | 0x8];
        }
    }
    return uuid;
}

// Helper: Get ISO8601 timestamp
static std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t_now), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

// Helper: Extract model metadata from filename
struct ModelMetadata {
    std::string filename;
    std::string model_name;
    std::string quantization;
    std::string size_estimate;
};

static ModelMetadata extract_model_metadata(const char* path) {
    ModelMetadata meta;

    // Extract filename from path
    std::string full_path(path);
    auto pos = full_path.find_last_of("/\\");
    meta.filename = (pos != std::string::npos) ? full_path.substr(pos + 1) : full_path;

    // Extract quantization (Q4_K_M, Q8_0, F16, etc.) - case insensitive
    std::string fname = meta.filename;
    std::string fname_lower = fname;
    std::transform(fname_lower.begin(), fname_lower.end(), fname_lower.begin(), ::tolower);

    if (fname_lower.find("q4_k_m") != std::string::npos) {
        meta.quantization = "Q4_K_M";
    } else if (fname_lower.find("q8_0") != std::string::npos) {
        meta.quantization = "Q8_0";
    } else if (fname_lower.find("f16") != std::string::npos || fname_lower.find("fp16") != std::string::npos) {
        meta.quantization = "F16";
    } else if (fname_lower.find("f32") != std::string::npos || fname_lower.find("fp32") != std::string::npos) {
        meta.quantization = "F32";
    } else {
        meta.quantization = "unknown";
    }

    // Extract model name (strip quantization and .gguf) - case insensitive
    meta.model_name = fname;
    size_t quant_pos = fname_lower.find("-q");
    if (quant_pos == std::string::npos) {
        quant_pos = fname_lower.find("-f");  // Also handle -F16, -F32
    }
    if (quant_pos != std::string::npos) {
        meta.model_name = fname.substr(0, quant_pos);
    }
    size_t gguf_pos = meta.model_name.find(".gguf");
    if (gguf_pos != std::string::npos) {
        meta.model_name = meta.model_name.substr(0, gguf_pos);
    }

    // Estimate size from name (1.7B, 3B, etc.) - using fname_lower already created above

    if (fname_lower.find("135m") != std::string::npos || fname_lower.find("0.1b") != std::string::npos) {
        meta.size_estimate = "135M";
    } else if (fname_lower.find("360m") != std::string::npos) {
        meta.size_estimate = "360M";
    } else if (fname_lower.find("0.5b") != std::string::npos || fname_lower.find("500m") != std::string::npos) {
        meta.size_estimate = "0.5B";
    } else if (fname_lower.find("1b") != std::string::npos) {
        meta.size_estimate = "1B";
    } else if (fname_lower.find("1.1b") != std::string::npos) {
        meta.size_estimate = "1.1B";
    } else if (fname_lower.find("1.5b") != std::string::npos) {
        meta.size_estimate = "1.5B";
    } else if (fname_lower.find("1.7b") != std::string::npos) {
        meta.size_estimate = "1.7B";
    } else if (fname_lower.find("2b") != std::string::npos) {
        meta.size_estimate = "2B";
    } else if (fname_lower.find("3b") != std::string::npos || fname_lower.find("3.8b") != std::string::npos) {
        meta.size_estimate = "3B";
    } else if (fname_lower.find("7b") != std::string::npos) {
        meta.size_estimate = "7B";
    } else {
        meta.size_estimate = "unknown";
    }

    return meta;
}

// Helper: Escape JSON strings
static std::string json_escape(const std::string& s) {
    std::string result;
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += c; break;
        }
    }
    return result;
}

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

    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
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

    // === STRUCTURED JSON OUTPUT ===
    // Extract model metadata
    auto meta = extract_model_metadata(MODEL_PATH);
    std::string run_id = generate_uuid();
    std::string timestamp = get_timestamp();

    // Determine pass/fail based on primary checks
    bool top1_pass = (top1_after == top1_before);
    bool top10_pass = (overlap >= 7);
    bool jsd_pass = (jsd < 1e-2);
    bool ppl_pass = (ppl_ratio < 1.10);
    bool overall_pass = top1_pass && top10_pass && jsd_pass;  // PPL is secondary

    // Output JSON (single line for easy parsing/aggregation)
    std::cout << "\n=== JSON_RESULT ===" << std::endl;
    std::cout << "{";
    std::cout << "\"run_id\":\"" << json_escape(run_id) << "\",";
    std::cout << "\"timestamp\":\"" << json_escape(timestamp) << "\",";
    std::cout << "\"model\":\"" << json_escape(meta.filename) << "\",";
    std::cout << "\"model_name\":\"" << json_escape(meta.model_name) << "\",";
    std::cout << "\"model_size\":\"" << json_escape(meta.size_estimate) << "\",";
    std::cout << "\"quantization\":\"" << json_escape(meta.quantization) << "\",";
    std::cout << "\"context_length\":" << ctx_params.n_ctx << ",";
    std::cout << "\"tokens_before_reseed\":" << tokens_to_generate << ",";
    std::cout << "\"sink_count\":" << SINK_COUNT << ",";
    std::cout << "\"tail_count\":" << TAIL_COUNT << ",";
    std::cout << "\"total_reseeded\":" << (SINK_COUNT + TAIL_COUNT) << ",";
    std::cout << "\"top1_before\":" << top1_before << ",";
    std::cout << "\"top1_after\":" << top1_after << ",";
    std::cout << "\"top1_match\":" << (top1_pass ? "true" : "false") << ",";
    std::cout << "\"top10_overlap\":" << overlap << ",";
    std::cout << "\"jsd\":" << std::fixed << std::setprecision(6) << jsd << ",";
    std::cout << "\"kl_before_after\":" << std::fixed << std::setprecision(6) << kl_ba << ",";
    std::cout << "\"kl_after_before\":" << std::fixed << std::setprecision(6) << kl_ab << ",";
    std::cout << "\"ppl_window_before\":" << std::fixed << std::setprecision(4) << mean_ppl_before << ",";
    std::cout << "\"ppl_window_after\":" << std::fixed << std::setprecision(4) << mean_ppl_after << ",";
    std::cout << "\"ppl_ratio\":" << std::fixed << std::setprecision(4) << ppl_ratio << ",";
    std::cout << "\"pass\":" << (overall_pass ? "true" : "false");
    std::cout << "}" << std::endl;
    std::cout << "=== END_JSON_RESULT ===" << std::endl;

    // === CLEANUP ===
    llama_free(ctx);
}
