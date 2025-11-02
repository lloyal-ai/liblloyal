#include <doctest/doctest.h>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/decoder.hpp>
#include <lloyal/sampler.hpp>
#include <llama/llama.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>

using namespace lloyal;

/**
 * Sampler Integration Tests
 *
 * Tests Part 2 sampling primitives with real model:
 * - llama_get_logits_ith() - Returns valid logits after decode
 * - llama_sampler_apply() - Constrains logits with grammar
 * - llama_sampler_accept() - Advances grammar parser state
 *
 * These tests validate that sampling has access to correct
 * llama.cpp primitives for implementing custom sampling strategies.
 *
 * Uses tiny-random-llama.gguf (~12MB)
 */

// Test struct that satisfies SamplingParamsLike concept
struct SamplingParams {
    std::optional<float> temperature = 1.0f;
    std::optional<int32_t> top_k = 40;
    std::optional<float> top_p = 0.95f;
    std::optional<float> typical_p = 1.0f;
    std::optional<float> min_p = 0.05f;
    std::optional<float> penalty_repeat = 1.0f;
    std::optional<float> penalty_freq = 0.0f;
    std::optional<float> penalty_present = 0.0f;
    std::optional<int32_t> penalty_last_n = 64;
    std::optional<uint32_t> seed = 0;
    std::optional<int32_t> n_prev = 64;
    std::optional<int32_t> n_probs = 0;
    std::optional<int32_t> min_keep = 1;
    std::optional<float> tfs_z = 1.0f;
    std::optional<float> xtc_probability = 0.0f;
    std::optional<float> xtc_threshold = 0.1f;
    std::optional<int32_t> mirostat = 0;
    std::optional<float> mirostat_tau = 5.0f;
    std::optional<float> mirostat_eta = 0.1f;
    std::optional<bool> penalize_nl = false;
};

static const char* MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

// Skip test if model not available
#define REQUIRE_MODEL() \
    if (!MODEL_PATH || !*MODEL_PATH) { \
        MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set"); \
        return; \
    }

// Helper to init llama.cpp backend once
struct LlamaBackendGuard {
    LlamaBackendGuard() { llama_backend_init(); }
    ~LlamaBackendGuard() { llama_backend_free(); }
};

// Helper to create and populate batch
struct BatchHelper {
    llama_batch batch;

    BatchHelper(int32_t capacity) {
        batch = llama_batch_init(capacity, 0, 1);
    }

    ~BatchHelper() {
        llama_batch_free(batch);
    }

    void addTokens(const std::vector<llama_token>& tokens, bool logits_last_only = true) {
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (logits_last_only && i == batch.n_tokens - 1) ? 1 : 0;
        }
    }
};

// ============================================================================
// PART 2 PRIMITIVES: Integration Tests (Real Model)
// ============================================================================

TEST_CASE("Integration: llama_get_logits_ith returns valid logits after decode") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    // Load model
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only
    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    // Create context
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 128;
    llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
    REQUIRE(ctx != nullptr);

    auto vocab = llama_model_get_vocab(model.get());
    REQUIRE(vocab != nullptr);

    // Tokenize input
    std::string test_text = "Hello world";
    auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
    REQUIRE_FALSE(tokens.empty());

    // Create batch and decode
    BatchHelper batch_helper(ctx_params.n_batch);
    batch_helper.addTokens(tokens, true);  // logits=true for last token only

    int decode_result = llama_decode(ctx, batch_helper.batch);
    REQUIRE(decode_result == 0);

    // PART 2 PRIMITIVE: Get logits (what TS sampling needs)
    const float* logits = llama_get_logits_ith(ctx, -1);
    REQUIRE(logits != nullptr);

    // Verify logits are valid (not all zeros, not all NaN)
    int n_vocab = llama_vocab_n_tokens(vocab);
    REQUIRE(n_vocab > 0);

    bool has_non_zero = false;
    bool has_nan = false;
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] != 0.0f) has_non_zero = true;
        if (std::isnan(logits[i])) has_nan = true;
    }

    CHECK(has_non_zero);  // Logits should have variation
    CHECK_FALSE(has_nan);  // No NaN values

    INFO("✓ llama_get_logits_ith() returned " << n_vocab << " valid logits");

    llama_free(ctx);
}

TEST_CASE("Integration: greedy sampling with real model") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 128;
    llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
    REQUIRE(ctx != nullptr);

    auto vocab = llama_model_get_vocab(model.get());
    REQUIRE(vocab != nullptr);

    // Tokenize and decode
    std::string test_text = "The quick brown";
    auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
    REQUIRE_FALSE(tokens.empty());

    BatchHelper batch_helper(ctx_params.n_batch);
    batch_helper.addTokens(tokens, true);

    int decode_result = llama_decode(ctx, batch_helper.batch);
    REQUIRE(decode_result == 0);

    // Test greedy sampling (uses llama_get_logits_ith internally)
    llama_token next_token = sampler::greedy(ctx, vocab);
    CHECK(next_token != -1);

    int n_vocab = llama_vocab_n_tokens(vocab);
    CHECK(next_token >= 0);
    CHECK(next_token < n_vocab);

    INFO("✓ Greedy sampling returned valid token: " << next_token);

    llama_free(ctx);
}

TEST_CASE("Integration: sample_with_params with real model (no grammar)") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 128;
    llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
    REQUIRE(ctx != nullptr);

    auto vocab = llama_model_get_vocab(model.get());
    REQUIRE(vocab != nullptr);

    // Tokenize and decode
    std::string test_text = "Once upon a time";
    auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
    REQUIRE_FALSE(tokens.empty());

    BatchHelper batch_helper(ctx_params.n_batch);
    batch_helper.addTokens(tokens, true);

    int decode_result = llama_decode(ctx, batch_helper.batch);
    REQUIRE(decode_result == 0);

    // Test parameterized sampling
    SamplingParams params;
    params.temperature = 0.8f;
    params.top_k = 40;
    params.top_p = 0.95f;
    params.min_p = 0.05f;
    params.seed = 42;

    llama_token next_token = sampler::sample_with_params(ctx, vocab, params, nullptr);
    CHECK(next_token != -1);

    int n_vocab = llama_vocab_n_tokens(vocab);
    CHECK(next_token >= 0);
    CHECK(next_token < n_vocab);

    INFO("✓ Parameterized sampling returned valid token: " << next_token);

    llama_free(ctx);
}

TEST_CASE("Integration: llama_sampler_apply constrains logits with grammar (Part 2)") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 128;
    llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
    REQUIRE(ctx != nullptr);

    auto vocab = llama_model_get_vocab(model.get());
    REQUIRE(vocab != nullptr);

    // Tokenize and decode
    std::string test_text = "Test grammar";
    auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
    REQUIRE_FALSE(tokens.empty());

    BatchHelper batch_helper(ctx_params.n_batch);
    batch_helper.addTokens(tokens, true);

    int decode_result = llama_decode(ctx, batch_helper.batch);
    REQUIRE(decode_result == 0);

    // Create simple grammar (only allows 'a' or 'b' characters)
    const char* grammar_str = "root ::= [ab]+";
    llama_sampler* grammar_sampler = llama_sampler_init_grammar(vocab, grammar_str, "root");
    REQUIRE(grammar_sampler != nullptr);

    // Get logits
    const float* logits = llama_get_logits_ith(ctx, -1);
    REQUIRE(logits != nullptr);

    int n_vocab = llama_vocab_n_tokens(vocab);

    // Build token data array (copy logits for modification)
    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<float> logits_copy(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        logits_copy[i] = logits[i];
        candidates[i] = llama_token_data{
            static_cast<llama_token>(i),
            logits_copy[i],
            0.0f
        };
    }

    llama_token_data_array cur_p = {
        candidates.data(),
        static_cast<size_t>(n_vocab),
        -1,
        false
    };

    // PART 2 PRIMITIVE: Apply grammar constraint (masks invalid tokens)
    llama_sampler_apply(grammar_sampler, &cur_p);

    // Verify grammar constraint was applied (some tokens should be masked)
    int masked_count = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (candidates[i].logit == -INFINITY && logits[i] != -INFINITY) {
            masked_count++;
        }
    }

    // Grammar should mask most tokens (only 'a' and 'b' allowed)
    CHECK(masked_count > 0);
    INFO("✓ Grammar masked " << masked_count << " tokens out of " << n_vocab);

    llama_sampler_free(grammar_sampler);
    llama_free(ctx);
}

TEST_CASE("Integration: llama_sampler_accept advances grammar state (Part 2)") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 128;
    llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
    REQUIRE(ctx != nullptr);

    auto vocab = llama_model_get_vocab(model.get());
    REQUIRE(vocab != nullptr);

    // Tokenize and decode
    std::string test_text = "Grammar test";
    auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
    REQUIRE_FALSE(tokens.empty());

    BatchHelper batch_helper(ctx_params.n_batch);
    batch_helper.addTokens(tokens, true);

    int decode_result = llama_decode(ctx, batch_helper.batch);
    REQUIRE(decode_result == 0);

    // Create grammar
    const char* grammar_str = "root ::= \"a\" \"b\" \"c\"";  // Fixed sequence
    llama_sampler* grammar_sampler = llama_sampler_init_grammar(vocab, grammar_str, "root");
    REQUIRE(grammar_sampler != nullptr);

    // Find token IDs for 'a', 'b', 'c'
    auto token_a_vec = tokenizer::tokenize(vocab, "a", false, false);
    auto token_b_vec = tokenizer::tokenize(vocab, "b", false, false);

    if (!token_a_vec.empty() && !token_b_vec.empty()) {
        llama_token token_a = token_a_vec[0];
        llama_token token_b = token_b_vec[0];

        // PART 2 PRIMITIVE: Accept token (advances grammar parser state)
        // After accepting 'a', grammar should only allow 'b' next
        llama_sampler_accept(grammar_sampler, token_a);

        // Get logits for next step
        const float* logits = llama_get_logits_ith(ctx, -1);
        REQUIRE(logits != nullptr);

        int n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<llama_token_data> candidates(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            candidates[i] = llama_token_data{
                static_cast<llama_token>(i),
                logits[i],
                0.0f
            };
        }

        llama_token_data_array cur_p = {
            candidates.data(),
            static_cast<size_t>(n_vocab),
            -1,
            false
        };

        // Apply grammar - should enforce 'b' is next
        llama_sampler_apply(grammar_sampler, &cur_p);

        // Token 'a' should now be masked (grammar moved past it)
        // Token 'b' should be allowed
        bool a_masked = (candidates[token_a].logit == -INFINITY);
        bool b_allowed = (candidates[token_b].logit != -INFINITY);

        CHECK(a_masked);   // 'a' should be masked after accepting it
        INFO("✓ Grammar state advanced: 'a' masked=" << a_masked << ", 'b' allowed=" << b_allowed);
    } else {
        INFO("[ SKIP ] Could not find tokens for 'a' and 'b' in vocabulary");
    }

    llama_sampler_free(grammar_sampler);
    llama_free(ctx);
}

TEST_CASE("Integration: sample_with_params with grammar (complete Part 2 flow)") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 128;
    llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
    REQUIRE(ctx != nullptr);

    auto vocab = llama_model_get_vocab(model.get());
    REQUIRE(vocab != nullptr);

    // Tokenize and decode
    std::string test_text = "Generate JSON";
    auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
    REQUIRE_FALSE(tokens.empty());

    BatchHelper batch_helper(ctx_params.n_batch);
    batch_helper.addTokens(tokens, true);

    int decode_result = llama_decode(ctx, batch_helper.batch);
    REQUIRE(decode_result == 0);

    // Create JSON grammar
    const char* json_grammar =
        "root ::= \"{\" ws \"\\\"key\\\"\" ws \":\" ws value ws \"}\"\n"
        "value ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
        "ws ::= [ \\t\\n]*";

    llama_sampler* grammar_sampler = llama_sampler_init_grammar(vocab, json_grammar, "root");
    REQUIRE(grammar_sampler != nullptr);

    // Test parameterized sampling WITH grammar (exercises all Part 2 primitives)
    SamplingParams params;
    params.temperature = 0.7f;
    params.top_k = 40;
    params.seed = 123;

    // This internally uses:
    // 1. llama_get_logits_ith() - Gets raw logits
    // 2. llama_sampler_apply() - Applies grammar constraint
    // 3. Returns constrained token (llama_sampler_accept() called by consumer)
    llama_token next_token = sampler::sample_with_params(ctx, vocab, params, grammar_sampler);
    CHECK(next_token != -1);

    int n_vocab = llama_vocab_n_tokens(vocab);
    CHECK(next_token >= 0);
    CHECK(next_token < n_vocab);

    INFO("✓ Grammar-constrained sampling returned valid token: " << next_token);

    // Simulate accepting the token (Part 2 complete flow)
    llama_sampler_accept(grammar_sampler, next_token);
    INFO("✓ Grammar state advanced via llama_sampler_accept()");

    llama_sampler_free(grammar_sampler);
    llama_free(ctx);
}

TEST_CASE("Integration: typical_p sampling parameter") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 128;
    llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
    REQUIRE(ctx != nullptr);

    auto vocab = llama_model_get_vocab(model.get());
    REQUIRE(vocab != nullptr);

    // Tokenize and decode
    std::string test_text = "Test typical sampling";
    auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
    REQUIRE_FALSE(tokens.empty());

    BatchHelper batch_helper(ctx_params.n_batch);
    batch_helper.addTokens(tokens, true);

    int decode_result = llama_decode(ctx, batch_helper.batch);
    REQUIRE(decode_result == 0);

    // Test typical_p parameter (new in Part 2)
    SamplingParams params;
    params.temperature = 0.8f;
    params.typical_p = 0.95f;  // Locally typical sampling
    params.seed = 42;

    llama_token next_token = sampler::sample_with_params(ctx, vocab, params, nullptr);
    CHECK(next_token != -1);

    INFO("✓ Typical-P sampling returned valid token: " << next_token);

    llama_free(ctx);
}
