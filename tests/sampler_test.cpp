#include "llama_stubs.h"
#include <doctest/doctest.h>
#include <lloyal/sampler.hpp>
#include <memory>
#include <optional>
#include <vector>

using namespace lloyal::sampler;

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

// ===== GREEDY SAMPLING TESTS =====

TEST_CASE("Sampler: null context guard") {
  resetStubConfig();

  llama_vocab vocab{};
  CHECK_THROWS(greedy(nullptr, &vocab));
}

TEST_CASE("Sampler: null vocab guard") {
  resetStubConfig();

  llama_context ctx{};
  CHECK_THROWS(greedy(&ctx, static_cast<const llama_vocab*>(nullptr)));
}

TEST_CASE("Sampler: null logits guard") {
  resetStubConfig();

  llama_context ctx{};
  llama_vocab vocab{};

  // Configure stub to return nullptr for logits
  llamaStubConfig().logits.clear(); // Empty vector → nullptr returned

  CHECK_THROWS(greedy(&ctx, &vocab));
}

TEST_CASE("Sampler: zero vocab size guard") {
  resetStubConfig();

  llama_context ctx{};
  llama_vocab vocab{};

  // Provide valid logits but zero vocab size
  llamaStubConfig().logits = {0.1f, 0.5f, 0.3f};
  llamaStubConfig().vocab_size_value = 0;

  CHECK_THROWS(greedy(&ctx, &vocab));
}

TEST_CASE("Sampler: argmax correctness") {
  resetStubConfig();

  llama_context ctx{};
  llama_vocab vocab{};

  // Configure stub with known logit distribution
  // Token 2 has highest probability
  llamaStubConfig().logits = {0.1f, 0.3f, 0.8f, 0.2f, 0.4f};
  llamaStubConfig().vocab_size_value = 5;

  llama_token result = greedy(&ctx, &vocab);
  CHECK(result == 2); // Index 2 has highest score (0.8)
}

TEST_CASE("Sampler: argmax tie-breaking (first wins)") {
  resetStubConfig();

  llama_context ctx{};
  llama_vocab vocab{};

  // Multiple tokens with same max score
  // Implementation should return the first occurrence
  llamaStubConfig().logits = {0.5f, 0.3f, 0.5f, 0.5f, 0.2f};
  llamaStubConfig().vocab_size_value = 5;

  llama_token result = greedy(&ctx, &vocab);
  CHECK(result == 0); // First token with score 0.5
}

// ===== PARAMETERIZED SAMPLING TESTS =====

TEST_CASE("Sampler: sample_with_params - null context guard") {
  resetStubConfig();

  llama_vocab vocab{};
  SamplingParams params;

  CHECK_THROWS(sample_with_params(nullptr, &vocab, params, nullptr));
}

TEST_CASE("Sampler: sample_with_params - null vocab guard") {
  resetStubConfig();

  llama_context ctx{};
  SamplingParams params;

  CHECK_THROWS(sample_with_params(&ctx, static_cast<const llama_vocab*>(nullptr), params, nullptr));
}

TEST_CASE("Sampler: sample_with_params - null logits guard") {
  resetStubConfig();

  llama_context ctx{};
  llama_vocab vocab{};
  SamplingParams params;

  // Configure stub to return nullptr for logits
  llamaStubConfig().logits.clear();

  CHECK_THROWS(sample_with_params(&ctx, &vocab, params, nullptr));
}

TEST_CASE("Sampler: sample_with_params - greedy sampling (temp=0, seed "
          "deterministic)") {
  resetStubConfig();

  llama_context ctx{};
  llama_vocab vocab{};

  // Configure stub with known logit distribution
  llamaStubConfig().logits = {0.1f, 0.3f, 0.8f, 0.2f, 0.4f};
  llamaStubConfig().vocab_size_value = 5;
  llamaStubConfig().sample_result = 2; // Stub returns token 2

  SamplingParams params;
  params.temperature = 0.0f; // Greedy mode
  params.seed = 42;

  llama_token result = sample_with_params(&ctx, &vocab, params, nullptr);
  CHECK(result == 2); // Should match stub sample_result
}

TEST_CASE("Sampler: sample_with_params - with temperature and top_k") {
  resetStubConfig();

  llama_context ctx{};
  llama_vocab vocab{};

  llamaStubConfig().logits = {0.1f, 0.3f, 0.8f, 0.2f, 0.4f};
  llamaStubConfig().vocab_size_value = 5;
  llamaStubConfig().sample_result = 4;

  SamplingParams params;
  params.temperature = 0.8f;
  params.top_k = 3;
  params.seed = 42;

  llama_token result = sample_with_params(&ctx, &vocab, params, nullptr);
  CHECK(result == 4); // Stub controls result
}

TEST_CASE("Sampler: sample_with_params - with penalties") {
  resetStubConfig();

  llama_context ctx{};
  llama_vocab vocab{};

  llamaStubConfig().logits = {0.1f, 0.3f, 0.8f, 0.2f, 0.4f};
  llamaStubConfig().vocab_size_value = 5;
  llamaStubConfig().sample_result = 1;

  SamplingParams params;
  params.penalty_repeat = 1.1f;
  params.penalty_freq = 0.1f;
  params.penalty_present = 0.05f;
  params.penalty_last_n = 64;

  llama_token result = sample_with_params(&ctx, &vocab, params, nullptr);
  CHECK(result == 1);
}

// ===== PART 2 PRIMITIVES: GRAMMAR SAMPLING TESTS =====
// NOTE: Grammar sampling tests require real llama.cpp and are in
// integration/Sampler.cpp Stubs cannot meaningfully test grammar sampling
// because llama_sampler_apply() logic requires actual parser state, which is
// too complex to mock accurately.

// ===== MODEL-ACCEPTING OVERLOAD TESTS =====

TEST_CASE("Sampler: greedy(ctx, model) null model guard") {
  resetStubConfig();

  llama_context ctx{};

  CHECK_THROWS(greedy(&ctx, static_cast<llama_model *>(nullptr)));
}

TEST_CASE("Sampler: greedy(ctx, model) successful") {
  resetStubConfig();

  llama_context ctx{};
  llama_model model{};

  // Configure stub with known logit distribution
  llamaStubConfig().logits = {0.1f, 0.3f, 0.8f, 0.2f, 0.4f};
  llamaStubConfig().vocab_size_value = 5;

  llama_token result = greedy(&ctx, &model);
  CHECK(result == 2); // Index 2 has highest score (0.8)
}

TEST_CASE("Sampler: sample_with_params(ctx, model, ...) null model guard") {
  resetStubConfig();

  llama_context ctx{};
  SamplingParams params;

  CHECK_THROWS(sample_with_params(&ctx, static_cast<llama_model *>(nullptr),
                                   params, nullptr));
}

TEST_CASE("Sampler: sample_with_params(ctx, model, ...) successful") {
  resetStubConfig();

  llama_context ctx{};
  llama_model model{};

  llamaStubConfig().logits = {0.1f, 0.3f, 0.8f, 0.2f, 0.4f};
  llamaStubConfig().vocab_size_value = 5;
  llamaStubConfig().sample_result = 2;

  SamplingParams params;
  params.temperature = 0.0f; // Greedy mode
  params.seed = 42;

  llama_token result = sample_with_params(&ctx, &model, params, nullptr);
  CHECK(result == 2); // Should match stub sample_result
}

TEST_CASE("Sampler: sample_with_params(ctx, model, ...) with grammar "
          "sampler") {
  resetStubConfig();

  llama_context ctx{};
  llama_model model{};

  llamaStubConfig().logits = {0.1f, 0.3f, 0.8f, 0.2f, 0.4f};
  llamaStubConfig().vocab_size_value = 5;
  llamaStubConfig().sample_result = 4;

  SamplingParams params;
  params.temperature = 0.8f;
  params.top_k = 3;

  // Note: Grammar sampler not fully testable with stubs (see line 199 comment)
  // Just verify overload accepts grammar parameter
  llama_token result = sample_with_params(&ctx, &model, params, nullptr);
  CHECK(result == 4);
}
