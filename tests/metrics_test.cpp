/**
 * Metrics Unit Tests
 *
 * Validates that the C++ implementation matches tsampler/metrics.ts behavior.
 * Tests numerical stability, edge cases, and handle-based perplexity tracking.
 */

#include <cmath>
#include <doctest/doctest.h>
#include <limits>
#include <lloyal/metrics.hpp>
#include <vector>

using namespace lloyal::metrics;

// ============================================================================
// Test helpers
// ============================================================================

constexpr float INF = std::numeric_limits<float>::infinity();
constexpr float EPSILON = 1e-5f;

bool approx_eq(float a, float b, float eps = EPSILON) {
  if (std::isinf(a) && std::isinf(b)) return true;
  if (std::isnan(a) || std::isnan(b)) return false;
  return std::abs(a - b) < eps;
}

// ============================================================================
// model_surprisal tests
// ============================================================================

TEST_CASE("metrics: model_surprisal with uniform distribution") {
  // Uniform logits -> all tokens equally likely -> surprisal = log(n_vocab)
  std::vector<float> logits(100, 0.0f);  // 100 tokens, all logit=0

  float s = model_surprisal(logits.data(), 100, 0);
  float expected = std::log(100.0f);  // ~4.605 nats

  CHECK(approx_eq(s, expected));
}

TEST_CASE("metrics: model_surprisal with peaked distribution") {
  // One token has high logit, others low -> low surprisal for high token
  std::vector<float> logits(100, -10.0f);
  logits[42] = 10.0f;  // Token 42 is very likely

  float s = model_surprisal(logits.data(), 100, 42);
  CHECK(s < 0.1f);  // Should be near 0 (very expected)

  // Picking a low-probability token should give high surprisal
  float s_low = model_surprisal(logits.data(), 100, 0);
  CHECK(s_low > 15.0f);  // Should be high (very surprising)
}

TEST_CASE("metrics: model_surprisal bits conversion") {
  std::vector<float> logits(8, 0.0f);  // 8 tokens uniform

  float s_nats = model_surprisal(logits.data(), 8, 0, Base::Nats);
  float s_bits = model_surprisal(logits.data(), 8, 0, Base::Bits);

  // For 8 tokens: surprisal = log(8) nats = log2(8) = 3 bits
  CHECK(approx_eq(s_nats, std::log(8.0f)));
  CHECK(approx_eq(s_bits, 3.0f));
}

TEST_CASE("metrics: model_surprisal edge cases") {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f};

  // Null logits
  CHECK(std::isinf(model_surprisal(nullptr, 3, 0)));

  // Empty vocab
  CHECK(std::isinf(model_surprisal(logits.data(), 0, 0)));

  // Out of range picked_id
  CHECK(std::isinf(model_surprisal(logits.data(), 3, -1)));
  CHECK(std::isinf(model_surprisal(logits.data(), 3, 3)));

  // -Infinity logit for picked token
  std::vector<float> masked = {-INF, 2.0f, 3.0f};
  CHECK(std::isinf(model_surprisal(masked.data(), 3, 0)));
}

TEST_CASE("metrics: model_surprisal numerical stability with large values") {
  // Large positive logits shouldn't overflow
  std::vector<float> large(100, 500.0f);
  large[0] = 510.0f;  // Slightly higher

  float s = model_surprisal(large.data(), 100, 0);
  CHECK(std::isfinite(s));
  CHECK(s >= 0.0f);
}

TEST_CASE("metrics: model_surprisal numerical stability with small values") {
  // Large negative logits shouldn't underflow to NaN
  std::vector<float> small(100, -500.0f);
  small[0] = -490.0f;  // Slightly less negative

  float s = model_surprisal(small.data(), 100, 0);
  CHECK(std::isfinite(s));
  CHECK(s >= 0.0f);
}

// ============================================================================
// model_entropy tests
// ============================================================================

TEST_CASE("metrics: model_entropy with uniform distribution") {
  // Uniform -> max entropy = log(n)
  std::vector<float> logits(100, 0.0f);

  float h = model_entropy(logits.data(), 100);
  float expected = std::log(100.0f);  // ~4.605 nats

  CHECK(approx_eq(h, expected));
}

TEST_CASE("metrics: model_entropy with peaked distribution") {
  // One dominant token -> low entropy (near 0)
  std::vector<float> logits(100, -100.0f);
  logits[42] = 100.0f;

  float h = model_entropy(logits.data(), 100);
  CHECK(h < 0.01f);  // Should be near 0
}

TEST_CASE("metrics: model_entropy bits conversion") {
  std::vector<float> logits(8, 0.0f);

  float h_nats = model_entropy(logits.data(), 8, Base::Nats);
  float h_bits = model_entropy(logits.data(), 8, Base::Bits);

  CHECK(approx_eq(h_nats, std::log(8.0f)));
  CHECK(approx_eq(h_bits, 3.0f));
}

TEST_CASE("metrics: model_entropy edge cases") {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f};

  CHECK(std::isinf(model_entropy(nullptr, 3)));
  CHECK(std::isinf(model_entropy(logits.data(), 0)));
}

TEST_CASE("metrics: model_entropy single token") {
  // Single token -> entropy = 0
  std::vector<float> logits = {5.0f};

  float h = model_entropy(logits.data(), 1);
  CHECK(approx_eq(h, 0.0f));
}

// ============================================================================
// sampling_surprisal tests
// ============================================================================

TEST_CASE("metrics: sampling_surprisal basic") {
  std::vector<float> logits = {1.0f, 2.0f, 3.0f};
  std::vector<int32_t> ids = {10, 20, 30};

  float s = sampling_surprisal(logits.data(), ids.data(), 3, 30);
  CHECK(std::isfinite(s));
  CHECK(s >= 0.0f);
  CHECK(s < 2.0f);  // Token 30 has highest logit
}

TEST_CASE("metrics: sampling_surprisal single candidate") {
  std::vector<float> logits = {5.0f};
  std::vector<int32_t> ids = {42};

  // Single candidate -> surprisal = 0
  float s = sampling_surprisal(logits.data(), ids.data(), 1, 42);
  CHECK(approx_eq(s, 0.0f));
}

TEST_CASE("metrics: sampling_surprisal picked not in candidates") {
  std::vector<float> logits = {1.0f, 2.0f};
  std::vector<int32_t> ids = {10, 20};

  float s = sampling_surprisal(logits.data(), ids.data(), 2, 999);
  CHECK(std::isinf(s));
}

// ============================================================================
// sampling_entropy tests
// ============================================================================

TEST_CASE("metrics: sampling_entropy basic") {
  std::vector<float> logits = {0.0f, 0.0f, 0.0f};

  float h = sampling_entropy(logits.data(), 3);
  CHECK(approx_eq(h, std::log(3.0f)));
}

TEST_CASE("metrics: sampling_entropy single candidate") {
  std::vector<float> logits = {5.0f};

  float h = sampling_entropy(logits.data(), 1);
  CHECK(approx_eq(h, 0.0f));
}

// ============================================================================
// RollingPerplexity handle-based tests
// ============================================================================

TEST_CASE("metrics: perplexity create and free") {
  auto h = create_perplexity();
  CHECK(h > 0);

  // Fresh handle should have no samples
  CHECK(get_count(h) == 0);
  CHECK(std::isinf(get_ppl(h)));

  free_perplexity(h);

  // After free, should return defaults
  CHECK(get_count(h) == 0);
}

TEST_CASE("metrics: perplexity add_surprisal and get_ppl") {
  auto h = create_perplexity();

  // Add surprisal of 2 nats for 3 tokens
  add_surprisal(h, 2.0f);
  add_surprisal(h, 2.0f);
  add_surprisal(h, 2.0f);

  CHECK(get_count(h) == 3);

  // ppl = exp(mean(surprisal)) = exp(2) ≈ 7.389
  float ppl = get_ppl(h);
  CHECK(approx_eq(ppl, std::exp(2.0f)));

  free_perplexity(h);
}

TEST_CASE("metrics: perplexity ignores non-finite values") {
  auto h = create_perplexity();

  add_surprisal(h, 1.0f);
  add_surprisal(h, INF);  // Should be ignored
  add_surprisal(h, std::nan(""));  // Should be ignored
  add_surprisal(h, 1.0f);

  CHECK(get_count(h) == 2);  // Only 2 valid samples
  CHECK(approx_eq(get_ppl(h), std::exp(1.0f)));

  free_perplexity(h);
}

TEST_CASE("metrics: perplexity reset") {
  auto h = create_perplexity();

  add_surprisal(h, 5.0f);
  add_surprisal(h, 5.0f);
  CHECK(get_count(h) == 2);

  reset_perplexity(h);

  CHECK(get_count(h) == 0);
  CHECK(std::isinf(get_ppl(h)));

  free_perplexity(h);
}

TEST_CASE("metrics: perplexity clone") {
  auto h1 = create_perplexity();

  add_surprisal(h1, 2.0f);
  add_surprisal(h1, 4.0f);
  // mean = 3, ppl = exp(3) ≈ 20.09

  auto h2 = clone_perplexity(h1);
  CHECK(h2 != h1);
  CHECK(h2 > 0);

  // Clone should have same state
  CHECK(get_count(h2) == 2);
  CHECK(approx_eq(get_ppl(h2), get_ppl(h1)));

  // Modifying clone shouldn't affect original
  add_surprisal(h2, 6.0f);
  CHECK(get_count(h1) == 2);  // Original unchanged
  CHECK(get_count(h2) == 3);  // Clone updated

  // Clone ppl: mean = (2+4+6)/3 = 4, ppl = exp(4) ≈ 54.6
  CHECK(approx_eq(get_ppl(h2), std::exp(4.0f)));

  free_perplexity(h1);
  free_perplexity(h2);
}

TEST_CASE("metrics: perplexity clone invalid handle") {
  auto h = clone_perplexity(99999);  // Invalid handle
  CHECK(h == 0);
}

TEST_CASE("metrics: multiple independent handles") {
  auto h1 = create_perplexity();
  auto h2 = create_perplexity();

  add_surprisal(h1, 1.0f);
  add_surprisal(h2, 5.0f);

  CHECK(get_count(h1) == 1);
  CHECK(get_count(h2) == 1);
  CHECK(approx_eq(get_ppl(h1), std::exp(1.0f)));
  CHECK(approx_eq(get_ppl(h2), std::exp(5.0f)));

  free_perplexity(h1);
  free_perplexity(h2);
}

// ============================================================================
// Integration: surprisal + perplexity workflow
// ============================================================================

TEST_CASE("metrics: end-to-end surprisal to perplexity") {
  // Simulate generating 3 tokens from a distribution
  std::vector<float> logits(100, 0.0f);  // Uniform

  auto ppl_handle = create_perplexity();

  // "Generate" 3 tokens (all have same probability in uniform)
  for (int token : {10, 50, 90}) {
    float s = model_surprisal(logits.data(), 100, token);
    add_surprisal(ppl_handle, s);
  }

  // For uniform distribution: surprisal = log(100), ppl = exp(log(100)) = 100
  float ppl = get_ppl(ppl_handle);
  CHECK(approx_eq(ppl, 100.0f, 0.1f));

  free_perplexity(ppl_handle);
}
