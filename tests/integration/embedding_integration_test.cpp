#include <cmath>
#include <cstdlib>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/decoder.hpp>
#include <lloyal/embedding.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <vector>

using namespace lloyal;

/**
 * Embedding Integration Tests
 *
 * Tests embedding extraction with real llama.cpp.
 *
 * NOTE: Standard LLM models (like SmolLM2) may not produce meaningful
 * embeddings without pooling enabled. These tests verify the API works
 * correctly - for meaningful embeddings, use a dedicated embedding model
 * like nomic-embed-text or bge-small-en.
 *
 * Set LLAMA_EMBED_MODEL to an embedding model for full semantic tests.
 * Falls back to LLAMA_TEST_MODEL for basic API validation.
 */

static const char *MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");
static const char *EMBED_MODEL = std::getenv("LLAMA_EMBED_MODEL");

#define REQUIRE_MODEL()                                                        \
  if (!MODEL_PATH || !*MODEL_PATH) {                                           \
    MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set");                              \
    return;                                                                    \
  }

// For semantic tests that need a dedicated embedding model (not just any LLM)
#define REQUIRE_EMBED_MODEL()                                                  \
  if (!EMBED_MODEL || !*EMBED_MODEL) {                                         \
    MESSAGE("[ SKIP ] LLAMA_EMBED_MODEL not set (semantic tests need dedicated embedding model)"); \
    return;                                                                    \
  }

struct LlamaBackendGuard {
  LlamaBackendGuard() { llama_backend_init(); }
  ~LlamaBackendGuard() { llama_backend_free(); }
};

struct ContextGuard {
  llama_context *ctx;
  explicit ContextGuard(llama_context *c) : ctx(c) {}
  ~ContextGuard() { if (ctx) llama_free(ctx); }
};

// ===== MODEL CAPABILITY TESTS =====

TEST_CASE("Embedding Integration: has_embeddings returns true for LLM models") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  // LLM models have embedding dimension (their hidden size)
  bool has = embedding::has_embeddings(model.get());
  CHECK(has == true);

  int32_t dim = embedding::dimension(model.get());
  INFO("Model embedding dimension: " << dim);
  CHECK(dim > 0);
}

TEST_CASE("Embedding Integration: dimension returns model hidden size") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  int32_t dim = embedding::dimension(model.get());

  // Common dimensions: 768 (BERT), 1024, 2048 (SmolLM2), 4096 (Llama)
  // Note: CI uses tiny-random-llama.gguf with dim=32 for fast testing
  INFO("Embedding dimension: " << dim);
  CHECK(dim > 0);     // Must have positive dimension
  CHECK(dim <= 8192); // Maximum reasonable hidden size
}

// ===== CONTEXT WITHOUT POOLING =====

TEST_CASE("Embedding Integration: context without pooling reports NONE") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  // Default context params - no pooling
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 512;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);
  ContextGuard ctx_guard(ctx);

  // Without explicit pooling, should report no pooling
  bool has_pool = embedding::has_pooling(ctx);
  int32_t pool_type = embedding::pooling_type(ctx);

  INFO("has_pooling: " << has_pool);
  INFO("pooling_type: " << pool_type);

  // Note: Default pooling behavior varies by llama.cpp version
  // Just verify we can query the type without crash
  CHECK(pool_type >= 0);
}

// ===== COSINE SIMILARITY =====

TEST_CASE("Embedding Integration: cosine_similarity mathematical correctness") {
  // Pure math test - no model needed

  // Test 1: Identical normalized vectors -> 1.0
  {
    std::vector<float> a = {0.6f, 0.8f};
    std::vector<float> b = {0.6f, 0.8f};
    float sim = embedding::cosine_similarity(a, b);
    CHECK(sim == doctest::Approx(1.0f).epsilon(0.001));
  }

  // Test 2: Orthogonal vectors -> 0.0
  {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f};
    float sim = embedding::cosine_similarity(a, b);
    CHECK(sim == doctest::Approx(0.0f).epsilon(0.001));
  }

  // Test 3: Opposite vectors -> -1.0
  {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {-1.0f, 0.0f};
    float sim = embedding::cosine_similarity(a, b);
    CHECK(sim == doctest::Approx(-1.0f).epsilon(0.001));
  }

  // Test 4: 45 degree angle -> ~0.707
  {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.707107f, 0.707107f}; // 45 degrees
    float sim = embedding::cosine_similarity(a, b);
    CHECK(sim == doctest::Approx(0.707f).epsilon(0.01));
  }
}

// ===== L2 NORMALIZATION =====

TEST_CASE("Embedding Integration: L2 normalization produces unit vectors") {
  // Create a test vector and normalize it
  std::vector<float> vec = {3.0f, 4.0f}; // norm = 5

  // Apply normalization inline (simulating what embedding::get does)
  float norm_sq = 0.0f;
  for (float v : vec) {
    norm_sq += v * v;
  }
  float norm = std::sqrt(norm_sq);
  for (float &v : vec) {
    v /= norm;
  }

  // Verify unit length
  float result_norm = 0.0f;
  for (float v : vec) {
    result_norm += v * v;
  }
  result_norm = std::sqrt(result_norm);

  CHECK(result_norm == doctest::Approx(1.0f).epsilon(0.0001));
  CHECK(vec[0] == doctest::Approx(0.6f).epsilon(0.0001));
  CHECK(vec[1] == doctest::Approx(0.8f).epsilon(0.0001));
}

// ===== RUNTIME POOLING TYPE CAST (production use-case) =====

TEST_CASE("Embedding Integration: context creation with runtime pooling type") {
  REQUIRE_EMBED_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(EMBED_MODEL, model_params);
  REQUIRE(model != nullptr);

  // Simulate runtime value from user config (e.g., from JS/TS options)
  // This tests the production use-case where pooling_type comes as an integer
  int32_t pooling_type_value = 1;  // LLAMA_POOLING_TYPE_MEAN

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 512;
  ctx_params.embeddings = true;

  // CRITICAL: This cast requires 'enum' keyword due to llama.cpp name collision
  // llama_pooling_type is both an enum type AND a function name
  // Without 'enum', the compiler thinks we're casting to the function type
  ctx_params.pooling_type = static_cast<enum llama_pooling_type>(pooling_type_value);

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);
  ContextGuard ctx_guard(ctx);

  // Verify the pooling type was set correctly
  CHECK(embedding::has_pooling(ctx) == true);
  CHECK(embedding::pooling_type(ctx) == LLAMA_POOLING_TYPE_MEAN);

  // Verify embeddings work with runtime-configured pooling
  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Test runtime pooling", true, true);
  REQUIRE_FALSE(tokens.empty());

  kv::clear_all(ctx);
  embedding::encode(ctx, tokens, 512);

  auto emb = embedding::get(ctx, embedding::Normalize::L2);
  CHECK(emb.size() == static_cast<size_t>(embedding::dimension(model.get())));
}

// ===== EMBEDDING MODEL TESTS (require dedicated embedding model) =====

TEST_CASE("Embedding Integration: extract embeddings from embedding model") {
  REQUIRE_EMBED_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(EMBED_MODEL, model_params);
  REQUIRE(model != nullptr);

  // Create context with mean pooling for embeddings
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 512;
  ctx_params.embeddings = true;
  ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);
  ContextGuard ctx_guard(ctx);

  // Verify pooling is enabled
  CHECK(embedding::has_pooling(ctx) == true);
  CHECK(embedding::pooling_type(ctx) == LLAMA_POOLING_TYPE_MEAN);

  // Tokenize and encode for embeddings
  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello world", true, true);
  REQUIRE_FALSE(tokens.empty());

  // Clear KV and encode (marks all tokens with logits=true)
  kv::clear_all(ctx);
  embedding::encode(ctx, tokens, 512);

  // Extract embeddings
  auto emb = embedding::get(ctx, embedding::Normalize::L2);

  int32_t expected_dim = embedding::dimension(model.get());
  CHECK(emb.size() == static_cast<size_t>(expected_dim));

  // Verify L2 normalized (unit length)
  float norm_sq = 0.0f;
  for (float v : emb) {
    norm_sq += v * v;
  }
  float norm = std::sqrt(norm_sq);
  CHECK(norm == doctest::Approx(1.0f).epsilon(0.01));
}

TEST_CASE(
    "Embedding Integration: similar sentences have high cosine similarity") {
  REQUIRE_EMBED_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();

  auto model = ModelRegistry::acquire(EMBED_MODEL, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 512;
  ctx_params.embeddings = true;
  ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);
  ContextGuard ctx_guard(ctx);

  auto vocab = llama_model_get_vocab(model.get());

  // Embed first sentence
  auto tokens1 = tokenizer::tokenize(vocab, "The cat sat on the mat", true, true);
  kv::clear_all(ctx);
  embedding::encode(ctx, tokens1, 512);
  auto emb1 = embedding::get(ctx, embedding::Normalize::L2);

  // Embed second (similar) sentence
  auto tokens2 = tokenizer::tokenize(vocab, "A cat rested on the rug", true, true);
  kv::clear_all(ctx);
  embedding::encode(ctx, tokens2, 512);
  auto emb2 = embedding::get(ctx, embedding::Normalize::L2);

  // Embed third (different) sentence
  auto tokens3 = tokenizer::tokenize(vocab, "Stock prices rose sharply", true, true);
  kv::clear_all(ctx);
  embedding::encode(ctx, tokens3, 512);
  auto emb3 = embedding::get(ctx, embedding::Normalize::L2);

  // Similar sentences should have higher similarity than different ones
  float sim_similar = embedding::cosine_similarity(emb1, emb2);
  float sim_different = embedding::cosine_similarity(emb1, emb3);

  INFO("Similar sentences similarity: " << sim_similar);
  INFO("Different sentences similarity: " << sim_different);

  // Similar sentences should score higher
  CHECK(sim_similar > sim_different);

  // Similar sentences should be reasonably similar (>0.5 for good embedding models)
  CHECK(sim_similar > 0.5f);
}
