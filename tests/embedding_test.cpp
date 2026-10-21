#include "llama_stubs.h"
#include <algorithm>
#include <cmath>
#include <doctest/doctest.h>
#include <lloyal/embedding.hpp>

using namespace lloyal::embedding;

// ===== MODEL CAPABILITY CHECKS =====

TEST_CASE("Embedding: has_embeddings - null model returns false") {
  resetStubConfig();

  CHECK(has_embeddings(nullptr) == false);
}

TEST_CASE("Embedding: has_embeddings - zero dimension returns false") {
  resetStubConfig();
  llamaStubConfig().n_embd = 0;

  llama_model model{};
  CHECK(has_embeddings(&model) == false);
}

TEST_CASE("Embedding: has_embeddings - positive dimension returns true") {
  resetStubConfig();
  llamaStubConfig().n_embd = 768;

  llama_model model{};
  CHECK(has_embeddings(&model) == true);
}

TEST_CASE("Embedding: dimension - null model returns 0") {
  resetStubConfig();

  CHECK(dimension(nullptr) == 0);
}

TEST_CASE("Embedding: dimension - returns configured value") {
  resetStubConfig();
  llamaStubConfig().n_embd = 1024;

  llama_model model{};
  CHECK(dimension(&model) == 1024);
}

// ===== CONTEXT CAPABILITY CHECKS =====

TEST_CASE("Embedding: has_pooling - null context returns false") {
  resetStubConfig();

  CHECK(has_pooling(nullptr) == false);
}

TEST_CASE("Embedding: has_pooling - NONE returns false") {
  resetStubConfig();
  llamaStubConfig().pooling_type = LLAMA_POOLING_TYPE_NONE;

  llama_context ctx{};
  CHECK(has_pooling(&ctx) == false);
}

TEST_CASE("Embedding: has_pooling - MEAN returns true") {
  resetStubConfig();
  llamaStubConfig().pooling_type = LLAMA_POOLING_TYPE_MEAN;

  llama_context ctx{};
  CHECK(has_pooling(&ctx) == true);
}

TEST_CASE("Embedding: pooling_type - null context returns NONE") {
  resetStubConfig();

  CHECK(pooling_type(nullptr) == LLAMA_POOLING_TYPE_NONE);
}

TEST_CASE("Embedding: pooling_type - returns configured value") {
  resetStubConfig();
  llamaStubConfig().pooling_type = LLAMA_POOLING_TYPE_CLS;

  llama_context ctx{};
  CHECK(pooling_type(&ctx) == LLAMA_POOLING_TYPE_CLS);
}

// ===== EMBEDDING EXTRACTION =====

TEST_CASE("Embedding: get - null context throws") {
  resetStubConfig();

  CHECK_THROWS_AS(get(nullptr), std::invalid_argument);
}

TEST_CASE("Embedding: get - unavailable embeddings throws") {
  resetStubConfig();
  llamaStubConfig().n_embd = 4;
  llamaStubConfig().embeddings_available = false;

  llama_context ctx{};
  CHECK_THROWS_AS(get(&ctx), std::runtime_error);
}

TEST_CASE("Embedding: get - returns embeddings without normalization") {
  resetStubConfig();
  llamaStubConfig().n_embd = 4;
  llamaStubConfig().embeddings_available = true;
  llamaStubConfig().embeddings = {1.0f, 2.0f, 3.0f, 4.0f};

  llama_context ctx{};
  auto emb = get(&ctx, Normalize::None);

  REQUIRE(emb.size() == 4);
  CHECK(emb[0] == doctest::Approx(1.0f));
  CHECK(emb[1] == doctest::Approx(2.0f));
  CHECK(emb[2] == doctest::Approx(3.0f));
  CHECK(emb[3] == doctest::Approx(4.0f));
}

TEST_CASE("Embedding: get - returns L2-normalized embeddings") {
  resetStubConfig();
  llamaStubConfig().n_embd = 3;
  llamaStubConfig().embeddings_available = true;
  llamaStubConfig().embeddings = {3.0f, 4.0f, 0.0f}; // norm = 5

  llama_context ctx{};
  auto emb = get(&ctx, Normalize::L2);

  REQUIRE(emb.size() == 3);
  CHECK(emb[0] == doctest::Approx(0.6f));  // 3/5
  CHECK(emb[1] == doctest::Approx(0.8f));  // 4/5
  CHECK(emb[2] == doctest::Approx(0.0f));

  // Verify unit length
  float norm_sq = emb[0] * emb[0] + emb[1] * emb[1] + emb[2] * emb[2];
  CHECK(std::sqrt(norm_sq) == doctest::Approx(1.0f));
}

TEST_CASE("Embedding: get - default normalization is L2") {
  resetStubConfig();
  llamaStubConfig().n_embd = 2;
  llamaStubConfig().embeddings_available = true;
  llamaStubConfig().embeddings = {3.0f, 4.0f}; // norm = 5

  llama_context ctx{};
  auto emb = get(&ctx); // Default normalize = L2

  REQUIRE(emb.size() == 2);
  CHECK(emb[0] == doctest::Approx(0.6f));
  CHECK(emb[1] == doctest::Approx(0.8f));
}

// ===== SEQUENCE EMBEDDING EXTRACTION =====

TEST_CASE("Embedding: get_seq - null context throws") {
  resetStubConfig();

  CHECK_THROWS_AS(get_seq(nullptr, 0), std::invalid_argument);
}

TEST_CASE("Embedding: get_seq - returns embeddings for sequence") {
  resetStubConfig();
  llamaStubConfig().n_embd = 3;
  llamaStubConfig().embeddings_available = true;
  llamaStubConfig().embeddings = {1.0f, 0.0f, 0.0f}; // Unit vector

  llama_context ctx{};
  auto emb = get_seq(&ctx, 0, Normalize::None);

  REQUIRE(emb.size() == 3);
  CHECK(emb[0] == doctest::Approx(1.0f));
}

// ===== PER-TOKEN EMBEDDING EXTRACTION =====

TEST_CASE("Embedding: get_ith - null context throws") {
  resetStubConfig();

  CHECK_THROWS_AS(get_ith(nullptr, 0), std::invalid_argument);
}

TEST_CASE("Embedding: get_ith - unavailable embeddings throws") {
  resetStubConfig();
  llamaStubConfig().n_embd = 4;
  llamaStubConfig().embeddings_available = false;

  llama_context ctx{};
  CHECK_THROWS_AS(get_ith(&ctx, 0), std::runtime_error);
}

TEST_CASE("Embedding: get_ith - returns embeddings for token index") {
  resetStubConfig();
  llamaStubConfig().n_embd = 2;
  llamaStubConfig().embeddings_available = true;
  llamaStubConfig().embeddings = {0.5f, 0.5f};

  llama_context ctx{};
  auto emb = get_ith(&ctx, 5, Normalize::None);

  REQUIRE(emb.size() == 2);
  CHECK(emb[0] == doctest::Approx(0.5f));
  CHECK(emb[1] == doctest::Approx(0.5f));
}

// ===== OUTPUT WIDTH =====
// A model whose output head differs from its hidden width (EmbeddingGemma 2:
// 768 out of a narrower backbone) is read at the width it outputs.

TEST_CASE("Embedding: dimension - is the model's output width") {
  resetStubConfig();
  llamaStubConfig().n_embd = 2;
  llamaStubConfig().n_embd_out = 3;

  llama_model model{};
  CHECK(dimension(&model) == 3);
}

TEST_CASE("Embedding: get, get_seq and get_ith - read the output width") {
  resetStubConfig();
  llamaStubConfig().n_embd = 2;
  llamaStubConfig().n_embd_out = 3;
  llamaStubConfig().pooling_type = LLAMA_POOLING_TYPE_MEAN;
  llamaStubConfig().embeddings_available = true;
  llamaStubConfig().embeddings = {1.0f, 2.0f, 3.0f};

  llama_context ctx{};
  CHECK(get(&ctx, Normalize::None).size() == 3);
  CHECK(get_seq(&ctx, 0, Normalize::None).size() == 3);
  CHECK(get_ith(&ctx, 0, Normalize::None).size() == 3);
}

// ===== COSINE SIMILARITY =====

TEST_CASE("Embedding: cosine_similarity - identical vectors return 1.0") {
  std::vector<float> a = {0.6f, 0.8f}; // Already normalized
  std::vector<float> b = {0.6f, 0.8f};

  float sim = cosine_similarity(a, b);
  CHECK(sim == doctest::Approx(1.0f));
}

TEST_CASE("Embedding: cosine_similarity - orthogonal vectors return 0.0") {
  std::vector<float> a = {1.0f, 0.0f};
  std::vector<float> b = {0.0f, 1.0f};

  float sim = cosine_similarity(a, b);
  CHECK(sim == doctest::Approx(0.0f));
}

TEST_CASE("Embedding: cosine_similarity - opposite vectors return -1.0") {
  std::vector<float> a = {1.0f, 0.0f};
  std::vector<float> b = {-1.0f, 0.0f};

  float sim = cosine_similarity(a, b);
  CHECK(sim == doctest::Approx(-1.0f));
}

TEST_CASE("Embedding: cosine_similarity - dimension mismatch throws") {
  std::vector<float> a = {1.0f, 0.0f};
  std::vector<float> b = {1.0f, 0.0f, 0.0f};

  CHECK_THROWS_AS(cosine_similarity(a, b), std::invalid_argument);
}

TEST_CASE("Embedding: cosine_similarity - empty vectors return 0.0") {
  std::vector<float> a = {};
  std::vector<float> b = {};

  float sim = cosine_similarity(a, b);
  CHECK(sim == doctest::Approx(0.0f));
}

TEST_CASE("Embedding: cosine_similarity - normalized vectors") {
  // cos(60°) = 0.5
  // Vector at 0°: (1, 0)
  // Vector at 60°: (cos(60°), sin(60°)) = (0.5, 0.866)
  std::vector<float> a = {1.0f, 0.0f};
  std::vector<float> b = {0.5f, 0.8660254f};

  float sim = cosine_similarity(a, b);
  CHECK(sim == doctest::Approx(0.5f).epsilon(0.001));
}

// ===== ENCODE ON THE RAIL =====

TEST_CASE("Embedding: encode refuses a context that would read an input in pieces") {
  // A model that keeps memory and attends causally, with pooling NONE, is
  // placed a segment at a time; an embedding reads every entry of one
  // dispatch, so it is refused before anything runs.
  resetStubConfig();
  llamaStubConfig().pooling_type = LLAMA_POOLING_TYPE_NONE;
  llama_context ctx{};
  const std::vector<llama_token> tokens = {1, 2, 3};

  CHECK_THROWS_AS(encode(&ctx, tokens, 512), std::runtime_error);
  CHECK(llamaStubConfig().decode_call_count == 0);
  CHECK(llamaStubConfig().process_call_count == 0);
}

TEST_CASE("Embedding: encode places a text in one dispatch, every output read") {
  resetStubConfig();
  llamaStubConfig().has_memory = false;
  llamaStubConfig().pooling_type = LLAMA_POOLING_TYPE_MEAN;
  llama_context ctx{};
  const std::vector<llama_token> tokens = {1, 2, 3};

  encode(&ctx, tokens, 512);

  CHECK(llamaStubConfig().process_call_count == 1);
  const auto& entries = llamaStubConfig().last_process_entries;
  REQUIRE(entries.size() == 3);
  for (size_t k = 0; k < entries.size(); ++k) {
    CHECK(entries[k].token == tokens[k]);
    CHECK(entries[k].pos == std::vector<llama_pos>{static_cast<llama_pos>(k)});
    CHECK(entries[k].output);
  }
}

TEST_CASE("Embedding: encode reads every token of a context without pooling") {
  // Per-token embeddings from an encoder model: no pooling, no memory, so the
  // stream is placed whole and every entry's output is on, as before the rail.
  resetStubConfig();
  llamaStubConfig().has_memory = false;
  llamaStubConfig().pooling_type = LLAMA_POOLING_TYPE_NONE;
  llama_context ctx{};
  const std::vector<llama_token> tokens = {1, 2, 3};

  encode(&ctx, tokens, 512);

  const auto& entries = llamaStubConfig().last_process_entries;
  REQUIRE(entries.size() == 3);
  CHECK(std::all_of(entries.begin(), entries.end(), [](const StubBatchEntry& e) { return e.output; }));
}
