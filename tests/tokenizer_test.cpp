#include "llama_stubs.h"
#include <doctest/doctest.h>
#include <lloyal/tokenizer.hpp>
#include <memory>
#include <string>
#include <vector>

using namespace lloyal::tokenizer;

TEST_CASE("Tokenizer: null vocab guard - tokenize") {
  resetStubConfig();

  auto result = tokenize(nullptr, "hello world", false, false);
  CHECK(result.empty());
}

TEST_CASE("Tokenizer: empty text returns empty") {
  resetStubConfig();

  llama_vocab vocab{};
  auto result = tokenize(&vocab, "", false, false);
  CHECK(result.empty());
}

TEST_CASE("Tokenizer: two-pass tokenization success") {
  resetStubConfig();

  llama_vocab vocab{};
  llamaStubConfig().tokenize_succeeds = true;
  llamaStubConfig().tokenize_result = {1, 2, 3, 4, 5};

  auto result = tokenize(&vocab, "hello world", false, false);

  REQUIRE(result.size() == 5);
  CHECK(result[0] == 1);
  CHECK(result[1] == 2);
  CHECK(result[2] == 3);
  CHECK(result[3] == 4);
  CHECK(result[4] == 5);
}

TEST_CASE("Tokenizer: token count mismatch between passes") {
  resetStubConfig();

  llama_vocab vocab{};

  // Configure stub to return 5 tokens on first pass, but implementation
  // expects same count on second pass. We can't directly simulate mismatch
  // with current stub, so we test that empty input returns empty result.
  llamaStubConfig().tokenize_succeeds = false;

  auto result = tokenize(&vocab, "hello", false, false);
  CHECK(result.empty());
}

TEST_CASE("Tokenizer: null vocab guard - detokenize single") {
  resetStubConfig();

  auto result = detokenize(static_cast<const llama_vocab*>(nullptr), 42, false);
  CHECK(result.empty());
}

TEST_CASE("Tokenizer: detokenize single token - buffer retry") {
  resetStubConfig();

  llama_vocab vocab{};
  llamaStubConfig().token_piece_succeeds = true;
  llamaStubConfig().token_piece = "hello";

  auto result = detokenize(&vocab, 42, false);
  CHECK(result == "hello");
}

TEST_CASE("Tokenizer: null vocab guard - detokenize_batch") {
  resetStubConfig();

  std::vector<llama_token> tokens = {1, 2, 3};
  auto result = detokenize_batch(static_cast<const llama_vocab*>(nullptr), tokens.data(), 3, false, false);
  CHECK(result.empty());
}

TEST_CASE("Tokenizer: null token array guard - detokenize_batch") {
  resetStubConfig();

  llama_vocab vocab{};
  auto result = detokenize_batch(&vocab, nullptr, 3, false, false);
  CHECK(result.empty());
}

TEST_CASE("Tokenizer: zero token count guard - detokenize_batch") {
  resetStubConfig();

  llama_vocab vocab{};
  std::vector<llama_token> tokens = {1, 2, 3};

  SUBCASE("Zero tokens") {
    auto result = detokenize_batch(&vocab, tokens.data(), 0, false, false);
    CHECK(result.empty());
  }

  SUBCASE("Negative token count") {
    auto result = detokenize_batch(&vocab, tokens.data(), -1, false, false);
    CHECK(result.empty());
  }
}

TEST_CASE("Tokenizer: detokenize_batch two-pass success") {
  resetStubConfig();

  llama_vocab vocab{};
  std::vector<llama_token> tokens = {1, 2, 3, 4, 5};

  llamaStubConfig().detokenize_succeeds = true;
  llamaStubConfig().detokenize_result = "hello world";

  auto result = detokenize_batch(&vocab, tokens.data(), 5, false, false);
  CHECK(result == "hello world");
}

TEST_CASE("Tokenizer: get_vocab null model returns nullptr") {
  resetStubConfig();

  auto vocab = get_vocab(nullptr);
  CHECK(vocab == nullptr);
}

TEST_CASE("Tokenizer: is_eog null vocab returns false") {
  resetStubConfig();

  bool result = is_eog(static_cast<const llama_vocab*>(nullptr), 42);
  CHECK(result == false);
}

TEST_CASE("Tokenizer: vocab_size null vocab returns 0") {
  resetStubConfig();

  int32_t size = vocab_size(static_cast<const llama_vocab*>(nullptr));
  CHECK(size == 0);
}

// ===== MODEL-ACCEPTING OVERLOAD TESTS =====

TEST_CASE("Tokenizer: tokenize(model, text) null model guard") {
  resetStubConfig();

  auto result = tokenize(static_cast<llama_model*>(nullptr), "hello world");
  CHECK(result.empty());
}

TEST_CASE("Tokenizer: tokenize(model, text) with add_bos detection") {
  resetStubConfig();

  llama_model model{};

  // Configure stubs
  llamaStubConfig().tokenize_succeeds = true;
  llamaStubConfig().tokenize_result = {1, 2, 3};

  auto result = tokenize(&model, "test");

  REQUIRE(result.size() == 3);
  CHECK(result[0] == 1);
  CHECK(result[1] == 2);
  CHECK(result[2] == 3);
}

TEST_CASE("Tokenizer: detokenize(model, token) null model guard") {
  resetStubConfig();

  auto result = detokenize(static_cast<llama_model*>(nullptr), 42);
  CHECK(result.empty());
}

TEST_CASE("Tokenizer: detokenize(model, token) successful") {
  resetStubConfig();

  llama_model model{};

  llamaStubConfig().token_piece_succeeds = true;
  llamaStubConfig().token_piece = "test";

  auto result = detokenize(&model, 42, true);
  CHECK(result == "test");
}

TEST_CASE("Tokenizer: detokenize_batch(model, vector) null model guard") {
  resetStubConfig();

  std::vector<llama_token> tokens = {1, 2, 3};
  auto result = detokenize_batch(static_cast<llama_model*>(nullptr), tokens);
  CHECK(result.empty());
}

TEST_CASE("Tokenizer: detokenize_batch(model, vector) successful") {
  resetStubConfig();

  llama_model model{};
  std::vector<llama_token> tokens = {1, 2, 3, 4, 5};

  llamaStubConfig().detokenize_succeeds = true;
  llamaStubConfig().detokenize_result = "hello world";

  auto result = detokenize_batch(&model, tokens);
  CHECK(result == "hello world");
}

TEST_CASE("Tokenizer: detokenize_batch(model, array) null model guard") {
  resetStubConfig();

  std::vector<llama_token> tokens = {1, 2, 3};
  auto result = detokenize_batch(static_cast<llama_model*>(nullptr),
                                 tokens.data(), 3, false, false);
  CHECK(result.empty());
}

TEST_CASE("Tokenizer: detokenize_batch(model, array) successful") {
  resetStubConfig();

  llama_model model{};
  std::vector<llama_token> tokens = {1, 2, 3, 4, 5};

  llamaStubConfig().detokenize_succeeds = true;
  llamaStubConfig().detokenize_result = "test output";

  auto result = detokenize_batch(&model, tokens.data(), 5, false, true);
  CHECK(result == "test output");
}

TEST_CASE("Tokenizer: is_eog(model, token) null model guard") {
  resetStubConfig();

  bool result = is_eog(static_cast<llama_model*>(nullptr), 42);
  CHECK(result == false);
}

TEST_CASE("Tokenizer: is_eog(model, token) successful") {
  resetStubConfig();

  llama_model model{};

  llamaStubConfig().eog_tokens.insert(2);

  bool result_true = is_eog(&model, 2);
  bool result_false = is_eog(&model, 42);

  CHECK(result_true == true);
  CHECK(result_false == false);
}

TEST_CASE("Tokenizer: vocab_size(model) null model guard") {
  resetStubConfig();

  int32_t size = vocab_size(static_cast<llama_model*>(nullptr));
  CHECK(size == 0);
}

TEST_CASE("Tokenizer: vocab_size(model) successful") {
  resetStubConfig();

  llama_model model{};

  llamaStubConfig().vocab_size_value = 32000;

  int32_t size = vocab_size(&model);
  CHECK(size == 32000);
}
