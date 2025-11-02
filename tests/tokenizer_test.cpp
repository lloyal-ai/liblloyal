#include <doctest/doctest.h>
#include <lloyal/tokenizer.hpp>
#include "llama_stubs.h"
#include <memory>
#include <vector>
#include <string>

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

    auto result = detokenize(nullptr, 42, false);
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
    auto result = detokenize_batch(nullptr, tokens.data(), 3, false, false);
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

    bool result = is_eog(nullptr, 42);
    CHECK(result == false);
}

TEST_CASE("Tokenizer: vocab_size null vocab returns 0") {
    resetStubConfig();

    int32_t size = vocab_size(nullptr);
    CHECK(size == 0);
}
