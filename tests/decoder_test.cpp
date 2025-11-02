#include "llama_stubs.h"
#include <doctest/doctest.h>
#include <lloyal/decoder.hpp>
#include <memory>
#include <vector>

using namespace lloyal::decoder;

TEST_CASE("Decoder: null context guard") {
  resetStubConfig();

  std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
  CHECK_THROWS(decode_tokens(nullptr, tokens.data(), 5, 0, 32));
}

TEST_CASE("Decoder: null token array guard") {
  resetStubConfig();

  llama_context ctx{};
  CHECK_THROWS(decode_tokens(&ctx, nullptr, 5, 0, 32));
}

TEST_CASE("Decoder: zero token count guard") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  CHECK_THROWS(decode_tokens(&ctx, tokens.data(), 0, 0, 32));
}

TEST_CASE("Decoder: negative token count guard") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  CHECK_THROWS(decode_tokens(&ctx, tokens.data(), -1, 0, 32));
}

TEST_CASE("Decoder: single batch processing") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
  llamaStubConfig().decode_result = 0; // Success

  // Should not throw
  decode_tokens(&ctx, tokens, 0, 32);

  // Verify: llama_decode called once (n_tokens=5 <= n_batch=32)
  CHECK(llamaStubConfig().decode_call_count == 1);
}

TEST_CASE("Decoder: multi-batch chunking") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens(100, 42); // 100 tokens, all with value 42
  llamaStubConfig().decode_result = 0;      // Success

  // Should not throw
  decode_tokens(&ctx, tokens, 0, 32);

  // Verify: llama_decode called 4 times (100/32 = 3.125 → ceil = 4 chunks)
  // Chunks: 32 + 32 + 32 + 4 = 100
  CHECK(llamaStubConfig().decode_call_count == 4);
}

TEST_CASE("Decoder: llama_decode failure propagates") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  llamaStubConfig().decode_result = -1; // Failure

  // Should throw on decode failure
  CHECK_THROWS(decode_tokens(&ctx, tokens, 0, 32));
}

TEST_CASE("Decoder: vector overload delegates to array version") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  llamaStubConfig().decode_result = 0; // Success

  // Vector overload should work
  decode_tokens(&ctx, tokens, 0, 32);

  // Should have called decode once
  CHECK(llamaStubConfig().decode_call_count == 1);
}

TEST_CASE("Decoder: BatchGuard RAII cleanup on exception") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  llamaStubConfig().decode_result = -1; // Force failure

  int initial_free_count = llamaStubConfig().batch_free_call_count;

  try {
    decode_tokens(&ctx, tokens, 0, 32);
  } catch (...) {
    // Expected to throw
  }

  // Verify: batch_free was called (RAII cleanup)
  CHECK(llamaStubConfig().batch_free_call_count == initial_free_count + 1);
}
