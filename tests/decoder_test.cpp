#include "llama_stubs.h"
#include <doctest/doctest.h>
#include <lloyal/decode.hpp>
#include <memory>
#include <tuple>
#include <vector>

using namespace lloyal::decode;

TEST_CASE("Decoder: null context guard") {
  resetStubConfig();

  std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
  CHECK_THROWS(many(nullptr, tokens.data(), 5, 0, 32));
}

TEST_CASE("Decoder: null token array guard") {
  resetStubConfig();

  llama_context ctx{};
  CHECK_THROWS(many(&ctx, nullptr, 5, 0, 32));
}

TEST_CASE("Decoder: zero token count guard") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  CHECK_THROWS(many(&ctx, tokens.data(), 0, 0, 32));
}

TEST_CASE("Decoder: negative token count guard") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  CHECK_THROWS(many(&ctx, tokens.data(), -1, 0, 32));
}

TEST_CASE("Decoder: single batch processing") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
  llamaStubConfig().decode_result = 0; // Success

  // Should succeed
  CHECK(many(&ctx, tokens, 0, 32) == 0);

  // Verify: llama_decode called once (n_tokens=5 <= n_batch=32)
  CHECK(llamaStubConfig().decode_call_count == 1);
}

TEST_CASE("Decoder: multi-batch chunking") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens(100, 42); // 100 tokens, all with value 42
  llamaStubConfig().decode_result = 0;      // Success

  // Should succeed
  CHECK(many(&ctx, tokens, 0, 32) == 0);

  // Verify: llama_decode called 4 times (100/32 = 3.125 → ceil = 4 chunks)
  // Chunks: 32 + 32 + 32 + 4 = 100
  CHECK(llamaStubConfig().decode_call_count == 4);
}

TEST_CASE("Decoder: llama_decode failure propagates") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  llamaStubConfig().decode_result = -1; // Failure

  // Should return non-zero on decode failure
  CHECK(many(&ctx, tokens, 0, 32) != 0);
}

TEST_CASE("Decoder: vector overload delegates to array version") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  llamaStubConfig().decode_result = 0; // Success

  // Vector overload should work
  CHECK(many(&ctx, tokens, 0, 32) == 0);

  // Should have called decode once
  CHECK(llamaStubConfig().decode_call_count == 1);
}

TEST_CASE("Decoder: many() returns error on decode failure") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  llamaStubConfig().decode_result = -1; // Force failure

  // Returns error code, thread_local batch stays alive for reuse
  CHECK(many(&ctx, tokens, 0, 32) != 0);

  // Subsequent call with success should work (batch is reusable)
  llamaStubConfig().decode_result = 0;
  CHECK(many(&ctx, tokens, 0, 32) == 0);
}

// ============================================================================
// n_batch validation (Fix 3)
// ============================================================================

TEST_CASE("Decoder: n_batch = 0 throws") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  CHECK_THROWS(many(&ctx, tokens.data(), 3, 0, 0));
}

TEST_CASE("Decoder: n_batch = -1 throws") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  CHECK_THROWS(many(&ctx, tokens.data(), 3, 0, -1));
}

// ============================================================================
// Error code propagation (Fix 2)
// ============================================================================

TEST_CASE("Decoder: many returns actual llama_decode error code") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};

  // Set a specific non-zero error code
  llamaStubConfig().decode_result = 2;

  int rc = many(&ctx, tokens, 0, 32);
  CHECK(rc == 2);
}

TEST_CASE("Decoder: many returns -1 error code from llama_decode") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};

  llamaStubConfig().decode_result = -1;

  int rc = many(&ctx, tokens, 0, 32);
  CHECK(rc == -1);
}

// ── The context's batch is a hard limit ─────────────────────────────────────
// llama_decode asserts n_tokens <= n_batch (an abort, not an rc). Chunk sizes
// come from the caller, so each primitive must refuse an oversize batch
// before it is built. The stub's context takes 8 here.

TEST_CASE("Decoder: many refuses a chunk larger than the context's batch") {
  resetStubConfig();
  llamaStubConfig().n_batch = 8;
  auto* ctx = reinterpret_cast<llama_context*>(0x1000);
  std::vector<llama_token> tokens(16, 1);
  CHECK_THROWS_WITH(std::ignore = many(ctx, tokens.data(), 16, 0, /*n_batch*/ 16),
                    doctest::Contains("exceeds the context's n_batch"));
  CHECK(llamaStubConfig().decode_call_count == 0);
  // The same tokens chunked at the context's own size go through.
  CHECK(many(ctx, tokens.data(), 16, 0, /*n_batch*/ 8) == 0);
}

TEST_CASE("Decoder: each refuses more items than the context's batch") {
  resetStubConfig();
  llamaStubConfig().n_batch = 8;
  auto* ctx = reinterpret_cast<llama_context*>(0x1000);
  std::vector<EachItem> items(9);
  for (int32_t i = 0; i < 9; ++i) items[i] = EachItem{1, 0, i, false};
  Scratch scratch;
  CHECK_THROWS_WITH(std::ignore = each(ctx, items, scratch),
                    doctest::Contains("exceeds the context's n_batch"));
  CHECK(llamaStubConfig().decode_call_count == 0);
}

TEST_CASE("Decoder: scatter refuses a total larger than the context's batch") {
  resetStubConfig();
  llamaStubConfig().n_batch = 8;
  auto* ctx = reinterpret_cast<llama_context*>(0x1000);
  std::vector<llama_token> a(5, 1), b(4, 2);
  ScatterItem items[] = {{a, 0, 0, true}, {b, 0, 1, true}};
  Scratch scratch;
  CHECK_THROWS_WITH(std::ignore = scatter(ctx, items, 2, scratch),
                    doctest::Contains("exceeds the context's n_batch"));
  CHECK(llamaStubConfig().decode_call_count == 0);
}

TEST_CASE("Decoder: embd refuses a row chunk larger than the context's batch") {
  resetStubConfig();
  llamaStubConfig().n_batch = 8;
  auto* ctx = reinterpret_cast<llama_context*>(0x1000);
  std::vector<float> rows(16 * 2, 0.5f);
  std::vector<llama_pos> pos(16, 0);
  EmbdItem item;
  item.rows = rows.data(); item.n_rows = 16; item.n_embd_inp = 2; item.pos = pos.data();
  Scratch scratch;
  CHECK_THROWS_WITH(std::ignore = embd(ctx, item, /*n_batch*/ 16, scratch),
                    doctest::Contains("exceeds the context's n_batch"));
  CHECK(llamaStubConfig().decode_call_count == 0);
}
