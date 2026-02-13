#include "llama_stubs.h"
#include <doctest/doctest.h>
#include <lloyal/decode.hpp>
#include <vector>

using namespace lloyal::decode;

/**
 * Tests for seq_id parameter in decode::many()
 *
 * Validates that the seq_id parameter is correctly propagated through
 * to the batch operations and that backward compatibility is maintained.
 */

TEST_CASE("Decoder seq_id: default seq_id is 0") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  llamaStubConfig().decode_result = 0;

  // Call without seq_id parameter (should default to 0)
  CHECK(many(&ctx, tokens, 0, 32) == 0);

  CHECK(llamaStubConfig().last_batch_seq_id == 0);
  CHECK(llamaStubConfig().all_batches_used_seq_id == 0);
}

TEST_CASE("Decoder seq_id: explicit seq_id = 5") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  llamaStubConfig().decode_result = 0;

  // Call with explicit seq_id = 5
  CHECK(many(&ctx, tokens, 0, 32, 5) == 0);

  CHECK(llamaStubConfig().last_batch_seq_id == 5);
  CHECK(llamaStubConfig().all_batches_used_seq_id == 5);
}

TEST_CASE("Decoder seq_id: seq_id preserved across batch chunks") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> long_tokens(100, 42);  // 100 tokens
  llamaStubConfig().decode_result = 0;

  // Decode with seq_id = 7
  CHECK(many(&ctx, long_tokens, 0, 32, 7) == 0);

  // Should have 4 decode calls (100/32 = 4 chunks)
  CHECK(llamaStubConfig().decode_call_count == 4);

  // All chunks should use seq_id = 7
  CHECK(llamaStubConfig().last_batch_seq_id == 7);
  CHECK(llamaStubConfig().all_batches_used_seq_id == 7);
}

TEST_CASE("Decoder seq_id: array overload with seq_id") {
  resetStubConfig();

  llama_context ctx{};
  llama_token tokens[] = {10, 20, 30};
  llamaStubConfig().decode_result = 0;

  // Use array overload with explicit seq_id
  CHECK(many(&ctx, tokens, 3, 0, 32, 3) == 0);

  CHECK(llamaStubConfig().last_batch_seq_id == 3);
}

TEST_CASE("Decoder seq_id: vector overload with seq_id") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {10, 20, 30};
  llamaStubConfig().decode_result = 0;

  // Use vector overload with explicit seq_id
  CHECK(many(&ctx, tokens, 0, 32, 9) == 0);

  CHECK(llamaStubConfig().last_batch_seq_id == 9);
}

TEST_CASE("Decoder seq_id: backward compatibility - old API works") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
  llamaStubConfig().decode_result = 0;

  // Old API call (no seq_id) should still work
  CHECK(many(&ctx, tokens, 0, 32) == 0);

  CHECK(llamaStubConfig().decode_call_count == 1);
  CHECK(llamaStubConfig().last_batch_seq_id == 0);  // Defaults to 0
}

TEST_CASE("Decoder seq_id: seq_id = 0 explicit is same as default") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  llamaStubConfig().decode_result = 0;

  // Explicit seq_id = 0
  CHECK(many(&ctx, tokens, 0, 32, 0) == 0);

  CHECK(llamaStubConfig().last_batch_seq_id == 0);
  CHECK(llamaStubConfig().all_batches_used_seq_id == 0);
}

TEST_CASE("Decoder seq_id: error handling preserves seq_id tracking") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  llamaStubConfig().decode_result = -1;  // Force failure

  // Should return error but seq_id should still be captured
  CHECK(many(&ctx, tokens, 0, 32, 42) != 0);

  // The batch was built before the decode failed
  CHECK(llamaStubConfig().last_batch_seq_id == 42);
}
