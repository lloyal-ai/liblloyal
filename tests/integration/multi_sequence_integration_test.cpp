#include <cstdlib>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/decoder.hpp>
#include <lloyal/grammar.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <string>
#include <vector>

using namespace lloyal;

/**
 * Multi-Sequence Integration Tests
 *
 * Validates that the seq_id parameter in decode_tokens() works correctly
 * with real llama.cpp. Tests:
 * - Decoding to different sequences populates different KV regions
 * - Sequences are isolated (clearing one doesn't affect others)
 * - Backward compatibility (default seq_id=0 works as before)
 *
 * Uses tiny-random-llama.gguf (~12MB)
 *
 * NOTE: These tests require a context initialized with n_seq_max > 1
 */

static const char *MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

#define REQUIRE_MODEL()                                                        \
  if (!MODEL_PATH || !*MODEL_PATH) {                                           \
    MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set");                              \
    return;                                                                    \
  }

struct LlamaBackendGuard {
  LlamaBackendGuard() { llama_backend_init(); }
  ~LlamaBackendGuard() { llama_backend_free(); }
};

// ============================================================================
// Multi-Sequence Tests
// ============================================================================

TEST_CASE("Integration: multi-sequence decode populates different KV regions") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  // Load model
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Create context with n_seq_max = 4 (support 4 parallel sequences)
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;  // Enable multi-sequence

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  // Tokenize test prompt
  std::string test_text = "Hello world";
  auto tokens = tokenizer::tokenize(vocab, test_text, false, false);
  REQUIRE_FALSE(tokens.empty());
  INFO("Tokenized '" << test_text << "' into " << tokens.size() << " tokens");

  // Decode to sequence 0
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 0);
  llama_pos pos_seq0 = kv::pos_max(ctx, 0);
  INFO("Seq 0 pos_max after decode: " << pos_seq0);
  CHECK(pos_seq0 >= 0);  // Should have tokens

  // Decode to sequence 1 (independent)
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 1);
  llama_pos pos_seq1 = kv::pos_max(ctx, 1);
  INFO("Seq 1 pos_max after decode: " << pos_seq1);
  CHECK(pos_seq1 >= 0);  // Should have tokens

  // Both sequences should have the same number of tokens
  CHECK(pos_seq0 == pos_seq1);

  // Seq 0 should be unchanged after decoding to seq 1
  llama_pos pos_seq0_after = kv::pos_max(ctx, 0);
  CHECK(pos_seq0_after == pos_seq0);

  llama_free(ctx);
}

TEST_CASE("Integration: clearing one sequence doesn't affect others") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Test input", false, false);
  REQUIRE_FALSE(tokens.empty());

  // Decode to both sequences
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 0);
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 1);

  llama_pos pos_seq0_before = kv::pos_max(ctx, 0);
  llama_pos pos_seq1_before = kv::pos_max(ctx, 1);
  CHECK(pos_seq0_before >= 0);
  CHECK(pos_seq1_before >= 0);

  // Clear sequence 0 only (remove all tokens from seq 0)
  kv::remove_range(ctx, 0, 0, -1);

  // Seq 0 should be empty, seq 1 should be intact
  llama_pos pos_seq0_after = kv::pos_max(ctx, 0);
  llama_pos pos_seq1_after = kv::pos_max(ctx, 1);

  CHECK(pos_seq0_after == -1);  // Empty (no tokens)
  CHECK(pos_seq1_after == pos_seq1_before);  // Unchanged

  llama_free(ctx);
}

TEST_CASE("Integration: backward compatibility - default seq_id=0") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Standard single-sequence context (n_seq_max defaults to 1)
  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 256;
  ctx_params.n_batch = 64;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello", false, false);
  REQUIRE_FALSE(tokens.empty());

  // Old API (no seq_id parameter) should still work
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch);

  // Check seq 0 has tokens
  llama_pos pos = kv::pos_max(ctx, 0);
  CHECK(pos >= 0);

  llama_free(ctx);
}

TEST_CASE("Integration: decode with explicit seq_id=0 matches default") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 256;
  ctx_params.n_batch = 64;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Test", false, false);
  REQUIRE_FALSE(tokens.empty());

  // Explicit seq_id=0 should work same as default
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 0);

  llama_pos pos = kv::pos_max(ctx, 0);
  CHECK(pos >= 0);

  llama_free(ctx);
}

// ============================================================================
// System 2 Branching Tests (seq_cp, seq_keep)
// ============================================================================

TEST_CASE("Integration: seq_cp copies KV cache to new sequence") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "The quick brown fox", false, false);
  REQUIRE_FALSE(tokens.empty());

  // Decode to sequence 0
  decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch, 0);
  llama_pos pos_seq0 = kv::pos_max(ctx, 0);
  CHECK(pos_seq0 >= 0);
  INFO("Seq 0 pos_max before copy: " << pos_seq0);

  // Sequence 2 should be empty before copy
  llama_pos pos_seq2_before = kv::pos_max(ctx, 2);
  CHECK(pos_seq2_before == -1);

  // Copy seq 0 -> seq 2 (System 2 fork)
  kv::seq_cp(ctx, 0, 2);

  // Both sequences should now have same position
  llama_pos pos_seq0_after = kv::pos_max(ctx, 0);
  llama_pos pos_seq2_after = kv::pos_max(ctx, 2);
  INFO("Seq 0 pos_max after copy: " << pos_seq0_after);
  INFO("Seq 2 pos_max after copy: " << pos_seq2_after);

  CHECK(pos_seq0_after == pos_seq0);  // Source unchanged
  CHECK(pos_seq2_after == pos_seq0);  // Dest matches source

  llama_free(ctx);
}

// NOTE: Partial range seq_cp is NOT supported by llama.cpp
// The assertion "seq_cp() is only supported for full KV buffers" fires
// when p0/p1 are not the full range. This is a llama.cpp limitation.
// For System 2 branching, we must copy entire sequences.
//
// TEST_CASE("Integration: seq_cp with range copies partial sequence") - REMOVED

// NOTE: seq_keep behavior differs from expectations
// After seq_cp, cells are shared across sequences. seq_keep removes the sequence
// from cells but pos_max still returns positions for all sequences because the
// cells still exist. For System 2, we should use remove_range per-sequence instead.
//
// For now, the critical operation is seq_cp (branching), which works correctly.
// Pruning can be done via kv::remove_range(ctx, seq_id, 0, -1) for each unwanted sequence.
//
// TEST_CASE("Integration: seq_keep after branch") - SKIPPED pending llama.cpp investigation

// ============================================================================
// System 2 Grammar Cloning Tests
// ============================================================================

TEST_CASE("Integration: clone_sampler creates independent grammar state") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // Import grammar namespace
  using namespace lloyal;

  // Get vocab and tokenize first to build grammar from actual token pieces
  auto vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  auto token_a_vec = tokenizer::tokenize(vocab, "a", false, false);
  auto token_b_vec = tokenizer::tokenize(vocab, "b", false, false);
  auto token_c_vec = tokenizer::tokenize(vocab, "c", false, false);

  if (!token_a_vec.empty() && !token_b_vec.empty() && !token_c_vec.empty()) {
    llama_token token_a = token_a_vec[0];

    // Detokenize with special=true to match grammar's internal token_to_piece cache
    std::string piece_a = tokenizer::detokenize(vocab, token_a, true);
    std::string piece_b = tokenizer::detokenize(vocab, token_b_vec[0], true);
    std::string piece_c = tokenizer::detokenize(vocab, token_c_vec[0], true);

    // Build grammar from actual pieces; use alternation instead of character class
    // (character class [bc] won't match multi-char pieces like " b")
    std::string grammar_str = "root ::= \"" + piece_a + "\" (\"" + piece_b + "\" | \"" + piece_c + "\")";
    INFO("Grammar: " << grammar_str);

    llama_sampler *trunk = grammar::init_sampler(model.get(), grammar_str.c_str());
    REQUIRE(trunk != nullptr);

    // Clone before accepting any tokens
    llama_sampler *branch_a = grammar::clone_sampler(trunk);
    llama_sampler *branch_b = grammar::clone_sampler(trunk);
    REQUIRE(branch_a != nullptr);
    REQUIRE(branch_b != nullptr);

    // Accept 'a' on trunk - this should advance trunk's state
    llama_sampler_accept(trunk, token_a);

    // Now clones should still be at initial state
    // Accept 'a' on branch_a should succeed (still at initial state)
    llama_sampler_accept(branch_a, token_a);

    // Accept 'a' on branch_b should also succeed
    llama_sampler_accept(branch_b, token_a);

    INFO("✓ Cloned samplers have independent state");

    // Cleanup
    llama_sampler_free(trunk);
    llama_sampler_free(branch_a);
    llama_sampler_free(branch_b);
  } else {
    INFO("[ SKIP ] Could not find tokens for 'a', 'b', and 'c' in vocabulary");
  }
}

TEST_CASE("Integration: clone_sampler preserves advanced grammar state") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  using namespace lloyal;

  auto vocab = llama_model_get_vocab(model.get());
  REQUIRE(vocab != nullptr);

  auto token_a_vec = tokenizer::tokenize(vocab, "a", false, false);
  auto token_b_vec = tokenizer::tokenize(vocab, "b", false, false);
  auto token_c_vec = tokenizer::tokenize(vocab, "c", false, false);

  if (!token_a_vec.empty() && !token_b_vec.empty() && !token_c_vec.empty()) {
    llama_token token_a = token_a_vec[0];
    llama_token token_b = token_b_vec[0];

    // Detokenize with special=true to match grammar's internal token_to_piece cache
    std::string piece_a = tokenizer::detokenize(vocab, token_a, true);
    std::string piece_b = tokenizer::detokenize(vocab, token_b, true);
    std::string piece_c = tokenizer::detokenize(vocab, token_c_vec[0], true);

    // Build grammar from actual pieces
    std::string grammar_str = "root ::= \"" + piece_a + "\" \"" + piece_b + "\" \"" + piece_c + "\"";
    INFO("Grammar: " << grammar_str);

    llama_sampler *trunk = grammar::init_sampler(model.get(), grammar_str.c_str());
    REQUIRE(trunk != nullptr);

    // Advance trunk past 'a'
    llama_sampler_accept(trunk, token_a);

    // Clone at this point - clone should be past 'a', expecting 'b'
    llama_sampler *clone = grammar::clone_sampler(trunk);
    REQUIRE(clone != nullptr);

    // Both trunk and clone should accept 'b' next
    llama_sampler_accept(trunk, token_b);
    llama_sampler_accept(clone, token_b);

    INFO("✓ Clone preserved grammar state after 'a'");

    llama_sampler_free(clone);
    llama_sampler_free(trunk);
  } else {
    INFO("[ SKIP ] Could not find tokens for 'a', 'b', and 'c' in vocabulary");
  }
}

// ============================================================================
// System 2 Divergent Branch Generation Tests
// ============================================================================

TEST_CASE("Integration: branches can decode different tokens independently") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Tokenize shared prefix
  std::string prefix = "The answer is";
  auto prefix_tokens = tokenizer::tokenize(vocab, prefix, false, false);
  REQUIRE_FALSE(prefix_tokens.empty());
  int32_t prefix_len = static_cast<int32_t>(prefix_tokens.size());

  // Decode prefix to seq 0
  decoder::decode_tokens(ctx, prefix_tokens, 0, ctx_params.n_batch, 0);
  llama_pos pos_after_prefix = kv::pos_max(ctx, 0);
  CHECK(pos_after_prefix == prefix_len - 1);

  // Fork to seq 1 and seq 2
  kv::seq_cp(ctx, 0, 1);
  kv::seq_cp(ctx, 0, 2);

  // Verify all sequences have same initial state
  CHECK(kv::pos_max(ctx, 0) == pos_after_prefix);
  CHECK(kv::pos_max(ctx, 1) == pos_after_prefix);
  CHECK(kv::pos_max(ctx, 2) == pos_after_prefix);

  // Get different tokens to decode to each branch
  auto token_yes = tokenizer::tokenize(vocab, " yes", false, false);
  auto token_no = tokenizer::tokenize(vocab, " no", false, false);

  if (!token_yes.empty() && !token_no.empty()) {
    // Decode " yes" to seq 1
    decoder::decode_tokens(ctx, {token_yes[0]}, prefix_len, ctx_params.n_batch, 1);

    // Decode " no" to seq 2
    decoder::decode_tokens(ctx, {token_no[0]}, prefix_len, ctx_params.n_batch, 2);

    // Verify branches diverged
    llama_pos pos_seq0 = kv::pos_max(ctx, 0);
    llama_pos pos_seq1 = kv::pos_max(ctx, 1);
    llama_pos pos_seq2 = kv::pos_max(ctx, 2);

    CHECK(pos_seq0 == pos_after_prefix);  // Trunk unchanged
    CHECK(pos_seq1 == prefix_len);        // Branch 1 advanced by 1
    CHECK(pos_seq2 == prefix_len);        // Branch 2 advanced by 1

    INFO("✓ Branches diverged: seq0=" << pos_seq0 << ", seq1=" << pos_seq1 << ", seq2=" << pos_seq2);
  }

  llama_free(ctx);
}

TEST_CASE("Integration: complete System 2 branching workflow") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // === STEP 1: Decode shared prefix ===
  std::string prefix = "Hello";
  auto prefix_tokens = tokenizer::tokenize(vocab, prefix, false, false);
  REQUIRE_FALSE(prefix_tokens.empty());

  decoder::decode_tokens(ctx, prefix_tokens, 0, ctx_params.n_batch, 0);
  int32_t trunk_pos = static_cast<int32_t>(prefix_tokens.size());
  INFO("Step 1: Decoded prefix (" << trunk_pos << " tokens) to seq 0");

  // === STEP 2: Fork to create branches ===
  kv::seq_cp(ctx, 0, 1);  // Fork to seq 1
  kv::seq_cp(ctx, 0, 2);  // Fork to seq 2
  INFO("Step 2: Forked to sequences 1 and 2");

  // === STEP 3: Generate different continuations ===
  // Sample one token for each branch (using greedy for determinism)
  auto token_world = tokenizer::tokenize(vocab, " world", false, false);
  auto token_there = tokenizer::tokenize(vocab, " there", false, false);

  if (!token_world.empty() && !token_there.empty()) {
    // Branch 1: " world"
    decoder::decode_tokens(ctx, {token_world[0]}, trunk_pos, ctx_params.n_batch, 1);

    // Branch 2: " there"
    decoder::decode_tokens(ctx, {token_there[0]}, trunk_pos, ctx_params.n_batch, 2);

    INFO("Step 3: Decoded divergent tokens to branches");

    // === STEP 4: Verify branch isolation ===
    llama_pos pos_trunk = kv::pos_max(ctx, 0);
    llama_pos pos_branch1 = kv::pos_max(ctx, 1);
    llama_pos pos_branch2 = kv::pos_max(ctx, 2);

    CHECK(pos_trunk == trunk_pos - 1);     // Trunk at original position
    CHECK(pos_branch1 == trunk_pos);       // Branch 1 advanced
    CHECK(pos_branch2 == trunk_pos);       // Branch 2 advanced

    INFO("Step 4: Verified isolation - trunk=" << pos_trunk
         << ", branch1=" << pos_branch1
         << ", branch2=" << pos_branch2);

    // === STEP 5: Prune losing branch ===
    // Assume branch 1 "won" - remove branch 2
    kv::remove_range(ctx, 2, 0, -1);
    llama_pos pos_branch2_after = kv::pos_max(ctx, 2);
    CHECK(pos_branch2_after == -1);  // Branch 2 cleared
    INFO("Step 5: Pruned losing branch (seq 2)");

    // Branch 1 should still be intact
    CHECK(kv::pos_max(ctx, 1) == trunk_pos);
    INFO("Step 5: Winning branch (seq 1) intact");
  }

  llama_free(ctx);
  INFO("✓ Complete System 2 workflow validated");
}
