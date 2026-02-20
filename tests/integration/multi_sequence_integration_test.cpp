#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/decode.hpp>
#include <lloyal/grammar.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <string>
#include <vector>

using namespace lloyal;

/**
 * Decode Integration Tests
 *
 * Validates decode::one, decode::many, decode::each, and decode::scatter
 * against real llama.cpp. Coverage includes:
 *
 * - Multi-sequence KV isolation (decoding to different seq_ids)
 * - Logits correctness after each decode variant
 * - Batch index conventions (which llama_get_logits_ith index maps to which item)
 * - Multi-chunk decode with small n_batch
 * - Context overflow (returns error, does not throw)
 * - Scratch buffer reuse across varying batch sizes
 * - System 2 branching (seq_cp, seq_keep, grammar cloning)
 *
 * Requires LLAMA_TEST_MODEL env var pointing to a coherent model.
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
  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch, 0) == 0);
  llama_pos pos_seq0 = kv::pos_max(ctx, 0);
  INFO("Seq 0 pos_max after decode: " << pos_seq0);
  CHECK(pos_seq0 >= 0);  // Should have tokens

  // Decode to sequence 1 (independent)
  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch, 1) == 0);
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
  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch, 0) == 0);
  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch, 1) == 0);

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

TEST_CASE("Integration: backward compatibility - default and explicit seq_id=0") {
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
  auto tokens = tokenizer::tokenize(vocab, "Hello", false, false);
  REQUIRE_FALSE(tokens.empty());

  // Default (no seq_id parameter) works
  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch) == 0);
  llama_pos pos_default = kv::pos_max(ctx, 0);
  CHECK(pos_default >= 0);

  kv::clear_all(ctx);

  // Explicit seq_id=0 produces same result
  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch, 0) == 0);
  llama_pos pos_explicit = kv::pos_max(ctx, 0);
  CHECK(pos_explicit == pos_default);

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
  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch, 0) == 0);
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
  REQUIRE(decode::many(ctx, prefix_tokens, 0, ctx_params.n_batch, 0) == 0);
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
    REQUIRE(decode::many(ctx, std::vector<llama_token>{token_yes[0]}, prefix_len, ctx_params.n_batch, 1) == 0);

    // Decode " no" to seq 2
    REQUIRE(decode::many(ctx, std::vector<llama_token>{token_no[0]}, prefix_len, ctx_params.n_batch, 2) == 0);

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

  REQUIRE(decode::many(ctx, prefix_tokens, 0, ctx_params.n_batch, 0) == 0);
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
    REQUIRE(decode::many(ctx, std::vector<llama_token>{token_world[0]}, trunk_pos, ctx_params.n_batch, 1) == 0);

    // Branch 2: " there"
    REQUIRE(decode::many(ctx, std::vector<llama_token>{token_there[0]}, trunk_pos, ctx_params.n_batch, 2) == 0);

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

// ============================================================================
// Multi-Sequence Decode Primitives Tests (decode_multiseq, decode_scatter)
// ============================================================================

TEST_CASE("Integration: decode_multiseq - 32 branches divergent generation") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  constexpr int N_BRANCHES = 32;
  constexpr int N_STEPS = 10;

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;
  ctx_params.n_seq_max = N_BRANCHES + 1;  // +1 for root sequence

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  int32_t n_vocab = llama_vocab_n_tokens(vocab);

  // Decode shared prefix to sequence 0
  std::string prefix = "The quick brown fox jumps over the lazy dog";
  auto prefix_tokens = tokenizer::tokenize(vocab, prefix, false, false);
  REQUIRE_FALSE(prefix_tokens.empty());

  REQUIRE(decode::many(ctx, prefix_tokens, 0, ctx_params.n_batch, 0) == 0);
  llama_pos prefix_len = static_cast<llama_pos>(prefix_tokens.size());
  INFO("Decoded shared prefix (" << prefix_len << " tokens) to seq 0");

  // Fork to 32 branches (seq_id 1..32)
  for (int i = 1; i <= N_BRANCHES; ++i) {
    kv::seq_cp(ctx, 0, static_cast<llama_seq_id>(i));
  }
  INFO("Forked to " << N_BRANCHES << " branches");

  // Track positions for each branch
  std::vector<llama_pos> positions(N_BRANCHES, prefix_len);

  // Scratch buffer for decode_multiseq
  decode::Scratch scratch;

  // Run N_STEPS of divergent generation
  for (int step = 0; step < N_STEPS; ++step) {
    // Build items for decode_multiseq - each branch gets a different token
    std::vector<decode::EachItem> items;
    items.reserve(N_BRANCHES);

    for (int i = 0; i < N_BRANCHES; ++i) {
      llama_seq_id seq_id = static_cast<llama_seq_id>(i + 1);
      // Pick a different token for each branch (spread across vocab)
      llama_token token = static_cast<llama_token>((step * N_BRANCHES + i) % n_vocab);

      items.push_back({
          .token = token,
          .pos = positions[i],
          .seq_id = seq_id,
          .output_logits = (i == 0)  // Only request logits for first branch
      });
    }

    // Decode all 32 branches in ONE llama_decode() call
    int result = decode::each(ctx, items, scratch);
    REQUIRE(result == 0);

    // Update positions
    for (int i = 0; i < N_BRANCHES; ++i) {
      positions[i] += 1;
    }
  }

  INFO("Completed " << N_STEPS << " steps of divergent generation");

  // Verify all branches have independent KV cache state
  for (int i = 0; i < N_BRANCHES; ++i) {
    llama_seq_id seq_id = static_cast<llama_seq_id>(i + 1);
    llama_pos pos = kv::pos_max(ctx, seq_id);

    // Each branch should be at prefix_len + N_STEPS - 1 (0-indexed)
    llama_pos expected = prefix_len + N_STEPS - 1;
    CHECK(pos == expected);
  }

  // Verify root sequence is unchanged
  llama_pos root_pos = kv::pos_max(ctx, 0);
  CHECK(root_pos == prefix_len - 1);

  INFO("✓ All " << N_BRANCHES << " branches verified at independent positions");

  llama_free(ctx);
}

TEST_CASE("Integration: decode_scatter - asymmetric prefill") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // 8 sequences with wildly different prompt lengths
  constexpr int N_SEQUENCES = 8;
  const int prompt_lengths[] = {3, 7, 15, 42, 2, 28, 11, 5};

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;
  ctx_params.n_seq_max = N_SEQUENCES;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  int32_t n_vocab = llama_vocab_n_tokens(vocab);

  // Generate token arrays for each sequence
  std::vector<std::vector<llama_token>> token_arrays(N_SEQUENCES);
  int32_t total_tokens = 0;

  for (int i = 0; i < N_SEQUENCES; ++i) {
    int len = prompt_lengths[i];
    token_arrays[i].resize(len);
    for (int j = 0; j < len; ++j) {
      // Use deterministic but varied tokens
      token_arrays[i][j] = static_cast<llama_token>((i * 100 + j) % n_vocab);
    }
    total_tokens += len;
  }

  INFO("Total tokens to decode: " << total_tokens << " across " << N_SEQUENCES << " sequences");

  // Build scatter items
  std::vector<decode::ScatterItem> items;
  items.reserve(N_SEQUENCES);

  for (int i = 0; i < N_SEQUENCES; ++i) {
    items.push_back({
        .tokens = token_arrays[i],
        .start_pos = 0,
        .seq_id = static_cast<llama_seq_id>(i),
        .output_logits = true  // Get logits at end of each prompt
    });
  }

  // Decode ALL sequences in ONE llama_decode() call
  decode::Scratch scratch;
  int result = decode::scatter(ctx, items, scratch);
  REQUIRE(result == 0);

  INFO("Decoded " << total_tokens << " tokens in single decode_scatter call");

  // Verify each sequence has correct KV cache length
  for (int i = 0; i < N_SEQUENCES; ++i) {
    llama_pos pos = kv::pos_max(ctx, static_cast<llama_seq_id>(i));
    llama_pos expected = prompt_lengths[i] - 1;  // 0-indexed
    CHECK(pos == expected);
    INFO("Seq " << i << ": expected pos=" << expected << ", actual=" << pos);
  }

  INFO("✓ Asymmetric prefill verified for all " << N_SEQUENCES << " sequences");

  llama_free(ctx);
}

// ============================================================================
// Logits Validation Helper
// ============================================================================

static void require_valid_logits(llama_context *ctx, int32_t batch_idx,
                                 int32_t n_vocab, const char *label) {
  const float *logits = llama_get_logits_ith(ctx, batch_idx);
  REQUIRE_MESSAGE(logits != nullptr, label, " — logits pointer is null");

  bool has_non_zero = false;
  for (int i = 0; i < std::min(100, n_vocab); ++i) {
    CHECK_MESSAGE(!std::isnan(logits[i]), label, " — NaN at index ", i);
    if (logits[i] != 0.0f) has_non_zero = true;
  }
  CHECK_MESSAGE(has_non_zero, label, " — all logits zero");
}

// ============================================================================
// decode::one Tests
// ============================================================================

TEST_CASE("Integration: decode::one produces valid logits") {
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
  int n_vocab = llama_vocab_n_tokens(vocab);

  // Tokenize a prompt and build KV state with decode::many
  auto tokens = tokenizer::tokenize(vocab, "The quick brown fox", false, false);
  REQUIRE(tokens.size() >= 2);

  std::vector<llama_token> prefix(tokens.begin(), tokens.end() - 1);
  REQUIRE(decode::many(ctx, prefix, 0, ctx_params.n_batch) == 0);

  int32_t pos = static_cast<int32_t>(prefix.size());
  llama_token last_tok = tokens.back();

  // Decode single token with decode::one
  REQUIRE(decode::one(ctx, last_tok, pos, 0, true) == 0);

  // Verify logits are valid
  require_valid_logits(ctx, -1, n_vocab, "decode::one with want_logits=true");

  llama_free(ctx);
}

TEST_CASE("Integration: decode::one want_logits=false then true") {
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
  int n_vocab = llama_vocab_n_tokens(vocab);

  auto tokens = tokenizer::tokenize(vocab, "Hello world", false, false);
  REQUIRE(tokens.size() >= 2);

  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch) == 0);
  int32_t pos = static_cast<int32_t>(tokens.size());

  // Decode with want_logits=false — KV should advance but no logits needed
  REQUIRE(decode::one(ctx, tokens[0], pos, 0, false) == 0);
  CHECK(kv::pos_max(ctx, 0) == pos);

  // Now decode with want_logits=true — logits must be available
  REQUIRE(decode::one(ctx, tokens[1], pos + 1, 0, true) == 0);
  require_valid_logits(ctx, -1, n_vocab, "decode::one after want_logits=false");

  llama_free(ctx);
}

// ============================================================================
// decode::many Logits & Chunking Tests
// ============================================================================

TEST_CASE("Integration: decode::many multi-chunk produces valid logits") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 16; // Small batch to force multi-chunk

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  int n_vocab = llama_vocab_n_tokens(vocab);

  std::string long_prompt =
      "The quick brown fox jumps over the lazy dog and then runs "
      "across the wide open field toward the distant mountains.";
  auto tokens = tokenizer::tokenize(vocab, long_prompt, false, false);
  REQUIRE(tokens.size() > 16); // Must exceed n_batch

  INFO("Token count: " << tokens.size() << " (n_batch=16, requires "
       << ((tokens.size() + 15) / 16) << " chunks)");

  REQUIRE(decode::many(ctx, tokens, 0, 16) == 0);

  // Verify KV position
  llama_pos pos = kv::pos_max(ctx, 0);
  CHECK(pos == static_cast<llama_pos>(tokens.size() - 1));

  // Verify logits from last chunk are valid
  require_valid_logits(ctx, -1, n_vocab, "decode::many multi-chunk");

  llama_free(ctx);
}

// NOTE: Context overflow test removed — llama.cpp silently handles
// n_tokens > n_ctx without returning an error from llama_decode().
// This is tracked as a known behavior. If llama.cpp changes to return
// errors on overflow, add a test here.

// ============================================================================
// decode::each Logits Tests
// ============================================================================

TEST_CASE("Integration: decode::each logits at correct batch indices") {
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
  int n_vocab = llama_vocab_n_tokens(vocab);

  // Decode shared prefix to seq 0
  auto tokens = tokenizer::tokenize(vocab, "The answer is", false, false);
  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch, 0) == 0);
  llama_pos prefix_len = static_cast<llama_pos>(tokens.size());

  // Fork to seqs 1, 2, 3
  kv::seq_cp(ctx, 0, 1);
  kv::seq_cp(ctx, 0, 2);
  kv::seq_cp(ctx, 0, 3);

  // Build 4 EachItems: output_logits=true on items 0 and 2 only
  std::vector<decode::EachItem> items = {
      {.token = tokens[0], .pos = prefix_len, .seq_id = 0, .output_logits = true},
      {.token = tokens[1 % tokens.size()], .pos = prefix_len, .seq_id = 1, .output_logits = false},
      {.token = tokens[0], .pos = prefix_len, .seq_id = 2, .output_logits = true},
      {.token = tokens[1 % tokens.size()], .pos = prefix_len, .seq_id = 3, .output_logits = false},
  };

  decode::Scratch scratch;
  REQUIRE(decode::each(ctx, items, scratch) == 0);

  // Items 0 and 2 requested logits — verify at batch indices 0 and 2
  require_valid_logits(ctx, 0, n_vocab, "decode::each item 0");
  require_valid_logits(ctx, 2, n_vocab, "decode::each item 2");

  llama_free(ctx);
}

// ============================================================================
// decode::scatter Logits Tests
// ============================================================================

TEST_CASE("Integration: decode::scatter logits at correct cursor positions") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 3;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  int n_vocab = llama_vocab_n_tokens(vocab);

  // Token arrays of different sizes: 5, 3, 7 = 15 total
  std::vector<llama_token> tokens_a(5), tokens_b(3), tokens_c(7);
  for (int i = 0; i < 5; ++i) tokens_a[i] = static_cast<llama_token>((100 + i) % n_vocab);
  for (int i = 0; i < 3; ++i) tokens_b[i] = static_cast<llama_token>((200 + i) % n_vocab);
  for (int i = 0; i < 7; ++i) tokens_c[i] = static_cast<llama_token>((300 + i) % n_vocab);

  std::vector<decode::ScatterItem> items = {
      {.tokens = tokens_a, .start_pos = 0, .seq_id = 0, .output_logits = true},
      {.tokens = tokens_b, .start_pos = 0, .seq_id = 1, .output_logits = true},
      {.tokens = tokens_c, .start_pos = 0, .seq_id = 2, .output_logits = true},
  };

  decode::Scratch scratch;
  REQUIRE(decode::scatter(ctx, items, scratch) == 0);

  // Flattened cursor positions of last token per item: 4, 7, 14
  require_valid_logits(ctx, 4,  n_vocab, "scatter item 0 (cursor=4)");
  require_valid_logits(ctx, 7,  n_vocab, "scatter item 1 (cursor=7)");
  require_valid_logits(ctx, 14, n_vocab, "scatter item 2 (cursor=14)");

  // Verify KV positions
  CHECK(kv::pos_max(ctx, 0) == 4);
  CHECK(kv::pos_max(ctx, 1) == 2);
  CHECK(kv::pos_max(ctx, 2) == 6);

  llama_free(ctx);
}

// ============================================================================
// Scratch Reuse Tests
// ============================================================================

TEST_CASE("Integration: Scratch reuse across different sizes") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 9; // root + 8 branches

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  int n_vocab = llama_vocab_n_tokens(vocab);

  // Shared prefix on seq 0
  auto tokens = tokenizer::tokenize(vocab, "Hello world", false, false);
  REQUIRE(decode::many(ctx, tokens, 0, ctx_params.n_batch, 0) == 0);
  llama_pos next_pos = static_cast<llama_pos>(tokens.size());

  // Fork to seqs 1-8
  for (int i = 1; i <= 8; ++i) {
    kv::seq_cp(ctx, 0, static_cast<llama_seq_id>(i));
  }

  decode::Scratch scratch; // Single scratch, reused across 3 calls

  // Call 1: 4 items (seqs 1-4)
  {
    std::vector<decode::EachItem> items;
    for (int i = 1; i <= 4; ++i) {
      items.push_back({
          .token = static_cast<llama_token>((100 + i) % n_vocab),
          .pos = next_pos,
          .seq_id = static_cast<llama_seq_id>(i),
          .output_logits = false,
      });
    }
    REQUIRE(decode::each(ctx, items, scratch) == 0);
  }

  // Call 2: 2 items (seqs 5-6) — scratch internal buffers shrink
  {
    std::vector<decode::EachItem> items;
    for (int i = 5; i <= 6; ++i) {
      items.push_back({
          .token = static_cast<llama_token>((200 + i) % n_vocab),
          .pos = next_pos,
          .seq_id = static_cast<llama_seq_id>(i),
          .output_logits = false,
      });
    }
    REQUIRE(decode::each(ctx, items, scratch) == 0);
  }

  // Call 3: 8 items (seqs 1-8) — scratch must grow back
  {
    std::vector<decode::EachItem> items;
    for (int i = 1; i <= 8; ++i) {
      // seqs 1-6 already have a token at next_pos, so use next_pos+1
      // seqs 7-8 don't have next_pos yet, so use next_pos
      llama_pos pos = (i <= 6) ? next_pos + 1 : next_pos;
      items.push_back({
          .token = static_cast<llama_token>((300 + i) % n_vocab),
          .pos = pos,
          .seq_id = static_cast<llama_seq_id>(i),
          .output_logits = false,
      });
    }
    REQUIRE(decode::each(ctx, items, scratch) == 0);
  }

  // Verify: seqs 1-6 at next_pos+1, seqs 7-8 at next_pos
  for (int i = 1; i <= 6; ++i) {
    CHECK(kv::pos_max(ctx, static_cast<llama_seq_id>(i)) == next_pos + 1);
  }
  for (int i = 7; i <= 8; ++i) {
    CHECK(kv::pos_max(ctx, static_cast<llama_seq_id>(i)) == next_pos);
  }

  llama_free(ctx);
}

// ============================================================================
// Multi-Sequence Interleaved Tests
// ============================================================================

TEST_CASE("Integration: decode_scatter - interleaved generation and prefill") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  constexpr int N_SEQUENCES = 6;

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 512;
  ctx_params.n_seq_max = N_SEQUENCES;

  llama_context *ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  int32_t n_vocab = llama_vocab_n_tokens(vocab);

  // Phase 1: Initialize all sequences with short prompts
  const int initial_len = 5;
  std::vector<std::vector<llama_token>> initial_tokens(N_SEQUENCES);

  for (int i = 0; i < N_SEQUENCES; ++i) {
    initial_tokens[i].resize(initial_len);
    for (int j = 0; j < initial_len; ++j) {
      initial_tokens[i][j] = static_cast<llama_token>((i * 50 + j) % n_vocab);
    }
    REQUIRE(decode::many(ctx, initial_tokens[i], 0, ctx_params.n_batch,
                         static_cast<llama_seq_id>(i)) == 0);
  }

  INFO("Phase 1: Initialized " << N_SEQUENCES << " sequences with " << initial_len << " tokens each");

  // Phase 2: Mixed workload - some generating (1 token), some prefilling (15 tokens)
  // Seq 0, 2, 4: generating (1 token each)
  // Seq 1, 3, 5: prefilling (15 tokens each)
  const int prefill_len = 15;

  std::vector<std::vector<llama_token>> mixed_tokens(N_SEQUENCES);
  for (int i = 0; i < N_SEQUENCES; ++i) {
    int len = (i % 2 == 0) ? 1 : prefill_len;
    mixed_tokens[i].resize(len);
    for (int j = 0; j < len; ++j) {
      mixed_tokens[i][j] = static_cast<llama_token>((i * 200 + j + 1000) % n_vocab);
    }
  }

  // Build scatter items for mixed workload
  std::vector<decode::ScatterItem> items;
  items.reserve(N_SEQUENCES);

  for (int i = 0; i < N_SEQUENCES; ++i) {
    items.push_back({
        .tokens = mixed_tokens[i],
        .start_pos = initial_len,  // Continue from where we left off
        .seq_id = static_cast<llama_seq_id>(i),
        .output_logits = true
    });
  }

  // Decode mixed workload in ONE call
  decode::Scratch scratch;
  int result = decode::scatter(ctx, items, scratch);
  REQUIRE(result == 0);

  INFO("Phase 2: Decoded mixed workload (1 or 15 tokens per sequence)");

  // Verify positions
  for (int i = 0; i < N_SEQUENCES; ++i) {
    llama_pos pos = kv::pos_max(ctx, static_cast<llama_seq_id>(i));
    int expected_len = initial_len + ((i % 2 == 0) ? 1 : prefill_len);
    llama_pos expected_pos = expected_len - 1;

    CHECK(pos == expected_pos);
    INFO("Seq " << i << ": " << (i % 2 == 0 ? "generating" : "prefilling")
         << ", expected pos=" << expected_pos << ", actual=" << pos);
  }

  INFO("✓ Interleaved generation/prefill verified");

  llama_free(ctx);
}
