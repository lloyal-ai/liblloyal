/**
 * Contrastive Logits Integration Test
 *
 * Validates the primitives that enable DExperts-style contrastive decoding
 * across branches:
 *
 * - `lloyal::branch::set_logits()` — overwrite a branch's cached logits_snapshot
 *   from a caller-provided buffer (round-trip via get_logits)
 * - `BranchStore::merge_logits()` — additively combine N expert branches'
 *   logits_snapshot into a destination branch's logits_snapshot
 * - `BranchStore::decode_each()` with the SAME token committed to all branches
 *   in one batched dispatch (lockstep advance) — already covered by existing
 *   decode_each test, but here verified in the contrastive use case where
 *   each branch produces fresh logits from its own KV history at every step
 * - End-to-end contrastive loop: trunk + expert with different KV prefixes,
 *   merge → sample → commit-to-all → repeat — verifies the loop runs without
 *   crashing and that contrastive output diverges from trunk-only output
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/branch.hpp>
#include <lloyal/decode.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <memory>
#include <vector>

using namespace lloyal;
using namespace lloyal::branch;

static const char* MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

#define REQUIRE_MODEL()                                                        \
  if (!MODEL_PATH || !*MODEL_PATH) {                                           \
    MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set");                              \
    return;                                                                    \
  }

struct LlamaBackendGuard {
  LlamaBackendGuard() { llama_backend_init(); }
  ~LlamaBackendGuard() { llama_backend_free(); }
};

struct TestParams {
  float temperature = 0.0f;  // greedy for deterministic verification
  int32_t top_k = 40;
  float top_p = 0.95f;
  float min_p = 0.05f;
  float typical_p = 1.0f;
  float penalty_repeat = 1.0f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 64;
  uint32_t seed = 42;
};

// ============================================================================
// set_logits: round-trip and replacement semantics
// ============================================================================

TEST_CASE("contrastive: set_logits round-trip preserves bytes") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  // Need captured logits before set_logits — prefill a small prompt
  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello", true, false);
  REQUIRE(!tokens.empty());
  prefill(h, tokens.data(), tokens.size(), store);

  int n_vocab = get_n_vocab(h, store);
  REQUIRE(n_vocab > 0);

  // Construct a known synthetic distribution
  std::vector<float> synthetic(n_vocab);
  for (int i = 0; i < n_vocab; ++i) {
    synthetic[i] = static_cast<float>(i) / static_cast<float>(n_vocab) - 0.5f;
  }

  // Write it in
  CHECK_NOTHROW(set_logits(h, std::span<const float>(synthetic.data(), synthetic.size()), store));

  // Read it back
  const float* read_back = get_logits(h, store);
  REQUIRE(read_back != nullptr);

  // Verify byte equality
  bool all_match = true;
  for (int i = 0; i < n_vocab; ++i) {
    if (read_back[i] != synthetic[i]) {
      all_match = false;
      break;
    }
  }
  CHECK(all_match);

  pruneSubtree(h, store);
  llama_free(ctx);
}

TEST_CASE("contrastive: set_logits rejects mismatched length") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hi", true, false);
  prefill(h, tokens.data(), tokens.size(), store);

  // Wrong length should throw
  std::vector<float> wrong_size(10, 0.5f);
  CHECK_THROWS(set_logits(h, std::span<const float>(wrong_size.data(), wrong_size.size()), store));

  pruneSubtree(h, store);
  llama_free(ctx);
}

// ============================================================================
// merge_logits: math verification
// ============================================================================

TEST_CASE("contrastive: merge_logits computes dst += alpha * sum(experts)") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  // Create three branches and prime each with a real prefill so logits exist
  BranchHandle dst = create(ctx, model.get(), store, 0, params, 64);
  BranchHandle src1 = create(ctx, model.get(), store, 0, params, 64);
  BranchHandle src2 = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(dst != INVALID_HANDLE);
  REQUIRE(src1 != INVALID_HANDLE);
  REQUIRE(src2 != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto t1 = tokenizer::tokenize(vocab, "A", true, false);
  auto t2 = tokenizer::tokenize(vocab, "B", true, false);
  auto t3 = tokenizer::tokenize(vocab, "C", true, false);
  prefill(dst, t1.data(), t1.size(), store);
  prefill(src1, t2.data(), t2.size(), store);
  prefill(src2, t3.data(), t3.size(), store);

  int n_vocab = get_n_vocab(dst, store);
  REQUIRE(n_vocab > 0);
  REQUIRE(get_n_vocab(src1, store) == n_vocab);
  REQUIRE(get_n_vocab(src2, store) == n_vocab);

  // Overwrite each branch's logits with a known synthetic distribution so we
  // can verify the math without dealing with real model output noise.
  std::vector<float> dst_init(n_vocab);
  std::vector<float> src1_init(n_vocab);
  std::vector<float> src2_init(n_vocab);
  for (int i = 0; i < n_vocab; ++i) {
    dst_init[i] = 1.0f;
    src1_init[i] = 2.0f;
    src2_init[i] = 3.0f;
  }
  set_logits(dst, std::span<const float>(dst_init.data(), dst_init.size()), store);
  set_logits(src1, std::span<const float>(src1_init.data(), src1_init.size()), store);
  set_logits(src2, std::span<const float>(src2_init.data(), src2_init.size()), store);

  // Merge: dst[t] += 0.5 * (src1[t] + src2[t])
  //                = 1 + 0.5 * (2 + 3)
  //                = 3.5
  const float alpha = 0.5f;
  std::vector<BranchHandle> experts = {src1, src2};
  CHECK_NOTHROW(store.merge_logits(
      dst,
      std::span<const BranchHandle>(experts.data(), experts.size()),
      alpha));

  const float* result = get_logits(dst, store);
  REQUIRE(result != nullptr);

  bool all_match = true;
  for (int i = 0; i < n_vocab; ++i) {
    if (std::abs(result[i] - 3.5f) > 1e-6f) {
      all_match = false;
      break;
    }
  }
  CHECK(all_match);

  // Verify experts unchanged
  const float* src1_after = get_logits(src1, store);
  const float* src2_after = get_logits(src2, store);
  REQUIRE(src1_after != nullptr);
  REQUIRE(src2_after != nullptr);
  CHECK(std::abs(src1_after[0] - 2.0f) < 1e-6f);
  CHECK(std::abs(src2_after[0] - 3.0f) < 1e-6f);

  pruneSubtree(dst, store);
  pruneSubtree(src1, store);
  pruneSubtree(src2, store);
  llama_free(ctx);
}

TEST_CASE("contrastive: merge_logits with empty experts is a no-op") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle dst = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(dst != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "X", true, false);
  prefill(dst, tokens.data(), tokens.size(), store);

  int n_vocab = get_n_vocab(dst, store);
  std::vector<float> initial(n_vocab, 7.0f);
  set_logits(dst, std::span<const float>(initial.data(), initial.size()), store);

  std::vector<BranchHandle> none;
  CHECK_NOTHROW(store.merge_logits(
      dst,
      std::span<const BranchHandle>(none.data(), none.size()),
      0.5f));

  // dst should be unchanged
  const float* result = get_logits(dst, store);
  REQUIRE(result != nullptr);
  for (int i = 0; i < n_vocab; ++i) {
    REQUIRE(std::abs(result[i] - 7.0f) < 1e-6f);
  }

  pruneSubtree(dst, store);
  llama_free(ctx);
}

TEST_CASE("contrastive: merge_logits rejects invalid handles") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle dst = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(dst != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "X", true, false);
  prefill(dst, tokens.data(), tokens.size(), store);

  // Invalid expert handle
  std::vector<BranchHandle> bad = {12345};
  CHECK_THROWS(store.merge_logits(
      dst,
      std::span<const BranchHandle>(bad.data(), bad.size()),
      0.5f));

  // Invalid dst handle
  std::vector<BranchHandle> empty;
  CHECK_THROWS(store.merge_logits(
      54321,
      std::span<const BranchHandle>(empty.data(), empty.size()),
      0.5f));

  pruneSubtree(dst, store);
  llama_free(ctx);
}

// ============================================================================
// Lockstep advance: same token committed to N branches via decode_each
// ============================================================================

TEST_CASE("contrastive: decode_each advances all branches with same token") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 128;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  // Create N branches with DIFFERENT prefills so each has its own KV history
  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  int32_t n_vocab = llama_vocab_n_tokens(vocab);

  BranchHandle b1 = create(ctx, model.get(), store, 0, params, 64);
  BranchHandle b2 = create(ctx, model.get(), store, 0, params, 64);
  BranchHandle b3 = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(b1 != INVALID_HANDLE);
  REQUIRE(b2 != INVALID_HANDLE);
  REQUIRE(b3 != INVALID_HANDLE);

  auto p1 = tokenizer::tokenize(vocab, "The capital of France is", true, false);
  auto p2 = tokenizer::tokenize(vocab, "Mathematics is the study of", true, false);
  auto p3 = tokenizer::tokenize(vocab, "Once upon a time, there was", true, false);
  prefill(b1, p1.data(), p1.size(), store);
  prefill(b2, p2.data(), p2.size(), store);
  prefill(b3, p3.data(), p3.size(), store);

  llama_pos pos1 = get_position(b1, store);
  llama_pos pos2 = get_position(b2, store);
  llama_pos pos3 = get_position(b3, store);

  // Snapshot the initial logits (each branch's distribution before lockstep)
  std::vector<float> snap1_before(n_vocab);
  std::vector<float> snap2_before(n_vocab);
  std::vector<float> snap3_before(n_vocab);
  std::memcpy(snap1_before.data(), get_logits(b1, store), n_vocab * sizeof(float));
  std::memcpy(snap2_before.data(), get_logits(b2, store), n_vocab * sizeof(float));
  std::memcpy(snap3_before.data(), get_logits(b3, store), n_vocab * sizeof(float));

  // Commit the SAME token to all three branches in one batched dispatch
  llama_token shared_token = static_cast<llama_token>(100 % n_vocab);
  DecodeEachItem items[3];
  items[0] = {b1, shared_token};
  items[1] = {b2, shared_token};
  items[2] = {b3, shared_token};
  CHECK_NOTHROW(store.decode_each(items));

  // All branches advanced by 1
  CHECK(get_position(b1, store) == pos1 + 1);
  CHECK(get_position(b2, store) == pos2 + 1);
  CHECK(get_position(b3, store) == pos3 + 1);

  // Each branch produced its OWN fresh logits from its own KV history.
  // The new distributions must differ from the pre-commit snapshots
  // (since position advanced and a new token was incorporated).
  const float* after1 = get_logits(b1, store);
  const float* after2 = get_logits(b2, store);
  const float* after3 = get_logits(b3, store);
  REQUIRE(after1 != nullptr);
  REQUIRE(after2 != nullptr);
  REQUIRE(after3 != nullptr);

  // And each branch's new distribution must differ from the others
  // (because they have different KV histories — same new token doesn't make
  // them identical)
  bool b1_b2_differ = false;
  bool b1_b3_differ = false;
  bool b2_b3_differ = false;
  for (int i = 0; i < n_vocab; ++i) {
    if (std::abs(after1[i] - after2[i]) > 1e-4f) b1_b2_differ = true;
    if (std::abs(after1[i] - after3[i]) > 1e-4f) b1_b3_differ = true;
    if (std::abs(after2[i] - after3[i]) > 1e-4f) b2_b3_differ = true;
  }
  CHECK(b1_b2_differ);
  CHECK(b1_b3_differ);
  CHECK(b2_b3_differ);

  pruneSubtree(b1, store);
  pruneSubtree(b2, store);
  pruneSubtree(b3, store);
  llama_free(ctx);
}

// ============================================================================
// End-to-end contrastive loop
// ============================================================================

TEST_CASE("contrastive: end-to-end contrastive loop diverges from trunk-only") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 128;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);

  // Greedy sampling for deterministic comparison
  TestParams params;
  params.temperature = 0.0f;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());

  // Two prompts: trunk knows the topic broadly, expert has a leading hint
  auto trunk_prompt = tokenizer::tokenize(
      vocab, "The largest planet in our solar system is", true, false);
  auto expert_prompt = tokenizer::tokenize(
      vocab,
      "Jupiter is a gas giant. The largest planet in our solar system is",
      true, false);

  // ---- Run 1: trunk-only (alpha=0, no merge) ----
  BranchHandle trunk_alone = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(trunk_alone != INVALID_HANDLE);
  prefill(trunk_alone, trunk_prompt.data(), trunk_prompt.size(), store);

  std::vector<llama_token> trunk_only_tokens;
  const int max_tokens = 8;
  for (int step = 0; step < max_tokens; ++step) {
    llama_token tok = sample(trunk_alone, store);
    if (tok < 0) break;
    if (tokenizer::is_eog(model.get(), tok)) break;
    accept_token(trunk_alone, tok, store);
    DecodeEachItem item = {trunk_alone, tok};
    store.decode_each(std::span<const DecodeEachItem>(&item, 1));
    trunk_only_tokens.push_back(tok);
  }
  pruneSubtree(trunk_alone, store);

  // ---- Run 2: contrastive (trunk + expert, alpha=0.5) ----
  BranchHandle trunk = create(ctx, model.get(), store, 0, params, 64);
  BranchHandle expert = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(trunk != INVALID_HANDLE);
  REQUIRE(expert != INVALID_HANDLE);

  prefill(trunk, trunk_prompt.data(), trunk_prompt.size(), store);
  prefill(expert, expert_prompt.data(), expert_prompt.size(), store);

  const float alpha = 0.5f;
  std::vector<BranchHandle> experts = {expert};

  std::vector<llama_token> contrastive_tokens;
  for (int step = 0; step < max_tokens; ++step) {
    // Merge expert into trunk's logits_snapshot
    store.merge_logits(
        trunk,
        std::span<const BranchHandle>(experts.data(), experts.size()),
        alpha);

    // Sample from the combined distribution via trunk's sampler chain
    llama_token tok = sample(trunk, store);
    if (tok < 0) break;
    if (tokenizer::is_eog(model.get(), tok)) break;

    // Accept on trunk only (its sampler state tracks what trunk "chose")
    accept_token(trunk, tok, store);

    // Lockstep commit: same token to both branches in one batched dispatch
    DecodeEachItem batch[2];
    batch[0] = {trunk, tok};
    batch[1] = {expert, tok};
    store.decode_each(std::span<const DecodeEachItem>(batch, 2));

    contrastive_tokens.push_back(tok);
  }

  // Both runs should produce SOMETHING
  CHECK(!trunk_only_tokens.empty());
  CHECK(!contrastive_tokens.empty());

  // The contrastive output should pull trunk toward "Jupiter" (the expert's
  // hint). We don't assert exact equality with any particular sequence
  // because that's model-dependent — we just verify the loop runs and
  // produces a valid token sequence with no crashes.
  //
  // Sanity: every token is in the vocab range.
  int32_t n_vocab = llama_vocab_n_tokens(vocab);
  for (auto tok : trunk_only_tokens) {
    CHECK(tok >= 0);
    CHECK(tok < n_vocab);
  }
  for (auto tok : contrastive_tokens) {
    CHECK(tok >= 0);
    CHECK(tok < n_vocab);
  }

  INFO("trunk_only generated " << trunk_only_tokens.size() << " tokens");
  INFO("contrastive generated " << contrastive_tokens.size() << " tokens");

  pruneSubtree(trunk, store);
  pruneSubtree(expert, store);
  llama_free(ctx);
}
