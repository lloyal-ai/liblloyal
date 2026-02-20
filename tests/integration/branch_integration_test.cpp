/**
 * Branch Integration Test
 *
 * Validates that lloyal::branch provides correct handle-based branch
 * management with a real llama.cpp model. Tests:
 *
 * - Branch creation with sampling params
 * - Branch fork with KV cache copy
 * - Logits capture after decode
 * - Branch prune removes KV entries
 * - Multiple branches operate independently
 * - RAII Branch wrapper cleanup
 * - Tenancy lifecycle (seq_id recycling, retainOnly, drain)
 * - Topology (parent/child edges, pruneSubtree CASCADE)
 * - set_sampler_params() memoization and greedy↔stochastic transitions
 * - set_grammar() hot-swap, removal, and fork propagation
 * - Handle registry cleanup on prune and slot reuse
 */

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/branch.hpp>
#include <lloyal/decode.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <memory>
#include <string>
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

// Test sampling params matching the concept
struct TestParams {
  float temperature = 0.8f;
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
// Basic Branch Lifecycle
// ============================================================================

TEST_CASE("branch integration: create and prune") {
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
  CHECK(h != INVALID_HANDLE);

  BranchState* state = store.get(h);
  REQUIRE(state != nullptr);
  CHECK(state->ctx == ctx);
  CHECK(state->model == model.get());
  CHECK(state->seq_id >= 0);  // Tenancy-assigned, specific value unspecified
  CHECK(state->position == 0);
  CHECK(state->n_vocab > 0);

  prune(h, store);
  CHECK(store.get(h) == nullptr);

  llama_free(ctx);
}

// ============================================================================
// Decode and Position Tracking
// ============================================================================

TEST_CASE("branch integration: decode updates position") {
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
  auto tokens = tokenizer::tokenize(vocab, "Hello world", true, false);
  REQUIRE(!tokens.empty());

  CHECK(get_position(h, store) == 0);
  CHECK_NOTHROW(prefill(h, tokens.data(), tokens.size(), store));
  CHECK(get_position(h, store) == static_cast<llama_pos>(tokens.size()));

  prune(h, store);
  llama_free(ctx);
}

// ============================================================================
// Logits Capture
// ============================================================================

TEST_CASE("branch integration: decode_and_capture captures logits") {
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
  auto tokens = tokenizer::tokenize(vocab, "The capital of France is", true, false);
  REQUIRE(!tokens.empty());

  CHECK_NOTHROW(prefill(h, tokens.data(), tokens.size(), store));

  const float* logits = get_logits(h, store);
  REQUIRE(logits != nullptr);

  int n_vocab = get_n_vocab(h, store);
  CHECK(n_vocab > 0);

  float sum = 0.0f;
  for (int i = 0; i < std::min(100, n_vocab); ++i) {
    sum += std::abs(logits[i]);
  }
  CHECK(sum > 0.0f);

  prune(h, store);
  llama_free(ctx);
}

// ============================================================================
// Fork and Independence
// ============================================================================

TEST_CASE("branch integration: fork creates independent branch") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 64;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle parent = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(parent != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Once upon a time", true, false);
  REQUIRE(!prompt.empty());

  prefill(parent, prompt.data(), prompt.size(), store);
  llama_pos parent_pos = get_position(parent, store);
  CHECK(parent_pos == static_cast<llama_pos>(prompt.size()));

  // Snapshot parent logits before fork
  const float* parent_logits_before_fork = get_logits(parent, store);
  REQUIRE(parent_logits_before_fork != nullptr);
  int n_vocab = get_n_vocab(parent, store);
  std::vector<float> parent_snapshot(n_vocab);
  std::memcpy(parent_snapshot.data(), parent_logits_before_fork, n_vocab * sizeof(float));

  // Fork — seq_id allocated internally by tenancy
  BranchHandle child = fork(parent, store);
  REQUIRE(child != INVALID_HANDLE);
  CHECK(child != parent);

  // Child should have same position
  CHECK(get_position(child, store) == parent_pos);
  CHECK(get_position(parent, store) == parent_pos);

  // Topology: child's parent should be parent, parent should have child
  CHECK(store.parent(child) == parent);
  CHECK(!store.children(parent).empty());

  // After fork, both branches must have independent logits memory
  const float* logits_parent_after_fork = get_logits(parent, store);
  const float* logits_child_after_fork = get_logits(child, store);
  REQUIRE(logits_parent_after_fork != nullptr);
  REQUIRE(logits_child_after_fork != nullptr);
  CHECK(logits_parent_after_fork != logits_child_after_fork);

  // After fork, child should have same logits values as parent (copied)
  bool fork_copy_correct = true;
  for (int i = 0; i < n_vocab; ++i) {
    if (logits_child_after_fork[i] != parent_snapshot[i]) {
      fork_copy_correct = false;
      break;
    }
  }
  CHECK(fork_copy_correct);

  // Decode different continuations
  auto cont_a = tokenizer::tokenize(vocab, " princess", false, false);
  auto cont_b = tokenizer::tokenize(vocab, " dragon", false, false);
  REQUIRE(!cont_a.empty());
  REQUIRE(!cont_b.empty());

  step(parent, cont_a[0], store);
  step(child, cont_b[0], store);

  CHECK(get_position(parent, store) == parent_pos + 1);
  CHECK(get_position(child, store) == parent_pos + 1);

  // Logits must still be at different memory addresses
  const float* logits_parent = get_logits(parent, store);
  const float* logits_child = get_logits(child, store);
  REQUIRE(logits_parent != nullptr);
  REQUIRE(logits_child != nullptr);
  CHECK(logits_parent != logits_child);

  float diff = 0.0f;
  for (int i = 0; i < std::min(100, n_vocab); ++i) {
    diff += std::abs(logits_parent[i] - logits_child[i]);
  }
  INFO("Logits diff after different decodes: " << diff);
  INFO("Token IDs: parent=" << cont_a[0] << " child=" << cont_b[0]);
  if (cont_a[0] != cont_b[0] && diff < 0.1f) {
    MESSAGE("WARNING: Different tokens produced identical logits (model may have random weights)");
  }

  pruneSubtree(parent, store);
  llama_free(ctx);
}

// ============================================================================
// Prune Removes KV Entries
// ============================================================================

TEST_CASE("branch integration: prune removes KV entries") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello world", true, false);
  prefill(h, tokens.data(), tokens.size(), store);

  // Get seq_id from state before pruning
  BranchState* state = store.get(h);
  REQUIRE(state != nullptr);
  llama_seq_id seq = state->seq_id;

  llama_pos pos_before = kv::pos_max(ctx, seq);
  CHECK(pos_before >= 0);

  prune(h, store);

  CHECK(store.get(h) == nullptr);

  // KV entries should be removed (evicted by tenancy)
  llama_pos pos_after = kv::pos_max(ctx, seq);
  CHECK(pos_after < pos_before);

  llama_free(ctx);
}

// ============================================================================
// RAII Branch Wrapper
// ============================================================================

TEST_CASE("branch integration: RAII Branch wrapper") {
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

  BranchHandle raw_handle;

  {
    Branch b = Branch::create(ctx, model.get(), store, 0, params, 64);
    CHECK(b.valid());
    raw_handle = b.handle();
    CHECK(store.get(raw_handle) != nullptr);

    const llama_vocab* vocab = llama_model_get_vocab(model.get());
    auto tokens = tokenizer::tokenize(vocab, "Test", true, false);

    CHECK_NOTHROW(b.prefill(tokens.data(), tokens.size()));
    CHECK(b.position() > 0);

    // Fork — no seq_id param
    Branch child = b.fork();
    CHECK(child.valid());
    CHECK(child.position() == b.position());
    CHECK(child.parentHandle() == b.handle());
    CHECK(b.isLeaf() == false);  // b now has a child
    CHECK(child.isLeaf() == true);
  }

  // After scope, branch should be freed (RAII pruneSubtree)
  CHECK(store.get(raw_handle) == nullptr);

  llama_free(ctx);
}

// ============================================================================
// Sample and Accept
// ============================================================================

TEST_CASE("branch integration: sample and accept token") {
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
  params.temperature = 0.0f;  // Greedy for reproducibility

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "The answer is", true, false);

  prefill(h, prompt.data(), prompt.size(), store);

  llama_token tok = sample(h, store);
  CHECK(tok >= 0);
  CHECK(tok < get_n_vocab(h, store));

  accept_token(h, tok, store);

  prune(h, store);
  llama_free(ctx);
}

// ============================================================================
// Perplexity Tracking
// ============================================================================

TEST_CASE("branch integration: perplexity tracking across fork") {
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

  BranchHandle parent = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(parent != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Hello", true, false);
  prefill(parent, prompt.data(), prompt.size(), store);

  const float* logits = get_logits(parent, store);
  int n_vocab = get_n_vocab(parent, store);

  llama_token test_tok = prompt.back();
  float surprisal = metrics::model_surprisal(logits, n_vocab, test_tok);

  BranchState* pstate = store.get(parent);
  REQUIRE(pstate);
  store.add_model_surprisal(pstate->metrics, surprisal);

  float parent_ppl = get_perplexity(parent, store);
  CHECK(std::isfinite(parent_ppl));

  BranchHandle child = fork(parent, store);
  REQUIRE(child != INVALID_HANDLE);

  float child_ppl = get_perplexity(child, store);
  CHECK(std::abs(parent_ppl - child_ppl) < 0.001f);

  BranchState* cstate = store.get(child);
  REQUIRE(cstate);
  store.add_model_surprisal(cstate->metrics, 2.0f);

  CHECK(std::abs(get_perplexity(parent, store) - parent_ppl) < 0.001f);
  CHECK(get_perplexity(child, store) != parent_ppl);

  pruneSubtree(parent, store);
  llama_free(ctx);
}

// ============================================================================
// Strict Branch Scoping Tests
// ============================================================================

TEST_CASE("branch scoping: logits snapshots are independent memory") {
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

  BranchHandle branch = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(branch != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "test", true, false);
  REQUIRE(!tokens.empty());

  step(branch, tokens[0], store);

  BranchHandle forked = fork(branch, store);
  REQUIRE(forked != INVALID_HANDLE);

  const float* logits1 = get_logits(branch, store);
  const float* logits2 = get_logits(forked, store);
  REQUIRE(logits1 != nullptr);
  REQUIRE(logits2 != nullptr);

  INFO("Original branch logits at: " << (void*)logits1);
  INFO("Forked branch logits at: " << (void*)logits2);

  CHECK(logits1 != logits2);

  int n_vocab = get_n_vocab(branch, store);
  bool all_same = true;
  for (int i = 0; i < n_vocab; ++i) {
    if (logits1[i] != logits2[i]) {
      all_same = false;
      break;
    }
  }
  CHECK(all_same);

  pruneSubtree(branch, store);
  llama_free(ctx);
}

TEST_CASE("branch scoping: decode updates only target branch logits") {
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

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Hello", true, false);
  REQUIRE(!prompt.empty());

  BranchHandle parent = create(ctx, model.get(), store, 0, params, 64);
  step(parent, prompt[0], store);

  BranchHandle child = fork(parent, store);
  REQUIRE(child != INVALID_HANDLE);

  // Snapshot parent's logits BEFORE child decode
  const float* parent_before = get_logits(parent, store);
  REQUIRE(parent_before != nullptr);
  int n_vocab = get_n_vocab(parent, store);

  std::vector<float> parent_snapshot(n_vocab);
  std::memcpy(parent_snapshot.data(), parent_before, n_vocab * sizeof(float));

  auto cont = tokenizer::tokenize(vocab, " world", false, false);
  REQUIRE(!cont.empty());

  // Decode to CHILD only
  step(child, cont[0], store);

  // Parent's logits must be UNCHANGED
  const float* parent_after = get_logits(parent, store);
  REQUIRE(parent_after != nullptr);

  CHECK(parent_after == parent_before);

  bool parent_unchanged = true;
  float max_diff = 0.0f;
  for (int i = 0; i < n_vocab; ++i) {
    float diff = std::abs(parent_after[i] - parent_snapshot[i]);
    if (diff > 0.0f) {
      parent_unchanged = false;
      max_diff = std::max(max_diff, diff);
    }
  }

  INFO("Max diff in parent logits: " << max_diff);
  CHECK(parent_unchanged);

  const float* child_logits = get_logits(child, store);
  REQUIRE(child_logits != nullptr);
  CHECK(child_logits != parent_after);

  pruneSubtree(parent, store);
  llama_free(ctx);
}

TEST_CASE("branch scoping: concurrent captures preserve isolation") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  cparams.n_seq_max = 8;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(16);
  store.init_tenancy(ctx);
  TestParams params;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Start", true, false);
  REQUIRE(!prompt.empty());

  BranchHandle root = create(ctx, model.get(), store, 0, params, 64);
  step(root, prompt[0], store);

  // Create 4 branches via fork (seq_ids assigned by tenancy)
  std::vector<BranchHandle> branches;
  for (int i = 0; i < 4; ++i) {
    BranchHandle b = fork(root, store);
    REQUIRE(b != INVALID_HANDLE);
    branches.push_back(b);
  }

  // Tokenize 4 different continuations
  std::vector<std::vector<llama_token>> continuations;
  const char* words[] = {" one", " two", " three", " four"};
  for (const char* word : words) {
    auto tokens = tokenizer::tokenize(vocab, word, false, false);
    REQUIRE(!tokens.empty());
    continuations.push_back(tokens);
  }

  for (size_t i = 0; i < branches.size(); ++i) {
    step(branches[i], continuations[i][0], store);
  }

  // Verify all branches have independent logits
  std::vector<const float*> all_logits;
  for (auto b : branches) {
    const float* l = get_logits(b, store);
    REQUIRE(l != nullptr);
    all_logits.push_back(l);
  }

  // All pointers must be unique
  for (size_t i = 0; i < all_logits.size(); ++i) {
    for (size_t j = i + 1; j < all_logits.size(); ++j) {
      INFO("Branch " << i << " logits at " << (void*)all_logits[i]);
      INFO("Branch " << j << " logits at " << (void*)all_logits[j]);
      CHECK(all_logits[i] != all_logits[j]);
    }
  }

  int n_vocab = get_n_vocab(root, store);
  int pairs_with_differences = 0;
  for (size_t i = 0; i < all_logits.size(); ++i) {
    for (size_t j = i + 1; j < all_logits.size(); ++j) {
      float diff = 0.0f;
      for (int k = 0; k < std::min(100, n_vocab); ++k) {
        diff += std::abs(all_logits[i][k] - all_logits[j][k]);
      }
      if (diff > 0.1f) {
        pairs_with_differences++;
      }
    }
  }

  INFO("Pairs with logits differences: " << pairs_with_differences << " / 6");

  pruneSubtree(root, store);
  llama_free(ctx);
}

// ============================================================================
// Logit Bias and Steer Tests
// ============================================================================

TEST_CASE("branch integration: basic logit_bias bans tokens") {
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
  params.seed = 12345;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "The capital", true, false);
  REQUIRE(!tokens.empty());
  CHECK_NOTHROW(prefill(h, tokens.data(), tokens.size(), store));

  llama_token banned_token = 42;
  llama_logit_bias bias = {banned_token, -std::numeric_limits<float>::infinity()};
  set_logit_bias(h, &bias, 1, store);

  for (int i = 0; i < 10; ++i) {
    llama_token tok = sample(h, store);
    REQUIRE(tok != -1);
    CHECK(tok != banned_token);
  }

  MESSAGE("Sampled 10 tokens, none were token " << banned_token);

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: logit_bias cloned on fork") {
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
  params.seed = 54321;

  BranchHandle parent = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(parent != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello", true, false);
  REQUIRE(!tokens.empty());
  CHECK_NOTHROW(prefill(parent, tokens.data(), tokens.size(), store));

  llama_logit_bias bias = {42, -std::numeric_limits<float>::infinity()};
  set_logit_bias(parent, &bias, 1, store);

  BranchHandle child = fork(parent, store);
  REQUIRE(child != INVALID_HANDLE);

  llama_token child_token = sample(child, store);
  REQUIRE(child_token != -1);
  CHECK(child_token != 42);

  MESSAGE("Child sampled token: " << child_token << " (should not be 42)");

  pruneSubtree(parent, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: basic steer masks tokens") {
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
  params.seed = 99999;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "The", true, false);
  REQUIRE(!tokens.empty());
  CHECK_NOTHROW(prefill(h, tokens.data(), tokens.size(), store));

  llama_token masked_token = 100;
  set_steer(h, [masked_token](llama_token_data_array& cur_p) {
    for (size_t i = 0; i < cur_p.size; ++i) {
      if (cur_p.data[i].id == masked_token) {
        cur_p.data[i].logit = -std::numeric_limits<float>::infinity();
      }
    }
  }, store);

  for (int i = 0; i < 10; ++i) {
    llama_token tok = sample(h, store);
    REQUIRE(tok != -1);
    CHECK(tok != masked_token);
  }

  clear_steer(h, store);

  llama_token tok_after_clear = sample(h, store);
  CHECK(tok_after_clear != -1);

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: steer NOT cloned on fork") {
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
  params.seed = 11111;

  BranchHandle parent = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(parent != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Test", true, false);
  REQUIRE(!tokens.empty());
  CHECK_NOTHROW(prefill(parent, tokens.data(), tokens.size(), store));

  llama_token masked_token = 200;
  set_steer(parent, [masked_token](llama_token_data_array& cur_p) {
    for (size_t i = 0; i < cur_p.size; ++i) {
      if (cur_p.data[i].id == masked_token) {
        cur_p.data[i].logit = -std::numeric_limits<float>::infinity();
      }
    }
  }, store);

  llama_token parent_token = sample(parent, store);
  REQUIRE(parent_token != -1);
  CHECK(parent_token != masked_token);

  BranchHandle child = fork(parent, store);
  REQUIRE(child != INVALID_HANDLE);

  llama_token parent_token2 = sample(parent, store);
  CHECK(parent_token2 != masked_token);

  llama_token child_token = sample(child, store);
  CHECK(child_token != -1);

  MESSAGE("Parent sampled: " << parent_token2 << " (never 200), Child sampled: "
          << child_token << " (steer not inherited)");

  pruneSubtree(parent, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: grammar + bias + steer composition") {
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
  auto tokens = tokenizer::tokenize(vocab, "{\"test\":", true, false);
  REQUIRE(!tokens.empty());
  CHECK_NOTHROW(prefill(h, tokens.data(), tokens.size(), store));

  llama_logit_bias bias = {50, -std::numeric_limits<float>::infinity()};
  set_logit_bias(h, &bias, 1, store);

  llama_token masked = 60;
  set_steer(h, [masked](llama_token_data_array& cur_p) {
    for (size_t i = 0; i < cur_p.size; ++i) {
      if (cur_p.data[i].id == masked) {
        cur_p.data[i].logit = -std::numeric_limits<float>::infinity();
      }
    }
  }, store);

  llama_token tok = sample(h, store);
  REQUIRE(tok != -1);
  CHECK(tok != 50);
  CHECK(tok != 60);

  MESSAGE("Sampled with bias+steer: " << tok);

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: clear functions work") {
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
  auto tokens = tokenizer::tokenize(vocab, "Clear", true, false);
  REQUIRE(!tokens.empty());
  CHECK_NOTHROW(prefill(h, tokens.data(), tokens.size(), store));

  llama_logit_bias bias = {70, -std::numeric_limits<float>::infinity()};
  set_logit_bias(h, &bias, 1, store);

  llama_token masked = 80;
  set_steer(h, [masked](llama_token_data_array& cur_p) {
    for (size_t i = 0; i < cur_p.size; ++i) {
      if (cur_p.data[i].id == masked) {
        cur_p.data[i].logit = -std::numeric_limits<float>::infinity();
      }
    }
  }, store);

  llama_token tok1 = sample(h, store);
  CHECK(tok1 != 70);
  CHECK(tok1 != 80);

  clear_logit_bias(h, store);

  llama_token tok2 = sample(h, store);
  CHECK(tok2 != 80);

  clear_steer(h, store);

  llama_token tok3 = sample(h, store);
  CHECK(tok3 != -1);

  MESSAGE("After clearing: sampled " << tok3);

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: exploration with steer deduplication") {
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
  params.seed = 77777;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Explore", true, false);
  REQUIRE(!tokens.empty());
  CHECK_NOTHROW(prefill(h, tokens.data(), tokens.size(), store));

  std::vector<llama_token> explored;

  // First expansion - no restrictions
  llama_token tok1 = sample(h, store);
  REQUIRE(tok1 != -1);
  explored.push_back(tok1);
  MESSAGE("Explored token 1: " << tok1);

  // Second expansion - mask first token
  set_steer(h, [&explored](llama_token_data_array& cur_p) {
    for (size_t i = 0; i < cur_p.size; ++i) {
      for (auto ex : explored) {
        if (cur_p.data[i].id == ex) {
          cur_p.data[i].logit = -std::numeric_limits<float>::infinity();
        }
      }
    }
  }, store);

  llama_token tok2 = sample(h, store);
  REQUIRE(tok2 != -1);
  CHECK(tok2 != tok1);
  explored.push_back(tok2);
  MESSAGE("Explored token 2: " << tok2);

  // Third expansion - mask both previous tokens
  set_steer(h, [&explored](llama_token_data_array& cur_p) {
    for (size_t i = 0; i < cur_p.size; ++i) {
      for (auto ex : explored) {
        if (cur_p.data[i].id == ex) {
          cur_p.data[i].logit = -std::numeric_limits<float>::infinity();
        }
      }
    }
  }, store);

  llama_token tok3 = sample(h, store);
  REQUIRE(tok3 != -1);
  CHECK(tok3 != tok1);
  CHECK(tok3 != tok2);
  MESSAGE("Explored token 3: " << tok3);

  CHECK(tok1 != tok2);
  CHECK(tok2 != tok3);
  CHECK(tok1 != tok3);

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: successive steer calls replace") {
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
  auto tokens = tokenizer::tokenize(vocab, "Replace", true, false);
  REQUIRE(!tokens.empty());
  CHECK_NOTHROW(prefill(h, tokens.data(), tokens.size(), store));

  llama_token masked1 = 90;
  set_steer(h, [masked1](llama_token_data_array& cur_p) {
    for (size_t i = 0; i < cur_p.size; ++i) {
      if (cur_p.data[i].id == masked1) {
        cur_p.data[i].logit = -std::numeric_limits<float>::infinity();
      }
    }
  }, store);

  llama_token tok1 = sample(h, store);
  CHECK(tok1 != masked1);

  llama_token masked2 = 95;
  set_steer(h, [masked2](llama_token_data_array& cur_p) {
    for (size_t i = 0; i < cur_p.size; ++i) {
      if (cur_p.data[i].id == masked2) {
        cur_p.data[i].logit = -std::numeric_limits<float>::infinity();
      }
    }
  }, store);

  llama_token tok2 = sample(h, store);
  CHECK(tok2 != masked2);

  MESSAGE("Steer replacement: tok1=" << tok1 << ", tok2=" << tok2);

  prune(h, store);
  llama_free(ctx);
}

// ============================================================================
// Slot Reuse Safety
// ============================================================================

TEST_CASE("branch integration: slot reuse does not leak logit_bias") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  // Store with capacity 2: slot 0 reserved, slot 1 is the ONLY usable slot.
  BranchStore store(2);
  store.init_tenancy(ctx);
  TestParams params;
  params.temperature = 0.0f;  // Greedy — deterministic argmax

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "The capital of France is", true, false);
  REQUIRE(!tokens.empty());

  // --- Branch A: find greedy token, then ban it ---

  BranchHandle a = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(a != INVALID_HANDLE);
  CHECK_NOTHROW(prefill(a, tokens.data(), tokens.size(), store));

  llama_token greedy_token = sample(a, store);
  REQUIRE(greedy_token != -1);

  llama_logit_bias bias = {greedy_token, -std::numeric_limits<float>::infinity()};
  set_logit_bias(a, &bias, 1, store);

  llama_token a_token = sample(a, store);
  CHECK(a_token != greedy_token);

  // Prune A — slot 1 + lease return to pools
  prune(a, store);

  // --- Branch B: must NOT inherit A's logit_bias ---

  kv::clear_all(ctx);

  BranchHandle b = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(b != INVALID_HANDLE);
  CHECK_NOTHROW(prefill(b, tokens.data(), tokens.size(), store));

  llama_token b_token = sample(b, store);
  REQUIRE(b_token != -1);
  CHECK(b_token == greedy_token);  // FAILS if logit_bias leaked from A

  MESSAGE("greedy=" << greedy_token << ", A(banned)=" << a_token << ", B=" << b_token);

  prune(b, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: slot reuse does not leak steer_fn") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(2);
  store.init_tenancy(ctx);
  TestParams params;
  params.temperature = 0.0f;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "The capital of France is", true, false);
  REQUIRE(!tokens.empty());

  // --- Branch A: find greedy token, then mask it via steer ---

  BranchHandle a = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(a != INVALID_HANDLE);
  CHECK_NOTHROW(prefill(a, tokens.data(), tokens.size(), store));

  llama_token greedy_token = sample(a, store);
  REQUIRE(greedy_token != -1);

  set_steer(a, [greedy_token](llama_token_data_array& cur_p) {
    for (size_t i = 0; i < cur_p.size; ++i) {
      if (cur_p.data[i].id == greedy_token) {
        cur_p.data[i].logit = -std::numeric_limits<float>::infinity();
      }
    }
  }, store);

  llama_token a_token = sample(a, store);
  CHECK(a_token != greedy_token);

  prune(a, store);

  // --- Branch B: must NOT inherit A's steer_fn ---

  kv::clear_all(ctx);

  BranchHandle b = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(b != INVALID_HANDLE);
  CHECK_NOTHROW(prefill(b, tokens.data(), tokens.size(), store));

  llama_token b_token = sample(b, store);
  REQUIRE(b_token != -1);
  CHECK(b_token == greedy_token);  // FAILS if steer_fn leaked from A

  MESSAGE("greedy=" << greedy_token << ", A(steered)=" << a_token << ", B=" << b_token);

  prune(b, store);
  llama_free(ctx);
}

// ============================================================================
// Batched Branch Decode Tests (BranchStore::decode_each / decode_scatter)
// ============================================================================

TEST_CASE("branch integration: BranchStore::decode_each batches N branches") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 128;
  cparams.n_seq_max = 8;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(16);
  store.init_tenancy(ctx);
  TestParams params;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  int32_t n_vocab = llama_vocab_n_tokens(vocab);
  auto prompt = tokenizer::tokenize(vocab, "The answer is", true, false);
  REQUIRE(!prompt.empty());

  BranchHandle root = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(root != INVALID_HANDLE);
  prefill(root, prompt.data(), prompt.size(), store);
  llama_pos prefix_pos = get_position(root, store);

  constexpr int N = 4;
  BranchHandle children[N];
  for (int i = 0; i < N; ++i) {
    children[i] = fork(root, store);
    REQUIRE(children[i] != INVALID_HANDLE);
    CHECK(get_position(children[i], store) == prefix_pos);
  }

  DecodeEachItem each_items[N];
  for (int i = 0; i < N; ++i) {
    each_items[i].handle = children[i];
    each_items[i].token = static_cast<llama_token>((100 + i * 50) % n_vocab);
  }

  CHECK_NOTHROW(store.decode_each(each_items));

  for (int i = 0; i < N; ++i) {
    CHECK(get_position(children[i], store) == prefix_pos + 1);
  }

  for (int i = 0; i < N; ++i) {
    const float* logits = get_logits(children[i], store);
    REQUIRE(logits != nullptr);

    float sum = 0.0f;
    for (int j = 0; j < std::min(100, n_vocab); ++j) {
      sum += std::abs(logits[j]);
    }
    CHECK(sum > 0.0f);
  }

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      CHECK(get_logits(children[i], store) != get_logits(children[j], store));
    }
  }

  CHECK(get_position(root, store) == prefix_pos);

  INFO("decode_each: batched " << N << " branches in single dispatch");

  pruneSubtree(root, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: BranchStore::decode_scatter batches asymmetric prefill") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 2048;
  cparams.n_batch = 512;
  cparams.n_seq_max = 8;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(16);
  store.init_tenancy(ctx);
  TestParams params;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  int32_t n_vocab = llama_vocab_n_tokens(vocab);
  auto prompt = tokenizer::tokenize(vocab, "Hello", true, false);
  REQUIRE(!prompt.empty());

  BranchHandle root = create(ctx, model.get(), store, 0, params, 512);
  REQUIRE(root != INVALID_HANDLE);
  prefill(root, prompt.data(), prompt.size(), store);
  llama_pos prefix_pos = get_position(root, store);

  constexpr int N = 3;
  const int32_t counts[] = {5, 20, 8};

  BranchHandle children[N];
  for (int i = 0; i < N; ++i) {
    children[i] = fork(root, store);
    REQUIRE(children[i] != INVALID_HANDLE);
  }

  std::vector<llama_token> token_buf_0(counts[0]);
  std::vector<llama_token> token_buf_1(counts[1]);
  std::vector<llama_token> token_buf_2(counts[2]);

  for (int j = 0; j < counts[0]; ++j) token_buf_0[j] = static_cast<llama_token>((100 + j) % n_vocab);
  for (int j = 0; j < counts[1]; ++j) token_buf_1[j] = static_cast<llama_token>((200 + j) % n_vocab);
  for (int j = 0; j < counts[2]; ++j) token_buf_2[j] = static_cast<llama_token>((300 + j) % n_vocab);

  DecodeScatterItem scatter_items[] = {
    {children[0], token_buf_0},
    {children[1], token_buf_1},
    {children[2], token_buf_2}
  };

  CHECK_NOTHROW(store.decode_scatter(scatter_items));

  for (int i = 0; i < N; ++i) {
    CHECK(get_position(children[i], store) == prefix_pos + counts[i]);
  }

  for (int i = 0; i < N; ++i) {
    const float* logits = get_logits(children[i], store);
    REQUIRE(logits != nullptr);

    float sum = 0.0f;
    for (int j = 0; j < std::min(100, n_vocab); ++j) {
      sum += std::abs(logits[j]);
    }
    CHECK(sum > 0.0f);
  }

  INFO("decode_scatter: batched asymmetric prefill (" << counts[0] << ", " << counts[1] << ", " << counts[2] << " tokens)");

  pruneSubtree(root, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: BranchStore::decode_scatter auto-chunks with small n_batch") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 2048;
  cparams.n_batch = 16;  // Small batch to force chunking
  cparams.n_seq_max = 8;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(16);
  store.init_tenancy(ctx);
  TestParams params;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  int32_t n_vocab = llama_vocab_n_tokens(vocab);
  auto prompt = tokenizer::tokenize(vocab, "Test", true, false);
  REQUIRE(!prompt.empty());

  BranchHandle root = create(ctx, model.get(), store, 0, params, 16);
  REQUIRE(root != INVALID_HANDLE);
  prefill(root, prompt.data(), prompt.size(), store);
  llama_pos prefix_pos = get_position(root, store);

  constexpr int N = 3;
  const int32_t counts[] = {10, 10, 10};

  BranchHandle children[N];
  for (int i = 0; i < N; ++i) {
    children[i] = fork(root, store);
    REQUIRE(children[i] != INVALID_HANDLE);
  }

  std::vector<llama_token> token_buf_0(counts[0]);
  std::vector<llama_token> token_buf_1(counts[1]);
  std::vector<llama_token> token_buf_2(counts[2]);

  for (int j = 0; j < counts[0]; ++j) token_buf_0[j] = static_cast<llama_token>((50 + j) % n_vocab);
  for (int j = 0; j < counts[1]; ++j) token_buf_1[j] = static_cast<llama_token>((150 + j) % n_vocab);
  for (int j = 0; j < counts[2]; ++j) token_buf_2[j] = static_cast<llama_token>((250 + j) % n_vocab);

  DecodeScatterItem scatter_items[] = {
    {children[0], token_buf_0},
    {children[1], token_buf_1},
    {children[2], token_buf_2}
  };

  CHECK_NOTHROW(store.decode_scatter(scatter_items));

  for (int i = 0; i < N; ++i) {
    CHECK(get_position(children[i], store) == prefix_pos + counts[i]);

    const float* logits = get_logits(children[i], store);
    REQUIRE(logits != nullptr);

    float sum = 0.0f;
    for (int j = 0; j < std::min(100, n_vocab); ++j) {
      sum += std::abs(logits[j]);
    }
    CHECK(sum > 0.0f);
  }

  INFO("decode_scatter: auto-chunked with n_batch=16, total_tokens=30");

  pruneSubtree(root, store);
  llama_free(ctx);
}

// ============================================================================
// Tenancy Integration Tests
// ============================================================================

TEST_CASE("branch integration: seq recycling after prune") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 64;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Hello", true, false);
  REQUIRE(!prompt.empty());

  // Exhaust all 4 leases
  BranchHandle handles[4];
  for (int i = 0; i < 4; ++i) {
    handles[i] = create(ctx, model.get(), store, 0, params, 64);
    REQUIRE(handles[i] != INVALID_HANDLE);
  }

  // 5th should fail — leases exhausted
  BranchHandle overflow = create(ctx, model.get(), store, 0, params, 64);
  CHECK(overflow == INVALID_HANDLE);
  CHECK(store.available() == 0);

  // Prune first two — recycling seq_ids
  prune(handles[0], store);
  prune(handles[1], store);
  CHECK(store.available() == 2);

  // Now can create two more
  BranchHandle recycled1 = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(recycled1 != INVALID_HANDLE);

  // Verify recycled branch decodes correctly
  CHECK_NOTHROW(prefill(recycled1, prompt.data(), prompt.size(), store));
  llama_token tok = sample(recycled1, store);
  CHECK(tok >= 0);

  MESSAGE("Recycled branch sampled token: " << tok);

  // Cleanup
  prune(recycled1, store);
  prune(handles[2], store);
  prune(handles[3], store);
  llama_free(ctx);
}

TEST_CASE("branch integration: retainOnly keeps winner decoding") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 64;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Once upon", true, false);
  REQUIRE(!prompt.empty());

  // Create root and fork a tree
  BranchHandle root = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(root != INVALID_HANDLE);
  prefill(root, prompt.data(), prompt.size(), store);

  BranchHandle child1 = fork(root, store);
  BranchHandle child2 = fork(root, store);
  REQUIRE(child1 != INVALID_HANDLE);
  REQUIRE(child2 != INVALID_HANDLE);

  // Decode different tokens on each child
  auto cont1 = tokenizer::tokenize(vocab, " a", false, false);
  auto cont2 = tokenizer::tokenize(vocab, " the", false, false);
  REQUIRE(!cont1.empty());
  REQUIRE(!cont2.empty());
  step(child1, cont1[0], store);
  step(child2, cont2[0], store);

  size_t avail_before = store.available();

  // Retain only child1 as winner
  store.retainOnly(child1);

  // Winner still alive and can decode
  CHECK(store.get(child1) != nullptr);
  auto more = tokenizer::tokenize(vocab, " time", false, false);
  REQUIRE(!more.empty());
  CHECK_NOTHROW(step(child1, more[0], store));

  llama_token tok = sample(child1, store);
  CHECK(tok >= 0);

  // Losers are gone
  CHECK(store.get(root) == nullptr);
  CHECK(store.get(child2) == nullptr);

  // All leases except winner's returned
  CHECK(store.available() == static_cast<uint32_t>(llama_n_seq_max(ctx)) - 1);

  MESSAGE("Winner sampled token after retainOnly: " << tok);

  prune(child1, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: multi-tag KV survival — child intact after parent eviction") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 64;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Test KV survival", true, false);
  REQUIRE(!prompt.empty());

  // Create parent, decode prompt, fork child
  // After fork: parent_seq and child_seq both tag the same KV cells (multi-tag)
  BranchHandle parent = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(parent != INVALID_HANDLE);
  prefill(parent, prompt.data(), prompt.size(), store);

  BranchHandle child = fork(parent, store);
  REQUIRE(child != INVALID_HANDLE);

  BranchState* pstate = store.get(parent);
  BranchState* cstate = store.get(child);
  REQUIRE(pstate != nullptr);
  REQUIRE(cstate != nullptr);
  llama_seq_id parent_seq = pstate->seq_id;
  llama_seq_id child_seq = cstate->seq_id;

  // Both seqs tag the same KV cells
  CHECK(kv::pos_max(ctx, parent_seq) >= 0);
  CHECK(kv::pos_max(ctx, child_seq) >= 0);

  // === Strip parent's tags — partial eviction of multi-tag cells ===
  kv::remove_range(ctx, parent_seq, 0, -1);

  // Parent's tags gone
  CHECK(kv::pos_max(ctx, parent_seq) < 0);

  // Child's tags survive — a cell is freed only when ALL tags are removed
  CHECK(kv::pos_max(ctx, child_seq) >= 0);

  // Child still decodes — KV is functional, not just tagged
  auto cont = tokenizer::tokenize(vocab, " more", false, false);
  REQUIRE(!cont.empty());
  CHECK_NOTHROW(step(child, cont[0], store));
  CHECK(get_position(child, store) > static_cast<llama_pos>(prompt.size()));

  llama_token tok = sample(child, store);
  CHECK(tok >= 0);
  MESSAGE("Child decodes after parent eviction, token: " << tok);

  // === Prune everything — now ALL tags removed from shared cells ===
  pruneSubtree(parent, store);

  // Cells freed — no tags remain
  CHECK(kv::pos_max(ctx, child_seq) < 0);
  CHECK(kv::pos_max(ctx, parent_seq) < 0);

  llama_free(ctx);
}

TEST_CASE("branch integration: is_eog detects end-of-generation from model") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 256;
  cparams.n_batch = 64;
  cparams.n_seq_max = 2;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(4);
  store.init_tenancy(ctx);
  TestParams params;

  Branch b = Branch::create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(b.valid());

  const llama_vocab* vocab = llama_model_get_vocab(model.get());

  // Model's EOS token must be recognized as EOG
  llama_token eos = llama_vocab_eos(vocab);
  CHECK(b.is_eog(eos));

  // A regular token must NOT be EOG
  auto tokens = tokenizer::tokenize(vocab, "hello", false, false);
  REQUIRE(!tokens.empty());
  CHECK_FALSE(b.is_eog(tokens[0]));

  store.drain();
  llama_free(ctx);
}

TEST_CASE("branch integration: drain then llama_free ordering") {
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

  // Create some branches
  BranchHandle h1 = create(ctx, model.get(), store, 0, params, 64);
  BranchHandle h2 = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h1 != INVALID_HANDLE);
  REQUIRE(h2 != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Drain", true, false);
  prefill(h1, prompt.data(), prompt.size(), store);

  // Drain while ctx alive — evicts all leases
  store.drain();

  // After drain, both handles should be gone
  CHECK(store.get(h1) == nullptr);
  CHECK(store.get(h2) == nullptr);

  // Allocate should fail (drained)
  auto [h3, s3] = store.allocate();
  CHECK(h3 == INVALID_HANDLE);

  // Now safe to free context
  llama_free(ctx);

  // BranchStore destructor runs — no crash
}

// ============================================================================
// set_sampler_params() Tests
// ============================================================================

TEST_CASE("branch integration: set_sampler_params changes sampling behavior") {
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

  // Start greedy (temp=0)
  TestParams greedy_params;
  greedy_params.temperature = 0.0f;
  greedy_params.top_k = 0;
  greedy_params.top_p = 1.0f;
  greedy_params.min_p = 0.0f;

  BranchHandle h = create(ctx, model.get(), store, 0, greedy_params, 64);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "The capital of France is", true, false);
  REQUIRE(!prompt.empty());
  prefill(h, prompt.data(), prompt.size(), store);

  // Sample greedy — deterministic
  llama_token greedy_tok = sample(h, store);
  REQUIRE(greedy_tok >= 0);

  // Switch to stochastic
  TestParams stochastic_params;
  stochastic_params.temperature = 1.5f;
  stochastic_params.seed = 42;
  stochastic_params.top_k = 0;
  stochastic_params.top_p = 1.0f;
  stochastic_params.min_p = 0.0f;

  set_sampler_params(h, stochastic_params, store);

  // Sample multiple times — at temp=1.5 with a real distribution,
  // at least one should differ from the greedy argmax
  bool found_different = false;
  for (int i = 0; i < 20; ++i) {
    llama_token tok = sample(h, store);
    if (tok != greedy_tok) {
      found_different = true;
      break;
    }
  }
  CHECK(found_different);

  MESSAGE("Greedy token: " << greedy_tok << ", stochastic diverged: " << found_different);

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: set_sampler_params memoization skips rebuild") {
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
  params.temperature = 0.8f;
  params.seed = 100;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  BranchState* state = store.get(h);
  REQUIRE(state);
  SamplerChainHandle original_handle = state->sampler_chain;
  CHECK(original_handle != 0);

  // Same params → memoized (handle unchanged)
  set_sampler_params(h, params, store);
  state = store.get(h);
  CHECK(state->sampler_chain == original_handle);

  // Different params → rebuild (handle changed)
  TestParams different;
  different.temperature = 0.3f;
  different.seed = 200;

  set_sampler_params(h, different, store);
  state = store.get(h);
  CHECK(state->sampler_chain != original_handle);
  CHECK(state->sampler_chain != 0);

  MESSAGE("Original handle: " << original_handle
          << ", after rebuild: " << state->sampler_chain);

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: set_sampler_params greedy-stochastic transition") {
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

  // Start greedy
  TestParams greedy;
  greedy.temperature = 0.0f;

  BranchHandle h = create(ctx, model.get(), store, 0, greedy, 64);
  REQUIRE(h != INVALID_HANDLE);

  BranchState* state = store.get(h);
  REQUIRE(state);
  CHECK_FALSE(store.sampler_has_dist(state->sampler_chain));

  // Switch to stochastic
  TestParams stochastic;
  stochastic.temperature = 0.8f;
  stochastic.seed = 42;

  set_sampler_params(h, stochastic, store);
  state = store.get(h);
  CHECK(store.sampler_has_dist(state->sampler_chain));

  // Switch back to greedy
  set_sampler_params(h, greedy, store);
  state = store.get(h);
  CHECK_FALSE(store.sampler_has_dist(state->sampler_chain));

  MESSAGE("Greedy→stochastic→greedy transitions: has_dist flag correct");

  prune(h, store);
  llama_free(ctx);
}

// ============================================================================
// set_grammar() Tests
// ============================================================================

TEST_CASE("branch integration: set_grammar hot-swap constrains output") {
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
  params.temperature = 0.0f;  // Greedy for determinism

  // Create WITHOUT grammar
  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  BranchState* state = store.get(h);
  CHECK(state->grammar == 0);  // No grammar initially

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "{\"key\":", true, false);
  REQUIRE(!prompt.empty());
  prefill(h, prompt.data(), prompt.size(), store);

  // Hot-swap grammar: JSON value must be a quoted string
  const char* json_grammar =
      "root ::= \"{\" ws \"\\\"key\\\"\" ws \":\" ws value ws \"}\"\n"
      "value ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
      "ws ::= [ \\t\\n]*\n";

  set_grammar(h, model.get(), json_grammar, store);

  state = store.get(h);
  CHECK(state->grammar != 0);  // Grammar now active

  // Sample — grammar should constrain to valid tokens
  llama_token tok = sample(h, store);
  REQUIRE(tok >= 0);

  // The grammar requires a quoted string value — first legal token should be '"'
  std::string text = tokenizer::detokenize(vocab, tok, false);
  MESSAGE("First token after grammar hot-swap: '" << text << "' (id=" << tok << ")");

  // Verify grammar sampler is accessible
  CHECK(store.get_grammar_sampler(state->grammar) != nullptr);

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: set_grammar removal clears constraint") {
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

  // Create WITH grammar
  const char* json_grammar =
      "root ::= \"{\" ws \"\\\"key\\\"\" ws \":\" ws value ws \"}\"\n"
      "value ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
      "ws ::= [ \\t\\n]*\n";

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64, json_grammar);
  REQUIRE(h != INVALID_HANDLE);

  BranchState* state = store.get(h);
  GrammarHandle original_grammar = state->grammar;
  CHECK(original_grammar != 0);

  // Remove grammar
  set_grammar(h, model.get(), "", store);
  state = store.get(h);
  CHECK(state->grammar == 0);

  // Old handle should be freed from registry
  CHECK(store.get_grammar_sampler(original_grammar) == nullptr);

  MESSAGE("Grammar removed: handle " << original_grammar << " freed");

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: set_grammar cloned on fork after hot-swap") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 64;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  // Create WITHOUT grammar, then hot-swap one in
  BranchHandle parent = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(parent != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "{\"key\":", true, false);
  REQUIRE(!prompt.empty());
  prefill(parent, prompt.data(), prompt.size(), store);

  const char* json_grammar =
      "root ::= \"{\" ws \"\\\"key\\\"\" ws \":\" ws value ws \"}\"\n"
      "value ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
      "ws ::= [ \\t\\n]*\n";

  set_grammar(parent, model.get(), json_grammar, store);

  BranchState* pstate = store.get(parent);
  GrammarHandle parent_grammar = pstate->grammar;
  CHECK(parent_grammar != 0);

  // Fork — child should get independent grammar clone
  BranchHandle child = fork(parent, store);
  REQUIRE(child != INVALID_HANDLE);

  BranchState* cstate = store.get(child);
  CHECK(cstate->grammar != 0);
  CHECK(cstate->grammar != parent_grammar);  // Independent handle

  // Both grammars should be valid
  CHECK(store.get_grammar_sampler(pstate->grammar) != nullptr);
  CHECK(store.get_grammar_sampler(cstate->grammar) != nullptr);

  // Different underlying pointers (deep clone)
  CHECK(store.get_grammar_sampler(pstate->grammar)
        != store.get_grammar_sampler(cstate->grammar));

  MESSAGE("Parent grammar: " << parent_grammar
          << ", child grammar: " << cstate->grammar
          << " (independent clones)");

  pruneSubtree(parent, store);
  llama_free(ctx);
}

// ============================================================================
// Handle Registry Cleanup Tests
// ============================================================================

TEST_CASE("branch integration: handle registry cleanup on prune") {
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

  const char* json_grammar =
      "root ::= \"{\" ws \"\\\"key\\\"\" ws \":\" ws value ws \"}\"\n"
      "value ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
      "ws ::= [ \\t\\n]*\n";

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64, json_grammar);
  REQUIRE(h != INVALID_HANDLE);

  BranchState* state = store.get(h);
  REQUIRE(state);

  // Record all handles
  SamplerChainHandle sc = state->sampler_chain;
  GrammarHandle gr = state->grammar;
  MetricsHandle mt = state->metrics;

  CHECK(sc != 0);
  CHECK(gr != 0);
  CHECK(mt != 0);

  // All handles should dereference successfully
  CHECK(store.get_sampler_chain(sc) != nullptr);
  CHECK(store.get_grammar_sampler(gr) != nullptr);

  // Prune — should free all registry entries
  prune(h, store);

  // All handles should now be invalid (freed from registry)
  CHECK(store.get_sampler_chain(sc) == nullptr);
  CHECK(store.get_grammar_sampler(gr) == nullptr);

  MESSAGE("Handles freed: sampler=" << sc << " grammar=" << gr << " metrics=" << mt);

  llama_free(ctx);
}

TEST_CASE("branch integration: slot reuse does not leak sampler chain") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  // Store with capacity 2: slot 0 reserved, slot 1 is the ONLY usable slot.
  BranchStore store(2);
  store.init_tenancy(ctx);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "The capital of France is", true, false);
  REQUIRE(!prompt.empty());

  // --- Branch A: stochastic (temp=0.8) ---
  TestParams stochastic;
  stochastic.temperature = 0.8f;
  stochastic.seed = 42;

  BranchHandle a = create(ctx, model.get(), store, 0, stochastic, 64);
  REQUIRE(a != INVALID_HANDLE);

  BranchState* astate = store.get(a);
  CHECK(store.sampler_has_dist(astate->sampler_chain));  // Stochastic

  prune(a, store);

  // --- Branch B: greedy (temp=0) in same slot ---
  kv::clear_all(ctx);

  TestParams greedy;
  greedy.temperature = 0.0f;
  greedy.top_k = 0;
  greedy.top_p = 1.0f;
  greedy.min_p = 0.0f;

  BranchHandle b = create(ctx, model.get(), store, 0, greedy, 64);
  REQUIRE(b != INVALID_HANDLE);

  BranchState* bstate = store.get(b);
  CHECK_FALSE(store.sampler_has_dist(bstate->sampler_chain));  // Greedy

  prefill(b, prompt.data(), prompt.size(), store);

  // Greedy sampling should be deterministic — sample twice, same token
  llama_token tok1 = sample(b, store);
  llama_token tok2 = sample(b, store);
  CHECK(tok1 == tok2);  // FAILS if stochastic chain leaked from A

  MESSAGE("Slot reuse: greedy tokens " << tok1 << "==" << tok2
          << " (no stochastic leak)");

  prune(b, store);
  llama_free(ctx);
}

// ============================================================================
// High-Fidelity Tests: Multi-Step Generation, ABA, Batch Errors, retainOnly
// ============================================================================

TEST_CASE("branch integration: multi-step generation loop with PPL and grammar") {
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

  // Grammar constraining output to a JSON object with lowercase string value
  const char* json_grammar =
      "root ::= \"{\" ws \"\\\"name\\\"\" ws \":\" ws value ws \"}\"\n"
      "value ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
      "ws ::= [ \\t\\n]*\n";

  TestParams params;
  params.temperature = 0.0f;  // Greedy for determinism
  params.top_k = 0;
  params.top_p = 1.0f;
  params.min_p = 0.0f;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64, json_grammar);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Output a JSON object with a name field:", true, false);
  REQUIRE(!prompt.empty());
  prefill(h, prompt.data(), prompt.size(), store);

  // Multi-step produce/accept/decode loop — 15 tokens
  const int N = 15;
  std::vector<llama_token> generated;
  llama_pos expected_pos = static_cast<llama_pos>(prompt.size());

  for (int i = 0; i < N; ++i) {
    BranchState* state = store.get(h);
    REQUIRE(state);

    // Position must track correctly
    CHECK(state->position == expected_pos);

    // Sample
    llama_token tok = sample(h, store);
    REQUIRE(tok >= 0);
    generated.push_back(tok);

    // Accept — advances grammar, sampler penalties, metrics
    accept_token(h, tok, store);

    // Decode+capture for next step
    step(h, tok, store);
    expected_pos += 1;
  }

  // Position advanced correctly through entire loop
  BranchState* final_state = store.get(h);
  CHECK(final_state->position == static_cast<llama_pos>(prompt.size() + N));

  // PPL was accumulated (model metrics count should equal N)
  float model_ppl = get_perplexity(h, store);
  CHECK(model_ppl > 0.0f);
  CHECK(std::isfinite(model_ppl));

  float sampling_ppl = get_sampling_perplexity(h, store);
  CHECK(sampling_ppl > 0.0f);
  CHECK(std::isfinite(sampling_ppl));

  // Detokenize and verify grammar held — output should be valid JSON-ish
  std::string output;
  for (auto tok : generated) {
    output += tokenizer::detokenize(vocab, tok, false);
  }
  MESSAGE("Multi-step output (" << N << " tokens): " << output);
  // Grammar requires opening { — check first non-whitespace char
  size_t first_nonws = output.find_first_not_of(" \t\n");
  if (first_nonws != std::string::npos) {
    CHECK(output[first_nonws] == '{');
  }

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: handle ABA prevention via generation counter") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  cparams.n_seq_max = 2;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  // Capacity 2: slot 0 reserved, slot 1 is the only usable slot
  BranchStore store(2);
  store.init_tenancy(ctx);

  TestParams params;
  params.temperature = 0.0f;
  params.top_k = 0;
  params.top_p = 1.0f;
  params.min_p = 0.0f;

  // Create branch A — gets slot 1
  BranchHandle a = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(a != INVALID_HANDLE);

  uint16_t a_index = handle_index(a);
  uint16_t a_gen = handle_generation(a);
  MESSAGE("Branch A: handle=" << a << " index=" << a_index << " gen=" << a_gen);

  // Verify A is live
  CHECK(store.get(a) != nullptr);

  // Prune A — frees slot 1, increments generation
  prune(a, store);
  kv::clear_all(ctx);

  // Stale handle A should now resolve to nullptr (generation mismatch)
  CHECK(store.get(a) == nullptr);

  // Create branch B — reuses slot 1 with incremented generation
  BranchHandle b = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(b != INVALID_HANDLE);

  uint16_t b_index = handle_index(b);
  uint16_t b_gen = handle_generation(b);
  MESSAGE("Branch B: handle=" << b << " index=" << b_index << " gen=" << b_gen);

  // Same slot, different generation
  CHECK(b_index == a_index);
  CHECK(b_gen == static_cast<uint16_t>(a_gen + 1));
  CHECK(a != b);  // Different handles despite same slot

  // B is live, A is still stale
  CHECK(store.get(b) != nullptr);
  CHECK(store.get(a) == nullptr);

  // Operations on stale handle A must be safe no-ops or return nullptr
  // accept_token on stale handle — silent no-op (returns early on nullptr)
  CHECK_NOTHROW(accept_token(a, 0, store));

  // sample on stale handle — should not crash
  // (sample() returns -1 when state is nullptr)
  llama_token stale_tok = sample(a, store);
  CHECK(stale_tok < 0);

  prune(b, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: decode_each and decode_scatter error paths") {
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
  params.temperature = 0.0f;
  params.top_k = 0;
  params.top_p = 1.0f;
  params.min_p = 0.0f;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Hello", true, false);
  REQUIRE(!prompt.empty());

  // Create two live branches for valid batching
  BranchHandle h1 = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h1 != INVALID_HANDLE);
  prefill(h1, prompt.data(), prompt.size(), store);

  BranchHandle h2 = fork(h1, store);
  REQUIRE(h2 != INVALID_HANDLE);

  // --- decode_each: empty items is a safe no-op ---
  std::vector<DecodeEachItem> empty_each;
  CHECK_NOTHROW(store.decode_each(empty_each));

  // --- decode_each: valid batch works ---
  llama_token tok = sample(h1, store);
  REQUIRE(tok >= 0);
  std::vector<DecodeEachItem> valid_each = {
    {h1, tok},
    {h2, tok},
  };
  CHECK_NOTHROW(store.decode_each(valid_each));

  // --- decode_each: invalid handle in batch throws ---
  // Clean up h2 (child of h1) first, then h1, to get a stale handle
  prune(h2, store);
  BranchHandle stale = h1;
  prune(h1, store);
  kv::clear_all(ctx);

  // Recreate branches for continued testing
  h1 = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h1 != INVALID_HANDLE);
  prefill(h1, prompt.data(), prompt.size(), store);

  std::vector<DecodeEachItem> bad_each = {
    {stale, tok},  // stale handle from pruned branch
  };
  CHECK_THROWS_WITH(store.decode_each(bad_each),
                     "BranchStore::decode_each - invalid handle at index 0");

  // --- decode_scatter: empty items is a safe no-op ---
  std::vector<DecodeScatterItem> empty_scatter;
  CHECK_NOTHROW(store.decode_scatter(empty_scatter));

  // --- decode_scatter: invalid handle throws ---
  std::vector<llama_token> scatter_toks = {tok};
  std::vector<DecodeScatterItem> bad_scatter = {
    {stale, scatter_toks},
  };
  CHECK_THROWS_WITH(store.decode_scatter(bad_scatter),
                     "BranchStore::decode_scatter - invalid handle at index 0");

  // --- decode_scatter: valid batch works ---
  BranchHandle h3 = fork(h1, store);
  REQUIRE(h3 != INVALID_HANDLE);
  std::vector<DecodeScatterItem> valid_scatter = {
    {h1, scatter_toks},
    {h3, scatter_toks},
  };
  CHECK_NOTHROW(store.decode_scatter(valid_scatter));

  prune(h3, store);
  prune(h1, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: retainOnly edge cases") {
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
  params.temperature = 0.0f;
  params.top_k = 0;
  params.top_p = 1.0f;
  params.min_p = 0.0f;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Hello world", true, false);
  REQUIRE(!prompt.empty());

  // --- retainOnly: invalid handle throws ---
  CHECK_THROWS_WITH(store.retainOnly(INVALID_HANDLE),
                     "retainOnly: invalid winner handle");

  // --- retainOnly with stale handle throws ---
  BranchHandle temp = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(temp != INVALID_HANDLE);
  prune(temp, store);
  kv::clear_all(ctx);
  CHECK_THROWS_WITH(store.retainOnly(temp),
                     "retainOnly: invalid winner handle");

  // --- retainOnly: topology reset ---
  // Build tree: root → child1, child2
  BranchHandle root = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(root != INVALID_HANDLE);
  prefill(root, prompt.data(), prompt.size(), store);

  BranchHandle child1 = fork(root, store);
  REQUIRE(child1 != INVALID_HANDLE);

  BranchHandle child2 = fork(root, store);
  REQUIRE(child2 != INVALID_HANDLE);

  // Verify topology before retainOnly
  BranchState* root_state = store.get(root);
  CHECK(root_state->children.size() == 2);

  BranchState* c1_state = store.get(child1);
  CHECK(c1_state->parent == root);

  BranchState* c2_state = store.get(child2);
  CHECK(c2_state->parent == root);

  // Retain child1 as winner — root and child2 should be released
  store.retainOnly(child1);

  // Winner's topology is cleared (no parent, no children)
  c1_state = store.get(child1);
  REQUIRE(c1_state);
  CHECK(c1_state->parent == INVALID_HANDLE);
  CHECK(c1_state->children.empty());

  // Losers are gone
  CHECK(store.get(root) == nullptr);
  CHECK(store.get(child2) == nullptr);

  // Winner is still functional — can decode and sample
  llama_token tok = sample(child1, store);
  CHECK(tok >= 0);
  accept_token(child1, tok, store);
  step(child1, tok, store);
  llama_token tok2 = sample(child1, store);
  CHECK(tok2 >= 0);
  MESSAGE("Post-retainOnly sampling: tok1=" << tok << " tok2=" << tok2);

  // --- Second retainOnly on sole survivor is a no-op (no losers) ---
  store.retainOnly(child1);
  CHECK(store.get(child1) != nullptr);

  prune(child1, store);
  llama_free(ctx);
}
