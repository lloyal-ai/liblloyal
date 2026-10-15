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
#include <lloyal/chat_in.hpp>
#include <lloyal/decode.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <memory>
#include <span>
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

// ============================================================================
// set_grammar_lazy() Tests
// ============================================================================

TEST_CASE("branch integration: set_grammar_lazy creates valid grammar handle") {
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
  params.temperature = 0.0f;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  BranchState* state = store.get(h);
  CHECK(state->grammar == 0);  // No grammar initially

  // Set lazy grammar — same GBNF as non-lazy tests, with a trigger pattern
  const char* json_grammar =
      "root ::= \"{\" ws \"\\\"key\\\"\" ws \":\" ws value ws \"}\"\n"
      "value ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
      "ws ::= [ \\t\\n]*\n";

  std::vector<std::string> trigger_patterns = {"\\{"};  // Trigger on opening brace
  std::vector<llama_token> trigger_tokens;  // No token triggers

  set_grammar_lazy(h, model.get(), json_grammar, trigger_patterns, trigger_tokens, store);

  state = store.get(h);
  CHECK(state->grammar != 0);  // Grammar handle allocated
  CHECK(store.get_grammar_sampler(state->grammar) != nullptr);

  MESSAGE("Lazy grammar handle: " << state->grammar);

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: set_grammar_lazy removal clears constraint") {
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

  const char* json_grammar =
      "root ::= \"{\" ws \"\\\"key\\\"\" ws \":\" ws value ws \"}\"\n"
      "value ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
      "ws ::= [ \\t\\n]*\n";

  std::vector<std::string> patterns = {"\\{"};
  std::vector<llama_token> tokens;

  set_grammar_lazy(h, model.get(), json_grammar, patterns, tokens, store);

  BranchState* state = store.get(h);
  GrammarHandle original = state->grammar;
  CHECK(original != 0);

  // Remove by setting empty grammar (uses set_grammar_lazy with empty string)
  set_grammar_lazy(h, model.get(), "", patterns, tokens, store);

  state = store.get(h);
  CHECK(state->grammar == 0);
  CHECK(store.get_grammar_sampler(original) == nullptr);

  MESSAGE("Lazy grammar removed: handle " << original << " freed");

  prune(h, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: set_grammar_lazy cloned on fork") {
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
  auto prompt = tokenizer::tokenize(vocab, "Hello", true, false);
  prefill(parent, prompt.data(), prompt.size(), store);

  const char* json_grammar =
      "root ::= \"{\" ws \"\\\"key\\\"\" ws \":\" ws value ws \"}\"\n"
      "value ::= \"\\\"\" [a-z]+ \"\\\"\"\n"
      "ws ::= [ \\t\\n]*\n";

  std::vector<std::string> patterns = {"\\{"};
  std::vector<llama_token> tokens;

  set_grammar_lazy(parent, model.get(), json_grammar, patterns, tokens, store);

  BranchState* pstate = store.get(parent);
  GrammarHandle parent_grammar = pstate->grammar;
  CHECK(parent_grammar != 0);

  // Fork — child should get independent lazy grammar clone
  BranchHandle child = fork(parent, store);
  REQUIRE(child != INVALID_HANDLE);

  BranchState* cstate = store.get(child);
  CHECK(cstate->grammar != 0);
  CHECK(cstate->grammar != parent_grammar);  // Independent handle

  // Both grammars valid and distinct sampler pointers
  CHECK(store.get_grammar_sampler(pstate->grammar) != nullptr);
  CHECK(store.get_grammar_sampler(cstate->grammar) != nullptr);
  CHECK(store.get_grammar_sampler(pstate->grammar)
        != store.get_grammar_sampler(cstate->grammar));

  MESSAGE("Parent lazy grammar: " << parent_grammar
          << ", child lazy grammar: " << cstate->grammar);

  prune(child, store);
  prune(parent, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: lazy grammar allows free generation before trigger") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 2048;
  cparams.n_batch = 512;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;
  params.temperature = 0.0f;  // Greedy for determinism
  params.top_k = 0;
  params.top_p = 1.0f;
  params.min_p = 0.0f;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());

  // Format a tool-call conversation to get grammar + triggers from the model's template
  lloyal::chat_in::FormatInputs inputs;
  inputs.messages_json = R"([{"role":"user","content":"Search for quantum computing papers"}])";
  inputs.tools_json = R"([{"type":"function","function":{"name":"search","description":"Search","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}])";

  auto fmt = lloyal::chat_in::format(model.get(), inputs);

  if (fmt.grammar.empty() || !fmt.grammar_lazy || fmt.grammar_triggers.empty()) {
    MESSAGE("[ SKIP ] Model template does not produce lazy grammar for tools");
    prune(h, store);
    llama_free(ctx);
    return;
  }

  MESSAGE("Grammar triggers: " << fmt.grammar_triggers.size());

  auto prompt = tokenizer::tokenize(vocab, fmt.prompt, true, false);
  REQUIRE(!prompt.empty());
  prefill(h, prompt.data(), prompt.size(), store);

  // Apply the lazy grammar from formatChat.
  // Mirrors the TS applyLazyGrammar fix: truncate WORD triggers to scope_start
  // (up to first \n) so the grammar activates before JSON/XML divergence.
  // Also regex-escape WORD values (matching common_sampler_init behavior).
  std::vector<std::string> trigger_patterns;
  std::vector<llama_token> trigger_tokens;
  auto regex_escape = [](const std::string& s) {
    std::string r;
    for (char c : s) {
      if (std::string(".+*?^${}()|[]\\").find(c) != std::string::npos)
        r += '\\';
      r += c;
    }
    return r;
  };
  for (const auto& t : fmt.grammar_triggers) {
    if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN) {
      trigger_tokens.push_back(t.token);
    } else if (t.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
      // Truncate to scope_start (up to first \n inclusive)
      std::string val = t.value;
      auto nl = val.find('\n');
      if (nl != std::string::npos && nl < val.size() - 1)
        val = val.substr(0, nl + 1);
      trigger_patterns.push_back(regex_escape(val));
    } else {
      trigger_patterns.push_back(t.value);
    }
  }
  set_grammar_lazy(h, model.get(), fmt.grammar.c_str(),
                   trigger_patterns, trigger_tokens, store);

  // Generate — lazy grammar should allow free text before tool call trigger
  std::string output;
  for (int i = 0; i < 200; ++i) {
    llama_token tok = sample(h, store);
    if (tokenizer::is_eog(model.get(), tok)) break;
    output += tokenizer::detokenize(vocab, tok, false);
    accept_token(h, tok, store);
    step(h, tok, store);
  }

  MESSAGE("Lazy grammar output (" << output.size() << " chars): " << output.substr(0, 300));

  // Generation must have produced output (grammar didn't block everything)
  CHECK(output.size() > 0);

  prune(h, store);
  llama_free(ctx);
}

// ============================================================================
// R5 (rerank hardening): ForkOpts{clone_logits=false} + get_logits_at
// ============================================================================

TEST_CASE("branch integration: fork ForkOpts{clone_logits=false} leaves child without logits") {
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

  // Parent has logits captured
  REQUIRE(get_logits(parent, store) != nullptr);

  // Default fork (clone_logits=true) — child has logits
  BranchHandle child_clone = fork(parent, store);
  REQUIRE(child_clone != INVALID_HANDLE);
  CHECK(get_logits(child_clone, store) != nullptr);

  // R5(a): Fork with clone_logits=false — child should NOT have logits captured
  BranchHandle child_skip = fork(parent, store, ForkOpts{ /*clone_logits=*/ false });
  REQUIRE(child_skip != INVALID_HANDLE);
  CHECK(get_logits(child_skip, store) == nullptr);

  // KV is still copied (forks share the same prefix), only the logits snapshot is skipped
  CHECK(get_position(child_skip, store) == get_position(parent, store));

  // After prefill on the skip-clone child, it captures fresh logits
  auto cont = tokenizer::tokenize(vocab, " princess", false, false);
  REQUIRE(!cont.empty());
  prefill(child_skip, cont.data(), cont.size(), store);
  CHECK(get_logits(child_skip, store) != nullptr);

  pruneSubtree(parent, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: get_logits_at bounds check + reads from logits_snapshot") {
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

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "The answer is", true, false);
  REQUIRE(!prompt.empty());
  prefill(h, prompt.data(), prompt.size(), store);

  int n_vocab = get_n_vocab(h, store);
  REQUIRE(n_vocab > 0);

  // R5(b): valid in-range reads match the full-buffer values from get_logits()
  const float* logits_full = get_logits(h, store);
  REQUIRE(logits_full != nullptr);

  const std::vector<int32_t> indices_v = {0, 1, n_vocab / 2, n_vocab - 1};
  std::vector<float> out(indices_v.size());
  CHECK_NOTHROW(get_logits_at(h,
                              std::span<const int32_t>(indices_v.data(), indices_v.size()),
                              std::span<float>(out.data(), out.size()),
                              store));
  for (size_t i = 0; i < indices_v.size(); ++i) {
    CHECK(out[i] == logits_full[indices_v[i]]);
  }

  // OOB negative index throws
  const std::vector<int32_t> oob_neg = { -1 };
  std::vector<float> out1(1);
  CHECK_THROWS_AS(
      get_logits_at(h,
                    std::span<const int32_t>(oob_neg.data(), oob_neg.size()),
                    std::span<float>(out1.data(), out1.size()),
                    store),
      std::runtime_error);

  // OOB >= n_vocab throws
  const std::vector<int32_t> oob_high = { n_vocab };
  CHECK_THROWS_AS(
      get_logits_at(h,
                    std::span<const int32_t>(oob_high.data(), oob_high.size()),
                    std::span<float>(out1.data(), out1.size()),
                    store),
      std::runtime_error);

  // Size mismatch (indices.size() != out.size()) throws
  const std::vector<int32_t> two_indices = { 0, 1 };
  std::vector<float> one_out(1);
  CHECK_THROWS_AS(
      get_logits_at(h,
                    std::span<const int32_t>(two_indices.data(), two_indices.size()),
                    std::span<float>(one_out.data(), one_out.size()),
                    store),
      std::runtime_error);

  // No-logits handle (fresh branch, no prefill) throws
  BranchHandle fresh = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(fresh != INVALID_HANDLE);
  REQUIRE(get_logits(fresh, store) == nullptr);
  const std::vector<int32_t> idx0 = { 0 };
  std::vector<float> out0(1);
  CHECK_THROWS_AS(
      get_logits_at(fresh,
                    std::span<const int32_t>(idx0.data(), idx0.size()),
                    std::span<float>(out0.data(), out0.size()),
                    store),
      std::runtime_error);

  pruneSubtree(h, store);
  prune(fresh, store);
  llama_free(ctx);
}

TEST_CASE("branch integration: get_logits_at reads per-branch snapshot after decode_each") {
  // Anchors the contract _branchLogitsAt depends on: when N branches decode
  // through the batched dispatch, each branch's get_logits_at must read from
  // its own logits_snapshot, NOT from a shared ctx-buffer. Mirrors decode_each
  // test but probes via get_logits_at instead of get_logits.
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

  constexpr int N = 4;
  BranchHandle children[N];
  for (int i = 0; i < N; ++i) {
    children[i] = fork(root, store);
    REQUIRE(children[i] != INVALID_HANDLE);
  }

  DecodeEachItem each_items[N];
  for (int i = 0; i < N; ++i) {
    each_items[i].handle = children[i];
    each_items[i].token = static_cast<llama_token>((100 + i * 50) % n_vocab);
  }
  CHECK_NOTHROW(store.decode_each(each_items));

  // R5(c): get_logits_at on each branch reads the per-branch snapshot.
  // Read 3 vocab indices per branch via get_logits_at; cross-check vs get_logits.
  const std::vector<int32_t> probe_idx = { 0, n_vocab / 2, n_vocab - 1 };
  std::vector<std::vector<float>> per_branch_reads(N, std::vector<float>(probe_idx.size()));
  for (int i = 0; i < N; ++i) {
    CHECK_NOTHROW(get_logits_at(children[i],
                                std::span<const int32_t>(probe_idx.data(), probe_idx.size()),
                                std::span<float>(per_branch_reads[i].data(), per_branch_reads[i].size()),
                                store));
    const float* logits_full = get_logits(children[i], store);
    REQUIRE(logits_full != nullptr);
    for (size_t k = 0; k < probe_idx.size(); ++k) {
      CHECK(per_branch_reads[i][k] == logits_full[probe_idx[k]]);
    }
  }

  // Adjacent branches received different tokens — at least one probe index
  // should differ across some branch pair (sanity: snapshots are not shared).
  bool any_diff = false;
  for (int i = 0; i < N && !any_diff; ++i) {
    for (int j = i + 1; j < N && !any_diff; ++j) {
      for (size_t k = 0; k < probe_idx.size(); ++k) {
        if (per_branch_reads[i][k] != per_branch_reads[j][k]) { any_diff = true; break; }
      }
    }
  }
  CHECK(any_diff);

  pruneSubtree(root, store);
  llama_free(ctx);
}

// ============================================================================
// b9581 ship-blocker (#443) — diagnostic: mirror lloyal-node's full
// SessionContext param combo in pure C++ to isolate "is it the params" from
// "is it the N-API dynamic-linking path". Test runs against the SAME model
// (LLAMA_TEST_MODEL → typically Qwen3.5-4B-Q4_K_M) as the other integration
// tests; ALL other branch_integration_test cases above pass at b9581 against
// this model using minimal params (n_ctx=512, no kv_unified, no quant KV).
// Smoke run reproduction is:
//
//   reasoning.run --query foo Qwen3.5-4B-Q4_K_M.gguf
//   → boot:error "Failed to create context"
//   → ggml-metal-device.m:622: GGML_ASSERT([rsets->data count] == 0) failed
//
// If this test ALSO crashes → the param combination triggers the b9581 Metal
// residency regression at the llama.cpp level (file upstream issue with this
// repro). If this test PASSES → the regression is in the N-API path
// (dynamic linking, Node lifecycle, AsyncWorker, etc.) and we need a
// different repro.
TEST_CASE("b9581 diagnostic: lloyal-node SessionContext full param combo") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  // EXACT copy of SessionContext.cpp:1773-1786 — reasoning.run's default
  // path when only the model arg is passed (no typeK/V override, no
  // explicit nBatch, so SDK defaults apply).
  //
  // Production constants:
  //   nCtx       = 4096   (LLAMA_CTX_SIZE=4096 from smoke; default 32768)
  //   nBatch     = 512    (SDK default — matches llama.cpp's default)
  //   nUbatch    = 512    (SessionContext sets ubatch = batch)
  //   nSeqMax    = 8      (typical for reasoning.run's agent pool)
  //   typeK/V    = F16    (NOT set by smoke → llama_context_default_params)
  //   kv_unified = true   (architecturally required — branch/spine depends)
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 4096;
  ctx_params.n_batch = 512;
  ctx_params.n_ubatch = 512;
  ctx_params.n_threads = 4;
  ctx_params.n_seq_max = 8;
  ctx_params.type_k = GGML_TYPE_F16;
  ctx_params.type_v = GGML_TYPE_F16;
  ctx_params.kv_unified = true;
  // No embeddings/pooling — main LLM, not reranker.

  MESSAGE("b9581 diagnostic: creating context with lloyal-node's param combo");
  MESSAGE("  n_ctx=" << ctx_params.n_ctx << " n_batch=" << ctx_params.n_batch
                    << " n_seq_max=" << ctx_params.n_seq_max
                    << " kv_unified=" << ctx_params.kv_unified);

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);

  // The crash from the smoke run manifests as llama_init_from_model returning
  // NULL (and a Metal residency assert firing at process exit). If this test
  // hits that path, REQUIRE(ctx) will fail loud and the process will likely
  // abort with the same Metal assert — that's the desired signal.
  REQUIRE(ctx != nullptr);

  MESSAGE("b9581 diagnostic: ctx created OK — params alone do not reproduce");

  // Optional smoke: do a single decode to ensure init didn't just lazy-defer
  // the failure. Use a minimal prompt to exercise the KV path.
  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;
  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "The quick brown fox", true, false);
  REQUIRE(!tokens.empty());
  prefill(h, tokens.data(), tokens.size(), store);

  CHECK(get_position(h, store) == static_cast<llama_pos>(tokens.size()));
  CHECK(get_logits(h, store) != nullptr);

  prune(h, store);
  llama_free(ctx);
}


// ============================================================================
// Fork fidelity — the state that must survive llama_sampler_clone
//
// fork() advertises that a child continues from the parent's exact position:
// penalty window, grammar parser position, and PRNG state. Every one of those
// lives inside a llama_sampler that we clone via the public API, and losing
// any of them fails SILENTLY — the child keeps generating, just from the wrong
// state. Nothing else in this suite asserts it.
//
// These cannot run on the stub tier: llama_sampler_apply and _accept are
// no-ops there and llama_sampler_clone returns a configured fake, so a stub
// test could only assert that clone was CALLED, never that state came with it.
// ============================================================================

namespace {

/// Greedy chain with repetition penalty and every filter disabled, so the
/// only thing between logits and the sampled token is the penalty window.
struct PenaltyOnlyParams {
  float temperature = 0.0f;      // greedy — no dist, deterministic
  int32_t top_k = 0;             // disabled
  float top_p = 1.0f;            // disabled
  float min_p = 0.0f;            // disabled
  float typical_p = 1.0f;        // disabled
  float penalty_repeat = 2.0f;   // active
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 64;
  uint32_t seed = 7;
};

/// Flat over the first `n_flat` tokens so a draw is genuinely random —
/// a peaked distribution would make every sampler agree by luck.
std::vector<float> flat_logits(int32_t n_vocab, int32_t n_flat) {
  std::vector<float> v(static_cast<size_t>(n_vocab), -60.0f);
  for (int32_t i = 0; i < n_flat && i < n_vocab; ++i) v[i] = 1.0f;
  return v;
}

std::vector<llama_token> legal_set(BranchHandle h, BranchStore& s) {
  std::vector<llama_token> out;
  for (const auto& [tok, prob] : get_legal_priors(h, s)) out.push_back(tok);
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

TEST_CASE("fork fidelity: the penalty window survives the clone") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cp);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  PenaltyOnlyParams params;

  BranchHandle parent = create(ctx, model.get(), store, 0, params);
  REQUIRE(parent != INVALID_HANDLE);
  const int32_t n_vocab = get_n_vocab(parent, store);
  REQUIRE(n_vocab > 2);

  // Token 0 wins outright on raw logits; token 1 wins once 0 is penalised.
  // With penalty_repeat = 2.0 llama.cpp divides a positive logit, so an
  // accepted 0 drops 10 -> 5, below 1's 6.
  std::vector<float> logits(static_cast<size_t>(n_vocab), -60.0f);
  logits[0] = 10.0f;
  logits[1] = 6.0f;

  set_logits(parent, std::span<const float>(logits), store);
  for (int i = 0; i < 3; ++i) accept_token(parent, 0, store);

  BranchHandle child = fork(parent, store);
  REQUIRE(child != INVALID_HANDLE);
  set_logits(child, std::span<const float>(logits), store);

  // Control: same params, same logits, NO accept history.
  BranchHandle fresh = create(ctx, model.get(), store, 0, params);
  REQUIRE(fresh != INVALID_HANDLE);
  set_logits(fresh, std::span<const float>(logits), store);

  // Greedy sampling advances no PRNG and accepts nothing, so drawing from
  // all three is free of ordering effects.
  const llama_token from_fresh  = sample(fresh, store);
  const llama_token from_parent = sample(parent, store);
  const llama_token from_child  = sample(child, store);

  // Control: without a penalty window token 0 is the argmax. If this fails
  // the logit margin is wrong for this model and everything below is vacuous.
  REQUIRE_MESSAGE(from_fresh == 0,
                  "control: an unpenalised branch must pick the argmax");

  // Does the penalty work at all on the branch that accepted the tokens?
  // Separating this from the fork assertion is what tells us whether a
  // failure is about cloning or about penalties generally.
  REQUIRE_MESSAGE(from_parent == 1,
                  "precondition: the accepting branch must itself be penalised");

  CHECK_MESSAGE(from_child == from_parent,
                "forked child must inherit the parent's penalty window");

  prune(child, store);
  prune(fresh, store);
  prune(parent, store);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("fork fidelity: the grammar parser position survives the clone") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cp);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  // A literal, so the legal set genuinely narrows as the parser advances.
  // Asserted rather than assumed below — a grammar whose legal set did not
  // change would make the fork assertion vacuous.
  const char* GBNF = "root ::= \"hello world\"";

  BranchHandle parent =
      create(ctx, model.get(), store, 0, params, DEFAULT_N_BATCH, GBNF);
  REQUIRE(parent != INVALID_HANDLE);
  const int32_t n_vocab = get_n_vocab(parent, store);
  REQUIRE(n_vocab > 0);

  auto uniform = flat_logits(n_vocab, n_vocab);
  set_logits(parent, std::span<const float>(uniform), store);

  const auto at_start = legal_set(parent, store);
  REQUIRE_MESSAGE(!at_start.empty(), "grammar must admit something at root");

  // Advance the parser by one legal token.
  accept_token(parent, at_start.front(), store);
  const auto advanced = legal_set(parent, store);
  REQUIRE_MESSAGE(advanced != at_start,
                  "the legal set must change after one token, or this "
                  "test cannot discriminate a cloned parser position");

  BranchHandle child = fork(parent, store);
  REQUIRE(child != INVALID_HANDLE);
  set_logits(child, std::span<const float>(uniform), store);

  const auto inherited = legal_set(child, store);
  CHECK_MESSAGE(inherited == advanced,
                "forked child must resume at the parent's grammar position");
  CHECK_MESSAGE(inherited != at_start,
                "forked child must NOT be back at the grammar root");

  prune(child, store);
  prune(parent, store);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("fork fidelity: PRNG state is cloned, and reseed breaks it") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_seq_max = 8;
  llama_context* ctx = llama_init_from_model(model.get(), cp);
  REQUIRE(ctx);

  BranchStore store(16);
  store.init_tenancy(ctx);
  TestParams params;   // temperature 0.8 → stochastic, ends with dist
  params.top_k = 0;    // keep the flat distribution intact
  params.min_p = 0.0f;
  params.top_p = 1.0f;

  BranchHandle parent = create(ctx, model.get(), store, 0, params);
  REQUIRE(parent != INVALID_HANDLE);
  const int32_t n_vocab = get_n_vocab(parent, store);
  REQUIRE(n_vocab > 64);

  // Flat over 64 tokens: agreement by luck is ~1/64 per child, so four
  // children agreeing is decisive rather than lucky.
  const auto uniform = flat_logits(n_vocab, 64);
  set_logits(parent, std::span<const float>(uniform), store);

  // Fork BEFORE sampling — sampling advances the PRNG, so the children must
  // be taken from the same position the parent is about to draw from.
  std::vector<BranchHandle> children;
  for (int i = 0; i < 4; ++i) {
    BranchHandle c = fork(parent, store);
    REQUIRE(c != INVALID_HANDLE);
    set_logits(c, std::span<const float>(uniform), store);
    children.push_back(c);
  }

  const llama_token from_parent = sample(parent, store);
  REQUIRE(from_parent >= 0);

  for (size_t i = 0; i < children.size(); ++i) {
    CHECK_MESSAGE(sample(children[i], store) == from_parent,
                  "child " << i << " must draw from the cloned PRNG position");
  }

  // Reseeding is what MAKES forks diverge — if it stopped working, every
  // agent in a pool would generate the same text.
  BranchHandle reseeded = fork(parent, store);
  REQUIRE(reseeded != INVALID_HANDLE);
  set_logits(reseeded, std::span<const float>(uniform), store);
  BranchState* rs = store.get(reseeded);
  REQUIRE(rs);
  llama_sampler* chain = store.get_sampler_chain(rs->sampler_chain);
  REQUIRE(chain);
  sampler::reseed_chain(chain, 9973);

  // In lockstep with an unreseeded sibling: both stand at the same PRNG position
  // (one draw past the fork), so an unreseeded clone would match it draw for
  // draw, and only a real reseed can make the two sequences part.
  BranchHandle control = fork(parent, store);
  REQUIRE(control != INVALID_HANDLE);
  int diverged = 0;
  int control_tracks = 0;
  for (int i = 0; i < 8; ++i) {
    set_logits(reseeded, std::span<const float>(uniform), store);
    set_logits(control, std::span<const float>(uniform), store);
    set_logits(children[0], std::span<const float>(uniform), store);
    const llama_token sibling = sample(children[0], store);
    if (sample(control, store) == sibling) ++control_tracks;
    if (sample(reseeded, store) != sibling) ++diverged;
  }
  CHECK_MESSAGE(control_tracks == 8,
                "an unreseeded fork draws in lockstep with its sibling (" << control_tracks << " of 8)");
  CHECK_MESSAGE(diverged > 0,
                "a reseeded fork must part from its sibling's draws");
  prune(control, store);

  for (auto c : children) prune(c, store);
  prune(reseeded, store);
  prune(parent, store);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("fork fidelity: state survives promote-then-fork (the README search loop)") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  // The library's own advertised idiom (README "Lifecycle"): fork candidates,
  // generate on them, retainOnly() the winner, promote it to root, repeat.
  //
  //   for (turn...) { leaf = root.fork(); generate; retainOnly(best); root = best; }
  //
  // Turn 0 is clean — the root was prefilled and never accepted. From turn 1
  // the root IS a branch that generated and accepted, so every property fork()
  // promises to carry is load-bearing on the promoted root. Nothing else in
  // this suite exercises state across a retainOnly().
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  llama_context_params cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cp);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  PenaltyOnlyParams params;

  BranchHandle root = create(ctx, model.get(), store, 0, params);
  REQUIRE(root != INVALID_HANDLE);
  const int32_t n_vocab = get_n_vocab(root, store);
  REQUIRE(n_vocab > 2);

  std::vector<float> logits(static_cast<size_t>(n_vocab), -60.0f);
  logits[0] = 10.0f;
  logits[1] = 6.0f;
  set_logits(root, std::span<const float>(logits), store);

  // --- turn 0: fork a candidate and "generate" on it ---
  BranchHandle leaf = fork(root, store);
  REQUIRE(leaf != INVALID_HANDLE);
  set_logits(leaf, std::span<const float>(logits), store);
  for (int i = 0; i < 3; ++i) accept_token(leaf, 0, store);

  REQUIRE_MESSAGE(sample(leaf, store) == 1,
                  "precondition: the candidate that generated must be penalised");

  // --- promote the winner; every other branch is vaporised ---
  store.retainOnly(leaf);

  // retainOnly() rewrites topology and KV but must not disturb the winner's
  // sampler registry entry. Asserted separately from the fork below so a future
  // failure is attributable to promotion rather than to cloning.
  set_logits(leaf, std::span<const float>(logits), store);
  CHECK_MESSAGE(sample(leaf, store) == 1,
                "retainOnly must preserve the promoted branch's own sampler state");

  // --- turn 1: the promoted root is now the fork source ---
  BranchHandle next = fork(leaf, store);
  REQUIRE(next != INVALID_HANDLE);
  set_logits(next, std::span<const float>(logits), store);

  CHECK_MESSAGE(sample(next, store) == 1,
                "a fork from the promoted root must inherit its penalty window; "
                "without this every search turn silently discards the previous "
                "turn's history");

  prune(next, store);
  prune(leaf, store);
  store.drain();
  llama_free(ctx);
}


// ============================================================================
// Fast-forward — a child's state becomes its parent's, the parent keeping its
// identity (git merge --ff-only). fork + prune + fast_forward give any branch
// a checkpoint it can keep or discard while the rest of the tree runs on.
//
// Every case runs on each model the suite runs on. On a hybrid recurrent model
// (Qwen3.5) a fork SHARES the parent's recurrent cell until one of the two
// decodes, so a fork that never decoded and a sibling that never decoded are
// the cases that reach that sharing; the KV is the witness throughout.
// ============================================================================

namespace {

/// Greedy and penalty-free: the sampled token is the argmax of what the
/// modifiers leave, so a branch's next tokens are data a case can compare.
struct GreedyParams {
  float temperature = 0.0f;
  int32_t top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float typical_p = 1.0f;
  float penalty_repeat = 1.0f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 64;
  uint32_t seed = 3;
};

/// A context shaped like the binding's: one unified KV pool, room for a tree.
struct FastForwardRig {
  std::shared_ptr<llama_model> model = TestConfig::acquire_test_model();
  llama_context* ctx = nullptr;
  BranchStore store{16};
  const llama_vocab* vocab = nullptr;

  explicit FastForwardRig(uint32_t n_ctx = 2048, uint32_t n_batch = 256) {
    REQUIRE(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = n_batch;
    cp.n_ubatch = n_batch;
    cp.n_seq_max = 8;
    cp.kv_unified = true;
    ctx = llama_init_from_model(model.get(), cp);
    REQUIRE(ctx);
    store.init_tenancy(ctx);
    vocab = llama_model_get_vocab(model.get());
  }
  ~FastForwardRig() {
    store.drain();
    if (ctx) llama_free(ctx);
  }

  std::vector<llama_token> tokens(const std::string& text, bool bos) const {
    return tokenizer::tokenize(vocab, text, bos, false);
  }

  /// A root holding `prompt`, created with `params` and an optional grammar.
  template <class P = GreedyParams>
  BranchHandle root(const std::string& prompt, P params = {}, const char* grammar = nullptr) {
    BranchHandle h = create(ctx, model.get(), store, 0, params, DEFAULT_N_BATCH, grammar);
    REQUIRE(h != INVALID_HANDLE);
    if (!prompt.empty()) {
      const auto t = tokens(prompt, true);
      prefill(h, t.data(), t.size(), store);
    }
    return h;
  }

  void extend(BranchHandle h, const std::string& text) {
    const auto t = tokens(text, false);
    REQUIRE(!t.empty());
    prefill(h, t.data(), t.size(), store);
  }

  llama_seq_id seq(BranchHandle h) { return store.get(h)->seq_id; }
  uint32_t cells() const { return store.kv_pressure().cells_used; }
};

/// What a branch says next, as data: greedy tokens, each accepted and decoded.
std::vector<llama_token> greedy_run(BranchHandle h, BranchStore& s, int n) {
  std::vector<llama_token> out;
  for (int i = 0; i < n; ++i) {
    const llama_token t = sample(h, s);
    REQUIRE(t >= 0);
    accept_token(h, t, s);
    step(h, t, s);
    out.push_back(t);
  }
  return out;
}

std::vector<float> logits_of(BranchHandle h, BranchStore& s) {
  const float* l = get_logits(h, s);
  REQUIRE(l != nullptr);
  return {l, l + get_n_vocab(h, s)};
}

/// Everything a refused fast-forward must leave exactly as it found it.
struct Books {
  llama_pos parent_position, child_position;
  llama_pos parent_kv, child_kv;
  llama_seq_id parent_seq, child_seq;
  uint32_t cells;
  std::vector<float> parent_logits;
};

Books books(FastForwardRig& r, BranchHandle parent, BranchHandle child) {
  return {get_position(parent, r.store), get_position(child, r.store),
          kv::pos_max(r.ctx, r.seq(parent)), kv::pos_max(r.ctx, r.seq(child)),
          r.seq(parent), r.seq(child), r.cells(), logits_of(parent, r.store)};
}

void check_untouched(FastForwardRig& r, BranchHandle parent, BranchHandle child, const Books& before) {
  REQUIRE(r.store.get(parent) != nullptr);
  REQUIRE(r.store.get(child) != nullptr);
  const Books after = books(r, parent, child);
  CHECK(after.parent_position == before.parent_position);
  CHECK(after.child_position == before.child_position);
  CHECK(after.parent_kv == before.parent_kv);
  CHECK(after.child_kv == before.child_kv);
  CHECK(after.parent_seq == before.parent_seq);
  CHECK(after.child_seq == before.child_seq);
  CHECK(after.cells == before.cells);
  CHECK(after.parent_logits == before.parent_logits);
}

constexpr const char* PROMPT = "The capital of France is";
constexpr const char* SUFFIX = " Paris. The capital of Germany is";

}  // namespace

TEST_CASE("fast forward: the parent continues exactly as its child would have") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r;

  BranchHandle a = r.root(PROMPT);
  BranchHandle p = fork(a, r.store);
  BranchHandle control = fork(a, r.store);  // the same work, never fast-forwarded
  REQUIRE(p != INVALID_HANDLE);
  REQUIRE(control != INVALID_HANDLE);
  r.extend(p, SUFFIX);
  r.extend(control, SUFFIX);

  const auto p_logits = logits_of(p, r.store);
  const llama_seq_id p_seq = r.seq(p);
  const llama_seq_id a_seq = r.seq(a);
  const llama_pos p_position = get_position(p, r.store);
  const llama_pos a_fork_head = get_fork_head(a, r.store);
  const uint32_t cells = r.cells();

  fast_forward(a, p, r.store);

  // The child is gone; the parent is the child, under its own handle.
  CHECK(r.store.get(p) == nullptr);
  REQUIRE(r.store.get(a) != nullptr);
  CHECK(r.seq(a) == p_seq);
  CHECK(get_position(a, r.store) == p_position);
  CHECK(get_fork_head(a, r.store) == a_fork_head);
  CHECK(logits_of(a, r.store) == p_logits);
  CHECK(r.store.children(a) == std::vector<BranchHandle>{control});
  CHECK(r.store.parent(control) == a);

  // The parent's former sequence is clean, and nothing was charged or refunded.
  CHECK(kv::pos_max(r.ctx, a_seq) == -1);
  CHECK(r.cells() == cells);

  // Lockstep with the control: one dispatch per token, the same KV content.
  float max_diff = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const llama_token ta = sample(a, r.store);
    const llama_token tc = sample(control, r.store);
    REQUIRE(ta >= 0);
    CHECK(ta == tc);
    accept_token(a, ta, r.store);
    accept_token(control, tc, r.store);
    DecodeEachItem items[] = {{a, ta}, {control, tc}};
    r.store.decode_each(items);
    const float* la = get_logits(a, r.store);
    const float* lc = get_logits(control, r.store);
    for (int k = 0; k < get_n_vocab(a, r.store); ++k) max_diff = std::max(max_diff, std::abs(la[k] - lc[k]));
  }
  MESSAGE("max |logit(a) - logit(control)| over 8 lockstep steps: " << max_diff);

  pruneSubtree(a, r.store);
  CHECK(r.cells() == 0);
}

TEST_CASE("fast forward: a sibling forked before the child is unaffected") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r;

  BranchHandle a = r.root(PROMPT);
  // Never decoded: on a recurrent model it shares the parent's cell.
  BranchHandle sibling = fork(a, r.store);
  REQUIRE(sibling != INVALID_HANDLE);

  // What the sibling says next, recorded on a probe forked from it.
  BranchHandle probe = fork(sibling, r.store);
  REQUIRE(probe != INVALID_HANDLE);
  const auto expected = greedy_run(probe, r.store, 8);
  prune(probe, r.store);

  BranchHandle p = fork(a, r.store);
  REQUIRE(p != INVALID_HANDLE);
  r.extend(p, SUFFIX);
  const llama_pos sibling_head = get_fork_head(sibling, r.store);
  const llama_pos sibling_kv = kv::pos_max(r.ctx, r.seq(sibling));

  fast_forward(a, p, r.store);

  CHECK(r.store.parent(sibling) == a);
  CHECK(get_fork_head(sibling, r.store) == sibling_head);
  CHECK(kv::pos_max(r.ctx, r.seq(sibling)) == sibling_kv);
  CHECK(greedy_run(sibling, r.store, 8) == expected);
  // And the parent goes on from the child's text, not the sibling's.
  CHECK_NOTHROW(greedy_run(a, r.store, 2));

  pruneSubtree(a, r.store);
  CHECK(r.cells() == 0);
}

TEST_CASE("fast forward: a fork that never decoded leaves the parent as it was") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r;

  BranchHandle a = r.root(PROMPT);
  BranchHandle reference = fork(a, r.store);
  REQUIRE(reference != INVALID_HANDLE);
  const auto expected = greedy_run(reference, r.store, 8);
  prune(reference, r.store);

  const llama_pos position = get_position(a, r.store);
  const uint32_t cells = r.cells();
  const auto logits = logits_of(a, r.store);

  // Shares the parent's recurrent cell outright: the parent's own tag is the one removed.
  BranchHandle p = fork(a, r.store);
  REQUIRE(p != INVALID_HANDLE);
  fast_forward(a, p, r.store);

  CHECK(get_position(a, r.store) == position);
  CHECK(r.cells() == cells);
  CHECK(logits_of(a, r.store) == logits);
  CHECK(greedy_run(a, r.store, 8) == expected);

  prune(a, r.store);
  CHECK(r.cells() == 0);
}

TEST_CASE("fast forward: the child's penalty window becomes the parent's") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(512, 64);
  PenaltyOnlyParams params;

  BranchHandle a = r.root("", params);
  const int32_t n_vocab = get_n_vocab(a, r.store);
  // Token 0 wins on raw logits; token 1 once 0 is penalised (10 -> 5 < 6).
  std::vector<float> logits(static_cast<size_t>(n_vocab), -60.0f);
  logits[0] = 10.0f;
  logits[1] = 6.0f;
  set_logits(a, std::span<const float>(logits), r.store);

  BranchHandle untouched = fork(a, r.store);  // the parent's window, as it was
  BranchHandle p = fork(a, r.store);
  REQUIRE(untouched != INVALID_HANDLE);
  REQUIRE(p != INVALID_HANDLE);
  for (int i = 0; i < 3; ++i) accept_token(p, 0, r.store);
  REQUIRE_MESSAGE(sample(untouched, r.store) == 0, "control: no window, the argmax wins");
  REQUIRE_MESSAGE(sample(p, r.store) == 1, "precondition: the child's window penalises 0");

  fast_forward(a, p, r.store);
  set_logits(a, std::span<const float>(logits), r.store);
  CHECK_MESSAGE(sample(a, r.store) == 1, "the parent samples under the child's window");

  pruneSubtree(a, r.store);
}

TEST_CASE("fast forward: the child's grammar position becomes the parent's") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(512, 64);

  BranchHandle a = r.root("", TestParams{}, "root ::= \"hello world\"");
  const int32_t n_vocab = get_n_vocab(a, r.store);
  const auto uniform = flat_logits(n_vocab, n_vocab);
  set_logits(a, std::span<const float>(uniform), r.store);
  const auto at_start = legal_set(a, r.store);
  REQUIRE(!at_start.empty());

  BranchHandle p = fork(a, r.store);
  REQUIRE(p != INVALID_HANDLE);
  accept_token(p, at_start.front(), r.store);
  const auto advanced = legal_set(p, r.store);
  REQUIRE_MESSAGE(advanced != at_start, "the parser must have moved for this to discriminate");

  fast_forward(a, p, r.store);
  CHECK(legal_set(a, r.store) == advanced);

  prune(a, r.store);
}

TEST_CASE("fast forward: the child's PRNG position, metrics and handles become the parent's") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(512, 64);
  TestParams params;   // stochastic: the chain ends in dist
  params.top_k = 0;
  params.top_p = 1.0f;
  params.min_p = 0.0f;

  BranchHandle a = r.root(PROMPT, params);
  const int32_t n_vocab = get_n_vocab(a, r.store);
  const auto flat = flat_logits(n_vocab, 64);

  BranchState* as = r.store.get(a);
  const SamplerChainHandle a_chain = as->sampler_chain;
  const MetricsHandle a_metrics = as->metrics;

  // Two forks taken together stand at one PRNG position.
  BranchHandle p = fork(a, r.store);
  BranchHandle twin = fork(a, r.store);
  REQUIRE(p != INVALID_HANDLE);
  REQUIRE(twin != INVALID_HANDLE);

  // The child accepts from real logits, so its metrics move.
  const llama_token t = sample(p, r.store);
  REQUIRE(t >= 0);
  accept_token(p, t, r.store);
  step(p, t, r.store);
  // Keep the twin in step: one draw, one accept.
  const llama_token tt = sample(twin, r.store);
  REQUIRE(tt == t);
  accept_token(twin, tt, r.store);

  const float p_ppl = get_perplexity(p, r.store);
  REQUIRE(std::isfinite(p_ppl));
  const SamplerChainHandle p_chain = r.store.get(p)->sampler_chain;
  const MetricsHandle p_metrics = r.store.get(p)->metrics;

  fast_forward(a, p, r.store);

  as = r.store.get(a);
  CHECK(as->sampler_chain == p_chain);
  CHECK(as->metrics == p_metrics);
  CHECK(r.store.get_sampler_chain(a_chain) == nullptr);
  CHECK(get_perplexity(a, r.store) == p_ppl);
  (void)a_metrics;

  for (int i = 0; i < 6; ++i) {
    set_logits(a, std::span<const float>(flat), r.store);
    set_logits(twin, std::span<const float>(flat), r.store);
    CHECK_MESSAGE(sample(a, r.store) == sample(twin, r.store),
                  "draw " << i << ": the parent draws from the child's PRNG position");
  }

  pruneSubtree(a, r.store);
}

TEST_CASE("fast forward: configuration set on the child comes with it; the parent keeps its steer") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(512, 64);

  BranchHandle a = r.root("");
  const int32_t n_vocab = get_n_vocab(a, r.store);
  REQUIRE(n_vocab > 20);
  // 10 > 11 > 12 > 13 on raw logits.
  std::vector<float> logits(static_cast<size_t>(n_vocab), -60.0f);
  logits[10] = 10.0f;
  logits[11] = 9.0f;
  logits[12] = 8.0f;
  logits[13] = 7.0f;
  const auto mask = [](llama_token banned) {
    return [banned](llama_token_data_array& cur) {
      for (size_t i = 0; i < cur.size; ++i)
        if (cur.data[i].id == banned) cur.data[i].logit = -std::numeric_limits<float>::infinity();
    };
  };
  set_steer(a, mask(10), r.store);  // the parent's own

  BranchHandle p = fork(a, r.store);  // steer is not cloned
  REQUIRE(p != INVALID_HANDLE);
  llama_logit_bias ban{11, -std::numeric_limits<float>::infinity()};
  set_logit_bias(p, &ban, 1, r.store);  // the child's configuration
  set_steer(p, mask(12), r.store);      // the child's own, which dies with it
  TestParams stochastic;
  stochastic.seed = 77;
  set_sampler_params(p, stochastic, r.store);
  set_sampler_params(p, GreedyParams{}, r.store);  // back to greedy, as a rebuilt chain

  fast_forward(a, p, r.store);
  set_logits(a, std::span<const float>(logits), r.store);
  // 10: the parent's steer. 11: the child's bias, adopted. 12: the child's steer, gone.
  CHECK(sample(a, r.store) == 12);

  BranchHandle q = fork(a, r.store);
  REQUIRE(q != INVALID_HANDLE);
  set_grammar(q, r.model.get(), "root ::= \"hello world\"", r.store);
  fast_forward(a, q, r.store);
  CHECK(r.store.get(a)->grammar != 0);
  set_logits(a, std::span<const float>(flat_logits(n_vocab, n_vocab)), r.store);
  CHECK(legal_set(a, r.store).size() < static_cast<size_t>(n_vocab));

  prune(a, r.store);
}

TEST_CASE("fast forward: the search loop keeps its history and leaves other branches alive") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(512, 64);
  PenaltyOnlyParams params;

  BranchHandle root = r.root("", params);
  const int32_t n_vocab = get_n_vocab(root, r.store);
  std::vector<float> logits(static_cast<size_t>(n_vocab), -60.0f);
  logits[0] = 10.0f;
  logits[1] = 6.0f;
  set_logits(root, std::span<const float>(logits), r.store);

  const auto prompt = r.tokens("Hello", true);
  BranchHandle other = fork(root, r.store);  // unrelated work, still running
  BranchHandle leaf = fork(root, r.store);
  REQUIRE(other != INVALID_HANDLE);
  REQUIRE(leaf != INVALID_HANDLE);
  for (int i = 0; i < 3; ++i) accept_token(leaf, 0, r.store);

  fast_forward(root, leaf, r.store);

  set_logits(root, std::span<const float>(logits), r.store);
  CHECK_MESSAGE(sample(root, r.store) == 1, "the root carries the leaf's window");
  BranchHandle next = fork(root, r.store);
  REQUIRE(next != INVALID_HANDLE);
  set_logits(next, std::span<const float>(logits), r.store);
  CHECK_MESSAGE(sample(next, r.store) == 1, "a fork from the root inherits it");

  REQUIRE(r.store.get(other) != nullptr);
  CHECK_NOTHROW(prefill(other, prompt.data(), prompt.size(), r.store));

  pruneSubtree(root, r.store);
}

TEST_CASE("fast forward: refused, with nothing changed, unless the child is the parent's own and the parent is as it was") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r;

  BranchHandle a = r.root(PROMPT);
  BranchHandle p = fork(a, r.store);
  BranchHandle sibling = fork(a, r.store);
  REQUIRE(p != INVALID_HANDLE);
  REQUIRE(sibling != INVALID_HANDLE);
  r.extend(p, SUFFIX);
  BranchHandle unrelated = r.root(PROMPT);

  const auto refused = [&](BranchHandle parent, BranchHandle child, const char* why) {
    INFO("refusal: " << why);
    const Books before = books(r, parent, child);
    CHECK_THROWS_WITH(fast_forward(parent, child, r.store), doctest::Contains(why));
    check_untouched(r, parent, child, before);
  };

  refused(a, unrelated, "was not forked from this parent");
  refused(sibling, p, "was not forked from this parent");
  refused(p, a, "was not forked from this parent");
  refused(a, a, "was not forked from this parent");

  BranchHandle grandchild = fork(p, r.store);
  REQUIRE(grandchild != INVALID_HANDLE);
  refused(a, grandchild, "was not forked from this parent");
  refused(a, p, "has children of its own");
  prune(grandchild, r.store);

  // The parent changed after the fork: decoded, or not decoded but no longer the same.
  BranchHandle decoded = fork(a, r.store);
  REQUIRE(decoded != INVALID_HANDLE);
  const llama_token t = sample(a, r.store);
  accept_token(a, t, r.store);
  step(a, t, r.store);
  refused(a, decoded, "the parent has decoded since the fork");
  prune(decoded, r.store);

  const auto changed = [&](const char* how, auto&& change) {
    BranchHandle child = fork(a, r.store);
    REQUIRE(child != INVALID_HANDLE);
    change();
    INFO("change: " << how);
    refused(a, child, "the parent has changed since the fork");
    prune(child, r.store);
  };
  changed("accept", [&] { accept_token(a, sample(a, r.store), r.store); });
  changed("set_logits", [&] { set_logits(a, std::span<const float>(logits_of(sibling, r.store)), r.store); });
  changed("set_grammar", [&] { set_grammar(a, r.model.get(), "root ::= \"x\"", r.store); });
  changed("set_sampler_params", [&] { TestParams t2; t2.seed = 1234; set_sampler_params(a, t2, r.store); });
  changed("merge_logits", [&] {
    const BranchHandle experts[] = {sibling};
    r.store.merge_logits(a, experts, 0.5f);
  });

  // A merge refused partway changes nothing, so a child forked before it still fast-forwards.
  {
    BranchHandle before_merge = fork(a, r.store);
    BranchHandle stale = fork(a, r.store);
    REQUIRE(before_merge != INVALID_HANDLE);
    REQUIRE(stale != INVALID_HANDLE);
    prune(stale, r.store);
    const std::vector<float> logits = logits_of(a, r.store);
    const BranchHandle experts[] = {sibling, stale};
    CHECK_THROWS_WITH(r.store.merge_logits(a, experts, 0.5f), doctest::Contains("invalid expert handle at index 1"));
    CHECK(logits_of(a, r.store) == logits);
    CHECK_NOTHROW(fast_forward(a, before_merge, r.store));
  }

  // A merge of no experts changes nothing, so it does not refuse either.
  {
    BranchHandle before_empty = fork(a, r.store);
    REQUIRE(before_empty != INVALID_HANDLE);
    r.store.merge_logits(a, std::span<const BranchHandle>{}, 0.5f);
    CHECK_NOTHROW(fast_forward(a, before_empty, r.store));
  }

  // A memoized no-op changes nothing, so it does not refuse.
  BranchHandle same = fork(a, r.store);
  REQUIRE(same != INVALID_HANDLE);
  TestParams t2;
  t2.seed = 1234;
  set_sampler_params(a, t2, r.store);
  CHECK_NOTHROW(fast_forward(a, same, r.store));

  // A stale handle is refused by name, on either side.
  BranchHandle gone = fork(a, r.store);
  REQUIRE(gone != INVALID_HANDLE);
  prune(gone, r.store);
  CHECK_THROWS_WITH(fast_forward(a, gone, r.store), doctest::Contains("invalid child handle"));
  CHECK_THROWS_WITH(fast_forward(gone, a, r.store), doctest::Contains("invalid parent handle"));

  pruneSubtree(a, r.store);
  pruneSubtree(unrelated, r.store);
  CHECK(r.cells() == 0);
}

TEST_CASE("fast forward: a parent reseeded after the fork has changed since it") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(512, 64);
  TestParams stochastic;  // a chain with a PRNG to reseed

  BranchHandle a = r.root(PROMPT, stochastic);
  BranchHandle p = fork(a, r.store);
  REQUIRE(p != INVALID_HANDLE);
  reseed_sampler(a, 9973, r.store);
  const Books before = books(r, a, p);
  CHECK_THROWS_WITH(fast_forward(a, p, r.store), doctest::Contains("the parent has changed since the fork"));
  check_untouched(r, a, p, before);

  // A greedy chain has no PRNG: reseeding it changes nothing, so nothing refuses.
  BranchHandle g = r.root(PROMPT);
  BranchHandle q = fork(g, r.store);
  REQUIRE(q != INVALID_HANDLE);
  reseed_sampler(g, 9973, r.store);
  CHECK_NOTHROW(fast_forward(g, q, r.store));

  pruneSubtree(a, r.store);
  pruneSubtree(g, r.store);
}

// ============================================================================
// A reseed is what set_sampler_params compares against. The memo compares
// against cached_params, so a reseed that leaves the cached seed behind makes
// the memo skip a real change and rebuild for none.
// ============================================================================

namespace {

/// A wide distribution, so two different seeds draw different streams.
struct WideParams {
  float temperature = 1.5f;
  int32_t top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float typical_p = 1.0f;
  float penalty_repeat = 1.3f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 64;
  uint32_t seed = 42;
};

WideParams seeded(uint32_t seed) {
  WideParams p;
  p.seed = seed;
  return p;
}

/// The next `n` draws from a branch's sampler over the same logits: the PRNG
/// advances, nothing is accepted or decoded.
std::vector<llama_token> draws(BranchHandle h, BranchStore& s, int n) {
  std::vector<llama_token> out;
  for (int i = 0; i < n; ++i) out.push_back(sample(h, s));
  return out;
}

SamplerChainHandle chain_of(FastForwardRig& r, BranchHandle h) { return r.store.get(h)->sampler_chain; }

}  // namespace

TEST_CASE("branch integration: set_sampler_params after a reseed applies the seed it asks for") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(512, 64);

  BranchHandle a = r.root(PROMPT, seeded(42));
  BranchHandle fresh = fork(a, r.store);  // the chain as seed 42 made it, over the same logits
  REQUIRE(fresh != INVALID_HANDLE);

  reseed_sampler(a, 9973, r.store);
  const SamplerChainHandle reseeded = chain_of(r, a);
  set_sampler_params(a, seeded(42), r.store);

  CHECK(chain_of(r, a) != reseeded);
  CHECK(draws(a, r.store, 8) == draws(fresh, r.store, 8));

  pruneSubtree(a, r.store);
}

TEST_CASE("branch integration: set_sampler_params after a reseed keeps a chain already on that seed") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(512, 64);

  BranchHandle a = r.root(PROMPT, seeded(42));
  for (int i = 0; i < 4; ++i) {  // tokens in the penalty window a rebuild would forget
    const llama_token t = sample(a, r.store);
    accept_token(a, t, r.store);
    step(a, t, r.store);
  }
  reseed_sampler(a, 9973, r.store);
  BranchHandle twin = fork(a, r.store);  // the same window and the same PRNG
  REQUIRE(twin != INVALID_HANDLE);

  const SamplerChainHandle before = chain_of(r, a);
  set_sampler_params(a, seeded(9973), r.store);

  CHECK(chain_of(r, a) == before);
  CHECK(draws(a, r.store, 8) == draws(twin, r.store, 8));

  pruneSubtree(a, r.store);
}

TEST_CASE("fast forward: a reseeded seed travels with fork and fast-forward") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(512, 64);

  BranchHandle a = r.root(PROMPT, seeded(42));
  reseed_sampler(a, 9973, r.store);
  BranchHandle c = fork(a, r.store);
  REQUIRE(c != INVALID_HANDLE);
  CHECK(r.store.get(c)->cached_params.seed == 9973);

  BranchHandle p = fork(a, r.store);
  REQUIRE(p != INVALID_HANDLE);
  reseed_sampler(p, 7, r.store);
  fast_forward(a, p, r.store);
  CHECK(r.store.get(a)->cached_params.seed == 7);

  const SamplerChainHandle adopted = chain_of(r, a);
  set_sampler_params(a, seeded(7), r.store);
  CHECK(chain_of(r, a) == adopted);

  pruneSubtree(a, r.store);
}

TEST_CASE("fast forward: a poisoned child is refused and the parent goes on untouched") {
  // The decode-failure suite's rig: one 256-cell pool, so a later chunk finds
  // no slot on demand while earlier chunks stand.
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(256, 64);

  BranchHandle a = create(r.ctx, r.model.get(), r.store, 0, GreedyParams{}, 64);
  REQUIRE(a != INVALID_HANDLE);
  const auto prompt = r.tokens("Hello", true);
  prefill(a, prompt.data(), prompt.size(), r.store);
  const int32_t n_vocab = get_n_vocab(a, r.store);

  BranchHandle p = fork(a, r.store);
  REQUIRE(p != INVALID_HANDLE);
  std::vector<llama_token> flood(300);
  for (size_t j = 0; j < flood.size(); ++j) flood[j] = static_cast<llama_token>((1000 + j) % n_vocab);
  bool partial = false;
  try {
    prefill(p, flood.data(), flood.size(), r.store);
  } catch (const decode::DecodeError& e) {
    partial = e.partial;
  }
  REQUIRE_MESSAGE(partial, "the prefill must have died with earlier chunks landed");
  CHECK(is_poisoned(p, r.store));
  CHECK_FALSE(is_poisoned(a, r.store));

  // A fork of a poisoned branch shares the damage, and says so.
  BranchHandle inherits = fork(p, r.store);
  REQUIRE(inherits != INVALID_HANDLE);
  CHECK(is_poisoned(inherits, r.store));
  prune(inherits, r.store);

  const Books before = books(r, a, p);
  CHECK_THROWS_WITH(fast_forward(a, p, r.store), doctest::Contains("poisoned"));
  check_untouched(r, a, p, before);

  // Pruning reclaims the orphaned chunks, and the parent prefills where it could not.
  prune(p, r.store);
  CHECK(kv::pos_max(r.ctx, before.child_seq) == -1);
  const auto more = r.tokens(" world", false);
  CHECK_NOTHROW(prefill(a, more.data(), more.size(), r.store));
  CHECK_FALSE(is_poisoned(a, r.store));

  prune(a, r.store);
  CHECK(r.cells() == 0);
}

TEST_CASE("fast forward: a refusal that left nothing behind does not poison") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(256, 64);

  BranchHandle a = create(r.ctx, r.model.get(), r.store, 0, GreedyParams{}, 64);
  REQUIRE(a != INVALID_HANDLE);
  const auto prompt = r.tokens("Hello", true);
  prefill(a, prompt.data(), prompt.size(), r.store);
  const int32_t n_vocab = get_n_vocab(a, r.store);

  // Fill the pool to within a few cells of full on an unrelated root.
  BranchHandle filler = create(r.ctx, r.model.get(), r.store, 0, GreedyParams{}, 64);
  REQUIRE(filler != INVALID_HANDLE);
  const int32_t room = 256 - static_cast<int32_t>(r.cells()) - 4;
  for (int32_t done = 0; done < room;) {
    const int32_t n = std::min<int32_t>(64, room - done);
    std::vector<llama_token> chunk(static_cast<size_t>(n));
    for (int32_t j = 0; j < n; ++j) chunk[static_cast<size_t>(j)] = static_cast<llama_token>((2000 + done + j) % n_vocab);
    prefill(filler, chunk.data(), chunk.size(), r.store);
    done += n;
  }

  // One chunk that cannot find a slot: the call restored itself, nothing landed.
  BranchHandle p = fork(a, r.store);
  REQUIRE(p != INVALID_HANDLE);
  std::vector<llama_token> ten(10, static_cast<llama_token>(5000 % n_vocab));
  int32_t rc = 0;
  bool partial = true;
  try {
    prefill(p, ten.data(), ten.size(), r.store);
  } catch (const decode::DecodeError& e) {
    rc = e.rc;
    partial = e.partial;
  }
  REQUIRE(rc == 1);
  REQUIRE_FALSE(partial);
  CHECK_FALSE(is_poisoned(p, r.store));
  CHECK_NOTHROW(fast_forward(a, p, r.store));

  pruneSubtree(a, r.store);
  prune(filler, r.store);
}

TEST_CASE("fast forward: the child's slot is reused clean and its handle goes stale") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  FastForwardRig r(512, 64);

  BranchHandle a = r.root(PROMPT);
  BranchHandle reference = fork(a, r.store);
  REQUIRE(reference != INVALID_HANDLE);
  const llama_token greedy = sample(reference, r.store);
  REQUIRE(greedy >= 0);
  prune(reference, r.store);

  BranchHandle p = fork(a, r.store);
  REQUIRE(p != INVALID_HANDLE);
  llama_logit_bias ban{greedy, -std::numeric_limits<float>::infinity()};
  set_logit_bias(p, &ban, 1, r.store);
  set_grammar(p, r.model.get(), "root ::= \"hello world\"", r.store);
  const GrammarHandle p_grammar = r.store.get(p)->grammar;
  const GrammarHandle a_grammar_before = r.store.get(a)->grammar;
  CHECK(a_grammar_before == 0);

  fast_forward(a, p, r.store);
  CHECK(r.store.get(p) == nullptr);
  CHECK(sample(p, r.store) < 0);
  CHECK(r.store.get(a)->grammar == p_grammar);

  // The freed slot comes back under a new generation and carries nothing over.
  BranchHandle fresh = r.root(PROMPT);
  CHECK(handle_index(fresh) == handle_index(p));
  CHECK(handle_generation(fresh) != handle_generation(p));
  CHECK(r.store.get(fresh)->grammar == 0);
  CHECK(r.store.get(fresh)->logit_bias.empty());
  CHECK(sample(fresh, r.store) == greedy);

  const llama_seq_id a_seq = r.seq(a);
  const llama_seq_id fresh_seq = r.seq(fresh);
  r.store.drain();
  CHECK(kv::pos_max(r.ctx, a_seq) == -1);
  CHECK(kv::pos_max(r.ctx, fresh_seq) == -1);
}
