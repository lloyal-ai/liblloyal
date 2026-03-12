/**
 * KV Pressure Integration Test
 *
 * Validates BranchStore::kv_pressure() monotonic counter at each lifecycle point:
 * - init_tenancy: cells_used == 0
 * - decode_scatter (prefill): cells_used == N
 * - fork (seq_cp): cells_used unchanged
 * - decode_each (generate): cells_used += K
 * - retainOnly: cells_used == winner.position
 * - drain: cells_used == 0
 * - Multi-fork scenario: prefill + fork + generate
 */

#include <cstdlib>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/branch.hpp>
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

struct TestParams {
  float temperature = 0.0f;
  int32_t top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float typical_p = 1.0f;
  float penalty_repeat = 1.0f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 64;
  uint32_t seed = 42;
};

// ============================================================================
// kv_pressure() lifecycle
// ============================================================================

TEST_CASE("kv_pressure: init → zero cells_used") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 128;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);

  auto p = store.kv_pressure();
  CHECK(p.n_ctx == static_cast<uint32_t>(llama_n_ctx(ctx)));
  CHECK(p.cells_used == 0);
  CHECK(p.remaining == p.n_ctx);

  store.drain();
  llama_free(ctx);
}

TEST_CASE("kv_pressure: decode_scatter increments cells_used") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 128;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(h != INVALID_HANDLE);

  // Tokenize a short prompt
  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello world, this is a test", true, false);
  REQUIRE(tokens.size() > 2);

  const uint32_t prefill_n = static_cast<uint32_t>(tokens.size());

  // Prefill via decode_scatter
  DecodeScatterItem item{h, std::span<const llama_token>(tokens)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&item, 1));

  auto p = store.kv_pressure();
  CHECK(p.cells_used == prefill_n);
  CHECK(p.remaining == p.n_ctx - prefill_n);

  prune(h, store);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("kv_pressure: fork does not change cells_used") {
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

  BranchHandle root = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(root != INVALID_HANDLE);

  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Test prompt for fork", true, false);
  DecodeScatterItem item{root, std::span<const llama_token>(tokens)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&item, 1));

  uint32_t before_fork = store.kv_pressure().cells_used;

  // Fork — seq_cp only, no new cells allocated
  BranchHandle child = fork(root, store);
  REQUIRE(child != INVALID_HANDLE);

  uint32_t after_fork = store.kv_pressure().cells_used;
  CHECK(after_fork == before_fork);

  prune(child, store);
  prune(root, store);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("kv_pressure: decode_each increments by branch count") {
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

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle root = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(root != INVALID_HANDLE);

  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello", true, false);
  DecodeScatterItem prefill{root, std::span<const llama_token>(tokens)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&prefill, 1));

  uint32_t after_prefill = store.kv_pressure().cells_used;

  // Fork 3 branches
  BranchHandle b1 = fork(root, store);
  BranchHandle b2 = fork(root, store);
  BranchHandle b3 = fork(root, store);
  REQUIRE(b1 != INVALID_HANDLE);
  REQUIRE(b2 != INVALID_HANDLE);
  REQUIRE(b3 != INVALID_HANDLE);

  // Sample one token per branch
  llama_token t1 = sample(b1, store);
  llama_token t2 = sample(b2, store);
  llama_token t3 = sample(b3, store);
  accept_token(b1, t1, store);
  accept_token(b2, t2, store);
  accept_token(b3, t3, store);

  // decode_each with 3 branches → cells_used += 3
  DecodeEachItem items[3] = {{b1, t1}, {b2, t2}, {b3, t3}};
  store.decode_each(std::span<const DecodeEachItem>(items, 3));

  uint32_t after_decode = store.kv_pressure().cells_used;
  CHECK(after_decode == after_prefill + 3);

  prune(b1, store);
  prune(b2, store);
  prune(b3, store);
  prune(root, store);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("kv_pressure: retainOnly resets to winner position") {
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

  BranchHandle root = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(root != INVALID_HANDLE);

  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Retain test prompt", true, false);
  DecodeScatterItem prefill{root, std::span<const llama_token>(tokens)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&prefill, 1));

  BranchHandle winner = fork(root, store);
  REQUIRE(winner != INVALID_HANDLE);

  // Generate a few tokens on the winner
  for (int step = 0; step < 5; ++step) {
    llama_token t = sample(winner, store);
    accept_token(winner, t, store);
    DecodeEachItem item{winner, t};
    store.decode_each(std::span<const DecodeEachItem>(&item, 1));
  }

  BranchState* ws = store.get(winner);
  REQUIRE(ws);
  int32_t winner_pos = ws->position;

  store.retainOnly(winner);

  auto p = store.kv_pressure();
  CHECK(p.cells_used == static_cast<uint32_t>(winner_pos));

  prune(winner, store);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("kv_pressure: drain resets to zero") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 128;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(h != INVALID_HANDLE);

  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Drain test", true, false);
  DecodeScatterItem item{h, std::span<const llama_token>(tokens)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&item, 1));

  CHECK(store.kv_pressure().cells_used > 0);

  store.drain();

  auto p = store.kv_pressure();
  CHECK(p.cells_used == 0);
  CHECK(p.n_ctx == 0);  // drained — no context
  CHECK(p.remaining == 0);

  llama_free(ctx);
}

TEST_CASE("kv_pressure: multi-fork generate scenario") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 2048;
  cparams.n_batch = 256;
  cparams.n_seq_max = 8;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(16);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle root = create(ctx, model.get(), store, 0, params, 256);
  REQUIRE(root != INVALID_HANDLE);

  // Prefill a prompt
  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab,
    "This is a longer test prompt to verify the multi-fork KV pressure accounting works correctly across branches",
    true, false);
  const uint32_t prefill_n = static_cast<uint32_t>(tokens.size());

  DecodeScatterItem prefill{root, std::span<const llama_token>(tokens)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&prefill, 1));

  CHECK(store.kv_pressure().cells_used == prefill_n);

  // Fork 3 branches
  BranchHandle b1 = fork(root, store);
  BranchHandle b2 = fork(root, store);
  BranchHandle b3 = fork(root, store);

  // Generate tokens on each branch
  const int gen_steps = 10;
  for (int step = 0; step < gen_steps; ++step) {
    llama_token t1 = sample(b1, store);
    llama_token t2 = sample(b2, store);
    llama_token t3 = sample(b3, store);
    accept_token(b1, t1, store);
    accept_token(b2, t2, store);
    accept_token(b3, t3, store);

    DecodeEachItem items[3] = {{b1, t1}, {b2, t2}, {b3, t3}};
    store.decode_each(std::span<const DecodeEachItem>(items, 3));
  }

  // prefill_n + (3 branches * gen_steps tokens each)
  uint32_t expected = prefill_n + 3 * gen_steps;
  CHECK(store.kv_pressure().cells_used == expected);

  // Prune branches one at a time — each subtracts its unique cells (gen_steps)
  prune(b1, store);
  CHECK(store.kv_pressure().cells_used == prefill_n + 2 * gen_steps);

  prune(b2, store);
  CHECK(store.kv_pressure().cells_used == prefill_n + 1 * gen_steps);

  prune(b3, store);
  CHECK(store.kv_pressure().cells_used == prefill_n);

  // Root owns all prefill_n cells (fork_head == 0)
  prune(root, store);
  CHECK(store.kv_pressure().cells_used == 0);

  store.drain();
  llama_free(ctx);
}

TEST_CASE("kv_pressure: auto-reset when all branches pruned") {
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

  // Phase 1: create branch, prefill, generate, prune
  BranchHandle h1 = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(h1 != INVALID_HANDLE);

  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Phase one prompt", true, false);
  DecodeScatterItem item{h1, std::span<const llama_token>(tokens)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&item, 1));

  uint32_t after_phase1 = store.kv_pressure().cells_used;
  CHECK(after_phase1 > 0);

  prune(h1, store);

  // All branches released → cells_used_ should auto-reset to 0
  auto p1 = store.kv_pressure();
  CHECK(p1.cells_used == 0);
  CHECK(p1.remaining == p1.n_ctx);

  // Phase 2: create new branch — counter starts fresh
  BranchHandle h2 = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(h2 != INVALID_HANDLE);

  auto tokens2 = tokenizer::tokenize(vocab, "Phase two", true, false);
  DecodeScatterItem item2{h2, std::span<const llama_token>(tokens2)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&item2, 1));

  auto p2 = store.kv_pressure();
  CHECK(p2.cells_used == static_cast<uint32_t>(tokens2.size()));

  prune(h2, store);
  CHECK(store.kv_pressure().cells_used == 0);

  store.drain();
  llama_free(ctx);
}

TEST_CASE("kv_pressure: fork_head accessor") {
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

  // Root branch: fork_head == 0
  BranchHandle root = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(root != INVALID_HANDLE);
  CHECK(get_fork_head(root, store) == 0);

  // Prefill root
  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Fork head test prompt", true, false);
  DecodeScatterItem prefill{root, std::span<const llama_token>(tokens)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&prefill, 1));

  BranchState* rs = store.get(root);
  REQUIRE(rs);
  llama_pos root_pos = rs->position;
  CHECK(root_pos > 0);

  // Forked branch: fork_head == parent's position at fork time
  BranchHandle child = fork(root, store);
  REQUIRE(child != INVALID_HANDLE);
  CHECK(get_fork_head(child, store) == root_pos);

  prune(child, store);
  prune(root, store);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("kv_pressure: pruneSubtree decrements correctly") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 2048;
  cparams.n_batch = 256;
  cparams.n_seq_max = 8;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(16);
  store.init_tenancy(ctx);
  TestParams params;

  // Root: prefill N tokens
  BranchHandle root = create(ctx, model.get(), store, 0, params, 256);
  REQUIRE(root != INVALID_HANDLE);

  const auto* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Subtree test prompt for cascading prune", true, false);
  DecodeScatterItem prefill{root, std::span<const llama_token>(tokens)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&prefill, 1));

  // Fork A from root, generate K1 tokens
  BranchHandle a = fork(root, store);
  REQUIRE(a != INVALID_HANDLE);
  const int k1 = 5;
  for (int i = 0; i < k1; ++i) {
    llama_token t = sample(a, store);
    accept_token(a, t, store);
    DecodeEachItem item{a, t};
    store.decode_each(std::span<const DecodeEachItem>(&item, 1));
  }

  // Fork B from A, generate K2 tokens
  BranchHandle b = fork(a, store);
  REQUIRE(b != INVALID_HANDLE);
  const int k2 = 3;
  for (int i = 0; i < k2; ++i) {
    llama_token t = sample(b, store);
    accept_token(b, t, store);
    DecodeEachItem item{b, t};
    store.decode_each(std::span<const DecodeEachItem>(&item, 1));
  }

  uint32_t prefill_n = static_cast<uint32_t>(tokens.size());
  CHECK(store.kv_pressure().cells_used == prefill_n + k1 + k2);

  // pruneSubtree(root) should free everything
  pruneSubtree(root, store);
  CHECK(store.kv_pressure().cells_used == 0);

  store.drain();
  llama_free(ctx);
}
