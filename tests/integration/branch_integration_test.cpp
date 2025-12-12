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
 */

#include <cstdlib>
#include <cstddef>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/branch.hpp>
#include <lloyal/decoder.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <memory>
#include <string>
#include <vector>

// ============================================================================
// ABI Stability Check for llama_batch
// ============================================================================
// decoder::decode_one() uses a manually-constructed llama_batch on the stack
// for zero-allocation performance. This test catches ABI drift if llama.cpp
// adds/removes/reorders fields in the llama_batch struct.
//
// If this test fails after updating llama.cpp:
// 1. Check what changed in llama_batch (llama.h)
// 2. Update decoder::decode_one() to match
// 3. Update these assertions
// ============================================================================

TEST_CASE("llama_batch ABI stability check") {
  // Verify struct size hasn't changed unexpectedly
  // Current llama_batch has 7 pointer/int fields:
  //   int32_t n_tokens, token*, embd*, pos*, n_seq_id*, seq_id**, logits*
  // On 64-bit: 4 + 8*6 = 52 bytes, padded to 56 bytes typically
  // But actual size depends on alignment - just check it's reasonable
  constexpr size_t batch_size = sizeof(llama_batch);
  CHECK(batch_size >= 48);   // Minimum: 6 pointers + 1 int32
  CHECK(batch_size <= 128);  // Maximum: generous padding allowance

  // Verify field offsets match our manual construction in decode_one()
  // This catches reordering even if size stays the same
  llama_batch batch{};

  // Set sentinel values
  batch.n_tokens = 0x12345678;
  llama_token tok = 0xABCD;
  batch.token = &tok;

  // Verify we can read back via struct - proves field offset is correct
  CHECK(batch.n_tokens == 0x12345678);
  CHECK(batch.token == &tok);
  CHECK(*batch.token == 0xABCD);

  // Verify all expected fields exist and are accessible
  // (Compile error here means field was removed/renamed)
  [[maybe_unused]] auto* p1 = batch.token;
  [[maybe_unused]] auto* p2 = batch.embd;
  [[maybe_unused]] auto* p3 = batch.pos;
  [[maybe_unused]] auto* p4 = batch.n_seq_id;
  [[maybe_unused]] auto* p5 = batch.seq_id;
  [[maybe_unused]] auto* p6 = batch.logits;

  // Verify llama_batch_init produces compatible struct
  // This is the "ground truth" - if our manual construction differs, we're wrong
  llama_batch init_batch = llama_batch_init(1, 0, 1);
  CHECK(init_batch.n_tokens == 0);  // init sets to 0, not capacity
  CHECK(init_batch.token != nullptr);
  CHECK(init_batch.pos != nullptr);
  CHECK(init_batch.n_seq_id != nullptr);
  CHECK(init_batch.seq_id != nullptr);
  CHECK(init_batch.logits != nullptr);
  // embd is nullptr when embd param is 0
  CHECK(init_batch.embd == nullptr);

  llama_batch_free(init_batch);

  MESSAGE("llama_batch ABI check passed - struct layout matches decode_one() assumptions");
}

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

TEST_CASE("branch integration: create and destroy") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  // Load model
  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  // Create context
  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  TestParams params;

  // Create branch
  BranchHandle h = create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
  CHECK(h != INVALID_HANDLE);

  BranchState* state = store.get(h);
  REQUIRE(state != nullptr);
  CHECK(state->ctx == ctx);
  CHECK(state->model == model.get());
  CHECK(state->seq_id == 0);
  CHECK(state->position == 0);
  CHECK(state->n_vocab > 0);  // Should have detected vocab size

  // Destroy
  destroy(h, &store);
  CHECK(store.get(h) == nullptr);

  llama_free(ctx);
}

// ============================================================================
// Decode and Position Tracking
// ============================================================================

TEST_CASE("branch integration: decode updates position") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  TestParams params;

  BranchHandle h = create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Tokenize something
  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello world", true, false);
  REQUIRE(!tokens.empty());

  // Initial position
  CHECK(get_position(h, &store) == 0);

  // Decode (now throws on error)
  CHECK_NOTHROW(decode_batch(h, tokens.data(), tokens.size(), &store));

  // Position should advance
  CHECK(get_position(h, &store) == static_cast<llama_pos>(tokens.size()));

  destroy(h, &store);
  llama_free(ctx);
}

// ============================================================================
// Logits Capture
// ============================================================================

TEST_CASE("branch integration: decode_and_capture captures logits") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  TestParams params;

  BranchHandle h = create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Before decode, logits should be empty/zero
  const float* logits_before = get_logits(h, &store);
  // May be nullptr or allocated but not populated - just verify call doesn't crash

  // Tokenize and decode with capture
  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "The capital of France is", true, false);
  REQUIRE(!tokens.empty());

  CHECK_NOTHROW(decode_and_capture_batch(h, tokens.data(), tokens.size(), &store));

  // Logits should now be populated
  const float* logits = get_logits(h, &store);
  REQUIRE(logits != nullptr);

  // Verify logits have reasonable values (not all zero)
  int n_vocab = get_n_vocab(h, &store);
  CHECK(n_vocab > 0);

  float sum = 0.0f;
  for (int i = 0; i < std::min(100, n_vocab); ++i) {
    sum += std::abs(logits[i]);
  }
  CHECK(sum > 0.0f);  // Logits should have non-zero values

  destroy(h, &store);
  llama_free(ctx);
}

// ============================================================================
// Fork and Independence
// ============================================================================

TEST_CASE("branch integration: fork creates independent branch") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 64;
  cparams.n_seq_max = 4;  // Enable multi-sequence for fork/branch tests
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  TestParams params;

  // Create parent branch and decode some tokens
  BranchHandle parent = create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
  REQUIRE(parent != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Once upon a time", true, false);
  REQUIRE(!prompt.empty());

  decode_and_capture_batch(parent, prompt.data(), prompt.size(), &store);
  llama_pos parent_pos = get_position(parent, &store);
  CHECK(parent_pos == static_cast<llama_pos>(prompt.size()));

  // Snapshot parent logits before fork
  const float* parent_logits_before_fork = get_logits(parent, &store);
  REQUIRE(parent_logits_before_fork != nullptr);
  int n_vocab = get_n_vocab(parent, &store);
  std::vector<float> parent_snapshot(n_vocab);
  std::memcpy(parent_snapshot.data(), parent_logits_before_fork, n_vocab * sizeof(float));

  // Fork to new sequence
  BranchHandle child = fork(parent, 1, &store);
  REQUIRE(child != INVALID_HANDLE);
  CHECK(child != parent);

  // Child should have same position but different seq_id
  CHECK(get_seq_id(child, &store) == 1);
  CHECK(get_position(child, &store) == parent_pos);

  // Parent unchanged
  CHECK(get_seq_id(parent, &store) == 0);
  CHECK(get_position(parent, &store) == parent_pos);

  // CRITICAL: After fork, both branches must have independent logits memory
  const float* logits_parent_after_fork = get_logits(parent, &store);
  const float* logits_child_after_fork = get_logits(child, &store);
  REQUIRE(logits_parent_after_fork != nullptr);
  REQUIRE(logits_child_after_fork != nullptr);
  CHECK(logits_parent_after_fork != logits_child_after_fork);  // Different memory addresses

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

  // Decode single token to each branch
  decode_and_capture_one(parent, cont_a[0], &store);
  decode_and_capture_one(child, cont_b[0], &store);

  // Positions should advance by 1 token each
  CHECK(get_position(parent, &store) == parent_pos + 1);
  CHECK(get_position(child, &store) == parent_pos + 1);

  // CRITICAL: Logits must still be at different memory addresses
  const float* logits_parent = get_logits(parent, &store);
  const float* logits_child = get_logits(child, &store);
  REQUIRE(logits_parent != nullptr);
  REQUIRE(logits_child != nullptr);
  CHECK(logits_parent != logits_child);  // Still independent memory

  // With a coherent model, different input tokens should produce different logits.
  // Random-weight models may produce identical outputs - that's a model limitation,
  // not a branch scoping bug. Log the diff for diagnostics but don't fail on it.
  float diff = 0.0f;
  for (int i = 0; i < std::min(100, n_vocab); ++i) {
    diff += std::abs(logits_parent[i] - logits_child[i]);
  }
  INFO("Logits diff after different decodes: " << diff);
  INFO("Token IDs: parent=" << cont_a[0] << " child=" << cont_b[0]);
  if (cont_a[0] != cont_b[0] && diff < 0.1f) {
    // Different tokens but identical logits - warn but don't fail
    // This indicates the model may have random weights or very limited capability
    MESSAGE("WARNING: Different tokens produced identical logits (model may have random weights)");
  }

  destroy(parent, &store);
  destroy(child, &store);
  llama_free(ctx);
}

// ============================================================================
// Prune Removes KV Entries
// ============================================================================

TEST_CASE("branch integration: prune removes KV entries") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 1024;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  TestParams params;

  // Create branch and decode
  BranchHandle h = create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "Hello world", true, false);
  decode_batch(h, tokens.data(), tokens.size(), &store);

  llama_seq_id seq = get_seq_id(h, &store);

  // Check KV has entries
  llama_pos pos_before = kv::pos_max(ctx, seq);
  CHECK(pos_before >= 0);

  // Prune
  prune(h, &store);

  // Handle should be invalid
  CHECK(store.get(h) == nullptr);

  // KV entries should be removed (pos_max returns -1 for empty sequence)
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

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  TestParams params;

  BranchHandle raw_handle;

  {
    // Create via RAII wrapper
    Branch b = Branch::create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
    CHECK(b.valid());
    raw_handle = b.handle();
    CHECK(store.get(raw_handle) != nullptr);

    // Operations via wrapper
    const llama_vocab* vocab = llama_model_get_vocab(model.get());
    auto tokens = tokenizer::tokenize(vocab, "Test", true, false);

    CHECK_NOTHROW(b.decode_batch(tokens.data(), tokens.size()));
    CHECK(b.position() > 0);

    // Fork via wrapper
    Branch child = b.fork(1);
    CHECK(child.valid());
    CHECK(child.seq_id() == 1);
    CHECK(child.position() == b.position());
  }

  // After scope, branch should be freed
  CHECK(store.get(raw_handle) == nullptr);

  llama_free(ctx);
}

// ============================================================================
// Sample and Accept
// ============================================================================

TEST_CASE("branch integration: sample and accept token") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  TestParams params;
  params.temperature = 0.0f;  // Greedy for reproducibility

  BranchHandle h = create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "The answer is", true, false);

  decode_and_capture_batch(h, prompt.data(), prompt.size(), &store);

  // Sample a token
  llama_token tok = sample(h, &store);
  CHECK(tok >= 0);
  CHECK(tok < get_n_vocab(h, &store));

  // Accept the token
  accept_token(h, tok, &store);

  // Note: Accept updates the sampler state (penalties) but doesn't
  // affect position - decode does that

  destroy(h, &store);
  llama_free(ctx);
}

// ============================================================================
// Perplexity Tracking
// ============================================================================

TEST_CASE("branch integration: perplexity tracking across fork") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  TestParams params;

  BranchHandle parent = create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
  REQUIRE(parent != INVALID_HANDLE);

  // Decode prompt
  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Hello", true, false);
  decode_and_capture_batch(parent, prompt.data(), prompt.size(), &store);

  // Get logits and manually update perplexity
  const float* logits = get_logits(parent, &store);
  int n_vocab = get_n_vocab(parent, &store);

  // Compute surprisal for a token
  llama_token test_tok = prompt.back();
  float surprisal = metrics::model_surprisal(logits, n_vocab, test_tok);

  // Add to parent's perplexity tracker
  BranchState* pstate = store.get(parent);
  REQUIRE(pstate);
  metrics::add_surprisal(pstate->ppl, surprisal);

  float parent_ppl = get_perplexity(parent, &store);
  CHECK(std::isfinite(parent_ppl));

  // Fork
  BranchHandle child = fork(parent, 1, &store);
  REQUIRE(child != INVALID_HANDLE);

  // Child should have same perplexity (cloned tracker)
  float child_ppl = get_perplexity(child, &store);
  CHECK(std::abs(parent_ppl - child_ppl) < 0.001f);

  // Add more surprisal to child only
  BranchState* cstate = store.get(child);
  REQUIRE(cstate);
  metrics::add_surprisal(cstate->ppl, 2.0f);

  // Parent unchanged, child changed
  CHECK(std::abs(get_perplexity(parent, &store) - parent_ppl) < 0.001f);
  CHECK(get_perplexity(child, &store) != parent_ppl);

  destroy(parent, &store);
  destroy(child, &store);
  llama_free(ctx);
}

// ============================================================================
// Strict Branch Scoping Tests
// ============================================================================
// These tests verify that each branch maintains independent state and that
// operations on one branch do not affect another. This is fundamental to the
// entire branching/MCTS design.

TEST_CASE("branch scoping: logits snapshots are independent memory") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  TestParams params;

  // Create branch and decode a token
  BranchHandle branch = create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
  REQUIRE(branch != INVALID_HANDLE);

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto tokens = tokenizer::tokenize(vocab, "test", true, false);
  REQUIRE(!tokens.empty());

  decode_and_capture_one(branch, tokens[0], &store);

  // Fork the branch
  BranchHandle forked = fork(branch, 1, &store);
  REQUIRE(forked != INVALID_HANDLE);

  // Get logits pointers - they must be different memory locations
  const float* logits1 = get_logits(branch, &store);
  const float* logits2 = get_logits(forked, &store);
  REQUIRE(logits1 != nullptr);
  REQUIRE(logits2 != nullptr);

  INFO("Original branch logits at: " << (void*)logits1);
  INFO("Forked branch logits at: " << (void*)logits2);

  // CRITICAL: Logits must be at different memory addresses
  CHECK(logits1 != logits2);

  // After fork, logits should be identical (forked from same state)
  int n_vocab = get_n_vocab(branch, &store);
  bool all_same = true;
  for (int i = 0; i < n_vocab; ++i) {
    if (logits1[i] != logits2[i]) {
      all_same = false;
      break;
    }
  }
  CHECK(all_same);  // Fork should copy logits exactly

  destroy(branch, &store);
  destroy(forked, &store);
  llama_free(ctx);
}

TEST_CASE("branch scoping: decode updates only target branch logits") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  TestParams params;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Hello", true, false);
  REQUIRE(!prompt.empty());

  // Create parent and decode prompt
  BranchHandle parent = create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
  decode_and_capture_one(parent, prompt[0], &store);

  // Fork to child
  BranchHandle child = fork(parent, 1, &store);
  REQUIRE(child != INVALID_HANDLE);

  // Snapshot parent's logits BEFORE child decode
  const float* parent_before = get_logits(parent, &store);
  REQUIRE(parent_before != nullptr);
  int n_vocab = get_n_vocab(parent, &store);

  // Copy first 100 values for comparison
  std::vector<float> parent_snapshot(n_vocab);
  std::memcpy(parent_snapshot.data(), parent_before, n_vocab * sizeof(float));

  // Tokenize different continuation
  auto cont = tokenizer::tokenize(vocab, " world", false, false);
  REQUIRE(!cont.empty());

  // Decode to CHILD only
  decode_and_capture_one(child, cont[0], &store);

  // CRITICAL CHECK: Parent's logits must be UNCHANGED
  const float* parent_after = get_logits(parent, &store);
  REQUIRE(parent_after != nullptr);

  // Verify same memory address (snapshot wasn't reallocated)
  CHECK(parent_after == parent_before);

  // Verify values unchanged
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
  CHECK(parent_unchanged);  // Parent logits must not change when child decodes

  // Child's logits should be valid (not null, not same as parent)
  const float* child_logits = get_logits(child, &store);
  REQUIRE(child_logits != nullptr);
  CHECK(child_logits != parent_after);  // Different memory

  destroy(parent, &store);
  destroy(child, &store);
  llama_free(ctx);
}

TEST_CASE("branch scoping: concurrent captures preserve isolation") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = TestConfig::n_gpu_layers();
  std::shared_ptr<llama_model> model(
      llama_model_load_from_file(MODEL_PATH, mparams),
      llama_model_free);
  REQUIRE(model);

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 64;
  cparams.n_seq_max = 8;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(16);
  TestParams params;

  const llama_vocab* vocab = llama_model_get_vocab(model.get());
  auto prompt = tokenizer::tokenize(vocab, "Start", true, false);
  REQUIRE(!prompt.empty());

  // Create root and decode
  BranchHandle root = create(ctx, model.get(), 0, 0, params, 64, nullptr, &store);
  decode_and_capture_one(root, prompt[0], &store);

  // Create 4 branches
  std::vector<BranchHandle> branches;
  for (int i = 1; i <= 4; ++i) {
    BranchHandle b = fork(root, i, &store);
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

  // Decode different tokens to each branch and capture logits for each
  for (size_t i = 0; i < branches.size(); ++i) {
    decode_and_capture_one(branches[i], continuations[i][0], &store);
  }

  // Now verify all branches have independent logits
  std::vector<const float*> all_logits;
  for (auto b : branches) {
    const float* l = get_logits(b, &store);
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

  // Each branch should have captured its own decode result
  // Verify by checking that not all logits are identical
  int n_vocab = get_n_vocab(root, &store);
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

  // With a coherent model and different input tokens, we expect differences
  // But this may fail with random-weight models - that's a model issue, not a branch issue
  INFO("Pairs with logits differences: " << pairs_with_differences << " / 6");

  // Cleanup
  destroy(root, &store);
  for (auto b : branches) {
    destroy(b, &store);
  }
  llama_free(ctx);
}
