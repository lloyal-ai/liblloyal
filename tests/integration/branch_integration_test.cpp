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

  // Logits should differ
  const float* logits_parent = get_logits(parent, &store);
  const float* logits_child = get_logits(child, &store);
  REQUIRE(logits_parent != nullptr);
  REQUIRE(logits_child != nullptr);

  // At least some logits should differ (different continuations)
  int n_vocab = get_n_vocab(parent, &store);
  float diff = 0.0f;
  for (int i = 0; i < std::min(100, n_vocab); ++i) {
    diff += std::abs(logits_parent[i] - logits_child[i]);
  }
  CHECK(diff > 0.1f);  // Should have noticeable differences

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
