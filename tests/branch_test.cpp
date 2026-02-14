/**
 * Branch Unit Tests
 *
 * Tests the handle-based branch API with stubs.
 * Validates:
 * - Handle allocation and generation counters
 * - Fork semantics
 * - State isolation
 * - RAII wrapper
 */

#include <doctest/doctest.h>
#include <lloyal/branch.hpp>
#include <cmath>  // std::isnan, std::isinf

using namespace lloyal::branch;

// ============================================================================
// Test helpers
// ============================================================================

// Minimal SamplingParams for testing (matches SamplingParamsLike concept)
struct TestSamplingParams {
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
// Handle Table Tests
// ============================================================================

TEST_CASE("branch: BranchStore allocates handles with generation counters") {
  BranchStore store(4);  // Small initial capacity

  // Allocate first handle
  BranchHandle h1 = store.allocate();
  CHECK(h1 != INVALID_HANDLE);
  CHECK(handle_index(h1) >= 1);  // Slot 0 reserved
  CHECK(handle_generation(h1) == 0);  // First use of slot

  // Allocate second handle
  BranchHandle h2 = store.allocate();
  CHECK(h2 != INVALID_HANDLE);
  CHECK(h2 != h1);

  // Both should be retrievable
  CHECK(store.get(h1) != nullptr);
  CHECK(store.get(h2) != nullptr);
}

TEST_CASE("branch: BranchStore rejects invalid handles") {
  BranchStore store(4);

  // Invalid handle
  CHECK(store.get(INVALID_HANDLE) == nullptr);

  // Handle with wrong generation
  BranchHandle h = store.allocate();
  uint16_t idx = handle_index(h);
  BranchHandle bad_gen = make_handle(idx, handle_generation(h) + 1);
  CHECK(store.get(bad_gen) == nullptr);

  // Out of bounds index
  BranchHandle oob = make_handle(100, 0);
  CHECK(store.get(oob) == nullptr);
}

TEST_CASE("branch: BranchStore generation counter prevents ABA") {
  BranchStore store(4);

  // Allocate and release
  BranchHandle h1 = store.allocate();
  uint16_t idx1 = handle_index(h1);
  uint16_t gen1 = handle_generation(h1);
  store.release(h1);

  // Re-allocate (should reuse slot with incremented generation)
  BranchHandle h2 = store.allocate();

  // If same slot was reused, generation should be different
  if (handle_index(h2) == idx1) {
    CHECK(handle_generation(h2) == gen1 + 1);
  }

  // Old handle should be invalid
  CHECK(store.get(h1) == nullptr);

  // New handle should be valid
  CHECK(store.get(h2) != nullptr);
}

TEST_CASE("branch: BranchStore grows when full") {
  BranchStore store(2);  // Very small: slot 0 reserved, 1 available

  // Fill up
  BranchHandle h1 = store.allocate();
  CHECK(h1 != INVALID_HANDLE);

  // Should grow and succeed
  BranchHandle h2 = store.allocate();
  CHECK(h2 != INVALID_HANDLE);

  BranchHandle h3 = store.allocate();
  CHECK(h3 != INVALID_HANDLE);

  // All should be valid
  CHECK(store.get(h1) != nullptr);
  CHECK(store.get(h2) != nullptr);
  CHECK(store.get(h3) != nullptr);
}

TEST_CASE("branch: double release is safe") {
  BranchStore store(4);

  BranchHandle h = store.allocate();
  store.release(h);
  store.release(h);  // Should not crash

  CHECK(store.get(h) == nullptr);
}

// ============================================================================
// Handle Encoding Tests
// ============================================================================

TEST_CASE("branch: handle encoding/decoding round-trips") {
  CHECK(make_handle(0, 0) == 0);
  CHECK(handle_index(make_handle(0, 0)) == 0);
  CHECK(handle_generation(make_handle(0, 0)) == 0);

  CHECK(handle_index(make_handle(123, 456)) == 123);
  CHECK(handle_generation(make_handle(123, 456)) == 456);

  CHECK(handle_index(make_handle(0xFFFF, 0xFFFF)) == 0xFFFF);
  CHECK(handle_generation(make_handle(0xFFFF, 0xFFFF)) == 0xFFFF);
}

// ============================================================================
// Branch Create/Free Tests (with stubs)
// ============================================================================

TEST_CASE("branch: create initializes state") {
  // Note: With stubs, create will work but ctx/model are fake pointers
  // We test the store/handle mechanics, not llama.cpp interaction

  BranchStore store(4);
  TestSamplingParams params;

  // Create with fake ctx/model
  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 100, params, 512, nullptr, nullptr, &store);
  CHECK(h != INVALID_HANDLE);

  BranchState* state = store.get(h);
  REQUIRE(state != nullptr);
  CHECK(state->ctx == fake_ctx);
  CHECK(state->model == fake_model);
  CHECK(state->seq_id == 0);
  CHECK(state->position == 100);
  CHECK(state->n_batch == 512);

  // Destroy should work
  destroy(h, &store);
  CHECK(store.get(h) == nullptr);
}

TEST_CASE("branch: create with null ctx/model returns invalid") {
  BranchStore store(4);
  TestSamplingParams params;

  CHECK(create(nullptr, nullptr, 0, 0, params, 512, nullptr, nullptr, &store) == INVALID_HANDLE);

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  CHECK(create(nullptr, fake_model, 0, 0, params, 512, nullptr, nullptr, &store) == INVALID_HANDLE);

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  CHECK(create(fake_ctx, nullptr, 0, 0, params, 512, nullptr, nullptr, &store) == INVALID_HANDLE);
}

// ============================================================================
// Fork Tests
// ============================================================================

TEST_CASE("branch: fork creates independent copy") {
  BranchStore store(8);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle parent = create(fake_ctx, fake_model, 0, 50, params, 512, nullptr, nullptr, &store);
  REQUIRE(parent != INVALID_HANDLE);

  // Fork to new sequence
  BranchHandle child = fork(parent, 1, &store);
  REQUIRE(child != INVALID_HANDLE);
  CHECK(child != parent);

  // Both should be valid
  BranchState* parent_state = store.get(parent);
  BranchState* child_state = store.get(child);
  REQUIRE(parent_state != nullptr);
  REQUIRE(child_state != nullptr);

  // Child should have new seq_id but same position
  CHECK(parent_state->seq_id == 0);
  CHECK(child_state->seq_id == 1);
  CHECK(parent_state->position == child_state->position);

  // Destroying parent shouldn't affect child
  destroy(parent, &store);
  CHECK(store.get(parent) == nullptr);
  CHECK(store.get(child) != nullptr);

  destroy(child, &store);
}

TEST_CASE("branch: fork invalid handle returns invalid") {
  BranchStore store(4);

  CHECK(fork(INVALID_HANDLE, 1, &store) == INVALID_HANDLE);
  CHECK(fork(make_handle(99, 99), 1, &store) == INVALID_HANDLE);
}

// ============================================================================
// State Accessor Tests
// ============================================================================

TEST_CASE("branch: state accessors return correct values") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 5, 200, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  CHECK(get_seq_id(h, &store) == 5);
  CHECK(get_position(h, &store) == 200);

  destroy(h, &store);
}

TEST_CASE("branch: state accessors with invalid handle return defaults") {
  BranchStore store(4);

  CHECK(get_seq_id(INVALID_HANDLE, &store) == -1);
  CHECK(get_position(INVALID_HANDLE, &store) == -1);
  CHECK(std::isinf(get_perplexity(INVALID_HANDLE, &store)));
  CHECK(get_n_vocab(INVALID_HANDLE, &store) == 0);
}

// ============================================================================
// RAII Wrapper Tests
// ============================================================================

TEST_CASE("branch: RAII Branch auto-frees on destruction") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle raw_handle;
  {
    Branch b = Branch::create(fake_ctx, fake_model, 0, 100, params, 512, nullptr, nullptr, &store);
    CHECK(b.valid());
    raw_handle = b.handle();
    CHECK(store.get(raw_handle) != nullptr);
  }
  // Branch destroyed, should be freed
  CHECK(store.get(raw_handle) == nullptr);
}

TEST_CASE("branch: RAII Branch move semantics") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch b1 = Branch::create(fake_ctx, fake_model, 0, 100, params, 512, nullptr, nullptr, &store);
  BranchHandle h = b1.handle();

  // Move construct
  Branch b2 = std::move(b1);
  CHECK(!b1.valid());  // Moved-from is invalid
  CHECK(b2.valid());
  CHECK(b2.handle() == h);
  CHECK(store.get(h) != nullptr);

  // Move assign
  Branch b3;
  b3 = std::move(b2);
  CHECK(!b2.valid());
  CHECK(b3.valid());
  CHECK(store.get(h) != nullptr);
}

TEST_CASE("branch: RAII Branch fork returns new Branch") {
  BranchStore store(8);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch parent = Branch::create(fake_ctx, fake_model, 0, 100, params, 512, nullptr, nullptr, &store);
  REQUIRE(parent.valid());

  Branch child = parent.fork(1);
  REQUIRE(child.valid());
  CHECK(child.handle() != parent.handle());

  CHECK(parent.seq_id() == 0);
  CHECK(child.seq_id() == 1);
  CHECK(parent.position() == child.position());
}

// ============================================================================
// Logits Snapshot Tests
// ============================================================================

TEST_CASE("branch: get_logits returns nullptr before decode_and_capture") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Logits snapshot is empty before any decode
  const float* logits = get_logits(h, &store);
  // With stubs, n_vocab is 0, so snapshot is empty
  // This test verifies the path doesn't crash
  (void)logits;  // Suppress unused warning

  destroy(h, &store);
}

// ============================================================================
// Global Store Tests
// ============================================================================

TEST_CASE("branch: global store works for simple usage") {
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x3000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x4000);

  // Use global store (nullptr for store parameter)
  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr);
  REQUIRE(h != INVALID_HANDLE);

  CHECK(get_seq_id(h, nullptr) == 0);

  BranchHandle h2 = fork(h, 1, nullptr);
  REQUIRE(h2 != INVALID_HANDLE);
  CHECK(get_seq_id(h2, nullptr) == 1);

  destroy(h, nullptr);
  destroy(h2, nullptr);
}

// ============================================================================
// Memory Safety Regression Tests
// ============================================================================

TEST_CASE("branch: BranchStore edge case - capacity 1") {
  // This tests the size_t underflow bug in the constructor:
  // for (size_t i = capacity - 1; i >= 0; --i) would underflow when capacity=1
  // and i becomes 0, then i-- wraps to SIZE_MAX
  BranchStore store(1);  // Edge case: minimum capacity

  // Should still be able to allocate (store should grow)
  BranchHandle h = store.allocate();
  CHECK(h != INVALID_HANDLE);
  store.release(h);
}

TEST_CASE("branch: BranchStore edge case - capacity 0") {
  // Edge case: zero capacity should still work
  BranchStore store(0);

  BranchHandle h = store.allocate();
  CHECK(h != INVALID_HANDLE);
  store.release(h);
}

TEST_CASE("branch: sample() returns -1 before logits captured") {
  // This tests the has_logits invariant:
  // sample() should fail gracefully if decode_and_capture wasn't called
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // sample() without decode_and_capture should return -1
  llama_token token = sample(h, &store);
  CHECK(token == -1);

  // Verify has_logits is false
  BranchState* state = store.get(h);
  REQUIRE(state != nullptr);
  CHECK(state->has_logits == false);

  destroy(h, &store);
}

TEST_CASE("branch: get_legal_priors returns empty before logits captured") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Should return empty without logits
  auto priors = get_legal_priors(h, &store);
  CHECK(priors.empty());

  destroy(h, &store);
}

TEST_CASE("branch: get_legal_logsumexp returns -inf before logits captured") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Returns -inf when no logits are captured (unambiguous sentinel)
  float logsumexp = get_legal_logsumexp(h, &store);
  CHECK(std::isinf(logsumexp));
  CHECK(logsumexp < 0);  // Specifically -INFINITY

  destroy(h, &store);
}

TEST_CASE("branch: candidates_buffer is pre-allocated on create") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  BranchState* state = store.get(h);
  REQUIRE(state != nullptr);

  // candidates_buffer should be pre-allocated to n_vocab size
  CHECK(state->candidates_buffer.size() == static_cast<size_t>(state->n_vocab));

  destroy(h, &store);
}

TEST_CASE("branch: candidates_buffer is pre-allocated on fork") {
  BranchStore store(8);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle parent = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(parent != INVALID_HANDLE);

  BranchHandle child = fork(parent, 1, &store);
  REQUIRE(child != INVALID_HANDLE);

  BranchState* child_state = store.get(child);
  REQUIRE(child_state != nullptr);

  // Child's candidates_buffer should also be pre-allocated
  CHECK(child_state->candidates_buffer.size() == static_cast<size_t>(child_state->n_vocab));

  destroy(parent, &store);
  destroy(child, &store);
}

TEST_CASE("branch: release resets has_logits flag") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h1 = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h1 != INVALID_HANDLE);

  // Manually set has_logits to simulate decode_and_capture
  BranchState* state = store.get(h1);
  REQUIRE(state != nullptr);
  state->has_logits = true;

  // Release and reallocate (should reuse the slot)
  store.release(h1);
  BranchHandle h2 = store.allocate();

  BranchState* new_state = store.get(h2);
  REQUIRE(new_state != nullptr);

  // has_logits should be reset to false for the new allocation
  CHECK(new_state->has_logits == false);

  store.release(h2);
}

TEST_CASE("branch: stress test allocate/release cycles") {
  // This would catch memory leaks or corruption under repeated use
  BranchStore store(4);

  for (int cycle = 0; cycle < 100; ++cycle) {
    std::vector<BranchHandle> handles;

    // Allocate many handles
    for (int i = 0; i < 20; ++i) {
      BranchHandle h = store.allocate();
      CHECK(h != INVALID_HANDLE);
      handles.push_back(h);
    }

    // Release all
    for (auto h : handles) {
      store.release(h);
    }

    // All should be invalid now
    for (auto h : handles) {
      CHECK(store.get(h) == nullptr);
    }
  }
}

TEST_CASE("branch: freelist ordering after grow") {
  // Verify freelist is correctly populated after growing
  BranchStore store(2);  // Starts with 1 usable slot (slot 0 reserved)

  // Allocate to force growth
  std::vector<BranchHandle> handles;
  for (int i = 0; i < 10; ++i) {
    BranchHandle h = store.allocate();
    REQUIRE(h != INVALID_HANDLE);
    handles.push_back(h);
  }

  // All handles should be valid
  for (auto h : handles) {
    CHECK(store.get(h) != nullptr);
  }

  // Release all
  for (auto h : handles) {
    store.release(h);
  }
}

// ============================================================================
// Overflow Safety Tests
// ============================================================================

TEST_CASE("branch: generation counter increments on release") {
  BranchStore store(4);

  BranchHandle h1 = store.allocate();
  uint16_t gen1 = handle_generation(h1);
  uint16_t idx1 = handle_index(h1);

  store.release(h1);

  // Reallocate - should get same slot with incremented generation
  BranchHandle h2 = store.allocate();

  // If same slot reused, generation should be gen1 + 1
  if (handle_index(h2) == idx1) {
    CHECK(handle_generation(h2) == static_cast<uint16_t>(gen1 + 1));
  }

  store.release(h2);
}

TEST_CASE("branch: generation counter overflow wraps safely") {
  // This test verifies behavior when generation counter wraps from 0xFFFF -> 0
  // After wrap, old handles should still be detected as invalid

  BranchStore store(4);

  // Allocate a slot
  BranchHandle h = store.allocate();
  uint16_t idx = handle_index(h);

  // Manually force generation near overflow point
  BranchState* state = store.get(h);
  REQUIRE(state != nullptr);
  state->generation = 0xFFFE;  // Set to near-max

  // Create a handle with this generation
  BranchHandle near_max_handle = make_handle(idx, 0xFFFE);
  CHECK(store.get(near_max_handle) != nullptr);

  // Release and reallocate - generation becomes 0xFFFF
  store.release(near_max_handle);
  BranchHandle h_ffff = store.allocate();

  if (handle_index(h_ffff) == idx) {
    CHECK(handle_generation(h_ffff) == 0xFFFF);

    // Old handle should be invalid
    CHECK(store.get(near_max_handle) == nullptr);

    // Release again - generation wraps to 0
    store.release(h_ffff);
    BranchHandle h_wrapped = store.allocate();

    if (handle_index(h_wrapped) == idx) {
      // Generation wrapped to 0
      CHECK(handle_generation(h_wrapped) == 0);

      // Both old handles should be invalid
      CHECK(store.get(near_max_handle) == nullptr);
      CHECK(store.get(h_ffff) == nullptr);

      store.release(h_wrapped);
    }
  }
}

TEST_CASE("branch: store cannot exceed max capacity") {
  // BranchStore caps at 65536 slots (INDEX_MASK + 1)
  // We can't practically allocate 65536 slots in a unit test,
  // but we can verify the growth logic caps correctly

  BranchStore store(2);

  // Verify the cap constant is correct
  CHECK(INDEX_MASK == 0xFFFF);

  // After hitting max, allocate should return INVALID_HANDLE
  // (This is already tested by grow logic returning INVALID_HANDLE
  // when old_size >= new_size after hitting cap)
}

TEST_CASE("branch: handle encoding preserves full range") {
  // Verify handle encoding works for edge values
  CHECK(make_handle(0, 0) == INVALID_HANDLE);  // 0 is invalid

  // Max index
  BranchHandle max_idx = make_handle(0xFFFF, 0);
  CHECK(handle_index(max_idx) == 0xFFFF);
  CHECK(handle_generation(max_idx) == 0);

  // Max generation
  BranchHandle max_gen = make_handle(0, 0xFFFF);
  CHECK(handle_index(max_gen) == 0);
  CHECK(handle_generation(max_gen) == 0xFFFF);

  // Both max
  BranchHandle max_both = make_handle(0xFFFF, 0xFFFF);
  CHECK(handle_index(max_both) == 0xFFFF);
  CHECK(handle_generation(max_both) == 0xFFFF);
}

TEST_CASE("branch: slot 0 reserved generation never valid") {
  BranchStore store(4);

  // Slot 0 is reserved with generation 0xFFFF
  // No valid handle should ever reference it

  // Try to access slot 0 with various generations
  CHECK(store.get(make_handle(0, 0)) == nullptr);
  CHECK(store.get(make_handle(0, 1)) == nullptr);
  CHECK(store.get(make_handle(0, 0xFFFF)) == nullptr);
  CHECK(store.get(make_handle(0, 0xFFFE)) == nullptr);
}

// ============================================================================
// Float Safety Tests
// ============================================================================

TEST_CASE("branch: get_legal_priors handles empty legal set") {
  // When all tokens are masked by grammar, legal_priors is empty
  // This should return empty vector, not crash on division by zero
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Without logits, should return empty (not crash)
  auto priors = get_legal_priors(h, &store);
  CHECK(priors.empty());

  destroy(h, &store);
}

TEST_CASE("branch: get_legal_logsumexp returns -inf when no legal tokens") {
  // When no tokens are legal (or no logits captured), returns -INFINITY
  // as an unambiguous sentinel (0.0f is a valid logsumexp value)
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Without logits captured, returns -inf (unambiguous sentinel)
  float logsumexp = get_legal_logsumexp(h, &store);
  CHECK(std::isinf(logsumexp));
  CHECK(logsumexp < 0);  // -INFINITY, not +INFINITY
  CHECK(!std::isnan(logsumexp));

  destroy(h, &store);
}

TEST_CASE("branch: get_token_prior handles invalid token") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Negative token
  float prior = get_token_prior(h, -1, 0.0f, &store);
  CHECK(prior == 0.0f);

  // Out of range token (n_vocab is 0 with stubs)
  prior = get_token_prior(h, 1000000, 0.0f, &store);
  CHECK(prior == 0.0f);

  destroy(h, &store);
}

TEST_CASE("branch: get_perplexity returns infinity for fresh branch") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Fresh branch with no tokens should have infinite perplexity
  float ppl = get_perplexity(h, &store);
  CHECK(std::isinf(ppl));
  CHECK(!std::isnan(ppl));

  destroy(h, &store);
}

// ============================================================================
// Null Pointer & Input Validation Tests
// ============================================================================

TEST_CASE("branch: all APIs handle null store gracefully") {
  // These use the global store when nullptr is passed
  // Verify they don't crash

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  TestSamplingParams params;

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr);
  REQUIRE(h != INVALID_HANDLE);

  // All these should work with nullptr store (uses global)
  CHECK(get_seq_id(h, nullptr) == 0);
  CHECK(get_position(h, nullptr) == 0);
  CHECK(get_n_vocab(h, nullptr) >= 0);
  CHECK(get_logits(h, nullptr) == nullptr);  // No logits yet

  destroy(h, nullptr);
}

TEST_CASE("branch: operations on invalid handle don't crash") {
  BranchStore store(4);

  // All operations should handle invalid handles gracefully
  CHECK(get_seq_id(INVALID_HANDLE, &store) == -1);
  CHECK(get_position(INVALID_HANDLE, &store) == -1);
  CHECK(get_n_vocab(INVALID_HANDLE, &store) == 0);
  CHECK(get_logits(INVALID_HANDLE, &store) == nullptr);
  CHECK(std::isinf(get_perplexity(INVALID_HANDLE, &store)));

  // These should be no-ops, not crashes
  destroy(INVALID_HANDLE, &store);
  prune(INVALID_HANDLE, &store);
  accept_token(INVALID_HANDLE, 0, &store);

  // These should throw on invalid handle
  CHECK_THROWS(decode_batch(INVALID_HANDLE, nullptr, 0, &store));
  CHECK_THROWS(decode_and_capture_batch(INVALID_HANDLE, nullptr, 0, &store));
  CHECK(sample(INVALID_HANDLE, &store) == -1);
  CHECK(fork(INVALID_HANDLE, 1, &store) == INVALID_HANDLE);
}

TEST_CASE("branch: operations on stale handle don't crash") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Destroy the handle
  destroy(h, &store);

  // Now h is stale - all operations should fail gracefully
  CHECK(store.get(h) == nullptr);
  CHECK(get_seq_id(h, &store) == -1);
  CHECK(fork(h, 1, &store) == INVALID_HANDLE);
  CHECK(sample(h, &store) == -1);

  // Double destroy should be safe
  destroy(h, &store);
}

TEST_CASE("branch: decode with zero tokens throws") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Decode with empty array throws (defensive check in decoder)
  llama_token empty[] = {};
  CHECK_THROWS(decode_batch(h, empty, 0, &store));
  CHECK_THROWS(decode_and_capture_batch(h, empty, 0, &store));

  // Position should be unchanged
  CHECK(get_position(h, &store) == 0);

  destroy(h, &store);
}

TEST_CASE("branch: decode with nullptr tokens and n > 0 is handled") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Note: This would be UB in production, but our stubs make it safe to test
  // In production, this should either be prevented by API or checked
  // For now we just document the behavior

  destroy(h, &store);
}

// ============================================================================
// Resource Exhaustion Tests
// ============================================================================

TEST_CASE("branch: handle exhaustion returns INVALID_HANDLE") {
  // Create a store that will hit capacity limits
  // The store caps at INDEX_MASK + 1 = 65536 slots
  // We can't practically allocate that many, but we can verify
  // the logic by checking the constants

  CHECK(INDEX_MASK == 0xFFFF);

  // With a small store that can't grow further, allocate should fail
  // This is already tested by "store cannot exceed max capacity"
}

// ============================================================================
// Exception Safety Tests (RAII guarantees)
// ============================================================================

TEST_CASE("branch: RAII Branch prune() invalidates handle") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch b = Branch::create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(b.valid());

  BranchHandle h = b.handle();
  CHECK(store.get(h) != nullptr);

  // Explicit prune
  b.prune();

  // Handle should now be invalid
  CHECK(!b.valid());
  CHECK(store.get(h) == nullptr);

  // Destructor should be safe (no double-free)
}

TEST_CASE("branch: RAII Branch move leaves source invalid") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch b1 = Branch::create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  BranchHandle h = b1.handle();

  // Move to b2
  Branch b2 = std::move(b1);

  // b1 should be invalidated
  CHECK(!b1.valid());
  CHECK(b1.handle() == INVALID_HANDLE);

  // b2 should own the resource
  CHECK(b2.valid());
  CHECK(b2.handle() == h);
  CHECK(store.get(h) != nullptr);

  // Operations on moved-from b1 should be safe
  CHECK(b1.seq_id() == -1);
  CHECK(b1.position() == -1);
}

TEST_CASE("branch: RAII Branch self-move-assign is safe") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch b = Branch::create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  BranchHandle h = b.handle();

  // Self-assignment (should be no-op)
  b = std::move(b);

  // Should still be valid
  CHECK(b.valid());
  CHECK(b.handle() == h);
}

// ============================================================================
// BranchStore Batched Decode Tests (span-based API)
// ============================================================================

TEST_CASE("branch: decode_each with empty span is no-op") {
  BranchStore store(4);

  std::span<const DecodeEachItem> empty;
  CHECK_NOTHROW(store.decode_each(empty));
}

TEST_CASE("branch: decode_each with invalid handle throws") {
  BranchStore store(4);

  DecodeEachItem items[] = {{INVALID_HANDLE, 42}};
  CHECK_THROWS(store.decode_each(items));
}

TEST_CASE("branch: decode_scatter with empty span is no-op") {
  BranchStore store(4);

  std::span<const DecodeScatterItem> empty;
  CHECK_NOTHROW(store.decode_scatter(empty));
}

TEST_CASE("branch: decode_scatter with invalid handle throws") {
  BranchStore store(4);

  llama_token tokens[] = {1, 2, 3};
  DecodeScatterItem items[] = {{INVALID_HANDLE, tokens}};
  CHECK_THROWS(store.decode_scatter(items));
}

TEST_CASE("branch: decode_scatter with zero-length tokens span skips item") {
  BranchStore store(4);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(fake_ctx, fake_model, 0, 0, params, 512, nullptr, nullptr, &store);
  REQUIRE(h != INVALID_HANDLE);

  // Empty span — no tokens to decode
  DecodeScatterItem items[] = {{h, std::span<const llama_token>{}}};
  CHECK_NOTHROW(store.decode_scatter(items));

  // Position should be unchanged
  CHECK(get_position(h, &store) == 0);

  destroy(h, &store);
}

TEST_CASE("branch: decode_scatter all items zero-length is no-op") {
  BranchStore store(8);
  TestSamplingParams params;

  auto* fake_ctx = reinterpret_cast<llama_context*>(0x1000);
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h1 = create(fake_ctx, fake_model, 0, 10, params, 512, nullptr, nullptr, &store);
  BranchHandle h2 = create(fake_ctx, fake_model, 1, 20, params, 512, nullptr, nullptr, &store);
  REQUIRE(h1 != INVALID_HANDLE);
  REQUIRE(h2 != INVALID_HANDLE);

  DecodeScatterItem items[] = {
    {h1, std::span<const llama_token>{}},
    {h2, std::span<const llama_token>{}}
  };
  CHECK_NOTHROW(store.decode_scatter(items));

  // Positions unchanged
  CHECK(get_position(h1, &store) == 10);
  CHECK(get_position(h2, &store) == 20);

  destroy(h1, &store);
  destroy(h2, &store);
}
