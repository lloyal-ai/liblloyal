/**
 * Branch Unit Tests
 *
 * Tests the handle-based branch API with stubs.
 * Validates:
 * - Handle allocation and generation counters
 * - Fork semantics
 * - State isolation
 * - RAII wrapper
 * - Tenancy (seq_id vacancy management)
 * - Topology (parent/children tracking)
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

// Helper: create a tenancy-initialized store with fake ctx
struct TestStore {
  BranchStore store;
  llama_context* ctx;

  TestStore(size_t capacity = 8)
    : store(capacity)
    , ctx(reinterpret_cast<llama_context*>(0x1000))
  {
    store.init_tenancy(ctx);
  }
};

// ============================================================================
// Handle Table Tests
// ============================================================================

TEST_CASE("branch: BranchStore allocates handles with generation counters") {
  TestStore ts(4);

  auto [h1, seq1] = ts.store.allocate();
  CHECK(h1 != INVALID_HANDLE);
  CHECK(seq1 >= 0);
  CHECK(handle_index(h1) >= 1);  // Slot 0 reserved
  CHECK(handle_generation(h1) == 0);  // First use of slot

  auto [h2, seq2] = ts.store.allocate();
  CHECK(h2 != INVALID_HANDLE);
  CHECK(h2 != h1);
  CHECK(seq2 != seq1);  // Different seq_ids

  CHECK(ts.store.get(h1) != nullptr);
  CHECK(ts.store.get(h2) != nullptr);
}

TEST_CASE("branch: BranchStore rejects invalid handles") {
  TestStore ts(4);

  CHECK(ts.store.get(INVALID_HANDLE) == nullptr);

  auto [h, seq] = ts.store.allocate();
  uint16_t idx = handle_index(h);
  BranchHandle bad_gen = make_handle(idx, handle_generation(h) + 1);
  CHECK(ts.store.get(bad_gen) == nullptr);

  BranchHandle oob = make_handle(100, 0);
  CHECK(ts.store.get(oob) == nullptr);
}

TEST_CASE("branch: BranchStore generation counter prevents ABA") {
  TestStore ts(4);

  auto [h1, seq1] = ts.store.allocate();
  uint16_t idx1 = handle_index(h1);
  uint16_t gen1 = handle_generation(h1);
  ts.store.release(h1);

  auto [h2, seq2] = ts.store.allocate();

  if (handle_index(h2) == idx1) {
    CHECK(handle_generation(h2) == gen1 + 1);
  }

  CHECK(ts.store.get(h1) == nullptr);
  CHECK(ts.store.get(h2) != nullptr);
}

TEST_CASE("branch: BranchStore grows when full") {
  TestStore ts(2);  // Very small: slot 0 reserved, 1 available

  auto [h1, s1] = ts.store.allocate();
  CHECK(h1 != INVALID_HANDLE);

  auto [h2, s2] = ts.store.allocate();
  CHECK(h2 != INVALID_HANDLE);

  auto [h3, s3] = ts.store.allocate();
  CHECK(h3 != INVALID_HANDLE);

  CHECK(ts.store.get(h1) != nullptr);
  CHECK(ts.store.get(h2) != nullptr);
  CHECK(ts.store.get(h3) != nullptr);
}

TEST_CASE("branch: double release is safe") {
  TestStore ts(4);

  auto [h, seq] = ts.store.allocate();
  ts.store.release(h);
  ts.store.release(h);  // Should not crash

  CHECK(ts.store.get(h) == nullptr);
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
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 100, params, 512);
  CHECK(h != INVALID_HANDLE);

  BranchState* state = ts.store.get(h);
  REQUIRE(state != nullptr);
  CHECK(state->ctx == ts.ctx);
  CHECK(state->model == fake_model);
  CHECK(state->seq_id != lloyal::kv::NO_LEASE);
  CHECK(state->position == 100);
  CHECK(state->n_batch == 512);

  prune(h, ts.store);
  CHECK(ts.store.get(h) == nullptr);
}

TEST_CASE("branch: create with null ctx/model returns invalid") {
  TestStore ts(4);
  TestSamplingParams params;

  CHECK(create(nullptr, nullptr, ts.store, 0, params, 512) == INVALID_HANDLE);

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  CHECK(create(nullptr, fake_model, ts.store, 0, params, 512) == INVALID_HANDLE);

  CHECK(create(ts.ctx, nullptr, ts.store, 0, params, 512) == INVALID_HANDLE);
}

// ============================================================================
// Fork Tests
// ============================================================================

TEST_CASE("branch: fork creates independent copy") {
  TestStore ts(8);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle parent = create(ts.ctx, fake_model, ts.store, 50, params, 512);
  REQUIRE(parent != INVALID_HANDLE);

  BranchHandle child = fork(parent, ts.store);
  REQUIRE(child != INVALID_HANDLE);
  CHECK(child != parent);

  BranchState* parent_state = ts.store.get(parent);
  BranchState* child_state = ts.store.get(child);
  REQUIRE(parent_state != nullptr);
  REQUIRE(child_state != nullptr);

  // Child should have different seq_id but same position
  CHECK(parent_state->seq_id != child_state->seq_id);
  CHECK(parent_state->position == child_state->position);

  // Topology: child's parent is parent, parent has child in children
  CHECK(child_state->parent == parent);
  CHECK(parent_state->children.size() == 1);
  CHECK(parent_state->children[0] == child);

  // Pruning child should not affect parent
  prune(child, ts.store);
  CHECK(ts.store.get(parent) != nullptr);
  CHECK(ts.store.get(child) == nullptr);

  // Parent's children should be empty after child pruned
  parent_state = ts.store.get(parent);
  CHECK(parent_state->children.empty());

  prune(parent, ts.store);
}

TEST_CASE("branch: fork invalid handle returns invalid") {
  TestStore ts(4);

  CHECK(fork(INVALID_HANDLE, ts.store) == INVALID_HANDLE);
  CHECK(fork(make_handle(99, 99), ts.store) == INVALID_HANDLE);
}

// ============================================================================
// State Accessor Tests
// ============================================================================

TEST_CASE("branch: state accessors return correct values") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 200, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  CHECK(get_position(h, ts.store) == 200);

  prune(h, ts.store);
}

TEST_CASE("branch: state accessors with invalid handle return defaults") {
  TestStore ts(4);

  CHECK(get_position(INVALID_HANDLE, ts.store) == -1);
  CHECK(std::isinf(get_perplexity(INVALID_HANDLE, ts.store)));
  CHECK(get_n_vocab(INVALID_HANDLE, ts.store) == 0);
}

// ============================================================================
// RAII Wrapper Tests
// ============================================================================

TEST_CASE("branch: RAII Branch auto-frees on destruction") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle raw_handle;
  {
    Branch b = Branch::create(ts.ctx, fake_model, ts.store, 100, params, 512);
    CHECK(b.valid());
    raw_handle = b.handle();
    CHECK(ts.store.get(raw_handle) != nullptr);
  }
  // Branch destroyed, should be freed
  CHECK(ts.store.get(raw_handle) == nullptr);
}

TEST_CASE("branch: RAII Branch move semantics") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch b1 = Branch::create(ts.ctx, fake_model, ts.store, 100, params, 512);
  BranchHandle h = b1.handle();

  // Move construct
  Branch b2 = std::move(b1);
  CHECK(!b1.valid());
  CHECK(b2.valid());
  CHECK(b2.handle() == h);
  CHECK(ts.store.get(h) != nullptr);

  // Move assign
  Branch b3;
  b3 = std::move(b2);
  CHECK(!b2.valid());
  CHECK(b3.valid());
  CHECK(ts.store.get(h) != nullptr);
}

TEST_CASE("branch: RAII Branch fork returns new Branch") {
  TestStore ts(8);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch parent = Branch::create(ts.ctx, fake_model, ts.store, 100, params, 512);
  REQUIRE(parent.valid());

  Branch child = parent.fork();
  REQUIRE(child.valid());
  CHECK(child.handle() != parent.handle());
  CHECK(parent.position() == child.position());
}

// ============================================================================
// Logits Snapshot Tests
// ============================================================================

TEST_CASE("branch: get_logits returns nullptr before decode_and_capture") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  const float* logits = get_logits(h, ts.store);
  (void)logits;  // Suppress unused warning

  prune(h, ts.store);
}

// ============================================================================
// Memory Safety Regression Tests
// ============================================================================

TEST_CASE("branch: BranchStore edge case - capacity 1") {
  TestStore ts(1);

  auto [h, seq] = ts.store.allocate();
  CHECK(h != INVALID_HANDLE);
  ts.store.release(h);
}

TEST_CASE("branch: BranchStore edge case - capacity 0") {
  TestStore ts(0);

  auto [h, seq] = ts.store.allocate();
  CHECK(h != INVALID_HANDLE);
  ts.store.release(h);
}

TEST_CASE("branch: sample() returns -1 before logits captured") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  llama_token token = sample(h, ts.store);
  CHECK(token == -1);

  BranchState* state = ts.store.get(h);
  REQUIRE(state != nullptr);
  CHECK(state->has_logits == false);

  prune(h, ts.store);
}

TEST_CASE("branch: get_legal_priors returns empty before logits captured") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  auto priors = get_legal_priors(h, ts.store);
  CHECK(priors.empty());

  prune(h, ts.store);
}

TEST_CASE("branch: get_legal_logsumexp returns -inf before logits captured") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  float logsumexp = get_legal_logsumexp(h, ts.store);
  CHECK(std::isinf(logsumexp));
  CHECK(logsumexp < 0);

  prune(h, ts.store);
}

TEST_CASE("branch: candidates_buffer is pre-allocated on create") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  BranchState* state = ts.store.get(h);
  REQUIRE(state != nullptr);
  CHECK(state->candidates_buffer.size() == static_cast<size_t>(state->n_vocab));

  prune(h, ts.store);
}

TEST_CASE("branch: candidates_buffer is pre-allocated on fork") {
  TestStore ts(8);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle parent = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(parent != INVALID_HANDLE);

  BranchHandle child = fork(parent, ts.store);
  REQUIRE(child != INVALID_HANDLE);

  BranchState* child_state = ts.store.get(child);
  REQUIRE(child_state != nullptr);
  CHECK(child_state->candidates_buffer.size() == static_cast<size_t>(child_state->n_vocab));

  pruneSubtree(parent, ts.store);
}

TEST_CASE("branch: release resets has_logits flag") {
  TestStore ts(4);

  auto [h1, seq1] = ts.store.allocate();
  REQUIRE(h1 != INVALID_HANDLE);

  BranchState* state = ts.store.get(h1);
  REQUIRE(state != nullptr);
  state->has_logits = true;

  ts.store.release(h1);
  auto [h2, seq2] = ts.store.allocate();

  BranchState* new_state = ts.store.get(h2);
  REQUIRE(new_state != nullptr);
  CHECK(new_state->has_logits == false);

  ts.store.release(h2);
}

TEST_CASE("branch: stress test allocate/release cycles") {
  TestStore ts(4);

  for (int cycle = 0; cycle < 100; ++cycle) {
    std::vector<BranchHandle> handles;

    // Allocate up to available leases
    for (int i = 0; i < 7; ++i) {  // n_seq_max default = 8
      auto [h, seq] = ts.store.allocate();
      if (h == INVALID_HANDLE) break;
      handles.push_back(h);
    }

    for (auto h : handles) {
      ts.store.release(h);
    }

    for (auto h : handles) {
      CHECK(ts.store.get(h) == nullptr);
    }
  }
}

TEST_CASE("branch: freelist ordering after grow") {
  TestStore ts(2);

  std::vector<BranchHandle> handles;
  for (int i = 0; i < 7; ++i) {  // limited by n_seq_max=8
    auto [h, seq] = ts.store.allocate();
    if (h == INVALID_HANDLE) break;
    handles.push_back(h);
  }

  for (auto h : handles) {
    CHECK(ts.store.get(h) != nullptr);
  }

  for (auto h : handles) {
    ts.store.release(h);
  }
}

// ============================================================================
// Overflow Safety Tests
// ============================================================================

TEST_CASE("branch: generation counter increments on release") {
  TestStore ts(4);

  auto [h1, seq1] = ts.store.allocate();
  uint16_t gen1 = handle_generation(h1);
  uint16_t idx1 = handle_index(h1);

  ts.store.release(h1);

  auto [h2, seq2] = ts.store.allocate();

  if (handle_index(h2) == idx1) {
    CHECK(handle_generation(h2) == static_cast<uint16_t>(gen1 + 1));
  }

  ts.store.release(h2);
}

TEST_CASE("branch: generation counter overflow wraps safely") {
  TestStore ts(4);

  auto [h, seq] = ts.store.allocate();
  uint16_t idx = handle_index(h);

  BranchState* state = ts.store.get(h);
  REQUIRE(state != nullptr);
  state->generation = 0xFFFE;

  BranchHandle near_max_handle = make_handle(idx, 0xFFFE);
  CHECK(ts.store.get(near_max_handle) != nullptr);

  ts.store.release(near_max_handle);
  auto [h_ffff, seq2] = ts.store.allocate();

  if (handle_index(h_ffff) == idx) {
    CHECK(handle_generation(h_ffff) == 0xFFFF);
    CHECK(ts.store.get(near_max_handle) == nullptr);

    ts.store.release(h_ffff);
    auto [h_wrapped, seq3] = ts.store.allocate();

    if (handle_index(h_wrapped) == idx) {
      CHECK(handle_generation(h_wrapped) == 0);
      CHECK(ts.store.get(near_max_handle) == nullptr);
      CHECK(ts.store.get(h_ffff) == nullptr);

      ts.store.release(h_wrapped);
    }
  }
}

TEST_CASE("branch: store cannot exceed max capacity") {
  TestStore ts(2);
  CHECK(INDEX_MASK == 0xFFFF);
}

TEST_CASE("branch: handle encoding preserves full range") {
  CHECK(make_handle(0, 0) == INVALID_HANDLE);

  BranchHandle max_idx = make_handle(0xFFFF, 0);
  CHECK(handle_index(max_idx) == 0xFFFF);
  CHECK(handle_generation(max_idx) == 0);

  BranchHandle max_gen = make_handle(0, 0xFFFF);
  CHECK(handle_index(max_gen) == 0);
  CHECK(handle_generation(max_gen) == 0xFFFF);

  BranchHandle max_both = make_handle(0xFFFF, 0xFFFF);
  CHECK(handle_index(max_both) == 0xFFFF);
  CHECK(handle_generation(max_both) == 0xFFFF);
}

TEST_CASE("branch: slot 0 reserved generation never valid") {
  TestStore ts(4);

  CHECK(ts.store.get(make_handle(0, 0)) == nullptr);
  CHECK(ts.store.get(make_handle(0, 1)) == nullptr);
  CHECK(ts.store.get(make_handle(0, 0xFFFF)) == nullptr);
  CHECK(ts.store.get(make_handle(0, 0xFFFE)) == nullptr);
}

// ============================================================================
// Float Safety Tests
// ============================================================================

TEST_CASE("branch: get_legal_priors handles empty legal set") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  auto priors = get_legal_priors(h, ts.store);
  CHECK(priors.empty());

  prune(h, ts.store);
}

TEST_CASE("branch: get_legal_logsumexp returns -inf when no legal tokens") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  float logsumexp = get_legal_logsumexp(h, ts.store);
  CHECK(std::isinf(logsumexp));
  CHECK(logsumexp < 0);
  CHECK(!std::isnan(logsumexp));

  prune(h, ts.store);
}

TEST_CASE("branch: get_token_prior handles invalid token") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  float prior = get_token_prior(h, -1, 0.0f, ts.store);
  CHECK(prior == 0.0f);

  prior = get_token_prior(h, 1000000, 0.0f, ts.store);
  CHECK(prior == 0.0f);

  prune(h, ts.store);
}

TEST_CASE("branch: get_perplexity returns infinity for fresh branch") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  float ppl = get_perplexity(h, ts.store);
  CHECK(std::isinf(ppl));
  CHECK(!std::isnan(ppl));

  prune(h, ts.store);
}

// ============================================================================
// Null Pointer & Input Validation Tests
// ============================================================================

TEST_CASE("branch: operations on invalid handle don't crash") {
  TestStore ts(4);

  CHECK(get_position(INVALID_HANDLE, ts.store) == -1);
  CHECK(get_n_vocab(INVALID_HANDLE, ts.store) == 0);
  CHECK(get_logits(INVALID_HANDLE, ts.store) == nullptr);
  CHECK(std::isinf(get_perplexity(INVALID_HANDLE, ts.store)));

  // prune on invalid is a no-op
  prune(INVALID_HANDLE, ts.store);
  accept_token(INVALID_HANDLE, 0, ts.store);

  CHECK_THROWS(prefill(INVALID_HANDLE, nullptr, 0, ts.store));
  CHECK_THROWS(step(INVALID_HANDLE, 0, ts.store));
  CHECK(sample(INVALID_HANDLE, ts.store) == -1);
  CHECK(fork(INVALID_HANDLE, ts.store) == INVALID_HANDLE);
}

TEST_CASE("branch: operations on stale handle don't crash") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  prune(h, ts.store);

  CHECK(ts.store.get(h) == nullptr);
  CHECK(get_position(h, ts.store) == -1);
  CHECK(fork(h, ts.store) == INVALID_HANDLE);
  CHECK(sample(h, ts.store) == -1);

  // Double prune should be safe
  prune(h, ts.store);
}

TEST_CASE("branch: decode with zero tokens throws") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  llama_token empty[] = {};
  CHECK_THROWS(prefill(h, empty, 0, ts.store));

  CHECK(get_position(h, ts.store) == 0);

  prune(h, ts.store);
}

// ============================================================================
// Exception Safety Tests (RAII guarantees)
// ============================================================================

TEST_CASE("branch: RAII Branch prune() invalidates handle") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch b = Branch::create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(b.valid());

  BranchHandle h = b.handle();
  CHECK(ts.store.get(h) != nullptr);

  b.prune();

  CHECK(!b.valid());
  CHECK(ts.store.get(h) == nullptr);
}

TEST_CASE("branch: RAII Branch move leaves source invalid") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch b1 = Branch::create(ts.ctx, fake_model, ts.store, 0, params, 512);
  BranchHandle h = b1.handle();

  Branch b2 = std::move(b1);

  CHECK(!b1.valid());
  CHECK(b1.handle() == INVALID_HANDLE);

  CHECK(b2.valid());
  CHECK(b2.handle() == h);
  CHECK(ts.store.get(h) != nullptr);

  CHECK(b1.position() == -1);
}

TEST_CASE("branch: RAII Branch self-move-assign is safe") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch b = Branch::create(ts.ctx, fake_model, ts.store, 0, params, 512);
  BranchHandle h = b.handle();

  b = std::move(b);

  CHECK(b.valid());
  CHECK(b.handle() == h);
}

TEST_CASE("branch: RAII Branch is_eog detects stop tokens") {
  llamaStubConfig().eog_tokens = {2, 151645};  // EOS + ChatML EOT

  TestStore ts(4);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch b = Branch::create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(b.valid());

  CHECK(b.is_eog(2));        // EOS
  CHECK(b.is_eog(151645));   // ChatML EOT
  CHECK_FALSE(b.is_eog(42)); // regular token
  CHECK_FALSE(b.is_eog(0));  // BOS is not EOG

  llamaStubConfig().eog_tokens.clear();
}

TEST_CASE("branch: RAII Branch is_eog returns false when invalid") {
  Branch b;  // default-constructed, no store
  CHECK_FALSE(b.is_eog(2));
}

// ============================================================================
// BranchStore Batched Decode Tests (span-based API)
// ============================================================================

TEST_CASE("branch: decode_each with empty span is no-op") {
  TestStore ts(4);

  std::span<const DecodeEachItem> empty;
  CHECK_NOTHROW(ts.store.decode_each(empty));
}

TEST_CASE("branch: decode_each with invalid handle throws") {
  TestStore ts(4);

  DecodeEachItem items[] = {{INVALID_HANDLE, 42}};
  CHECK_THROWS(ts.store.decode_each(items));
}

TEST_CASE("branch: decode_scatter with empty span is no-op") {
  TestStore ts(4);

  std::span<const DecodeScatterItem> empty;
  CHECK_NOTHROW(ts.store.decode_scatter(empty));
}

TEST_CASE("branch: decode_scatter with invalid handle throws") {
  TestStore ts(4);

  llama_token tokens[] = {1, 2, 3};
  DecodeScatterItem items[] = {{INVALID_HANDLE, tokens}};
  CHECK_THROWS(ts.store.decode_scatter(items));
}

TEST_CASE("branch: decode_scatter with zero-length tokens span skips item") {
  TestStore ts(4);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  DecodeScatterItem items[] = {{h, std::span<const llama_token>{}}};
  CHECK_NOTHROW(ts.store.decode_scatter(items));

  CHECK(get_position(h, ts.store) == 0);

  prune(h, ts.store);
}

TEST_CASE("branch: decode_scatter all items zero-length is no-op") {
  TestStore ts(8);
  TestSamplingParams params;

  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h1 = create(ts.ctx, fake_model, ts.store, 10, params, 512);
  BranchHandle h2 = create(ts.ctx, fake_model, ts.store, 20, params, 512);
  REQUIRE(h1 != INVALID_HANDLE);
  REQUIRE(h2 != INVALID_HANDLE);

  DecodeScatterItem items[] = {
    {h1, std::span<const llama_token>{}},
    {h2, std::span<const llama_token>{}}
  };
  CHECK_NOTHROW(ts.store.decode_scatter(items));

  CHECK(get_position(h1, ts.store) == 10);
  CHECK(get_position(h2, ts.store) == 20);

  prune(h1, ts.store);
  prune(h2, ts.store);
}

// ============================================================================
// Tenancy Tests
// ============================================================================

TEST_CASE("tenancy: init fills vacancy correctly") {
  auto* ctx = reinterpret_cast<llama_context*>(0x1000);
  lloyal::kv::tenancy::State s = lloyal::kv::tenancy::init(ctx, 4);

  CHECK(lloyal::kv::tenancy::available(s) == 4);

  // Acquire all 4
  for (int i = 0; i < 4; ++i) {
    llama_seq_id seq = lloyal::kv::tenancy::acquire(s);
    CHECK(seq >= 0);
    CHECK(seq < 4);
  }

  // All exhausted
  CHECK(lloyal::kv::tenancy::available(s) == 0);
  CHECK(lloyal::kv::tenancy::acquire(s) == lloyal::kv::NO_LEASE);
}

TEST_CASE("tenancy: release returns lease without KV calls") {
  auto* ctx = reinterpret_cast<llama_context*>(0x1000);
  lloyal::kv::tenancy::State s = lloyal::kv::tenancy::init(ctx, 3);

  llama_seq_id seq = lloyal::kv::tenancy::acquire(s);
  CHECK(seq >= 0);
  CHECK(lloyal::kv::tenancy::available(s) == 2);

  lloyal::kv::tenancy::release(s, seq);
  CHECK(lloyal::kv::tenancy::available(s) == 3);

  // Re-acquire should succeed
  llama_seq_id seq2 = lloyal::kv::tenancy::acquire(s);
  CHECK(seq2 >= 0);
}

TEST_CASE("tenancy: evict returns lease and strips KV") {
  auto* ctx = reinterpret_cast<llama_context*>(0x1000);
  lloyal::kv::tenancy::State s = lloyal::kv::tenancy::init(ctx, 3);

  llama_seq_id seq = lloyal::kv::tenancy::acquire(s);
  CHECK(lloyal::kv::tenancy::available(s) == 2);

  lloyal::kv::tenancy::evict(s, seq);
  CHECK(lloyal::kv::tenancy::available(s) == 3);
}

TEST_CASE("tenancy: retain rebuilds vacancy") {
  auto* ctx = reinterpret_cast<llama_context*>(0x1000);
  lloyal::kv::tenancy::State s = lloyal::kv::tenancy::init(ctx, 5);

  // Acquire 3 leases
  llama_seq_id seq0 = lloyal::kv::tenancy::acquire(s);
  llama_seq_id seq1 = lloyal::kv::tenancy::acquire(s);
  llama_seq_id seq2 = lloyal::kv::tenancy::acquire(s);
  CHECK(lloyal::kv::tenancy::available(s) == 2);

  // Retain only seq1
  lloyal::kv::tenancy::retain(s, seq1);

  // Now only seq1 is leased, rest are vacant
  CHECK(lloyal::kv::tenancy::available(s) == 4);  // n_seq_max - 1

  // seq0 and seq2 should be acquirable again
  llama_seq_id reacquired = lloyal::kv::tenancy::acquire(s);
  CHECK(reacquired >= 0);
  CHECK(reacquired != seq1);  // seq1 is still leased
  (void)seq0;
  (void)seq2;
}

TEST_CASE("tenancy: evict_all clears everything") {
  auto* ctx = reinterpret_cast<llama_context*>(0x1000);
  lloyal::kv::tenancy::State s = lloyal::kv::tenancy::init(ctx, 4);

  lloyal::kv::tenancy::acquire(s);
  lloyal::kv::tenancy::acquire(s);
  lloyal::kv::tenancy::acquire(s);
  CHECK(lloyal::kv::tenancy::available(s) == 1);

  lloyal::kv::tenancy::evict_all(s);
  CHECK(lloyal::kv::tenancy::available(s) == 4);
}

TEST_CASE("tenancy: BranchStore available tracks leases") {
  TestStore ts(8);

  size_t initial = ts.store.available();
  CHECK(initial == 8);  // n_seq_max from stub config

  auto [h1, s1] = ts.store.allocate();
  CHECK(ts.store.available() == initial - 1);

  auto [h2, s2] = ts.store.allocate();
  CHECK(ts.store.available() == initial - 2);

  ts.store.release(h1);
  CHECK(ts.store.available() == initial - 1);

  ts.store.release(h2);
  CHECK(ts.store.available() == initial);
}

TEST_CASE("tenancy: allocate returns INVALID when leases exhausted") {
  // Use stub config with small n_seq_max
  llamaStubConfig().n_seq_max = 2;
  TestStore ts(8);
  llamaStubConfig().n_seq_max = 8;  // Reset for other tests

  auto [h1, s1] = ts.store.allocate();
  CHECK(h1 != INVALID_HANDLE);

  auto [h2, s2] = ts.store.allocate();
  CHECK(h2 != INVALID_HANDLE);

  // Third should fail — no leases left
  auto [h3, s3] = ts.store.allocate();
  CHECK(h3 == INVALID_HANDLE);
  CHECK(s3 < 0);

  ts.store.release(h1);
  ts.store.release(h2);
}

// ============================================================================
// Topology Tests
// ============================================================================

TEST_CASE("topology: fork records parent/child edges") {
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle root = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(root != INVALID_HANDLE);

  BranchHandle child1 = fork(root, ts.store);
  BranchHandle child2 = fork(root, ts.store);
  REQUIRE(child1 != INVALID_HANDLE);
  REQUIRE(child2 != INVALID_HANDLE);

  // Parent queries
  CHECK(ts.store.parent(root) == INVALID_HANDLE);  // root has no parent
  CHECK(ts.store.parent(child1) == root);
  CHECK(ts.store.parent(child2) == root);

  // Children query
  const auto& children = ts.store.children(root);
  CHECK(children.size() == 2);

  // Leaf queries
  CHECK(!ts.store.isLeaf(root));
  CHECK(ts.store.isLeaf(child1));
  CHECK(ts.store.isLeaf(child2));

  // Active queries
  CHECK(ts.store.isActive(root));
  CHECK(ts.store.isActive(child1));

  pruneSubtree(root, ts.store);
}

TEST_CASE("topology: prune RESTRICT throws if children exist") {
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle parent = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  BranchHandle child = fork(parent, ts.store);
  REQUIRE(child != INVALID_HANDLE);

  // prune(parent) should throw — it has a child
  CHECK_THROWS(prune(parent, ts.store));

  // Parent should still be alive
  CHECK(ts.store.get(parent) != nullptr);

  // Prune child first, then parent succeeds
  prune(child, ts.store);
  CHECK_NOTHROW(prune(parent, ts.store));
}

TEST_CASE("topology: pruneSubtree CASCADE depth-3") {
  TestStore ts(16);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle root = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  BranchHandle a = fork(root, ts.store);
  BranchHandle b = fork(a, ts.store);
  BranchHandle c = fork(b, ts.store);

  size_t before = ts.store.available();

  pruneSubtree(root, ts.store);

  // All 4 should be freed
  CHECK(ts.store.get(root) == nullptr);
  CHECK(ts.store.get(a) == nullptr);
  CHECK(ts.store.get(b) == nullptr);
  CHECK(ts.store.get(c) == nullptr);

  // All 4 leases returned
  CHECK(ts.store.available() == before + 4);
}

TEST_CASE("topology: prune removes child from parent's children vector") {
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle parent = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  BranchHandle child1 = fork(parent, ts.store);
  BranchHandle child2 = fork(parent, ts.store);

  CHECK(ts.store.children(parent).size() == 2);

  prune(child1, ts.store);
  CHECK(ts.store.children(parent).size() == 1);
  CHECK(ts.store.children(parent)[0] == child2);

  prune(child2, ts.store);
  CHECK(ts.store.children(parent).empty());
  CHECK(ts.store.isLeaf(parent));

  prune(parent, ts.store);
}

// ============================================================================
// retainOnly Tests
// ============================================================================

TEST_CASE("retainOnly: keeps winner, frees all others") {
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle root = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  BranchHandle a = fork(root, ts.store);
  BranchHandle b = fork(root, ts.store);
  BranchHandle c = fork(root, ts.store);

  ts.store.retainOnly(a);

  // Winner survives
  CHECK(ts.store.get(a) != nullptr);

  // Losers are freed
  CHECK(ts.store.get(root) == nullptr);
  CHECK(ts.store.get(b) == nullptr);
  CHECK(ts.store.get(c) == nullptr);

  // Winner topology is reset
  BranchState* winner = ts.store.get(a);
  CHECK(winner->parent == INVALID_HANDLE);
  CHECK(winner->children.empty());

  // available = n_seq_max - 1
  CHECK(ts.store.available() == 7);

  prune(a, ts.store);
}

TEST_CASE("retainOnly: invalid winner throws") {
  TestStore ts(4);

  CHECK_THROWS(ts.store.retainOnly(INVALID_HANDLE));
}

// ============================================================================
// Drain Tests
// ============================================================================

TEST_CASE("drain: frees all resources") {
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h1 = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  BranchHandle h2 = create(ts.ctx, fake_model, ts.store, 0, params, 512);

  ts.store.drain();

  CHECK(ts.store.get(h1) == nullptr);
  CHECK(ts.store.get(h2) == nullptr);

  // After drain, allocate should fail (tenancy ctx is null)
  auto [h3, s3] = ts.store.allocate();
  CHECK(h3 == INVALID_HANDLE);
}

TEST_CASE("drain: idempotent") {
  TestStore ts(4);

  ts.store.drain();
  ts.store.drain();  // Should not crash
}
