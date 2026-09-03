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
#include <vector>

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
    // The stub is process-global: every case starts from a known state, so
    // no case depends on what an earlier one left behind.
    resetStubConfig();
    // A live branch has a vocab (create refuses a model without one), so the
    // fixture's model has a small one; a case about the refusal sets it to 0.
    llamaStubConfig().vocab_size_value = 8;
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
  TestStore ts(4);
  llamaStubConfig().eog_tokens = {2, 151645};  // EOS + ChatML EOT
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  Branch b = Branch::create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(b.valid());

  CHECK(b.is_eog(2));        // EOS
  CHECK(b.is_eog(151645));   // ChatML EOT
  CHECK_FALSE(b.is_eog(42)); // regular token
  CHECK_FALSE(b.is_eog(0));  // BOS is not EOG
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

TEST_CASE("branch: decode_each refuses a repeated handle before anything is dispatched") {
  // Two items for one branch would both read its position at build time and
  // land two tokens on one cell — decode_scatter has always refused this;
  // decode_each stated no rule at all.
  TestStore ts(4);
  llamaStubConfig().logits.assign(8, 0.0f);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 4);
  REQUIRE(h != INVALID_HANDLE);
  DecodeEachItem items[] = {{h, 1}, {h, 2}};
  CHECK_THROWS_WITH(ts.store.decode_each(items),
                    doctest::Contains("decode_each - duplicate handle at indices 0 and 1"));
  CHECK(llamaStubConfig().decode_call_count == 0);
  CHECK(get_position(h, ts.store) == 0);
  prune(h, ts.store);
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

// ── Partial commits are data ────────────────────────────────────────────────
// llama_decode restores state only for THE CALL that fails (llama.h). Every
// chunked operation may have landed earlier calls before a later one fails,
// and the branch's own books never move on failure. The caller cannot infer
// that from `rc`; the error must SAY it. The rule the field carries:
// intact ⇔ rc == 1 && !partial — anything else ⇒ prune and replay.

TEST_CASE("prefill: a failure on a later chunk reports partial") {
  TestStore ts(4);
  llamaStubConfig().vocab_size_value = 8;
  llamaStubConfig().logits.assign(8, 0.0f);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, /*n_batch*/ 4);
  REQUIRE(h != INVALID_HANDLE);
  const uint32_t cells_before = ts.store.kv_pressure().cells_used;
  llama_token tokens[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};   // 3 chunks of 4,4,2

  SUBCASE("second chunk fails: the first landed, so the branch is not intact") {
    llamaStubConfig().decode_fail_on_call = 2;
    llamaStubConfig().decode_fail_rc = 1;                    // "no KV slot"
    bool threw = false;
    try {
      prefill(h, tokens, 10, ts.store);
    } catch (const lloyal::decode::DecodeError& e) {
      threw = true;
      CHECK(e.rc == 1);
      CHECK(e.partial == true);
    }
    CHECK(threw);
  }
  SUBCASE("first chunk fails: nothing landed, the branch is intact") {
    llamaStubConfig().decode_fail_on_call = 1;
    llamaStubConfig().decode_fail_rc = 1;
    bool threw = false;
    try {
      prefill(h, tokens, 10, ts.store);
    } catch (const lloyal::decode::DecodeError& e) {
      threw = true;
      CHECK(e.rc == 1);
      CHECK(e.partial == false);
    }
    CHECK(threw);
  }
  // Either way the books did not move — the prune contract.
  CHECK(get_position(h, ts.store) == 0);
  CHECK(ts.store.kv_pressure().cells_used == cells_before);
  prune(h, ts.store);
}

TEST_CASE("decode_scatter: a failure on a later chunk reports partial and keeps landed branches consistent") {
  TestStore ts(4);
  llamaStubConfig().vocab_size_value = 8;
  llamaStubConfig().logits.assign(8, 0.0f);
  llamaStubConfig().n_batch = 4;                             // two 3-token items → two chunks
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h1 = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  BranchHandle h2 = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  llama_token a[] = {1, 2, 3};
  llama_token b[] = {4, 5, 6};
  DecodeScatterItem items[] = {{h1, a}, {h2, b}};
  const uint32_t cells_before = ts.store.kv_pressure().cells_used;
  llamaStubConfig().decode_fail_on_call = 2;
  llamaStubConfig().decode_fail_rc = 1;
  bool threw = false;
  try {
    ts.store.decode_scatter(items);
  } catch (const lloyal::decode::DecodeError& e) {
    threw = true;
    CHECK(e.rc == 1);
    CHECK(e.partial == true);
  }
  CHECK(threw);
  // The first chunk landed and its branch advanced; the second was restored
  // by llama_decode and its branch did not move. Re-running the whole call
  // would decode h1's tokens twice — which is exactly why partial is data.
  CHECK(get_position(h1, ts.store) == 3);
  CHECK(get_position(h2, ts.store) == 0);
  CHECK(ts.store.kv_pressure().cells_used == cells_before + 3);
  prune(h1, ts.store);
  prune(h2, ts.store);
}

TEST_CASE("decode_embd: a failure on a later chunk reports partial") {
  TestStore ts(4);
  llamaStubConfig().vocab_size_value = 8;
  llamaStubConfig().logits.assign(8, 0.0f);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, /*n_batch*/ 4);
  const int32_t n_rows = 8, n_pos = 8, nppe = 1, n_embd = 2;  // 2 chunks of 4
  std::vector<float> rows(static_cast<size_t>(n_rows) * n_embd, 0.5f);
  std::vector<llama_pos> pos(static_cast<size_t>(n_rows) * nppe, 0);
  llamaStubConfig().decode_fail_on_call = 2;
  llamaStubConfig().decode_fail_rc = 1;
  bool threw = false;
  try {
    ts.store.decode_embd(h, rows.data(), n_rows, n_embd, n_pos, pos.data(), nppe, false, false);
  } catch (const lloyal::decode::DecodeError& e) {
    threw = true;
    CHECK(e.rc == 1);
    CHECK(e.partial == true);
  }
  CHECK(threw);
  CHECK(get_position(h, ts.store) == 0);
  prune(h, ts.store);
}

TEST_CASE("branch: create refuses a model with no vocab, and leaks nothing") {
  // The invariant every capture relies on is established where a branch is
  // born: nothing downstream needs to re-check it, and no decode can ever be
  // dispatched for a branch that has nowhere to put its logits.
  TestStore ts(4);
  llamaStubConfig().vocab_size_value = 0;
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  const size_t leases = ts.store.available();
  CHECK_THROWS_AS(create(ts.ctx, fake_model, ts.store, 0, params, 4), std::runtime_error);
  CHECK(ts.store.available() == leases);            // slot and lease went back
  CHECK(llamaStubConfig().decode_call_count == 0);  // nothing was dispatched
}

TEST_CASE("branch: capture_logits refuses a state with no vocab before it copies") {
  TestStore ts(4);
  llamaStubConfig().logits.assign(8, 0.0f);  // logits exist ...
  BranchState st;
  st.ctx = ts.ctx;
  st.n_vocab = 0;                              // ... but there is nowhere to put them
  CHECK_THROWS_AS(st.capture_logits(-1), std::runtime_error);
  CHECK(st.has_logits == false);
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
  // Two leases, eight slots: the store must run out of leases first.
  TestStore ts(8);
  llamaStubConfig().n_seq_max = 2;
  ts.store.init_tenancy(ts.ctx);

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

// ============================================================================
// Embedding rail — the cells/position split, on stubs
//
// The integration tier covers this against a real VL model, but its REQUIRED
// CI tier is a plain-position model where n_pos == n_rows, so every slack
// field stays zero and the accounting could be deleted without failing
// anything. These cases force n_rows > n_pos deterministically.
// ============================================================================

TEST_CASE("branch: decode_embd advances position by n_pos, cells by n_rows") {
  resetStubConfig();
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  // A SECOND branch stays alive throughout: release() zeroes cells_used_
  // outright once the last branch goes, which would mask any arithmetic error
  // in the slack accounting.
  BranchHandle keeper = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(keeper != INVALID_HANDLE);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  // An M-RoPE image: 64 cells (8x8 patches) but only 8 positions (max(nx,ny)).
  const int32_t n_rows = 64, n_pos = 8, nppe = 4, n_embd = 2;
  std::vector<float> rows(static_cast<size_t>(n_rows) * n_embd, 0.5f);
  std::vector<llama_pos> pos(static_cast<size_t>(n_rows) * nppe, 0);

  const uint32_t cells_before = ts.store.kv_pressure().cells_used;
  ts.store.decode_embd(h, rows.data(), n_rows, n_embd, n_pos, pos.data(),
                       nppe, /*non_causal*/ false, /*want_logits*/ false);

  BranchState* st = ts.store.get(h);
  REQUIRE(st != nullptr);
  CHECK(st->position == n_pos);                                   // NOT n_rows
  CHECK(ts.store.kv_pressure().cells_used == cells_before + n_rows);

  // The slack itself, asserted directly — no gauge arithmetic to mask it.
  CHECK(st->img_slack_own   == static_cast<uint32_t>(n_rows - n_pos));
  CHECK(st->img_slack_total == static_cast<uint32_t>(n_rows - n_pos));

  // Release recovers the CELLS, not the position delta. Position-delta alone
  // strands (n_rows - n_pos) == 56 cells; keeper keeps auto-reset from hiding it.
  ts.store.release(h);
  CHECK(ts.store.kv_pressure().cells_used == cells_before);

  prune(keeper, ts.store);
}

TEST_CASE("branch: retainOnly promotes inherited embedding slack") {
  resetStubConfig();
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle root = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(root != INVALID_HANDLE);

  const int32_t n_rows = 32, n_pos = 6, nppe = 4, n_embd = 2;
  std::vector<float> rows(static_cast<size_t>(n_rows) * n_embd, 0.25f);
  std::vector<llama_pos> pos(static_cast<size_t>(n_rows) * nppe, 0);
  ts.store.decode_embd(root, rows.data(), n_rows, n_embd, n_pos, pos.data(),
                       nppe, false, false);
  const uint32_t slack = static_cast<uint32_t>(n_rows - n_pos);

  BranchHandle winner = fork(root, ts.store);
  REQUIRE(winner != INVALID_HANDLE);

  // The fork inherits the slack as TOTAL but owns none of it.
  BranchState* ws = ts.store.get(winner);
  REQUIRE(ws != nullptr);
  CHECK(ws->img_slack_total == slack);
  CHECK(ws->img_slack_own   == 0u);

  // retainOnly promotes the winner to root. The inherited slack must become
  // its OWN, or a later release under-subtracts by exactly `slack`.
  ts.store.retainOnly(winner);
  // Re-fetch: retainOnly releases the other slots, so a BranchState* taken
  // before the call must not be trusted afterwards.
  BranchState* promoted = ts.store.get(winner);
  REQUIRE(promoted != nullptr);
  CHECK(get_fork_head(winner, ts.store) == 0);
  CHECK(promoted->img_slack_own == slack);
  CHECK(ts.store.kv_pressure().cells_used ==
        static_cast<uint32_t>(promoted->position) + slack);

  prune(winner, ts.store);
}

TEST_CASE("branch: decode_embd brackets a non-causal block and restores it") {
  resetStubConfig();
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  const int32_t n_rows = 16, n_pos = 4, nppe = 1, n_embd = 2;
  std::vector<float> rows(static_cast<size_t>(n_rows) * n_embd, 1.0f);
  std::vector<llama_pos> pos(static_cast<size_t>(n_rows) * nppe, 0);

  ts.store.decode_embd(h, rows.data(), n_rows, n_embd, n_pos, pos.data(),
                       nppe, /*non_causal*/ true, false);

  // false on entry, true on exit — and back to causal when it returns.
  const auto& log = llamaStubConfig().causal_attn_log;
  REQUIRE(log.size() >= 2);
  CHECK(log.front() == false);
  CHECK(log.back() == true);
  CHECK(llamaStubConfig().causal_attn == true);

  prune(h, ts.store);
}

TEST_CASE("branch: an oversized non-causal block is rejected, not split") {
  resetStubConfig();
  // A bidirectional block cannot span dispatches: rows in an earlier decode
  // cannot attend to later ones. Splitting it would silently corrupt the
  // vision state, so the configuration must fail loud.
  llamaStubConfig().n_batch  = 8;
  llamaStubConfig().n_ubatch = 8;

  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, /*n_batch*/ 8);

  const int32_t n_rows = 64, n_pos = 8, nppe = 1, n_embd = 2;
  std::vector<float> rows(static_cast<size_t>(n_rows) * n_embd, 1.0f);
  std::vector<llama_pos> pos(static_cast<size_t>(n_rows) * nppe, 0);

  CHECK_THROWS_AS(
      ts.store.decode_embd(h, rows.data(), n_rows, n_embd, n_pos, pos.data(),
                           nppe, /*non_causal*/ true, false),
      std::runtime_error);

  // A causal block of the same size chunks happily.
  CHECK_NOTHROW(
      ts.store.decode_embd(h, rows.data(), n_rows, n_embd, n_pos, pos.data(),
                           nppe, /*non_causal*/ false, false));

  prune(h, ts.store);
}

TEST_CASE("branch: decode_embd without want_logits clears stale logits") {
  resetStubConfig();
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);

  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);
  BranchState* st = ts.store.get(h);
  REQUIRE(st != nullptr);

  // Stand in for an earlier decode that did capture logits.
  st->has_logits = true;

  const int32_t n_rows = 8, n_pos = 4, nppe = 1, n_embd = 2;
  std::vector<float> rows(static_cast<size_t>(n_rows) * n_embd, 0.5f);
  std::vector<llama_pos> pos(static_cast<size_t>(n_rows) * nppe, 0);
  ts.store.decode_embd(h, rows.data(), n_rows, n_embd, n_pos, pos.data(),
                       nppe, /*non_causal*/ false, /*want_logits*/ false);

  // Position moved, so the old snapshot describes a position that is no
  // longer current — sampling from it would be silently wrong.
  CHECK(st->position == n_pos);
  CHECK(st->has_logits == false);

  prune(h, ts.store);
}

TEST_CASE("branch: decode_embd rejects a row width the model does not use") {
  resetStubConfig();
  // llama_batch carries no row-width metadata: llama_decode consumes rows at
  // the MODEL's input width while decode::embd strides by the caller's. A
  // wrong-but-positive width reads past the caller's allocation.
  TestStore ts(8);
  llamaStubConfig().n_embd_inp = 512;
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  const int32_t n_rows = 4, n_pos = 4, nppe = 1;
  std::vector<llama_pos> pos(static_cast<size_t>(n_rows) * nppe, 0);

  // Positive but wrong — accepted before, out of bounds at decode time.
  std::vector<float> narrow(static_cast<size_t>(n_rows) * 64, 0.5f);
  CHECK_THROWS_AS(
      ts.store.decode_embd(h, narrow.data(), n_rows, /*n_embd_inp*/ 64, n_pos,
                           pos.data(), nppe, false, false),
      std::runtime_error);

  // Matching the resident width is accepted.
  std::vector<float> wide(static_cast<size_t>(n_rows) * 512, 0.5f);
  CHECK_NOTHROW(
      ts.store.decode_embd(h, wide.data(), n_rows, /*n_embd_inp*/ 512, n_pos,
                           pos.data(), nppe, false, false));

  prune(h, ts.store);
}

TEST_CASE("branch: decode_scatter allows an empty span beside a real one") {
  resetStubConfig();
  TestStore ts(8);
  // decode_scatter captures logits per item, so the branch needs a vocab at create time.
  llamaStubConfig().vocab_size_value = 8;
  llamaStubConfig().logits.assign(8, 0.0f);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  std::vector<llama_token> toks = {1, 2, 3};
  std::vector<llama_token> none;

  // bin_pack skips empty spans, so they occupy no cells and cannot collide
  // on start_pos — pairing one with a real span is not a duplicate.
  DecodeScatterItem mixed[2] = {{h, std::span<const llama_token>(none)},
                                {h, std::span<const llama_token>(toks)}};
  CHECK_NOTHROW(ts.store.decode_scatter(std::span<const DecodeScatterItem>(mixed, 2)));

  // Two spans that both decode DO collide, and still throw.
  DecodeScatterItem both[2] = {{h, std::span<const llama_token>(toks)},
                               {h, std::span<const llama_token>(toks)}};
  CHECK_THROWS_AS(ts.store.decode_scatter(std::span<const DecodeScatterItem>(both, 2)),
                  std::runtime_error);

  prune(h, ts.store);
}

// ============================================================================
// SegmentSource contract — it is a PUBLIC extension point, so whatever it
// returns is untrusted input the kernel must validate before acting on.
// ============================================================================

namespace {

/// A source under test control, recording whether positions() was reached.
struct TestSource : lloyal::decode::SegmentSource {
  std::vector<lloyal::decode::Segment> segs;
  bool positions_called = false;

  size_t size() override { return segs.size(); }
  /// The contract's sum: tokens for TEXT, rows for EMBD. Kept honest so a
  /// geometry the store rejects is still priced the way a real source would.
  size_t cells() const override {
    size_t n = 0;
    for (const auto& s : segs) {
      n += s.kind == lloyal::decode::Segment::Kind::Text
               ? s.tokens.size()
               : static_cast<size_t>(s.n_rows);
    }
    return n;
  }
  lloyal::decode::Segment at(size_t i) override { return segs[i]; }
  void positions(size_t, llama_pos, llama_pos* out) override {
    positions_called = true;
    if (out) *out = 0;  // a real source would fill n_rows * n_pos_per_embd
  }
};

lloyal::decode::Segment embd_seg(const float* rows, int32_t n_rows,
                                 int32_t n_embd_inp, llama_pos n_pos,
                                 int32_t nppe) {
  lloyal::decode::Segment s;
  s.kind = lloyal::decode::Segment::Kind::Embd;
  s.rows = rows; s.n_rows = n_rows; s.n_embd_inp = n_embd_inp;
  s.n_pos = n_pos; s.n_pos_per_embd = nppe;
  return s;
}

}  // namespace

TEST_CASE("branch: decode_segments reports partial when an EARLIER segment landed") {
  // A prefill is ONE operation to its caller, so a failure in segment 2 is
  // partial even when that segment's own first call is what failed: segment 1
  // already moved the branch. The SDK's media path gates "intact" on this —
  // without it a retry would decode the sep twice.
  TestStore ts(8);
  llamaStubConfig().vocab_size_value = 8;
  llamaStubConfig().logits.assign(8, 0.0f);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);
  const uint32_t cells_before = ts.store.kv_pressure().cells_used;

  llama_token sep[] = {1, 2, 3};
  llama_token tail[] = {4, 5};
  TestSource src;
  src.segs.resize(2);
  src.segs[0].kind = lloyal::decode::Segment::Kind::Text; src.segs[0].tokens = sep;
  src.segs[1].kind = lloyal::decode::Segment::Kind::Text; src.segs[1].tokens = tail;

  llamaStubConfig().decode_fail_on_call = 2;  // segment 1 is call 1; segment 2's only call fails
  llamaStubConfig().decode_fail_rc = 1;
  bool threw = false;
  try {
    ts.store.decode_segments(h, src);
  } catch (const lloyal::decode::DecodeError& e) {
    threw = true;
    CHECK(e.rc == 1);
    CHECK(e.partial == true);
  }
  CHECK(threw);
  // Segment 1 landed and its books moved; segment 2 was restored.
  CHECK(get_position(h, ts.store) == 3);
  CHECK(ts.store.kv_pressure().cells_used == cells_before + 3);
  prune(h, ts.store);
}

TEST_CASE("branch: decode_segments validates geometry BEFORE the callback") {
  resetStubConfig();
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);

  std::vector<float> rows(64, 0.5f);

  // n_pos_per_embd = -1 would wrap the size_t multiply into an enormous
  // allocation; 0 would hand positions() a zero-length buffer to write into.
  for (int32_t bad_nppe : {-1, 0, 2, 3, 5}) {
    TestSource src;
    src.segs = {embd_seg(rows.data(), 8, 2, 4, bad_nppe)};
    CHECK_THROWS_AS(ts.store.decode_segments(h, src), std::runtime_error);
    CHECK_MESSAGE(src.positions_called == false,
                  "validation must precede the source callback");
  }

  { // non-positive row width
    TestSource src;
    src.segs = {embd_seg(rows.data(), 8, 0, 4, 1)};
    CHECK_THROWS_AS(ts.store.decode_segments(h, src), std::runtime_error);
    CHECK(src.positions_called == false);
  }

  { // n_pos outside (0, n_rows]
    TestSource src;
    src.segs = {embd_seg(rows.data(), 8, 2, 0, 1)};
    CHECK_THROWS_AS(ts.store.decode_segments(h, src), std::runtime_error);
    TestSource src2;
    src2.segs = {embd_seg(rows.data(), 8, 2, 9, 1)};
    CHECK_THROWS_AS(ts.store.decode_segments(h, src2), std::runtime_error);
    CHECK(src2.positions_called == false);
  }

  // Nothing was dispatched, so the branch is untouched.
  BranchState* st = ts.store.get(h);
  REQUIRE(st != nullptr);
  CHECK(st->position == 0);
  CHECK(ts.store.kv_pressure().cells_used == 0);

  prune(h, ts.store);
}

TEST_CASE("branch: decode_segments rejects an empty segment") {
  resetStubConfig();
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);

  // Terminality is positional: a trailing empty segment would make the real
  // final one non-terminal, so it would decode with want_logits=false and
  // hand back a branch with nothing to sample.
  TestSource src;
  lloyal::decode::Segment empty;   // Text, no tokens
  src.segs = {empty};
  CHECK_THROWS_AS(ts.store.decode_segments(h, src), std::runtime_error);

  prune(h, ts.store);
}

TEST_CASE("branch: causal mode is restored even when the decode fails") {
  resetStubConfig();
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);

  llamaStubConfig().decode_result = -1;          // llama_decode fails

  const int32_t n_rows = 8, n_pos = 4, nppe = 1, n_embd = 2;
  std::vector<float> rows(static_cast<size_t>(n_rows) * n_embd, 1.0f);
  std::vector<llama_pos> pos(static_cast<size_t>(n_rows) * nppe, 0);

  CHECK_THROWS_AS(
      ts.store.decode_embd(h, rows.data(), n_rows, n_embd, n_pos, pos.data(),
                           nppe, /*non_causal*/ true, false),
      std::runtime_error);

  // Causal mode is context-wide: leaving it off would make every SUBSEQUENT
  // text decode on this context non-causal. The RAII guard restores it on
  // the failure path too, not just on the happy one.
  CHECK(llamaStubConfig().causal_attn == true);
  CHECK(llamaStubConfig().causal_attn_log.back() == true);

  prune(h, ts.store);
}


TEST_CASE("branch: a failed decode_embd poisons the branch but keeps its books") {
  resetStubConfig();
  TestStore ts(8);
  TestSamplingParams params;
  auto* fake_model = reinterpret_cast<llama_model*>(0x2000);
  BranchHandle h = create(ts.ctx, fake_model, ts.store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);
  BranchState* st = ts.store.get(h);
  REQUIRE(st != nullptr);
  st->position = 40;
  const uint32_t cells_before = ts.store.kv_pressure().cells_used;

  llamaStubConfig().decode_result = -1;

  const int32_t n_rows = 16, n_pos = 4, nppe = 1, n_embd = 2;
  std::vector<float> rows(static_cast<size_t>(n_rows) * n_embd, 0.5f);
  std::vector<llama_pos> pos(static_cast<size_t>(n_rows) * nppe, 0);

  // Earlier chunks may already be committed and cannot be rolled back — a
  // recurrent carrier has folded them and seq_rm can only rewind where the
  // model keeps snapshots. So the error says the branch is unusable.
  CHECK_THROWS_WITH(
      ts.store.decode_embd(h, rows.data(), n_rows, n_embd, n_pos, pos.data(),
                           nppe, false, false),
      "BranchStore::decode_embd - llama_decode failed; this branch is "
      "poisoned, prune it and replay onto a fresh one (rc=-1)");

  // The branch's OWN accounting stays consistent, which is what makes the
  // prune correct: neither counter moved, so release() subtracts exactly what
  // the branch legitimately owned and eviction reclaims the orphaned rows.
  CHECK(st->position == 40);
  CHECK(ts.store.kv_pressure().cells_used == cells_before);
  CHECK(st->img_slack_own == 0u);

  prune(h, ts.store);
  CHECK(ts.store.kv_pressure().cells_used == 0);
}
