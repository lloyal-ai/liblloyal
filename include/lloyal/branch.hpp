#pragma once

/**
 * Branch Primitive for Tree Search
 *
 * Consolidates all forkable state (KV cache, grammar, penalties, metrics, logits)
 * into a single handle-based API. Branch is the foundational primitive for
 * MCTS/LATS tree search and multi-sequence generation.
 *
 * Handle Design:
 * - Handle = (generation << 16) | index
 * - Generation counter prevents ABA bugs on slot reuse
 * - Freelist enables branch pooling without malloc churn (MCTS optimization)
 *
 * Fork Semantics:
 * - fork() clones ALL state atomically: KV, grammar, sampler chain, metrics, logits
 * - No manual juggling of individual components
 * - Each branch is fully independent after fork
 *
 * Usage:
 *   auto root = branch::create(ctx, model, 0, prompt_len, params, grammar);
 *   branch::decode_and_capture(root, tokens, n);
 *
 *   auto child = branch::fork(root, new_seq_id);  // Clone everything
 *   branch::decode(child, &token, 1);
 *   branch::accept_token(child, token);
 *
 *   branch::prune(child);  // Remove from KV + free resources
 *   branch::free(root);
 */

#include "common.hpp"
#include "decoder.hpp"
#include "grammar.hpp"
#include "kv.hpp"
#include "logits.hpp"
#include "metrics.hpp"
#include "sampler.hpp"

#include <llama/llama.h>
#include <cmath>      // std::exp, std::log, std::isinf, std::isfinite
#include <cstdint>
#include <cstring>    // std::memcpy
#include <ctime>      // std::time
#include <limits>     // std::numeric_limits
#include <mutex>
#include <stdexcept>  // std::runtime_error
#include <utility>    // std::pair, std::exchange
#include <vector>

namespace lloyal::branch {

// ============================================================================
// Handle Type
// ============================================================================

/**
 * Branch handle: (generation << 16) | index
 * - Upper 16 bits: generation counter (prevents ABA bugs)
 * - Lower 16 bits: slot index (max 65535 branches)
 * - Value 0 is invalid/null handle
 */
using BranchHandle = uint32_t;

constexpr BranchHandle INVALID_HANDLE = 0;
constexpr uint32_t GEN_SHIFT = 16;
constexpr uint32_t INDEX_MASK = 0xFFFF;

inline uint16_t handle_index(BranchHandle h) {
  return static_cast<uint16_t>(h & INDEX_MASK);
}

inline uint16_t handle_generation(BranchHandle h) {
  return static_cast<uint16_t>(h >> GEN_SHIFT);
}

inline BranchHandle make_handle(uint16_t index, uint16_t generation) {
  return (static_cast<uint32_t>(generation) << GEN_SHIFT) | index;
}

// ============================================================================
// Branch State
// ============================================================================

struct BranchState {
  // Context reference (not owned)
  llama_context* ctx = nullptr;
  const llama_model* model = nullptr;

  // Sequence identity
  llama_seq_id seq_id = 0;
  llama_pos position = 0;

  // Sampling chain (penalties + PRNG + filters) - owned
  llama_sampler* sampler_chain = nullptr;

  // Grammar sampler (optional) - owned
  llama_sampler* grammar = nullptr;

  // Perplexity tracker - owned via handle
  metrics::PerplexityHandle ppl = 0;

  // Logits snapshot (owned, copied on decode_and_capture)
  std::vector<float> logits_snapshot;
  bool has_logits = false;  // True only after capture_logits/decode_and_capture

  // Reusable buffer for sampling (avoids O(n_vocab) allocs per sample)
  // MEMORY NOTE: For large vocab models (128k tokens), each branch uses:
  //   - logits_snapshot: n_vocab * 4 bytes (~512KB)
  //   - candidates_buffer: n_vocab * sizeof(llama_token_data) (~1.5-2MB)
  // For deep MCTS trees, consider using a shared scratch arena instead.
  // TODO: Move to per-thread or SessionContext scratch for MCTS scaling.
  std::vector<llama_token_data> candidates_buffer;

  // Batch size for decode operations
  int n_batch = 512;

  // Vocab size for buffer sizing
  int n_vocab = 0;

  // Slot management
  uint16_t generation = 0;
  bool in_use = false;
};

// ============================================================================
// Branch Store (Handle Table)
// ============================================================================

/**
 * Handle table with generation counters and freelist.
 *
 * Thread safety: External synchronization required (caller's mutex).
 * Typically SessionContext holds _decodeMutex for decode operations.
 */
class BranchStore {
public:
  explicit BranchStore(size_t initial_capacity = 16) {
    // Ensure minimum capacity of 2 (slot 0 reserved + at least 1 usable)
    if (initial_capacity < 2) {
      initial_capacity = 2;
    }
    slots_.resize(initial_capacity);
    // Slot 0 is reserved (handle 0 = invalid)
    slots_[0].in_use = true;
    slots_[0].generation = 0xFFFF;  // Never valid

    // Initialize freelist with remaining slots
    // NOTE: Use i-- > 1 pattern to avoid size_t underflow
    for (size_t i = initial_capacity; i-- > 1; ) {
      freelist_.push_back(static_cast<uint16_t>(i));
    }
  }

  ~BranchStore() {
    // Free all active branches
    for (size_t i = 1; i < slots_.size(); ++i) {
      if (slots_[i].in_use) {
        free_branch_resources(slots_[i]);
      }
    }
  }

  /**
   * Allocate a new branch slot
   * Returns INVALID_HANDLE if no slots available
   */
  BranchHandle allocate() {
    if (freelist_.empty()) {
      // Grow the store
      size_t old_size = slots_.size();
      size_t new_size = old_size * 2;
      if (new_size > INDEX_MASK) {
        new_size = INDEX_MASK + 1;  // Max 65536 slots
      }
      if (old_size >= new_size) {
        LLOYAL_LOG_DEBUG("[branch::allocate] Store full, cannot allocate");
        return INVALID_HANDLE;
      }

      slots_.resize(new_size);
      // NOTE: Use i-- > old_size pattern to avoid size_t underflow
      for (size_t i = new_size; i-- > old_size; ) {
        freelist_.push_back(static_cast<uint16_t>(i));
      }
    }

    uint16_t index = freelist_.back();
    freelist_.pop_back();

    BranchState& slot = slots_[index];
    slot.in_use = true;
    // Generation already incremented on free, or 0 for fresh slot

    return make_handle(index, slot.generation);
  }

  /**
   * Free a branch slot, returning it to the freelist
   */
  void release(BranchHandle handle) {
    if (handle == INVALID_HANDLE) return;

    uint16_t index = handle_index(handle);
    uint16_t gen = handle_generation(handle);

    if (index >= slots_.size()) return;

    BranchState& slot = slots_[index];
    if (!slot.in_use || slot.generation != gen) {
      LLOYAL_LOG_DEBUG("[branch::release] Invalid handle: stale or double-free");
      return;
    }

    // Free owned resources
    free_branch_resources(slot);

    // Mark slot as free and increment generation (wrap is intentional)
    slot.in_use = false;
    slot.generation = static_cast<uint16_t>(slot.generation + 1);  // Prevent ABA
    slot.ctx = nullptr;
    slot.model = nullptr;
    slot.seq_id = 0;
    slot.position = 0;
    slot.sampler_chain = nullptr;
    slot.grammar = nullptr;
    slot.ppl = 0;
    slot.logits_snapshot.clear();
    slot.has_logits = false;
    slot.candidates_buffer.clear();  // Clear contents, capacity preserved for reuse
    slot.n_vocab = 0;

    freelist_.push_back(index);
  }

  /**
   * Get branch state by handle (with validation)
   * Returns nullptr if handle is invalid
   */
  BranchState* get(BranchHandle handle) {
    if (handle == INVALID_HANDLE) return nullptr;

    uint16_t index = handle_index(handle);
    uint16_t gen = handle_generation(handle);

    // Slot 0 is reserved and never valid for external use
    if (index == 0) return nullptr;

    if (index >= slots_.size()) return nullptr;

    BranchState& slot = slots_[index];
    if (!slot.in_use || slot.generation != gen) {
      return nullptr;
    }

    return &slot;
  }

  const BranchState* get(BranchHandle handle) const {
    return const_cast<BranchStore*>(this)->get(handle);
  }

private:
  void free_branch_resources(BranchState& slot) {
    if (slot.sampler_chain) {
      sampler::free_chain(slot.sampler_chain);
      slot.sampler_chain = nullptr;
    }
    if (slot.grammar) {
      grammar::free_sampler(slot.grammar);
      slot.grammar = nullptr;
    }
    if (slot.ppl != 0) {
      metrics::free_perplexity(slot.ppl);
      slot.ppl = 0;
    }
  }

  std::vector<BranchState> slots_;
  std::vector<uint16_t> freelist_;
};

// ============================================================================
// Global Store (for simple usage without explicit store management)
// ============================================================================

namespace detail {
inline BranchStore*& global_store_ptr() {
  static BranchStore* ptr = nullptr;
  return ptr;
}

inline BranchStore& global_store() {
  BranchStore*& ptr = global_store_ptr();
  if (!ptr) {
    ptr = new BranchStore();
  }
  return *ptr;
}
}  // namespace detail

/**
 * Shutdown the global branch store
 *
 * Call this BEFORE llama_backend_free() to ensure proper teardown order.
 * After calling this, any branches using the global store become invalid.
 *
 * Safe to call multiple times or if global store was never used.
 *
 * USAGE:
 *   // At application shutdown:
 *   branch::shutdown_global_store();
 *   llama_backend_free();
 */
inline void shutdown_global_store() {
  BranchStore*& ptr = detail::global_store_ptr();
  if (ptr) {
    delete ptr;
    ptr = nullptr;
  }
}

// ============================================================================
// Branch API (Free Functions)
// ============================================================================

/**
 * Create a new branch
 *
 * @param ctx Llama context (not owned, must outlive branch)
 * @param model Llama model for sampler/grammar initialization
 * @param seq_id Sequence ID for KV cache
 * @param start_pos Starting position (typically after prefill)
 * @param n_batch Batch size for decode operations
 * @param grammar_str Optional grammar string (nullptr = no grammar)
 * @param store Optional store (nullptr = use global store)
 * @return Branch handle, or INVALID_HANDLE on failure
 */
template <SamplingParamsLike P>
inline BranchHandle create(
    llama_context* ctx,
    const llama_model* model,
    llama_seq_id seq_id,
    llama_pos start_pos,
    const P& params,
    int n_batch = 512,
    const char* grammar_str = nullptr,
    BranchStore* store = nullptr) {
  if (!ctx || !model) {
    LLOYAL_LOG_DEBUG("[branch::create] NULL ctx or model");
    return INVALID_HANDLE;
  }

  BranchStore& s = store ? *store : detail::global_store();
  BranchHandle handle = s.allocate();
  if (handle == INVALID_HANDLE) {
    return INVALID_HANDLE;
  }

  BranchState* state = s.get(handle);
  if (!state) {
    s.release(handle);  // Fix: release slot on failure
    return INVALID_HANDLE;
  }

  state->ctx = ctx;
  state->model = model;
  state->seq_id = seq_id;
  state->position = start_pos;
  state->n_batch = n_batch;

  const llama_vocab* vocab = llama_model_get_vocab(model);
  state->n_vocab = llama_vocab_n_tokens(vocab);
  state->logits_snapshot.resize(state->n_vocab);
  state->has_logits = false;  // Must call capture_logits/decode_and_capture first
  state->candidates_buffer.resize(state->n_vocab);  // Pre-allocate for sampling

  // Create sampler chain via anti-corruption layer
  // Handles temp <= 0 as greedy mode automatically
  state->sampler_chain = sampler::create_chain(params);

  // Create grammar sampler if grammar string provided
  if (grammar_str && grammar_str[0] != '\0') {
    state->grammar = grammar::init_sampler(model, grammar_str);
  }

  // Create perplexity tracker
  state->ppl = metrics::create_perplexity();

  LLOYAL_LOG_DEBUG("[branch::create] Created branch handle=%u seq=%d pos=%d",
                   handle, seq_id, start_pos);

  return handle;
}

/**
 * Destroy a branch and all its resources
 * Note: Named 'destroy' to avoid conflict with std::free
 */
inline void destroy(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  s.release(handle);
}

/**
 * Fork a branch to a new sequence
 *
 * Clones all state:
 * - KV cache (via seq_cp)
 * - Sampler chain
 * - Grammar sampler
 * - Perplexity tracker
 * - Logits snapshot
 */
inline BranchHandle fork(
    BranchHandle source,
    llama_seq_id new_seq_id,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* src = s.get(source);
  if (!src) {
    LLOYAL_LOG_DEBUG("[branch::fork] Invalid source handle");
    return INVALID_HANDLE;
  }

  BranchHandle new_handle = s.allocate();
  if (new_handle == INVALID_HANDLE) {
    return INVALID_HANDLE;
  }

  BranchState* dst = s.get(new_handle);
  if (!dst) {
    s.release(new_handle);  // Fix: release slot on failure
    return INVALID_HANDLE;
  }

  // Copy basic state
  dst->ctx = src->ctx;
  dst->model = src->model;
  dst->seq_id = new_seq_id;
  dst->position = src->position;
  dst->n_batch = src->n_batch;
  dst->n_vocab = src->n_vocab;

  // Clear destination sequence first to handle seq_id reuse in MCTS
  // Without this, stale KV residue from previous use could corrupt branch state
  // (prune() clears on release, but this is defensive for seq_id recycling)
  kv::remove_range(src->ctx, new_seq_id, 0, -1);

  // Fork KV cache (use default p1=-1 to copy all positions)
  kv::seq_cp(src->ctx, src->seq_id, new_seq_id);

  // Clone sampler chain via anti-corruption layer
  if (src->sampler_chain) {
    dst->sampler_chain = sampler::clone_chain(src->sampler_chain);
  }

  // Clone grammar
  if (src->grammar) {
    dst->grammar = grammar::clone_sampler(src->grammar);
  }

  // Clone perplexity tracker
  if (src->ppl != 0) {
    dst->ppl = metrics::clone_perplexity(src->ppl);
  }

  // Copy logits snapshot and validity flag
  dst->logits_snapshot = src->logits_snapshot;
  dst->has_logits = src->has_logits;

  // Pre-allocate candidates buffer (don't copy contents, just capacity)
  dst->candidates_buffer.resize(dst->n_vocab);

  LLOYAL_LOG_DEBUG("[branch::fork] Forked handle=%u -> handle=%u seq=%d->%d",
                   source, new_handle, src->seq_id, new_seq_id);

  return new_handle;
}

/**
 * Prune a branch (remove from KV cache and free)
 */
inline void prune(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) return;

  // Remove from KV cache
  kv::remove_range(state->ctx, state->seq_id, 0, -1);

  // Free the branch
  s.release(handle);
}

/**
 * Decode multiple tokens into the branch's sequence
 *
 * For single-token decode, prefer decode_one() which has zero heap allocation.
 *
 * @throws std::runtime_error if handle is invalid or decode fails
 */
inline void decode_batch(
    BranchHandle handle,
    const llama_token* tokens,
    size_t n_tokens,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("decode_batch: invalid branch handle");
  }

  // Pass raw pointer directly - no vector copy needed
  decoder::decode_tokens(
      state->ctx, tokens, static_cast<int32_t>(n_tokens),
      state->position, state->n_batch, state->seq_id);

  state->position += static_cast<llama_pos>(n_tokens);
}

/**
 * Decode a single token (zero-allocation fast path)
 *
 * Uses stack-allocated llama_batch - no heap allocation.
 * Prefer this over decode_batch() for single-token MCTS expansion.
 *
 * @throws std::runtime_error if handle is invalid or decode fails
 */
inline void decode_one(
    BranchHandle handle,
    llama_token token,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("decode_one: invalid branch handle");
  }

  decoder::decode_one(state->ctx, token, state->position, state->seq_id, false);
  state->position += 1;
}

/**
 * Capture current logits into the branch's snapshot (no decode)
 * Use after prefill to initialize root branch logits
 *
 * @throws std::runtime_error if handle is invalid, state is null,
 *         or logits are unavailable (no decode with logits=true)
 */
inline void capture_logits(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("capture_logits: invalid branch handle");
  }

  // logits::get() throws if ctx is null or logits unavailable
  const float* raw_logits = logits::get(state->ctx, -1);

  if (state->n_vocab <= 0) {
    throw std::runtime_error("capture_logits: invalid vocab size");
  }

  std::memcpy(state->logits_snapshot.data(), raw_logits,
              state->n_vocab * sizeof(float));
  state->has_logits = true;
}

/**
 * Decode multiple tokens and capture logits atomically
 *
 * For single-token decode, prefer decode_and_capture_one() which has zero heap allocation.
 *
 * @throws std::runtime_error if handle is invalid, decode fails,
 *         or logits capture fails
 */
inline void decode_and_capture_batch(
    BranchHandle handle,
    const llama_token* tokens,
    size_t n_tokens,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("decode_and_capture_batch: invalid branch handle");
  }

  // Pass raw pointer directly - no vector copy needed
  decoder::decode_tokens(
      state->ctx, tokens, static_cast<int32_t>(n_tokens),
      state->position, state->n_batch, state->seq_id);

  state->position += static_cast<llama_pos>(n_tokens);

  // logits::get() throws if logits unavailable
  const float* raw_logits = logits::get(state->ctx, -1);

  if (state->n_vocab <= 0) {
    throw std::runtime_error("decode_and_capture_batch: invalid vocab size");
  }

  std::memcpy(state->logits_snapshot.data(), raw_logits,
              state->n_vocab * sizeof(float));
  state->has_logits = true;
}

/**
 * Decode a single token and capture logits (zero-allocation fast path)
 *
 * Uses stack-allocated llama_batch - no heap allocation.
 * Prefer this over decode_and_capture_batch() for single-token MCTS expansion.
 *
 * @throws std::runtime_error if handle is invalid, decode fails,
 *         or logits capture fails
 */
inline void decode_and_capture_one(
    BranchHandle handle,
    llama_token token,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("decode_and_capture_one: invalid branch handle");
  }

  decoder::decode_one(state->ctx, token, state->position, state->seq_id, true);
  state->position += 1;

  // logits::get() throws if logits unavailable
  const float* raw_logits = logits::get(state->ctx, -1);

  if (state->n_vocab <= 0) {
    throw std::runtime_error("decode_and_capture_one: invalid vocab size");
  }

  std::memcpy(state->logits_snapshot.data(), raw_logits,
              state->n_vocab * sizeof(float));
  state->has_logits = true;
}

/**
 * Get the branch's logits snapshot
 *
 * @return Pointer to logits, or nullptr if no logits have been captured
 */
inline const float* get_logits(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  const BranchState* state = s.get(handle);
  // Must check has_logits, not just empty() - buffer is pre-allocated in create()
  if (!state || !state->has_logits) {
    return nullptr;
  }

  return state->logits_snapshot.data();
}

/**
 * Sample from the branch using its sampler chain
 *
 * @return Sampled token, or -1 if no logits captured or sampling fails
 */
inline llama_token sample(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state || !state->sampler_chain) {
    return -1;
  }

  // Must have logits captured before sampling
  if (!state->has_logits) {
    LLOYAL_LOG_DEBUG("[branch::sample] No logits captured - call decode_and_capture first");
    return -1;
  }

  // Reuse pre-allocated candidates buffer (avoids O(n_vocab) allocs per sample)
  for (int i = 0; i < state->n_vocab; i++) {
    state->candidates_buffer[i] = llama_token_data{
        static_cast<llama_token>(i),
        state->logits_snapshot[i],
        0.0f};
  }

  llama_token_data_array cur_p = {
      state->candidates_buffer.data(),
      static_cast<size_t>(state->n_vocab),
      -1,
      false};

  // Apply grammar first if present (via anti-corruption layer)
  if (state->grammar) {
    grammar::apply(state->grammar, &cur_p);
  }

  // Apply sampler chain (via anti-corruption layer)
  sampler::apply(state->sampler_chain, &cur_p);

  if (cur_p.selected == -1) {
    return -1;
  }

  return cur_p.data[cur_p.selected].id;
}

/**
 * Accept a token (advance grammar and sampler chain state)
 */
inline void accept_token(
    BranchHandle handle,
    llama_token token,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) return;

  // Accept in grammar (via anti-corruption layer)
  if (state->grammar) {
    grammar::accept(state->grammar, token);
  }

  // Accept in sampler chain for penalty tracking (via anti-corruption layer)
  if (state->sampler_chain) {
    sampler::accept(state->sampler_chain, token);
  }

  // Update perplexity (only if logits have been captured)
  // Guard on has_logits to avoid computing surprisal from zero-filled buffer
  if (state->ppl != 0 && state->has_logits) {
    float surprisal = metrics::model_surprisal(
        state->logits_snapshot.data(), state->n_vocab, token);
    metrics::add_surprisal(state->ppl, surprisal);
  }
}

/**
 * Apply grammar constraints to a logits buffer
 *
 * Note: This function works on an external logits buffer, so it uses
 * the internal candidates_buffer for the grammar application.
 */
inline void apply_grammar(
    BranchHandle handle,
    float* logits,
    int n_vocab,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state || !state->grammar) return;

  // Use pre-allocated candidates buffer if size matches, otherwise allocate
  std::vector<llama_token_data>* candidates_ptr;
  std::vector<llama_token_data> temp_buffer;

  if (n_vocab == state->n_vocab && !state->candidates_buffer.empty()) {
    candidates_ptr = &state->candidates_buffer;
  } else {
    temp_buffer.resize(n_vocab);
    candidates_ptr = &temp_buffer;
  }

  auto& candidates = *candidates_ptr;
  for (int i = 0; i < n_vocab; i++) {
    candidates[i] = llama_token_data{
        static_cast<llama_token>(i), logits[i], 0.0f};
  }

  llama_token_data_array cur_p = {
      candidates.data(),
      static_cast<size_t>(n_vocab),
      -1,
      false};

  grammar::apply(state->grammar, &cur_p);

  // Copy masked logits back
  for (int i = 0; i < n_vocab; i++) {
    logits[i] = candidates[i].logit;
  }
}

/**
 * Get grammar-masked candidate tokens with their renormalized probabilities
 *
 * Returns vector of (token, probability) pairs for legal moves only.
 * Probabilities are renormalized over the legal set (sum to 1.0).
 *
 * This is essential for proper PUCT: policy priors should only cover legal moves.
 */
inline std::vector<std::pair<llama_token, float>> get_legal_priors(
    BranchHandle handle,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state || !state->has_logits) {
    return {};
  }

  // Reuse pre-allocated candidates buffer
  for (int i = 0; i < state->n_vocab; i++) {
    state->candidates_buffer[i] = llama_token_data{
        static_cast<llama_token>(i),
        state->logits_snapshot[i],
        0.0f};
  }

  llama_token_data_array cur_p = {
      state->candidates_buffer.data(),
      static_cast<size_t>(state->n_vocab),
      -1,
      false};

  // Apply grammar to mask illegal tokens
  if (state->grammar) {
    grammar::apply(state->grammar, &cur_p);
  }

  // Collect legal candidates (logit is finite after grammar masking)
  // Grammar masking sets illegal tokens to -INFINITY
  std::vector<std::pair<llama_token, float>> legal_priors;
  float max_logit = -std::numeric_limits<float>::infinity();

  for (size_t i = 0; i < cur_p.size; i++) {
    if (std::isfinite(cur_p.data[i].logit)) {  // Not masked
      legal_priors.emplace_back(cur_p.data[i].id, cur_p.data[i].logit);
      if (cur_p.data[i].logit > max_logit) {
        max_logit = cur_p.data[i].logit;
      }
    }
  }

  if (legal_priors.empty()) {
    return {};
  }

  // Compute softmax over legal moves only (numerically stable)
  float sum_exp = 0.0f;
  for (auto& [token, logit] : legal_priors) {
    float exp_val = std::exp(logit - max_logit);
    logit = exp_val;  // Temporarily store exp value
    sum_exp += exp_val;
  }

  // Normalize to probabilities
  for (auto& [token, prob] : legal_priors) {
    prob /= sum_exp;
  }

  return legal_priors;
}

/**
 * Compute logsumexp over legal (grammar-masked) logits
 * Used for efficient prior computation: P(token) = exp(logit - logsumexp)
 *
 * @return logsumexp value, or -INFINITY if no legal tokens or invalid state
 */
inline float get_legal_logsumexp(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state || !state->has_logits) {
    return -std::numeric_limits<float>::infinity();
  }

  // Reuse pre-allocated candidates buffer
  for (int i = 0; i < state->n_vocab; i++) {
    state->candidates_buffer[i] = llama_token_data{
        static_cast<llama_token>(i),
        state->logits_snapshot[i],
        0.0f};
  }

  llama_token_data_array cur_p = {
      state->candidates_buffer.data(),
      static_cast<size_t>(state->n_vocab),
      -1,
      false};

  // Apply grammar to mask illegal tokens
  if (state->grammar) {
    grammar::apply(state->grammar, &cur_p);
  }

  // Numerically stable logsumexp over legal tokens (finite logits only)
  float max_logit = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < cur_p.size; i++) {
    if (std::isfinite(cur_p.data[i].logit) && cur_p.data[i].logit > max_logit) {
      max_logit = cur_p.data[i].logit;
    }
  }

  if (!std::isfinite(max_logit)) {
    return -std::numeric_limits<float>::infinity();  // No legal tokens
  }

  float sum_exp = 0.0f;
  for (size_t i = 0; i < cur_p.size; i++) {
    if (std::isfinite(cur_p.data[i].logit)) {
      sum_exp += std::exp(cur_p.data[i].logit - max_logit);
    }
  }

  return max_logit + std::log(sum_exp);
}

/**
 * Check if a single token is legal under grammar constraints
 *
 * Uses a 1-element candidate array for O(grammar_complexity) check
 * instead of O(n_vocab) full scan.
 *
 * @param token The token to check
 * @return true if token is legal (or no grammar), false if illegal
 */
inline bool is_token_legal(
    BranchHandle handle,
    llama_token token,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state || token < 0 || token >= state->n_vocab) {
    return false;
  }

  // No grammar = all tokens legal
  if (!state->grammar) {
    return true;
  }

  // Build 1-element candidate array (stack allocated, no heap)
  llama_token_data single_candidate = {
      token,
      state->has_logits ? state->logits_snapshot[token] : 0.0f,
      0.0f
  };

  llama_token_data_array cur_p = {
      &single_candidate,
      1,
      -1,
      false
  };

  // Apply grammar - will set logit to -INFINITY if illegal
  grammar::apply(state->grammar, &cur_p);

  return std::isfinite(single_candidate.logit);
}

/**
 * Compute prior probability assuming token is already known to be legal
 *
 * O(1) - use this in MCTS inner loops where sample() already enforced grammar.
 * Does NOT check grammar legality - caller must ensure token is legal.
 *
 * @param token The token (must be legal under grammar)
 * @param logsumexp Pre-computed logsumexp from get_legal_logsumexp()
 * @return Probability in [0,1]
 */
inline float get_token_prior_assume_legal(
    BranchHandle handle,
    llama_token token,
    float logsumexp,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state || !state->has_logits || token < 0 || token >= state->n_vocab) {
    return 0.0f;
  }

  float logit = state->logits_snapshot[token];
  return std::exp(logit - logsumexp);
}

/**
 * Compute prior probability for a specific token, checking grammar legality
 *
 * O(grammar_complexity) - uses is_token_legal() for single-token check.
 * Safe for ad-hoc callers who don't know if token is legal.
 *
 * For MCTS inner loops: Use get_token_prior_assume_legal() instead since
 * sample() already enforced grammar constraints.
 *
 * @param token The token to compute prior for
 * @param logsumexp Pre-computed logsumexp from get_legal_logsumexp()
 * @return Probability in [0,1], or 0 if token is illegal
 */
inline float get_token_prior(
    BranchHandle handle,
    llama_token token,
    float logsumexp,
    BranchStore* store = nullptr) {
  if (!is_token_legal(handle, token, store)) {
    return 0.0f;
  }
  return get_token_prior_assume_legal(handle, token, logsumexp, store);
}

// ============================================================================
// State Accessors
// ============================================================================

inline llama_seq_id get_seq_id(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  const BranchState* state = s.get(handle);
  return state ? state->seq_id : -1;
}

inline llama_pos get_position(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  const BranchState* state = s.get(handle);
  return state ? state->position : -1;
}

inline float get_perplexity(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  const BranchState* state = s.get(handle);
  if (!state || state->ppl == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return metrics::get_ppl(state->ppl);
}

inline int get_n_vocab(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  const BranchState* state = s.get(handle);
  return state ? state->n_vocab : 0;
}

// ============================================================================
// Release Structures (for winner commit pattern)
// ============================================================================

/**
 * Result of release_kv() - KV cache preserved, resources freed
 * Use this for "Commit+Reset" mode where you want to restart sampling fresh.
 */
struct ReleasedKV {
  llama_context* ctx;
  llama_seq_id seq_id;
  llama_pos position;
};

/**
 * Result of release_full() - KV cache preserved, ownership of resources transferred
 * Use this for "Commit+Continue" mode where you want to keep sampling state.
 *
 * IMPORTANT: Caller owns these resources and must free them:
 * - sampler::free_chain(sampler_chain)
 * - grammar::free_sampler(grammar)  // if non-null
 * - metrics::free_perplexity(ppl)   // if non-zero
 */
struct ReleasedFull {
  llama_context* ctx;
  const llama_model* model;
  llama_seq_id seq_id;
  llama_pos position;
  llama_sampler* sampler_chain;  // Caller owns - must call sampler::free_chain()
  llama_sampler* grammar;        // Caller owns - must call grammar::free_sampler() if non-null
  metrics::PerplexityHandle ppl; // Caller owns - must call metrics::free_perplexity() if non-zero
};

// ============================================================================
// RAII Wrapper (Optional Convenience for C++ Users)
// ============================================================================

/**
 * RAII wrapper around BranchHandle
 *
 * Usage:
 *   Branch root = Branch::create(ctx, model, 0, pos, params);
 *   Branch child = root.fork(1);
 *   child.decode(&token, 1);
 *
 * Winner commit pattern:
 *   auto released = winner.release_kv();  // KV survives, resources freed
 *   // winner is now invalid, but KV at released.seq_id is preserved
 */
class Branch {
public:
  Branch() : store_(nullptr), handle_(INVALID_HANDLE) {}

  Branch(BranchStore* store, BranchHandle handle)
      : store_(store), handle_(handle) {}

  ~Branch() {
    if (handle_ != INVALID_HANDLE && store_) {
      // Use branch::prune() to remove from KV cache AND free resources
      // destroy() only frees resources without cleaning up KV cache
      branch::prune(handle_, store_);
    }
  }

  // Move-only
  Branch(Branch&& other) noexcept
      : store_(other.store_), handle_(other.handle_) {
    other.handle_ = INVALID_HANDLE;
  }

  Branch& operator=(Branch&& other) noexcept {
    if (this != &other) {
      if (handle_ != INVALID_HANDLE && store_) {
        branch::prune(handle_, store_);  // Clean up KV cache + free resources
      }
      store_ = other.store_;
      handle_ = other.handle_;
      other.handle_ = INVALID_HANDLE;
    }
    return *this;
  }

  // Non-copyable
  Branch(const Branch&) = delete;
  Branch& operator=(const Branch&) = delete;

  // Factory
  template <SamplingParamsLike P>
  static Branch create(
      llama_context* ctx,
      const llama_model* model,
      llama_seq_id seq_id,
      llama_pos start_pos,
      const P& params,
      int n_batch = 512,
      const char* grammar_str = nullptr,
      BranchStore* store = nullptr) {
    BranchStore* s = store ? store : &detail::global_store();
    BranchHandle h = branch::create(ctx, model, seq_id, start_pos, params, n_batch, grammar_str, s);
    return Branch(s, h);
  }

  // Operations
  Branch fork(llama_seq_id new_seq_id) {
    BranchHandle h = branch::fork(handle_, new_seq_id, store_);
    return Branch(store_, h);
  }

  void prune() {
    branch::prune(handle_, store_);
    handle_ = INVALID_HANDLE;
  }

  /**
   * Release KV cache ownership without pruning (Commit+Reset mode)
   *
   * - KV cache is PRESERVED at seq_id
   * - Branch resources (sampler, grammar, metrics) are FREED
   * - Branch handle is invalidated
   *
   * Use when you want to keep the KV cache but restart sampling fresh.
   * Returns {ctx, seq_id, position} so caller can continue from there.
   */
  ReleasedKV release_kv() {
    if (handle_ == INVALID_HANDLE || !store_) {
      return ReleasedKV{nullptr, -1, -1};
    }

    BranchState* st = store_->get(handle_);
    if (!st) {
      return ReleasedKV{nullptr, -1, -1};
    }

    ReleasedKV out{st->ctx, st->seq_id, st->position};

    // Free resources but NOT KV cache
    branch::destroy(handle_, store_);
    handle_ = INVALID_HANDLE;

    return out;
  }

  /**
   * Release full ownership including resources (Commit+Continue mode)
   *
   * - KV cache is PRESERVED at seq_id
   * - Branch resources are TRANSFERRED to caller (not freed)
   * - Branch handle is invalidated
   *
   * Use when you want to keep both KV cache AND sampling state.
   * Caller is responsible for freeing resources in ReleasedFull.
   */
  ReleasedFull release_full() {
    if (handle_ == INVALID_HANDLE || !store_) {
      return ReleasedFull{nullptr, nullptr, -1, -1, nullptr, nullptr, 0};
    }

    BranchState* st = store_->get(handle_);
    if (!st) {
      return ReleasedFull{nullptr, nullptr, -1, -1, nullptr, nullptr, 0};
    }

    // Steal ownership using std::exchange - nullifies slot's pointers
    // so destroy() won't free them
    ReleasedFull out{
      st->ctx,
      st->model,
      st->seq_id,
      st->position,
      std::exchange(st->sampler_chain, nullptr),
      std::exchange(st->grammar, nullptr),
      std::exchange(st->ppl, 0),
    };

    // Now safe to destroy - nothing left to free
    branch::destroy(handle_, store_);
    handle_ = INVALID_HANDLE;

    return out;
  }

  void capture_logits() {
    branch::capture_logits(handle_, store_);
  }

  void decode_batch(const llama_token* tokens, size_t n) {
    branch::decode_batch(handle_, tokens, n, store_);
  }

  void decode_one(llama_token token) {
    branch::decode_one(handle_, token, store_);
  }

  void decode_and_capture_batch(const llama_token* tokens, size_t n) {
    branch::decode_and_capture_batch(handle_, tokens, n, store_);
  }

  void decode_and_capture_one(llama_token token) {
    branch::decode_and_capture_one(handle_, token, store_);
  }

  const float* logits() const {
    return branch::get_logits(handle_, store_);
  }

  llama_token sample() {
    return branch::sample(handle_, store_);
  }

  void accept(llama_token token) {
    branch::accept_token(handle_, token, store_);
  }

  // Accessors
  llama_seq_id seq_id() const { return branch::get_seq_id(handle_, store_); }
  llama_pos position() const { return branch::get_position(handle_, store_); }
  float perplexity() const { return branch::get_perplexity(handle_, store_); }
  int n_vocab() const { return branch::get_n_vocab(handle_, store_); }

  bool valid() const { return handle_ != INVALID_HANDLE; }
  BranchHandle handle() const { return handle_; }

private:
  BranchStore* store_;
  BranchHandle handle_;
};

}  // namespace lloyal::branch
