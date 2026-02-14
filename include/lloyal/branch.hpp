#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

/**
 * @file branch.hpp
 * @brief Branch Primitive for Tree Search and Multi-Sequence Generation
 *
 * Consolidates all forkable state into a single handle-based API:
 * - KV cache sequence (via kv::seq_cp)
 * - Sampler chain (penalties, PRNG, top-k/p filters)
 * - Grammar constraints (GBNF parser state)
 * - Metrics (model + sampling perplexity trackers)
 * - Logits snapshot (captured distribution for deferred sampling)
 * - Logit bias (static token-level adjustments)
 *
 * Handle design:
 * - Handle = (generation << 16) | index
 * - Generation counter prevents ABA bugs on slot reuse
 * - Freelist enables branch pooling without malloc churn
 *
 * KV cache constraint:
 * - Each live branch occupies one llama_seq_id in the KV cache
 * - Max simultaneous branches = llama_context_params.n_seq_max (<= LLAMA_MAX_SEQ = 256)
 * - Slot table (65535) > n_seq_max — the store manages slots, not seq_ids
 * - Caller must allocate and recycle seq_ids to stay within n_seq_max
 *
 * Fork semantics:
 * - fork() clones: KV, grammar, sampler chain, metrics, logits, logit_bias
 * - fork() does NOT clone: steer callback (captures references, unsafe to copy)
 * - Each branch is fully independent after fork
 *
 * These primitives compose into inference patterns including:
 * - Best-of-N sampling (fork N candidates, select by perplexity)
 * - Speculative decoding (draft/verify with branch fork/prune)
 * - MCTS/LATS tree search (expand/backup/select with grammar priors)
 * - Beam search (top-k branches at each step)
 *
 * @example Basic usage (best-of-N)
 * @code
 *   auto root = branch::create(ctx, model, 0, prompt_len, params);
 *   branch::capture_logits(root);
 *
 *   auto child = branch::fork(root, new_seq_id);
 *   auto token = branch::sample(child);
 *   branch::accept_token(child, token);
 *   branch::decode_and_capture_one(child, token);
 *
 *   branch::prune(child);   // Remove KV + free resources
 *   branch::destroy(root);  // Free resources only
 * @endcode
 */

#include "boundaries.hpp"
#include "common.hpp"
#include "decode.hpp"
#include "grammar.hpp"
#include "kv.hpp"
#include "logits.hpp"
#include "metrics.hpp"
#include "sampler.hpp"

#include <llama/llama.h>
#include <cassert>    // assert
#include <cmath>      // std::exp, std::log, std::isinf, std::isfinite
#include <cstdint>
#include <cstring>    // std::memcpy
#include <ctime>      // std::time
#include <deque>      // std::deque (pointer stability for BranchStore)
#include <functional> // std::function
#include <limits>     // std::numeric_limits
#include <mutex>
#include <span>       // std::span (C++20)
#include <stdexcept>  // std::runtime_error
#include <string>     // std::to_string
#include <utility>    // std::pair, std::exchange
#include <vector>

namespace lloyal::branch {

// ===== HANDLE TYPE =====

/**
 * @brief Opaque handle to a branch slot
 *
 * Encoded as (generation << 16) | index:
 * - Upper 16 bits: generation counter (prevents ABA bugs on slot reuse)
 * - Lower 16 bits: slot index (max 65535 branches)
 * - Value 0 is reserved as the invalid/null handle
 */
using BranchHandle = uint32_t;

constexpr BranchHandle INVALID_HANDLE = 0;  ///< Null handle sentinel
constexpr int DEFAULT_N_BATCH = 512;        ///< Default batch size for decode operations
constexpr uint32_t GEN_SHIFT = 16;          ///< Bit shift for generation field
constexpr uint32_t INDEX_MASK = 0xFFFF;     ///< Mask for slot index field

/**
 * @brief Extract slot index from a branch handle
 * @param h Branch handle
 * @return Slot index (lower 16 bits)
 */
inline uint16_t handle_index(BranchHandle h) {
  return static_cast<uint16_t>(h & INDEX_MASK);
}

/**
 * @brief Extract generation counter from a branch handle
 * @param h Branch handle
 * @return Generation counter (upper 16 bits)
 */
inline uint16_t handle_generation(BranchHandle h) {
  return static_cast<uint16_t>(h >> GEN_SHIFT);
}

/**
 * @brief Construct a branch handle from index and generation
 * @param index Slot index (0–65535)
 * @param generation Generation counter
 * @return Encoded branch handle
 */
inline BranchHandle make_handle(uint16_t index, uint16_t generation) {
  return (static_cast<uint32_t>(generation) << GEN_SHIFT) | index;
}

// ===== BRANCH STATE =====

/**
 * @brief Consolidated mutable state for a single branch
 *
 * Each branch encapsulates all state needed for independent generation:
 *
 * Forkable state (cloned by fork()):
 * - KV cache sequence (via llama_memory_seq_cp)
 * - Sampler chain (penalties, PRNG, top-k/p filters)
 * - Grammar constraints (GBNF parser state)
 * - Boundary tracker (token boundary detection)
 * - Metrics (model + sampling perplexity)
 * - Logits snapshot (captured distribution for deferred sampling)
 * - Logit bias (static token-level adjustments)
 *
 * Non-forkable state (NOT cloned by fork()):
 * - steer_fn (dynamic callback — may capture references, unsafe to copy)
 *
 * Sampling application order in sample():
 *   Grammar → Logit Bias → Steer → Sampler Chain
 */
struct BranchState {
  llama_context* ctx = nullptr;       ///< Llama context (not owned, must outlive branch)
  const llama_model* model = nullptr; ///< Llama model (not owned, must outlive branch)

  llama_seq_id seq_id = 0;   ///< KV cache sequence identifier
  llama_pos position = 0;    ///< Current decode position in the sequence

  llama_sampler* sampler_chain = nullptr;  ///< Sampling chain: penalties + PRNG + filters (owned)
  bool has_dist_sampler = false;           ///< True if chain ends with dist (temp > 0), false if greedy

  llama_sampler* grammar = nullptr;  ///< Grammar sampler for GBNF constraints (owned, optional)

  boundaries::BoundaryTracker* boundary_tracker = nullptr;  ///< Token boundary detector (owned, optional)

  std::vector<llama_logit_bias> logit_bias;  ///< Static token biases, cloned on fork
  std::function<void(llama_token_data_array&)> steer_fn;  ///< Dynamic logit callback, NOT cloned on fork

  metrics::BranchMetricsHandle metrics = 0;  ///< Unified perplexity tracker (owned via handle)

  llama_token last_token = -1;                   ///< Last token returned by sample()
  std::vector<llama_token_data> last_candidates; ///< Filtered candidates from last sample()

  std::vector<float> logits_snapshot;  ///< Captured logit distribution (n_vocab floats)
  bool has_logits = false;             ///< True only after capture_logits() or decode_and_capture()

  /// Reusable scratch buffer for sampling (avoids O(n_vocab) allocs per sample call).
  ///
  /// Memory footprint per branch for large vocab models (128k tokens):
  /// - logits_snapshot: n_vocab * 4 bytes (~512KB)
  /// - candidates_buffer: n_vocab * sizeof(llama_token_data) (~1.5–2MB)
  /// - last_candidates: typically ~40 entries with top_k=40 (~480 bytes)
  ///
  /// @todo Move to per-thread or SessionContext scratch arena for deep MCTS trees.
  std::vector<llama_token_data> candidates_buffer;

  int n_batch = DEFAULT_N_BATCH;  ///< Batch size for decode operations
  int n_vocab = 0;    ///< Vocabulary size (cached for buffer pre-allocation)

  uint16_t generation = 0;  ///< Slot generation counter (for ABA prevention)
  bool in_use = false;      ///< True when slot is allocated to an active branch
};

// ===== BATCHED DECODE ITEM TYPES =====

/**
 * @brief Item for decode_each: one token per branch
 *
 * Replaces parallel (handle[], token[]) arrays with a single struct,
 * eliminating length-mismatch bugs structurally.
 */
struct DecodeEachItem {
  BranchHandle handle;
  llama_token token;
};

/**
 * @brief Item for decode_scatter: variable tokens per branch
 *
 * Uses std::span for zero-copy non-owning view of tokens.
 * Structural wins over parallel arrays:
 * - Can't have a handle without its tokens
 * - span::size() is size_t — negative counts impossible
 * - span with size > 0 can't have null data
 */
struct DecodeScatterItem {
  BranchHandle handle;
  std::span<const llama_token> tokens;
};

// ===== BRANCH STORE (HANDLE TABLE) =====

/**
 * @brief Handle table and batched decode orchestrator for branch management
 *
 * Provides two concerns:
 *
 * **Slot management** — A pool of BranchState slots addressed by opaque handles
 * with generation counters for ABA prevention. Slot 0 is permanently reserved
 * (handle 0 = INVALID_HANDLE). Auto-grows by doubling up to 65535 slots.
 * Methods: allocate(), release(), get().
 *
 * **Batched decode** — Orchestrates multi-branch GPU dispatches that amortize
 * llama_decode() overhead across N branches. Each method validates handles,
 * builds the appropriate decode primitive's input, dispatches, captures logits
 * into per-branch snapshots, and advances positions atomically.
 * Methods: decode_each(), decode_scatter().
 *
 * Batched decode methods vs free-function decode:
 * | Method              | Tokens/branch | Chunking    | Logit capture |
 * |---------------------|---------------|-------------|---------------|
 * | decode_each()       | 1             | No (1 call) | Per-branch    |
 * | decode_scatter()    | Variable      | Auto        | Per-branch    |
 * | branch::decode_and_capture_one() | 1 | No       | Single branch |
 *
 * @warning **n_seq_max constraint:** Each live branch consumes one KV cache
 *          sequence ID (llama_seq_id). The llama_context must be created with
 *          `llama_context_params.n_seq_max` >= max simultaneous branches.
 *          The hard ceiling is `LLAMA_MAX_SEQ` (256). BranchStore only manages
 *          slots — it has no awareness of seq_ids or n_seq_max. The caller is
 *          responsible for allocating and recycling seq_ids (e.g. via a
 *          freelist) to stay within the context's limit. No runtime validation
 *          is performed; exceeding n_seq_max causes undefined behavior in
 *          the KV cache.
 *
 * @warning Thread safety: External synchronization required (caller's mutex).
 *          Typically SessionContext holds _decodeMutex for decode operations.
 *
 * @see branch::create() to initialize a branch in this store
 * @see branch::fork() to clone a branch into a new sequence
 * @see branch::prune() / branch::destroy() for teardown
 */
class BranchStore {
public:
  /**
   * @brief Construct a branch store with initial slot capacity
   * @param initial_capacity Number of slots to pre-allocate (minimum 2)
   */
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

  /// @brief Destructor — frees all active branch resources
  ~BranchStore() {
    for (size_t i = 1; i < slots_.size(); ++i) {
      if (slots_[i].in_use) {
        free_branch_resources(slots_[i]);
      }
    }
  }

  /**
   * @brief Allocate a new branch slot from the freelist
   *
   * Auto-grows the store by doubling if the freelist is empty.
   *
   * @return Valid BranchHandle, or INVALID_HANDLE if store is at max capacity (65535)
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
   * @brief Release a branch slot back to the freelist
   *
   * Frees owned resources (sampler, grammar, metrics) and increments the
   * generation counter to invalidate any stale handles to this slot.
   *
   * @param handle Branch handle to release (INVALID_HANDLE is a safe no-op)
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
    slot.metrics = 0;
    slot.has_dist_sampler = false;
    slot.last_token = -1;
    slot.last_candidates.clear();
    slot.logits_snapshot.clear();
    slot.has_logits = false;
    slot.logit_bias.clear();
    slot.steer_fn = nullptr;
    slot.candidates_buffer.clear();  // Clear contents, capacity preserved for reuse
    slot.n_batch = DEFAULT_N_BATCH;
    slot.n_vocab = 0;

    freelist_.push_back(index);
  }

  /**
   * @brief Look up branch state by handle
   *
   * Validates the handle's index, generation, and in-use flag.
   * Slot 0 always returns nullptr (reserved for INVALID_HANDLE).
   *
   * @param handle Branch handle to look up
   * @return Pointer to BranchState, or nullptr if handle is invalid/stale
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

  /// @copydoc get(BranchHandle)
  const BranchState* get(BranchHandle handle) const {
    return const_cast<BranchStore*>(this)->get(handle);
  }

  // ===== BATCHED DECODE =====

  /**
   * @brief Decode one token per branch in a single GPU dispatch
   *
   * Packs N tokens (one per branch) into a single llama_batch and calls
   * decode::each(), amortizing GPU dispatch overhead across all branches.
   * After decode, captures logits from the batch into each branch's
   * logits_snapshot and advances each branch's position by 1.
   *
   * @note Batch index mapping: item[i] in the batch corresponds to
   *       `llama_get_logits_ith(ctx, i)`. This is a 1:1 mapping because
   *       decode::each places exactly one token per batch slot.
   *
   * @note Uses an internal scratch buffer. Since BranchStore requires external
   *       synchronization (caller's mutex), no concurrent access is possible.
   *
   * @param items    Span of {handle, token} pairs (all handles must be valid)
   * @throws std::runtime_error if any handle is invalid, contexts don't match,
   *         or decode fails
   *
   * @see decode::each() for the underlying single-batch primitive
   * @see decode_scatter() for variable token counts per branch
   */
  void decode_each(std::span<const DecodeEachItem> items) {
    if (items.empty()) return;

    const int32_t n = static_cast<int32_t>(items.size());

    // Validate handles and collect states
    std::vector<BranchState*> states(n);
    for (int32_t i = 0; i < n; ++i) {
      states[i] = get(items[i].handle);
      if (!states[i]) {
        throw std::runtime_error("BranchStore::decode_each - invalid handle at index " + std::to_string(i));
      }
      if (i > 0 && states[i]->ctx != states[0]->ctx) {
        throw std::runtime_error("BranchStore::decode_each - all branches must share the same context");
      }
    }

    // Build EachItem array from branch states
    std::vector<decode::EachItem> decode_items(n);
    for (int32_t i = 0; i < n; ++i) {
      decode_items[i].token = items[i].token;
      decode_items[i].pos = states[i]->position;
      decode_items[i].seq_id = states[i]->seq_id;
      decode_items[i].output_logits = true;
    }

    // Single GPU dispatch
    if (decode::each(states[0]->ctx, decode_items.data(), n, scratch_) != 0) {
      throw std::runtime_error("BranchStore::decode_each - llama_decode failed");
    }

    // Capture logits and update positions
    llama_context* ctx = states[0]->ctx;
    for (int32_t i = 0; i < n; ++i) {
      const float* raw_logits = logits::get(ctx, i);  // throws on null
      if (states[i]->n_vocab <= 0) {
        throw std::runtime_error("BranchStore::decode_each - invalid vocab size at index " + std::to_string(i));
      }
      assert(states[i]->logits_snapshot.size() >= static_cast<size_t>(states[i]->n_vocab));
      std::memcpy(states[i]->logits_snapshot.data(), raw_logits,
                  states[i]->n_vocab * sizeof(float));
      states[i]->has_logits = true;
      states[i]->position += 1;
    }
  }

  /**
   * @brief Decode variable token counts per branch with auto-chunking
   *
   * Two-pass algorithm:
   *
   * **Pass 1 — Build chunks:** Greedily bin-packs items into chunks up to
   * `llama_n_batch(ctx)` tokens. Oversized items (tokens.size() > n_batch)
   * get their own chunk and are dispatched via decode::many(). Zero-length
   * items are silently skipped.
   *
   * **Pass 2 — Dispatch:** Iterates chunks, dispatching normal chunks via
   * decode::scatter() and oversized chunks via decode::many(). Captures
   * logits into per-branch snapshots and advances positions.
   *
   * @note Uses an internal scratch buffer. Since BranchStore requires external
   *       synchronization (caller's mutex), no concurrent access is possible.
   *
   * @param items    Span of {handle, tokens} pairs (all handles must be valid)
   * @throws std::runtime_error if any handle is invalid, contexts don't match,
   *         or decode fails
   *
   * @see decode::scatter() for the underlying single-batch primitive
   * @see decode::many() for the oversized-item fallback
   * @see decode_each() for the simpler one-token-per-branch variant
   */
  void decode_scatter(std::span<const DecodeScatterItem> items) {
    if (items.empty()) return;

    const int32_t n = static_cast<int32_t>(items.size());

    // Validate handles and collect states
    std::vector<BranchState*> states(n);
    for (int32_t i = 0; i < n; ++i) {
      states[i] = get(items[i].handle);
      if (!states[i]) {
        throw std::runtime_error("BranchStore::decode_scatter - invalid handle at index " + std::to_string(i));
      }
      if (i > 0 && states[i]->ctx != states[0]->ctx) {
        throw std::runtime_error("BranchStore::decode_scatter - all branches must share the same context");
      }
    }

    llama_context* ctx = states[0]->ctx;
    const int32_t batch_limit = static_cast<int32_t>(llama_n_batch(ctx));

    // --- Pass 1: Build chunks ---
    struct Chunk {
      std::vector<int32_t> item_indices;
      bool oversized = false;
    };

    std::vector<Chunk> chunks;
    int32_t chunk_total = 0;

    for (int32_t i = 0; i < n; ++i) {
      int32_t tc = static_cast<int32_t>(items[i].tokens.size());
      if (tc == 0) continue;

      if (tc > batch_limit) {
        chunks.push_back({{i}, true});
        continue;
      }

      if (chunks.empty() || chunks.back().oversized ||
          chunk_total + tc > batch_limit) {
        chunks.push_back({{i}, false});
        chunk_total = tc;
      } else {
        chunks.back().item_indices.push_back(i);
        chunk_total += tc;
      }
    }

    // --- Pass 2: Dispatch each chunk ---
    for (const auto& chunk : chunks) {
      if (chunk.oversized) {
        // Single oversized item — dispatch via decode::many
        int32_t idx = chunk.item_indices[0];
        int32_t tc = static_cast<int32_t>(items[idx].tokens.size());

        if (decode::many(ctx, items[idx].tokens.data(), tc,
                         states[idx]->position, states[idx]->n_batch,
                         states[idx]->seq_id) != 0) {
          throw std::runtime_error("BranchStore::decode_scatter - decode::many failed for oversized item " + std::to_string(idx));
        }

        // Capture logits (many() outputs logits for last token at index -1)
        const float* raw_logits = logits::get(ctx, -1);
        if (states[idx]->n_vocab <= 0) {
          throw std::runtime_error("BranchStore::decode_scatter - invalid vocab size");
        }
        assert(states[idx]->logits_snapshot.size() >= static_cast<size_t>(states[idx]->n_vocab));
        std::memcpy(states[idx]->logits_snapshot.data(), raw_logits,
                    states[idx]->n_vocab * sizeof(float));
        states[idx]->has_logits = true;
        states[idx]->position += tc;
        continue;
      }

      // Normal chunk — build ScatterItems and dispatch
      std::vector<decode::ScatterItem> scatter_items(chunk.item_indices.size());
      for (size_t k = 0; k < chunk.item_indices.size(); ++k) {
        int32_t idx = chunk.item_indices[k];
        scatter_items[k].tokens = items[idx].tokens;
        scatter_items[k].start_pos = states[idx]->position;
        scatter_items[k].seq_id = states[idx]->seq_id;
        scatter_items[k].output_logits = true;
      }

      if (decode::scatter(ctx, scatter_items.data(),
                          static_cast<int32_t>(scatter_items.size()),
                          scratch_) != 0) {
        throw std::runtime_error("BranchStore::decode_scatter - decode::scatter failed");
      }

      // Capture logits for each item in the chunk
      int32_t cursor = 0;
      for (size_t k = 0; k < scatter_items.size(); ++k) {
        int32_t idx = chunk.item_indices[k];
        int32_t item_n = static_cast<int32_t>(scatter_items[k].tokens.size());
        int32_t logit_pos = cursor + item_n - 1;

        const float* raw_logits = logits::get(ctx, logit_pos);
        if (states[idx]->n_vocab <= 0) {
          throw std::runtime_error("BranchStore::decode_scatter - invalid vocab size for item " + std::to_string(idx));
        }
        assert(states[idx]->logits_snapshot.size() >= static_cast<size_t>(states[idx]->n_vocab));
        std::memcpy(states[idx]->logits_snapshot.data(), raw_logits,
                    states[idx]->n_vocab * sizeof(float));
        states[idx]->has_logits = true;
        states[idx]->position += static_cast<int32_t>(items[idx].tokens.size());

        cursor += item_n;
      }
    }
  }

private:
  /// @brief Free owned resources (sampler, grammar, boundary tracker, metrics)
  void free_branch_resources(BranchState& slot) {
    if (slot.sampler_chain) {
      sampler::free_chain(slot.sampler_chain);
      slot.sampler_chain = nullptr;
    }
    if (slot.grammar) {
      grammar::free_sampler(slot.grammar);
      slot.grammar = nullptr;
    }
    if (slot.boundary_tracker) {
      delete slot.boundary_tracker;
      slot.boundary_tracker = nullptr;
    }
    if (slot.metrics != 0) {
      metrics::free_branch_metrics(slot.metrics);
      slot.metrics = 0;
    }
  }

  /// Slot array. Uses std::deque (not std::vector) for pointer stability —
  /// get() returns BranchState* that remain valid across allocate()/grow().
  std::deque<BranchState> slots_;
  std::vector<uint16_t> freelist_;  ///< Available slot indices (LIFO)

  /// Reusable scratch buffers for batched decode. Safe without locking because
  /// BranchStore requires external synchronization (caller's mutex).
  decode::Scratch scratch_;
};

// ===== GLOBAL STORE =====

/// @cond INTERNAL
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
/// @endcond

/**
 * @brief Tear down the global branch store
 *
 * Frees all branches in the global store and deallocates it.
 * Call this BEFORE llama_backend_free() to ensure proper teardown order.
 *
 * Safe to call multiple times or if the global store was never used.
 *
 * @note After calling this, any handles obtained from the global store are invalid.
 *
 * @example
 * @code
 *   branch::shutdown_global_store();
 *   llama_backend_free();
 * @endcode
 */
inline void shutdown_global_store() {
  BranchStore*& ptr = detail::global_store_ptr();
  if (ptr) {
    delete ptr;
    ptr = nullptr;
  }
}

// ===== BRANCH API =====

/**
 * @brief Create a new branch with sampler chain, optional grammar, and metrics
 *
 * Allocates a slot from the store, initializes the sampler chain from @p params,
 * optionally attaches a GBNF grammar and boundary tracker, and pre-allocates
 * logits/candidates buffers sized to the model's vocabulary.
 *
 * @tparam P Any type satisfying the SamplingParamsLike concept
 * @param ctx Llama context (not owned, must outlive branch)
 * @param model Llama model (not owned, used for vocab size and sampler init)
 * @param seq_id KV cache sequence ID for this branch. Must be < n_seq_max
 *               configured on the context. Caller manages seq_id allocation.
 * @param start_pos Starting decode position (typically prompt length after prefill)
 * @param params Sampling parameters (temperature, top_k, top_p, penalties, etc.)
 * @param n_batch Batch size for decode operations (default 512)
 * @param grammar_str GBNF grammar string, or nullptr for unconstrained generation
 * @param boundary_tracker Boundary detector (ownership transferred), or nullptr
 * @param store Branch store to allocate from (nullptr = global store)
 * @return Valid BranchHandle, or INVALID_HANDLE on failure
 *
 * @see destroy() to free without KV cleanup, prune() to free with KV cleanup
 */
template <SamplingParamsLike P>
inline BranchHandle create(
    llama_context* ctx,
    const llama_model* model,
    llama_seq_id seq_id,
    llama_pos start_pos,
    const P& params,
    int n_batch = DEFAULT_N_BATCH,
    const char* grammar_str = nullptr,
    boundaries::BoundaryTracker* boundary_tracker = nullptr,
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

  // Track chain type for safe reseeding (general branch property, not MCTS-specific)
  // Greedy chains (temp <= 0) don't have dist sampler, reseeding would corrupt them
  float temperature = ::lloyal::detail::as_value(params.temperature, 0.8f);
  state->has_dist_sampler = (temperature > 0.0f);

  // Create grammar sampler if grammar string provided
  if (grammar_str && grammar_str[0] != '\0') {
    state->grammar = grammar::init_sampler(model, grammar_str);
  }

  // Take ownership of boundary tracker if provided
  state->boundary_tracker = boundary_tracker;

  // Create unified metrics tracker (model + sampling)
  state->metrics = metrics::create_branch_metrics();

  LLOYAL_LOG_DEBUG("[branch::create] Created branch handle=%u seq=%d pos=%d",
                   handle, seq_id, start_pos);

  return handle;
}

/**
 * @brief Destroy a branch, freeing resources but preserving KV cache
 *
 * Releases the sampler chain, grammar, metrics, and slot back to the store.
 * The KV cache entries for this branch's seq_id are NOT removed.
 * Use prune() instead if you also want to clear the KV cache.
 *
 * @param handle Branch to destroy (INVALID_HANDLE is a safe no-op)
 * @param store Branch store (nullptr = global store)
 *
 * @see prune() to remove KV cache entries and free resources
 */
inline void destroy(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  s.release(handle);
}

/**
 * @brief Fork a branch into a new independent sequence
 *
 * Creates a deep copy of the source branch under a new KV cache sequence ID.
 *
 * Cloned state:
 * - KV cache (via kv::seq_cp)
 * - Sampler chain (penalties, PRNG, filters)
 * - Grammar (parser state)
 * - Boundary tracker
 * - Metrics (model + sampling perplexity)
 * - Logits snapshot and logit bias
 *
 * NOT cloned:
 * - steer_fn (may capture references — call set_steer() on the child if needed)
 *
 * @param source Handle of the branch to fork from
 * @param new_seq_id KV cache sequence ID for the child branch. Must be
 *                   < n_seq_max configured on the context.
 * @param store Branch store (nullptr = global store)
 * @return Handle to the new child branch, or INVALID_HANDLE on failure
 *
 * @note Clears @p new_seq_id before copying to handle seq_id reuse safely.
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
    dst->has_dist_sampler = src->has_dist_sampler;  // Preserve chain type
  }

  // Clone grammar
  if (src->grammar) {
    dst->grammar = grammar::clone_sampler(src->grammar);
  }

  // Clone boundary tracker
  if (src->boundary_tracker) {
    dst->boundary_tracker = src->boundary_tracker->clone().release();
  }

  // Clone unified metrics (model + sampling)
  if (src->metrics != 0) {
    dst->metrics = metrics::clone_branch_metrics(src->metrics);
  }

  // Copy last token and candidates (shallow copy)
  dst->last_token = src->last_token;
  dst->last_candidates = src->last_candidates;

  // Copy logits snapshot and validity flag
  dst->logits_snapshot = src->logits_snapshot;
  dst->has_logits = src->has_logits;

  // Clone logit bias (safe - just vector of structs)
  dst->logit_bias = src->logit_bias;

  // DO NOT clone steer_fn (callbacks may capture references)
  // dst->steer_fn remains default-constructed (empty)

  // Pre-allocate candidates buffer (don't copy contents, just capacity)
  dst->candidates_buffer.resize(dst->n_vocab);

  LLOYAL_LOG_DEBUG("[branch::fork] Forked handle=%u -> handle=%u seq=%d->%d",
                   source, new_handle, src->seq_id, new_seq_id);

  return new_handle;
}

/**
 * @brief Set static logit bias for specific tokens
 *
 * Adds additive bias to token logits before sampling. Use -INFINITY to ban tokens.
 * Replaces any existing biases. Logit bias is CLONED when forking.
 *
 * Applied in sample() order: Grammar → Logit Bias → Steer → Sampler Chain
 *
 * @param handle Branch to modify
 * @param biases Array of llama_logit_bias structs {token, bias}
 * @param n_biases Number of biases in the array
 * @param store Branch store (nullptr = global store)
 * @throws std::runtime_error if handle is invalid
 *
 * @example
 * @code
 *   llama_logit_bias biases[] = {{42, -INFINITY}, {100, 2.5f}};
 *   branch::set_logit_bias(handle, biases, 2);  // Ban token 42, boost token 100
 * @endcode
 */
inline void set_logit_bias(
    BranchHandle handle,
    const llama_logit_bias* biases,
    size_t n_biases,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("set_logit_bias: invalid branch handle");
  }

  // Replace existing biases with new set
  state->logit_bias.assign(biases, biases + n_biases);

  LLOYAL_LOG_DEBUG("[branch::set_logit_bias] Set %zu biases on handle=%u",
                   n_biases, handle);
}

/**
 * @brief Clear all logit biases from a branch
 *
 * @param handle Branch to modify
 * @param store Branch store (nullptr = global store)
 * @throws std::runtime_error if handle is invalid
 */
inline void clear_logit_bias(
    BranchHandle handle,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("clear_logit_bias: invalid branch handle");
  }

  state->logit_bias.clear();

  LLOYAL_LOG_DEBUG("[branch::clear_logit_bias] Cleared biases on handle=%u", handle);
}

/**
 * @brief Set dynamic steer callback for runtime logit modification
 *
 * The callback receives a llama_token_data_array and can modify logits directly.
 * Steer is NOT cloned when forking — must be re-set on the child if needed.
 *
 * Applied in sample() order: Grammar → Logit Bias → Steer → Sampler Chain
 *
 * @param handle Branch to modify
 * @param steer_fn Callback: void(llama_token_data_array&)
 * @param store Branch store (nullptr = global store)
 * @throws std::runtime_error if handle is invalid
 *
 * @warning Callback may capture references — ensure their lifetime exceeds branch usage.
 *
 * @example MCTS action deduplication
 * @code
 *   branch::set_steer(child, [&explored](llama_token_data_array& cur_p) {
 *     for (size_t i = 0; i < cur_p.size; ++i) {
 *       if (explored.count(cur_p.data[i].id)) {
 *         cur_p.data[i].logit = -INFINITY;
 *       }
 *     }
 *   });
 * @endcode
 */
inline void set_steer(
    BranchHandle handle,
    std::function<void(llama_token_data_array&)> steer_fn,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("set_steer: invalid branch handle");
  }

  state->steer_fn = std::move(steer_fn);

  LLOYAL_LOG_DEBUG("[branch::set_steer] Set steer callback on handle=%u", handle);
}

/**
 * @brief Clear the steer callback from a branch
 *
 * @param handle Branch to modify
 * @param store Branch store (nullptr = global store)
 * @throws std::runtime_error if handle is invalid
 */
inline void clear_steer(
    BranchHandle handle,
    BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();

  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("clear_steer: invalid branch handle");
  }

  state->steer_fn = nullptr;

  LLOYAL_LOG_DEBUG("[branch::clear_steer] Cleared steer callback on handle=%u", handle);
}

/**
 * @brief Remove branch from KV cache and free all resources
 *
 * Clears the branch's KV cache entries (all positions for its seq_id),
 * then releases the slot and owned resources back to the store.
 *
 * @param handle Branch to prune (INVALID_HANDLE is a safe no-op)
 * @param store Branch store (nullptr = global store)
 *
 * @see destroy() to free resources without clearing KV cache
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
 * @brief Decode multiple tokens into the branch's KV cache sequence
 *
 * Feeds tokens through the model in n_batch-sized chunks, advancing
 * the branch position. Does NOT capture logits — call capture_logits()
 * or use decode_and_capture_batch() if you need them.
 *
 * @param handle Branch to decode into
 * @param tokens Array of token IDs
 * @param n_tokens Number of tokens in the array
 * @param store Branch store (nullptr = global store)
 * @throws std::runtime_error if handle is invalid or decode fails
 *
 * @note For single-token decode, prefer decode_one() (zero heap allocation).
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
  if (decode::many(state->ctx, tokens, static_cast<int32_t>(n_tokens),
                   state->position, state->n_batch, state->seq_id) != 0) {
    throw std::runtime_error("decode_batch: llama_decode failed");
  }

  state->position += static_cast<llama_pos>(n_tokens);
}

/**
 * @brief Decode a single token (no per-call allocation)
 *
 * Uses decode::one() which maintains a thread_local batch — heap-allocated
 * once per thread, reused across calls. No per-call allocation.
 * Does NOT capture logits; use decode_and_capture_one() if needed.
 *
 * @param handle Branch to decode into
 * @param token Token ID to decode
 * @param store Branch store (nullptr = global store)
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

  if (decode::one(state->ctx, token, state->position, state->seq_id, false) != 0) {
    throw std::runtime_error("decode_one: llama_decode failed");
  }
  state->position += 1;
}

/**
 * @brief Capture current context logits into the branch's snapshot
 *
 * Copies the logits from the llama_context into the branch's internal buffer
 * without performing a decode. Call this after prefill to initialize the root
 * branch for sampling.
 *
 * @param handle Branch to capture logits for
 * @param store Branch store (nullptr = global store)
 * @throws std::runtime_error if handle is invalid, vocab size is zero,
 *         or no logits are available (no prior decode with logits enabled)
 *
 * @note Sets has_logits = true, enabling sample() and get_logits().
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
 * @brief Decode multiple tokens and capture logits atomically
 *
 * Combines decode_batch() + capture_logits() in a single call.
 * After this call, sample() and get_logits() are available.
 *
 * @param handle Branch to decode into
 * @param tokens Array of token IDs
 * @param n_tokens Number of tokens in the array
 * @param store Branch store (nullptr = global store)
 * @throws std::runtime_error if handle is invalid, decode fails,
 *         or logits capture fails
 *
 * @note For single-token decode, prefer decode_and_capture_one() (zero heap allocation).
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
  if (decode::many(state->ctx, tokens, static_cast<int32_t>(n_tokens),
                   state->position, state->n_batch, state->seq_id) != 0) {
    throw std::runtime_error("decode_and_capture_batch: llama_decode failed");
  }

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
 * @brief Decode a single token and capture logits (no per-call allocation)
 *
 * Uses decode::one() which maintains a thread_local batch — heap-allocated
 * once per thread, reused across calls. No per-call allocation.
 * Combines decode_one() + capture_logits() in a single call.
 *
 * @param handle Branch to decode into
 * @param token Token ID to decode
 * @param store Branch store (nullptr = global store)
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

  if (decode::one(state->ctx, token, state->position, state->seq_id, true) != 0) {
    throw std::runtime_error("decode_and_capture_one: llama_decode failed");
  }
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
 * @brief Get the branch's captured logits snapshot
 *
 * Returns a pointer to the internal logits buffer (n_vocab floats).
 * Only valid after capture_logits() or a decode_and_capture call.
 *
 * @param handle Branch to read logits from
 * @param store Branch store (nullptr = global store)
 * @return Pointer to n_vocab floats, or nullptr if no logits captured
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
 * @brief Sample a token from the branch's captured logits
 *
 * Applies modifiers in order: Grammar → Logit Bias → Steer → Sampler Chain,
 * then selects a token. Also records filtered candidates for metrics.
 *
 * Requires prior capture_logits() or decode_and_capture call.
 *
 * @param handle Branch to sample from
 * @param store Branch store (nullptr = global store)
 * @return Sampled token ID, or -1 if no logits captured or sampling fails
 *
 * @note Call accept_token() after sampling to advance grammar and penalty state.
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

  // Apply logit bias if present — O(n_biases) via direct index
  // (candidates are in token-ID order; grammar::apply preserves order)
  if (!state->logit_bias.empty()) {
    for (const auto& bias : state->logit_bias) {
      if (bias.token >= 0 && bias.token < state->n_vocab) {
        cur_p.data[bias.token].logit += bias.bias;
      }
    }
  }

  // Apply steer callback if present
  if (state->steer_fn) {
    try {
      state->steer_fn(cur_p);
    } catch (const std::exception& e) {
      LLOYAL_LOG_DEBUG("[branch::sample] Steer exception: %s", e.what());
      // Continue sampling without steer on exception
    }
  }

  // Apply sampler chain (via anti-corruption layer)
  sampler::apply(state->sampler_chain, &cur_p);

  if (cur_p.selected == -1) {
    return -1;
  }

  llama_token token = cur_p.data[cur_p.selected].id;

  // Capture filtered candidates for sampling metrics
  // After BOTH grammar and sampler chain - this is the actual sampling distribution
  state->last_token = token;
  state->last_candidates.clear();
  state->last_candidates.reserve(cur_p.size);

  for (size_t i = 0; i < cur_p.size; i++) {
    state->last_candidates.push_back(cur_p.data[i]);
  }

  return token;
}

/**
 * @brief Accept a sampled token, advancing grammar and sampler state
 *
 * Updates:
 * - Grammar parser state (if grammar is attached)
 * - Sampler chain penalty tracking (repetition/frequency penalties)
 * - Model-level perplexity (from raw logits, if available)
 * - Sampling-level perplexity (from filtered candidate distribution)
 *
 * @param handle Branch that produced the token
 * @param token Token ID returned by sample()
 * @param store Branch store (nullptr = global store)
 *
 * @note Safe to call with invalid handle (silent no-op).
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

  // Update model-level perplexity (from raw logits)
  // Guard on has_logits to avoid computing surprisal from zero-filled buffer
  if (state->metrics != 0 && state->has_logits) {
    float model_surprisal = metrics::model_surprisal(
        state->logits_snapshot.data(), state->n_vocab, token);
    metrics::add_model_surprisal(state->metrics, model_surprisal);
  }

  // Update sampling-level perplexity (from filtered candidates)
  if (state->metrics != 0 && !state->last_candidates.empty() &&
      token == state->last_token) {
    // Extract filtered logits and IDs from candidates
    std::vector<float> candidate_logits;
    std::vector<int32_t> candidate_ids;
    candidate_logits.reserve(state->last_candidates.size());
    candidate_ids.reserve(state->last_candidates.size());

    for (const auto& cand : state->last_candidates) {
      candidate_logits.push_back(cand.logit);
      candidate_ids.push_back(cand.id);
    }

    // Compute sampling-level surprisal
    float sampling_surprisal = metrics::sampling_surprisal(
        candidate_logits.data(),
        candidate_ids.data(),
        static_cast<int>(candidate_logits.size()),
        token
    );
    metrics::add_sampling_surprisal(state->metrics, sampling_surprisal);
  }
}

/**
 * @brief Apply grammar constraints to an external logits buffer
 *
 * Sets logits of grammar-illegal tokens to -INFINITY in the provided buffer.
 * Uses the branch's internal candidates_buffer as scratch space when vocab
 * sizes match; allocates a temporary buffer otherwise.
 *
 * @param handle Branch with grammar to apply
 * @param logits Logits buffer to modify in place (n_vocab floats)
 * @param n_vocab Number of entries in the logits buffer
 * @param store Branch store (nullptr = global store)
 *
 * @note No-op if handle is invalid or branch has no grammar attached.
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
 * @brief Get grammar-legal tokens with renormalized probabilities
 *
 * Returns (token, probability) pairs for tokens that pass grammar constraints.
 * Probabilities are softmax-normalized over the legal set only (sum to 1.0).
 *
 * Essential for PUCT in MCTS: policy priors must only cover legal moves.
 *
 * @param handle Branch with captured logits and optional grammar
 * @param store Branch store (nullptr = global store)
 * @return Vector of (token_id, probability) pairs, empty if no logits or no legal tokens
 *
 * @note If no grammar is attached, all tokens with finite logits are included.
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
 * @brief Compute log-sum-exp over grammar-legal logits
 *
 * Returns log(sum(exp(logit_i))) over tokens that pass grammar constraints.
 * Use for efficient per-token prior computation:
 *   P(token) = exp(logit[token] - logsumexp)
 *
 * Numerically stable (max-subtraction trick).
 *
 * @param handle Branch with captured logits and optional grammar
 * @param store Branch store (nullptr = global store)
 * @return Log-sum-exp value, or -INFINITY if no legal tokens or invalid state
 *
 * @see get_token_prior_assume_legal() for O(1) per-token prior using this value
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
 * @brief Check if a token is legal under grammar constraints
 *
 * Uses a 1-element candidate array for O(grammar_complexity) check
 * instead of the O(n_vocab) full scan used by get_legal_priors().
 *
 * @param handle Branch with optional grammar
 * @param token Token ID to check
 * @param store Branch store (nullptr = global store)
 * @return true if token is legal (or no grammar attached), false if illegal
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
 * @brief Compute prior probability for a token known to be grammar-legal
 *
 * O(1) operation — use in MCTS inner loops where sample() already enforced grammar.
 * Does NOT validate grammar legality; caller must ensure token is legal.
 *
 * @param handle Branch with captured logits
 * @param token Token ID (must be legal under grammar)
 * @param logsumexp Pre-computed value from get_legal_logsumexp()
 * @param store Branch store (nullptr = global store)
 * @return Probability in [0, 1], or 0 if state is invalid
 *
 * @see get_token_prior() for a safe version that checks grammar legality
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
 * @brief Compute prior probability for a token, checking grammar legality first
 *
 * O(grammar_complexity) — uses is_token_legal() before computing the prior.
 * Safe for ad-hoc callers who don't know whether the token is grammar-legal.
 *
 * @param handle Branch with captured logits and optional grammar
 * @param token Token ID to compute prior for
 * @param logsumexp Pre-computed value from get_legal_logsumexp()
 * @param store Branch store (nullptr = global store)
 * @return Probability in [0, 1], or 0 if token is illegal
 *
 * @note For MCTS inner loops, prefer get_token_prior_assume_legal() since
 *       sample() already enforces grammar constraints.
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

// ===== STATE ACCESSORS =====

/**
 * @brief Get the branch's KV cache sequence ID
 * @param handle Branch handle
 * @param store Branch store (nullptr = global store)
 * @return Sequence ID, or -1 if handle is invalid
 */
inline llama_seq_id get_seq_id(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  const BranchState* state = s.get(handle);
  return state ? state->seq_id : -1;
}

/**
 * @brief Get the branch's current decode position
 * @param handle Branch handle
 * @param store Branch store (nullptr = global store)
 * @return Token position, or -1 if handle is invalid
 */
inline llama_pos get_position(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  const BranchState* state = s.get(handle);
  return state ? state->position : -1;
}

/**
 * @brief Get model-level perplexity (from raw logits)
 *
 * Returns perplexity computed from the full logit distribution before any
 * sampler filtering. For the distribution actually sampled from, use
 * get_sampling_perplexity().
 *
 * @param handle Branch handle
 * @param store Branch store (nullptr = global store)
 * @return Model perplexity, or INFINITY if no tokens accepted
 */
inline float get_perplexity(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  const BranchState* state = s.get(handle);
  if (!state || state->metrics == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return metrics::get_model_ppl(state->metrics);
}

/**
 * @brief Get sampling-level perplexity (from filtered distribution)
 *
 * Returns perplexity from the distribution actually sampled from
 * (after top-k/p/temp/penalties). Useful for PUCT priors and
 * monitoring sampler chain impact.
 *
 * @param handle Branch handle
 * @param store Branch store (nullptr = global store)
 * @return Sampling-level perplexity, or INFINITY if no tokens accepted
 */
inline float get_sampling_perplexity(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  const BranchState* state = s.get(handle);
  if (!state || state->metrics == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return metrics::get_sampling_ppl(state->metrics);
}

/**
 * @brief Get the last sampled token's prior from the filtered distribution
 *
 * Returns P(token) from the post-filter sampling distribution.
 * This is the correct prior for PUCT since it matches what was actually sampled.
 *
 * @param handle Branch handle
 * @param store Branch store (nullptr = global store)
 * @return Probability of last sampled token in [0, 1], or 0 if unavailable
 */
inline float get_last_sampling_prior(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  const BranchState* state = s.get(handle);

  if (!state || state->last_candidates.empty() || state->last_token < 0) {
    return 0.0f;
  }

  // Extract candidates
  std::vector<float> candidate_logits;
  std::vector<int32_t> candidate_ids;
  candidate_logits.reserve(state->last_candidates.size());
  candidate_ids.reserve(state->last_candidates.size());

  for (const auto& cand : state->last_candidates) {
    candidate_logits.push_back(cand.logit);
    candidate_ids.push_back(cand.id);
  }

  // Compute surprisal from filtered distribution
  float surprisal = metrics::sampling_surprisal(
      candidate_logits.data(),
      candidate_ids.data(),
      static_cast<int>(candidate_logits.size()),
      state->last_token
  );

  // Convert to probability: P = exp(-surprisal)
  return std::exp(-surprisal);
}

/**
 * @brief Get the branch's vocabulary size
 * @param handle Branch handle
 * @param store Branch store (nullptr = global store)
 * @return Vocabulary size, or 0 if handle is invalid
 */
inline int get_n_vocab(BranchHandle handle, BranchStore* store = nullptr) {
  BranchStore& s = store ? *store : detail::global_store();
  const BranchState* state = s.get(handle);
  return state ? state->n_vocab : 0;
}

// ===== RELEASE STRUCTURES =====

/**
 * @brief Result of Branch::release_kv() — KV cache preserved, resources freed
 *
 * Use for "Commit+Reset" mode: keep the KV cache but restart sampling fresh
 * with a new sampler chain / grammar.
 */
struct ReleasedKV {
  llama_context* ctx;   ///< Context the KV cache lives in
  llama_seq_id seq_id;  ///< Sequence ID with preserved KV entries
  llama_pos position;   ///< Decode position (token count)
};

/**
 * @brief Result of Branch::release_full() — KV cache preserved, resource ownership transferred
 *
 * Use for "Commit+Continue" mode: keep both KV cache and sampling state.
 *
 * @warning Caller takes ownership and must free these resources:
 *          - sampler::free_chain(sampler_chain)
 *          - grammar::free_sampler(grammar) if non-null
 *          - metrics::free_branch_metrics(metrics) if non-zero
 */
struct ReleasedFull {
  llama_context* ctx;          ///< Context the KV cache lives in
  const llama_model* model;    ///< Model reference (not owned)
  llama_seq_id seq_id;         ///< Sequence ID with preserved KV entries
  llama_pos position;          ///< Decode position (token count)
  llama_sampler* sampler_chain;              ///< Caller must call sampler::free_chain()
  llama_sampler* grammar;                    ///< Caller must call grammar::free_sampler() if non-null
  metrics::BranchMetricsHandle metrics;      ///< Caller must call metrics::free_branch_metrics() if non-zero
};

// ===== RAII WRAPPER =====

/**
 * @brief RAII wrapper around BranchHandle for automatic resource management
 *
 * Move-only value type. Destructor calls prune() (clears KV + frees resources).
 * Use release_kv() or release_full() to detach ownership before destruction.
 *
 * @example Basic usage
 * @code
 *   Branch root = Branch::create(ctx, model, 0, pos, params);
 *   Branch child = root.fork(1);
 *
 *   child.decode_and_capture_one(token);
 *   auto next = child.sample();
 *   child.accept(next);
 * @endcode
 *
 * @example Winner commit pattern
 * @code
 *   auto released = winner.release_kv();  // KV survives, resources freed
 *   // winner is now invalid, but KV at released.seq_id is preserved
 * @endcode
 */
class Branch {
public:
  /// @brief Construct an invalid (empty) branch
  Branch() : store_(nullptr), handle_(INVALID_HANDLE) {}

  /// @brief Construct from an existing store and handle (takes ownership)
  Branch(BranchStore* store, BranchHandle handle)
      : store_(store), handle_(handle) {}

  /// @brief Destructor — prunes KV cache and frees all resources
  ~Branch() {
    if (handle_ != INVALID_HANDLE && store_) {
      branch::prune(handle_, store_);
    }
  }

  /// @brief Move constructor (transfers ownership, source becomes invalid)
  Branch(Branch&& other) noexcept
      : store_(other.store_), handle_(other.handle_) {
    other.handle_ = INVALID_HANDLE;
  }

  /// @brief Move assignment (prunes current branch, then transfers ownership)
  Branch& operator=(Branch&& other) noexcept {
    if (this != &other) {
      if (handle_ != INVALID_HANDLE && store_) {
        branch::prune(handle_, store_);
      }
      store_ = other.store_;
      handle_ = other.handle_;
      other.handle_ = INVALID_HANDLE;
    }
    return *this;
  }

  Branch(const Branch&) = delete;             ///< Non-copyable
  Branch& operator=(const Branch&) = delete;  ///< Non-copyable

  /**
   * @brief Create a new branch (factory method)
   * @see branch::create() for parameter documentation
   */
  template <SamplingParamsLike P>
  static Branch create(
      llama_context* ctx,
      const llama_model* model,
      llama_seq_id seq_id,
      llama_pos start_pos,
      const P& params,
      int n_batch = DEFAULT_N_BATCH,
      const char* grammar_str = nullptr,
      boundaries::BoundaryTracker* boundary_tracker = nullptr,
      BranchStore* store = nullptr) {
    BranchStore* s = store ? store : &detail::global_store();
    BranchHandle h = branch::create(ctx, model, seq_id, start_pos, params, n_batch, grammar_str, boundary_tracker, s);
    return Branch(s, h);
  }

  /**
   * @brief Fork this branch into a new independent sequence
   * @param new_seq_id KV cache sequence ID for the child
   * @return New Branch owning the forked state
   * @see branch::fork()
   */
  Branch fork(llama_seq_id new_seq_id) {
    BranchHandle h = branch::fork(handle_, new_seq_id, store_);
    return Branch(store_, h);
  }

  /// @brief Prune this branch (clear KV + free resources), invalidating the handle
  void prune() {
    branch::prune(handle_, store_);
    handle_ = INVALID_HANDLE;
  }

  /**
   * @brief Release KV cache ownership without pruning (Commit+Reset mode)
   *
   * KV cache entries are PRESERVED. Branch resources (sampler, grammar,
   * metrics) are FREED. The Branch object becomes invalid after this call.
   *
   * @return ReleasedKV with ctx, seq_id, and position for continued use
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
   * @brief Release full ownership including resources (Commit+Continue mode)
   *
   * KV cache entries are PRESERVED. Resource ownership is TRANSFERRED to
   * the caller (not freed). The Branch object becomes invalid after this call.
   *
   * @return ReleasedFull with all state — caller must free owned resources
   * @warning Caller takes ownership of sampler_chain, grammar, and metrics.
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
      std::exchange(st->metrics, 0),
    };

    // Now safe to destroy - nothing left to free
    branch::destroy(handle_, store_);
    handle_ = INVALID_HANDLE;

    return out;
  }

  /// @brief Capture current context logits into this branch's snapshot
  /// @see branch::capture_logits()
  void capture_logits() {
    branch::capture_logits(handle_, store_);
  }

  /// @brief Decode multiple tokens into this branch's KV cache
  /// @see branch::decode_batch()
  void decode_batch(const llama_token* tokens, size_t n) {
    branch::decode_batch(handle_, tokens, n, store_);
  }

  /// @brief Decode a single token (zero-allocation fast path)
  /// @see branch::decode_one()
  void decode_one(llama_token token) {
    branch::decode_one(handle_, token, store_);
  }

  /// @brief Decode multiple tokens and capture logits atomically
  /// @see branch::decode_and_capture_batch()
  void decode_and_capture_batch(const llama_token* tokens, size_t n) {
    branch::decode_and_capture_batch(handle_, tokens, n, store_);
  }

  /// @brief Decode a single token and capture logits (zero-allocation fast path)
  /// @see branch::decode_and_capture_one()
  void decode_and_capture_one(llama_token token) {
    branch::decode_and_capture_one(handle_, token, store_);
  }

  /// @brief Get captured logits snapshot (n_vocab floats), or nullptr
  /// @see branch::get_logits()
  const float* logits() const {
    return branch::get_logits(handle_, store_);
  }

  /// @brief Sample a token using Grammar → Logit Bias → Steer → Sampler Chain
  /// @see branch::sample()
  llama_token sample() {
    return branch::sample(handle_, store_);
  }

  /// @brief Accept a sampled token, advancing grammar and sampler state
  /// @see branch::accept_token()
  void accept(llama_token token) {
    branch::accept_token(handle_, token, store_);
  }

  // -- Accessors --

  llama_seq_id seq_id() const { return branch::get_seq_id(handle_, store_); }  ///< @see branch::get_seq_id()
  llama_pos position() const { return branch::get_position(handle_, store_); }  ///< @see branch::get_position()
  float perplexity() const { return branch::get_perplexity(handle_, store_); }  ///< @see branch::get_perplexity()
  int n_vocab() const { return branch::get_n_vocab(handle_, store_); }          ///< @see branch::get_n_vocab()

  bool valid() const { return handle_ != INVALID_HANDLE; }  ///< True if this branch holds a valid handle
  BranchHandle handle() const { return handle_; }           ///< Get the underlying raw handle

private:
  BranchStore* store_;    ///< Store this branch was allocated from
  BranchHandle handle_;   ///< Opaque handle to the branch slot
};

}  // namespace lloyal::branch
