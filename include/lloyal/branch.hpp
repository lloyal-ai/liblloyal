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
 * - Max simultaneous branches = llama_n_seq_max(ctx) (<= 256)
 * - Slot table (65535) > n_seq_max — slots are abundant, leases are scarce
 * - kv::tenancy manages seq_id lifecycle (allocate/evict/retain/drain)
 *
 * Fork semantics:
 * - fork() clones: KV, grammar, sampler chain, metrics, logits, logit_bias
 * - fork() does NOT clone: steer callback (captures references, unsafe to copy)
 * - Each branch is fully independent after fork
 *
 * These primitives compose into inference patterns including:
 * - Best-of-N sampling (fork N candidates, select by perplexity)
 * - Speculative decoding (draft/verify with branch fork/prune)
 * - Tree search (expand/backup/select with grammar priors)
 * - Beam search (top-k branches at each step)
 *
 * @example Basic usage (best-of-N)
 * @code
 *   store.init_tenancy(ctx);
 *   auto root = branch::create(ctx, model, store, prompt_len, params);
 *   branch::capture_logits(root, store);
 *
 *   auto child = branch::fork(root, store);
 *   auto token = branch::sample(child, store);
 *   branch::accept_token(child, token, store);
 *   branch::decode_and_capture_one(child, token, store);
 *
 *   branch::prune(child, store);          // RESTRICT leaf prune
 *   store.retainOnly(root);               // Keep winner, nuke rest
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
#include <algorithm>  // std::remove
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
constexpr llama_seq_id NO_LEASE = kv::NO_LEASE;  ///< Branch has no KV residency
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

// ===== REGISTRY HANDLE TYPES =====

/// Handle to a sampler chain in BranchStore's registry (0 = invalid/none)
using SamplerChainHandle = int32_t;

/// Handle to a grammar sampler in BranchStore's registry (0 = invalid/none)
using GrammarHandle = int32_t;

/// Handle to a metrics tracker in BranchStore's registry (0 = invalid/none)
using MetricsHandle = int32_t;

/**
 * @brief RAII entry for a sampler chain in the registry
 *
 * Owns the llama_sampler* and tracks whether the chain ends with dist (stochastic)
 * or greedy. Move-only to prevent double-free.
 */
struct SamplerChainEntry {
  llama_sampler* chain = nullptr;
  bool has_dist = false;   ///< True if chain ends with dist (temp > 0), false if greedy

  SamplerChainEntry() = default;
  ~SamplerChainEntry() { if (chain) sampler::free_chain(chain); }

  SamplerChainEntry(SamplerChainEntry&& o) noexcept
      : chain(o.chain), has_dist(o.has_dist) { o.chain = nullptr; }
  SamplerChainEntry& operator=(SamplerChainEntry&& o) noexcept {
    if (this != &o) {
      if (chain) sampler::free_chain(chain);
      chain = o.chain; has_dist = o.has_dist; o.chain = nullptr;
    }
    return *this;
  }
  SamplerChainEntry(const SamplerChainEntry&) = delete;
  SamplerChainEntry& operator=(const SamplerChainEntry&) = delete;
};

/**
 * @brief RAII entry for a grammar sampler in the registry
 *
 * Owns the llama_sampler* for GBNF grammar constraints. Move-only.
 */
struct GrammarEntry {
  llama_sampler* sampler = nullptr;

  GrammarEntry() = default;
  ~GrammarEntry() { if (sampler) grammar::free_sampler(sampler); }

  GrammarEntry(GrammarEntry&& o) noexcept : sampler(o.sampler) { o.sampler = nullptr; }
  GrammarEntry& operator=(GrammarEntry&& o) noexcept {
    if (this != &o) {
      if (sampler) grammar::free_sampler(sampler);
      sampler = o.sampler; o.sampler = nullptr;
    }
    return *this;
  }
  GrammarEntry(const GrammarEntry&) = delete;
  GrammarEntry& operator=(const GrammarEntry&) = delete;
};

/**
 * @brief Concrete sampling params snapshot for memoization
 *
 * Captures resolved parameter values at chain creation time. Used by
 * set_sampler_params() to skip rebuilding the chain when params haven't changed.
 * Satisfies SamplingParamsLike for re-creation.
 */
struct CachedSamplingParams {
  float temperature = 0.8f;
  int32_t top_k = 40;
  float top_p = 0.95f;
  float typical_p = 1.0f;
  float min_p = 0.05f;
  float penalty_repeat = 1.0f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 64;
  uint32_t seed = 0;
  bool operator==(const CachedSamplingParams&) const = default;
};

/**
 * @brief Snapshot sampling params for memoization comparison
 *
 * Extracts resolved values from any SamplingParamsLike type using the same
 * defaults as sampler::create_chain(), except seed defaults to 0 (not time)
 * to avoid false cache misses from non-deterministic defaults.
 */
template <SamplingParamsLike P>
inline CachedSamplingParams snapshot_params(const P& p) {
  using ::lloyal::detail::as_value;
  return CachedSamplingParams{
    as_value(p.temperature, 0.8f),
    as_value(p.top_k, static_cast<int32_t>(40)),
    as_value(p.top_p, 0.95f),
    as_value(p.typical_p, 1.0f),
    as_value(p.min_p, 0.05f),
    as_value(p.penalty_repeat, 1.0f),
    as_value(p.penalty_freq, 0.0f),
    as_value(p.penalty_present, 0.0f),
    as_value(p.penalty_last_n, static_cast<int32_t>(64)),
    as_value(p.seed, static_cast<uint32_t>(0)),
  };
}

// ===== BRANCH STATE =====

/**
 * @brief Consolidated mutable state for a single branch
 *
 * Each branch encapsulates all state needed for independent generation:
 *
 * Forkable state (cloned by fork()):
 * - KV cache sequence (via llama_memory_seq_cp)
 * - Sampler chain (handle into BranchStore registry)
 * - Grammar constraints (handle into BranchStore registry)
 * - Boundary tracker (token boundary detection)
 * - Metrics (handle into BranchStore registry)
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

  llama_seq_id seq_id = NO_LEASE;  ///< KV cache sequence identifier (NO_LEASE when inactive)
  llama_pos position = 0;    ///< Current decode position in the sequence

  SamplerChainHandle sampler_chain = 0;  ///< Handle into BranchStore's sampler registry
  GrammarHandle grammar = 0;             ///< Handle into BranchStore's grammar registry

  CachedSamplingParams cached_params;    ///< Params used to create current chain (for memoization)

  boundaries::BoundaryTracker* boundary_tracker = nullptr;  ///< Token boundary detector (owned, optional)

  std::vector<llama_logit_bias> logit_bias;  ///< Static token biases, cloned on fork
  std::function<void(llama_token_data_array&)> steer_fn;  ///< Dynamic logit callback, NOT cloned on fork

  MetricsHandle metrics = 0;  ///< Handle into BranchStore's metrics registry

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
  /// @todo Move to per-thread or SessionContext scratch arena for deep search trees.
  std::vector<llama_token_data> candidates_buffer;

  int n_batch = DEFAULT_N_BATCH;  ///< Batch size for decode operations
  int n_vocab = 0;    ///< Vocabulary size (cached for buffer pre-allocation)

  uint16_t generation = 0;  ///< Slot generation counter (for ABA prevention)
  bool in_use = false;      ///< True when slot is allocated to an active branch

  // Topology — maintained by fork/prune/pruneSubtree
  BranchHandle parent = INVALID_HANDLE;     ///< Parent branch (INVALID_HANDLE if root)
  std::vector<BranchHandle> children;       ///< Child branches forked from this one
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
 * - `tokens` is a non-owning view (pointer + length). If `size() > 0`,
 *   `data()` must point to valid, dereferenceable memory.
 *
 * @warning Caller must keep the pointed-to token data alive until
 *          decode_scatter() returns. Do not pass spans of temporaries.
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
 *          sequence ID (llama_seq_id) managed by kv::tenancy. Call
 *          init_tenancy(ctx) after context creation to set the ceiling.
 *          The hard limit is `llama_n_seq_max(ctx)` (typically 256).
 *          allocate() acquires both a slot and a lease atomically;
 *          release()/drain() return both resources symmetrically.
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

  /// @brief Destructor — frees CPU resources. drain() must be called first
  /// while the llama_context is still alive.
  ~BranchStore() {
    if (tenancy_.ctx != nullptr) {
      LLOYAL_LOG_DEBUG("[BranchStore] WARNING: not drained before destruction");
    }
    for (size_t i = 1; i < slots_.size(); ++i) {
      if (slots_[i].in_use) {
        free_branch_resources(slots_[i]);
      }
    }
  }

  /// Result of allocate(): a slot handle + its leased seq_id.
  struct Allocation { BranchHandle handle; llama_seq_id seq_id; };

  /**
   * @brief Allocate a branch slot + KV lease atomically
   *
   * Acquires a seq_id from tenancy, then a slot from the freelist.
   * If either fails, both are rolled back cleanly.
   *
   * @return {handle, seq_id}, or {INVALID_HANDLE, -1} if exhausted
   */
  Allocation allocate() {
    if (tenancy_.ctx == nullptr) return {INVALID_HANDLE, NO_LEASE};  // drained
    llama_seq_id seq = kv::tenancy::acquire(tenancy_);
    if (seq < 0) return {INVALID_HANDLE, NO_LEASE};
    BranchHandle handle = allocate_slot();
    if (handle == INVALID_HANDLE) {
      // Seq was never used — bookkeeping-only return, no KV calls
      kv::tenancy::release(tenancy_, seq);
      return {INVALID_HANDLE, NO_LEASE};
    }
    // Stamp seq_id on slot so release() can evict properly
    BranchState* st = get(handle);
    st->seq_id = seq;
    return {handle, seq};
  }

  /**
   * @brief Release a branch slot + evict its KV lease
   *
   * Removes parent→child edge, evicts the seq_id (stripping KV tags),
   * frees CPU resources, and returns the slot to the freelist.
   *
   * @param handle Branch handle to release (INVALID_HANDLE is a safe no-op)
   */
  void release(BranchHandle handle) {
    if (handle == INVALID_HANDLE) return;
    BranchState* st = get(handle);
    if (!st) return;
    // Eager edge cleanup: remove from parent's children
    if (st->parent != INVALID_HANDLE) {
      BranchState* p = get(st->parent);
      if (p) {
        auto& c = p->children;
        c.erase(std::remove(c.begin(), c.end(), handle), c.end());
      }
    }
    // Evict lease (KV strip + bookkeeping)
    if (st->seq_id != NO_LEASE)
      kv::tenancy::evict(tenancy_, st->seq_id);
    free_branch_resources(*st);
    reset_slot(*st);
    freelist_.push_back(handle_index(handle));
  }

  // ===== TENANCY LIFECYCLE =====

  /**
   * @brief Initialize KV tenancy after context creation
   * @param ctx Llama context (must outlive BranchStore or call drain() first)
   */
  void init_tenancy(llama_context* ctx) {
    tenancy_ = kv::tenancy::init(ctx, llama_n_seq_max(ctx));
  }

  /**
   * @brief Explicit teardown — evict all leases while context is alive
   *
   * Must be called before llama_free(ctx). Idempotent.
   * Terminal — BranchStore is not reusable after drain(). freelist_ is not
   * repopulated; call init_tenancy() on a fresh store if you need a new cycle.
   * After drain(), allocate() returns {INVALID_HANDLE, NO_LEASE}.
   */
  void drain() {
    if (tenancy_.ctx == nullptr) return;  // idempotent
    kv::tenancy::evict_all(tenancy_);
    for (size_t i = 1; i < slots_.size(); ++i) {
      if (slots_[i].in_use) {
        free_branch_resources(slots_[i]);
        reset_slot(slots_[i]);
      }
    }
    tenancy_.ctx = nullptr;  // marks as drained
  }

  /**
   * @brief Keep only the winner — nuclear KV + CPU cleanup
   *
   * Calls seq_keep(winner_seq) for a single KV pass, then releases
   * all other slots (CPU only — KV already stripped by seq_keep).
   *
   * @param winner Handle to the branch to retain (must be valid + leased)
   * @throws std::runtime_error if winner is invalid or has no lease
   */
  void retainOnly(BranchHandle winner) {
    BranchState* w = get(winner);
    if (!w) throw std::runtime_error("retainOnly: invalid winner handle");
    if (w->seq_id == NO_LEASE) throw std::runtime_error("retainOnly: winner has no lease");
    kv::tenancy::retain(tenancy_, w->seq_id);  // nuclear KV pass
    // Collect losers first — don't mutate while iterating
    std::vector<BranchHandle> losers;
    for (size_t i = 1; i < slots_.size(); ++i) {
      if (!slots_[i].in_use) continue;
      BranchHandle h = make_handle(static_cast<uint16_t>(i), slots_[i].generation);
      if (h == winner) continue;
      losers.push_back(h);
    }
    for (auto h : losers)
      release_slot_only(h);  // CPU only, KV already stripped
    w->parent = INVALID_HANDLE;
    w->children.clear();
  }

  // ===== TOPOLOGY QUERIES =====

  /**
   * @brief Number of vacant seq_ids available for acquisition
   * @return Count of seq_ids in the tenancy vacant pool
   */
  size_t available() const { return kv::tenancy::available(tenancy_); }

  /**
   * @brief Get a branch's parent handle
   * @param h Branch handle
   * @return Parent handle, or INVALID_HANDLE if root or handle is invalid
   */
  BranchHandle parent(BranchHandle h) const {
    const BranchState* st = get(h);
    return st ? st->parent : INVALID_HANDLE;
  }

  /**
   * @brief Get a branch's child handles
   * @param h Branch handle
   * @return Reference to child handle vector (empty if leaf or invalid)
   */
  const std::vector<BranchHandle>& children(BranchHandle h) const {
    static const std::vector<BranchHandle> empty;
    const BranchState* st = get(h);
    return st ? st->children : empty;
  }

  /**
   * @brief Test whether a branch is a leaf (no children)
   * @param h Branch handle
   * @return true if branch has no children or handle is invalid
   */
  bool isLeaf(BranchHandle h) const {
    const BranchState* st = get(h);
    return st ? st->children.empty() : true;
  }

  /**
   * @brief Test whether a branch holds a KV lease
   * @param h Branch handle
   * @return true if seq_id != NO_LEASE, false if inactive or handle is invalid
   */
  bool isActive(BranchHandle h) const {
    const BranchState* st = get(h);
    return st ? (st->seq_id != NO_LEASE) : false;
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

  // ===== SAMPLER CHAIN REGISTRY =====

  /**
   * @brief Create a sampler chain and register it
   * @param params Sampling parameters (any SamplingParamsLike type)
   * @return Handle to the new sampler chain (never 0)
   */
  template <SamplingParamsLike P>
  SamplerChainHandle create_sampler(const P& params) {
    SamplerChainHandle h = next_sampler_handle_++;
    SamplerChainEntry entry;
    entry.chain = sampler::create_chain(params);
    float temperature = ::lloyal::detail::as_value(params.temperature, 0.8f);
    entry.has_dist = (temperature > 0.0f);
    sampler_chains_.emplace(h, std::move(entry));
    return h;
  }

  /**
   * @brief Clone a sampler chain (for fork)
   * @param h Source sampler chain handle
   * @return New handle with cloned chain, or 0 if source is invalid
   */
  SamplerChainHandle clone_sampler(SamplerChainHandle h) {
    if (h == 0) return 0;
    auto it = sampler_chains_.find(h);
    if (it == sampler_chains_.end()) return 0;
    SamplerChainHandle nh = next_sampler_handle_++;
    SamplerChainEntry entry;
    entry.chain = sampler::clone_chain(it->second.chain);
    entry.has_dist = it->second.has_dist;
    sampler_chains_.emplace(nh, std::move(entry));
    return nh;
  }

  /**
   * @brief Free a sampler chain
   * @param h Handle to free (0 is a safe no-op)
   */
  void free_sampler(SamplerChainHandle h) {
    if (h != 0) sampler_chains_.erase(h);
  }

  /**
   * @brief Dereference a sampler chain handle (non-owning)
   * @param h Sampler chain handle
   * @return Pointer to the chain, or nullptr if invalid
   */
  llama_sampler* get_sampler_chain(SamplerChainHandle h) const {
    if (h == 0) return nullptr;
    auto it = sampler_chains_.find(h);
    return it != sampler_chains_.end() ? it->second.chain : nullptr;
  }

  /**
   * @brief Check if a sampler chain ends with dist (stochastic) or greedy
   * @param h Sampler chain handle
   * @return true if chain has dist sampler, false if greedy or invalid
   */
  bool sampler_has_dist(SamplerChainHandle h) const {
    if (h == 0) return false;
    auto it = sampler_chains_.find(h);
    return it != sampler_chains_.end() ? it->second.has_dist : false;
  }

  // ===== GRAMMAR REGISTRY =====

  /**
   * @brief Create a grammar sampler and register it
   * @param model Llama model (for vocab)
   * @param grammar_str GBNF grammar string
   * @param root Root rule name (default "root")
   * @return Handle to the new grammar (never 0)
   */
  GrammarHandle create_grammar(const llama_model* model,
                               const char* grammar_str,
                               const char* root = "root") {
    GrammarHandle h = next_grammar_handle_++;
    GrammarEntry entry;
    entry.sampler = grammar::init_sampler(model, grammar_str, root);
    grammars_.emplace(h, std::move(entry));
    return h;
  }

  /**
   * @brief Clone a grammar (for fork)
   * @param h Source grammar handle
   * @return New handle with cloned grammar, or 0 if source is invalid
   */
  GrammarHandle clone_grammar(GrammarHandle h) {
    if (h == 0) return 0;
    auto it = grammars_.find(h);
    if (it == grammars_.end()) return 0;
    GrammarHandle nh = next_grammar_handle_++;
    GrammarEntry entry;
    entry.sampler = grammar::clone_sampler(it->second.sampler);
    grammars_.emplace(nh, std::move(entry));
    return nh;
  }

  /**
   * @brief Free a grammar
   * @param h Handle to free (0 is a safe no-op)
   */
  void free_grammar(GrammarHandle h) {
    if (h != 0) grammars_.erase(h);
  }

  /**
   * @brief Dereference a grammar handle (non-owning)
   * @param h Grammar handle
   * @return Pointer to the grammar sampler, or nullptr if invalid
   */
  llama_sampler* get_grammar_sampler(GrammarHandle h) const {
    if (h == 0) return nullptr;
    auto it = grammars_.find(h);
    return it != grammars_.end() ? it->second.sampler : nullptr;
  }

  // ===== METRICS REGISTRY =====

  /**
   * @brief Create a metrics tracker and register it
   * @return Handle to the new tracker (never 0)
   */
  MetricsHandle create_metrics() {
    MetricsHandle h = next_metrics_handle_++;
    metrics_registry_[h] = metrics::BranchMetricsState{};
    return h;
  }

  /**
   * @brief Clone a metrics tracker (for fork)
   * @param h Source metrics handle
   * @return New handle with cloned state, or 0 if source is invalid
   */
  MetricsHandle clone_metrics(MetricsHandle h) {
    if (h == 0) return 0;
    auto it = metrics_registry_.find(h);
    if (it == metrics_registry_.end()) return 0;
    MetricsHandle nh = next_metrics_handle_++;
    metrics_registry_[nh] = it->second;
    return nh;
  }

  /**
   * @brief Free a metrics tracker
   * @param h Handle to free (0 is a safe no-op)
   */
  void free_metrics(MetricsHandle h) {
    if (h != 0) metrics_registry_.erase(h);
  }

  /**
   * @brief Add model-level surprisal to a metrics tracker
   * @param h Metrics handle
   * @param surprisal Surprisal in nats
   */
  void add_model_surprisal(MetricsHandle h, float surprisal) {
    if (h == 0) return;
    auto it = metrics_registry_.find(h);
    if (it == metrics_registry_.end()) return;
    if (!std::isfinite(surprisal)) return;
    it->second.model.nll_sum_nats += std::max(0.0f, surprisal);
    it->second.model.count++;
  }

  /**
   * @brief Add sampling-level surprisal to a metrics tracker
   * @param h Metrics handle
   * @param surprisal Surprisal in nats
   */
  void add_sampling_surprisal(MetricsHandle h, float surprisal) {
    if (h == 0) return;
    auto it = metrics_registry_.find(h);
    if (it == metrics_registry_.end()) return;
    if (!std::isfinite(surprisal)) return;
    it->second.sampling.nll_sum_nats += std::max(0.0f, surprisal);
    it->second.sampling.count++;
  }

  /**
   * @brief Get model-level perplexity from a metrics tracker
   * @param h Metrics handle
   * @return exp(average surprisal), INFINITY if no samples or invalid
   */
  float get_model_ppl(MetricsHandle h) const {
    if (h == 0) return std::numeric_limits<float>::infinity();
    auto it = metrics_registry_.find(h);
    if (it == metrics_registry_.end() || it->second.model.count == 0)
      return std::numeric_limits<float>::infinity();
    return std::exp(it->second.model.nll_sum_nats /
                    static_cast<float>(it->second.model.count));
  }

  /**
   * @brief Get sampling-level perplexity from a metrics tracker
   * @param h Metrics handle
   * @return exp(average surprisal), INFINITY if no samples or invalid
   */
  float get_sampling_ppl(MetricsHandle h) const {
    if (h == 0) return std::numeric_limits<float>::infinity();
    auto it = metrics_registry_.find(h);
    if (it == metrics_registry_.end() || it->second.sampling.count == 0)
      return std::numeric_limits<float>::infinity();
    return std::exp(it->second.sampling.nll_sum_nats /
                    static_cast<float>(it->second.sampling.count));
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

        // Clamp to context batch limit — branch n_batch may exceed it
        const int32_t safe_batch = std::min(states[idx]->n_batch, batch_limit);
        if (decode::many(ctx, items[idx].tokens.data(), tc,
                         states[idx]->position, safe_batch,
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
    if (slot.sampler_chain != 0) {
      free_sampler(slot.sampler_chain);
      slot.sampler_chain = 0;
    }
    if (slot.grammar != 0) {
      free_grammar(slot.grammar);
      slot.grammar = 0;
    }
    if (slot.boundary_tracker) {
      delete slot.boundary_tracker;
      slot.boundary_tracker = nullptr;
    }
    if (slot.metrics != 0) {
      free_metrics(slot.metrics);
      slot.metrics = 0;
    }
  }

  /// Reset slot to default state. Increments generation for ABA prevention.
  void reset_slot(BranchState& slot) {
    slot.in_use = false;
    slot.generation = static_cast<uint16_t>(slot.generation + 1);  // Prevent ABA
    slot.ctx = nullptr;
    slot.model = nullptr;
    slot.seq_id = NO_LEASE;
    slot.position = 0;
    slot.sampler_chain = 0;
    slot.grammar = 0;
    slot.metrics = 0;
    slot.cached_params = CachedSamplingParams{};
    slot.last_token = -1;
    slot.last_candidates.clear();
    slot.logits_snapshot.clear();
    slot.has_logits = false;
    slot.logit_bias.clear();
    slot.steer_fn = nullptr;
    slot.candidates_buffer.clear();
    slot.n_batch = DEFAULT_N_BATCH;
    slot.n_vocab = 0;
    slot.parent = INVALID_HANDLE;
    slot.children.clear();
  }

  /// Allocate a slot from the freelist (no tenancy). Auto-grows by doubling.
  BranchHandle allocate_slot() {
    if (freelist_.empty()) {
      size_t old_size = slots_.size();
      size_t new_size = old_size * 2;
      if (new_size > INDEX_MASK) {
        new_size = INDEX_MASK + 1;
      }
      if (old_size >= new_size) {
        LLOYAL_LOG_DEBUG("[branch::allocate_slot] Store full, cannot allocate");
        return INVALID_HANDLE;
      }
      slots_.resize(new_size);
      for (size_t i = new_size; i-- > old_size; ) {
        freelist_.push_back(static_cast<uint16_t>(i));
      }
    }
    uint16_t index = freelist_.back();
    freelist_.pop_back();
    BranchState& slot = slots_[index];
    slot.in_use = true;
    return make_handle(index, slot.generation);
  }

  /// Release slot (CPU resources only, no KV calls). Used by retainOnly()
  /// after seq_keep has already stripped all KV tags.
  void release_slot_only(BranchHandle handle) {
    if (handle == INVALID_HANDLE) return;
    BranchState* st = get(handle);
    if (!st) return;
    // Eager edge cleanup
    if (st->parent != INVALID_HANDLE) {
      BranchState* p = get(st->parent);
      if (p) {
        auto& c = p->children;
        c.erase(std::remove(c.begin(), c.end(), handle), c.end());
      }
    }
    free_branch_resources(*st);
    reset_slot(*st);
    freelist_.push_back(handle_index(handle));
  }

  /// Slot array. Uses std::deque (not std::vector) for pointer stability —
  /// get() returns BranchState* that remain valid across allocate()/grow().
  std::deque<BranchState> slots_;
  std::vector<uint16_t> freelist_;  ///< Available slot indices (LIFO — locality heuristic)

  /// KV lease manager. Initialized by init_tenancy(), drained by drain().
  /// seq_id ownership invariant: only the BranchState that received a seq_id
  /// from allocate() may hold it, and only while its handle+generation is live.
  kv::tenancy::State tenancy_;

  /// Reusable scratch buffers for batched decode. Safe without locking because
  /// BranchStore requires external synchronization (caller's mutex).
  decode::Scratch scratch_;

  // ===== Handle registries (instance-scoped, not global static) =====

  std::unordered_map<SamplerChainHandle, SamplerChainEntry> sampler_chains_;
  SamplerChainHandle next_sampler_handle_ = 1;

  std::unordered_map<GrammarHandle, GrammarEntry> grammars_;
  GrammarHandle next_grammar_handle_ = 1;

  std::unordered_map<MetricsHandle, metrics::BranchMetricsState> metrics_registry_;
  MetricsHandle next_metrics_handle_ = 1;
};


// ===== BRANCH API =====

/**
 * @brief Create a new branch with sampler chain, optional grammar, and metrics
 *
 * Allocates a slot + KV lease from the store, initializes the sampler chain
 * from @p params, optionally attaches a GBNF grammar and boundary tracker,
 * and pre-allocates logits/candidates buffers sized to the model's vocabulary.
 *
 * @tparam P Any type satisfying the SamplingParamsLike concept
 * @param ctx Llama context (not owned, must outlive branch)
 * @param model Llama model (not owned, used for vocab size and sampler init)
 * @param s Branch store to allocate from
 * @param start_pos Starting decode position (typically prompt length after prefill)
 * @param params Sampling parameters (temperature, top_k, top_p, penalties, etc.)
 * @param n_batch Batch size for decode operations (default 512)
 * @param grammar_str GBNF grammar string, or nullptr for unconstrained generation
 * @param boundary_tracker Boundary detector (ownership transferred), or nullptr
 * @return Valid BranchHandle, or INVALID_HANDLE on failure
 *
 * @see prune() to free with KV cleanup, pruneSubtree() for CASCADE
 */
template <SamplingParamsLike P>
inline BranchHandle create(
    llama_context* ctx,
    const llama_model* model,
    BranchStore& s,
    llama_pos start_pos,
    const P& params,
    int n_batch = DEFAULT_N_BATCH,
    const char* grammar_str = nullptr,
    boundaries::BoundaryTracker* boundary_tracker = nullptr) {
  if (!ctx || !model) {
    LLOYAL_LOG_DEBUG("[branch::create] NULL ctx or model");
    return INVALID_HANDLE;
  }

  auto [handle, seq_id] = s.allocate();
  if (handle == INVALID_HANDLE) {
    return INVALID_HANDLE;
  }

  BranchState* state = s.get(handle);
  if (!state) {
    s.release(handle);
    return INVALID_HANDLE;
  }

  state->ctx = ctx;
  state->model = model;
  // seq_id already stamped by allocate()
  state->position = start_pos;
  state->n_batch = n_batch;

  const llama_vocab* vocab = llama_model_get_vocab(model);
  state->n_vocab = llama_vocab_n_tokens(vocab);
  state->logits_snapshot.resize(state->n_vocab);
  state->has_logits = false;
  state->candidates_buffer.resize(state->n_vocab);

  state->sampler_chain = s.create_sampler(params);
  state->cached_params = snapshot_params(params);

  if (grammar_str && grammar_str[0] != '\0') {
    state->grammar = s.create_grammar(model, grammar_str);
  }

  state->boundary_tracker = boundary_tracker;
  state->metrics = s.create_metrics();

  LLOYAL_LOG_DEBUG("[branch::create] Created branch handle=%u seq=%d pos=%d",
                   handle, seq_id, start_pos);

  return handle;
}


/**
 * @brief Fork a branch into a new independent sequence
 *
 * Allocates a slot + KV lease, deep copies source state under the new seq_id.
 * Records parent→child topology edge.
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
 * @param s Branch store
 * @return Handle to the new child branch, or INVALID_HANDLE on failure
 */
inline BranchHandle fork(BranchHandle source, BranchStore& s) {
  BranchState* src = s.get(source);
  if (!src) {
    LLOYAL_LOG_DEBUG("[branch::fork] Invalid source handle");
    return INVALID_HANDLE;
  }

  auto [new_handle, new_seq_id] = s.allocate();
  if (new_handle == INVALID_HANDLE) {
    return INVALID_HANDLE;
  }

  BranchState* dst = s.get(new_handle);
  if (!dst) {
    s.release(new_handle);
    return INVALID_HANDLE;
  }

  // Copy basic state
  dst->ctx = src->ctx;
  dst->model = src->model;
  dst->seq_id = new_seq_id;
  dst->position = src->position;
  dst->n_batch = src->n_batch;
  dst->n_vocab = src->n_vocab;

#ifndef NDEBUG
  assert(kv::pos_max(src->ctx, new_seq_id) < 0 && "tenancy: acquired seq must be clean");
  assert(dst->parent == INVALID_HANDLE && dst->children.empty() && "fresh slot must have no topology");
#endif

  // Fork KV cache
  kv::seq_cp(src->ctx, src->seq_id, new_seq_id);

  // Record topology
  dst->parent = source;
  src->children.push_back(new_handle);

  // Clone sampler chain
  if (src->sampler_chain != 0) {
    dst->sampler_chain = s.clone_sampler(src->sampler_chain);
  }
  dst->cached_params = src->cached_params;

  if (src->grammar != 0) {
    dst->grammar = s.clone_grammar(src->grammar);
  }

  if (src->boundary_tracker) {
    dst->boundary_tracker = src->boundary_tracker->clone().release();
  }

  if (src->metrics != 0) {
    dst->metrics = s.clone_metrics(src->metrics);
  }

  dst->last_token = src->last_token;
  dst->last_candidates = src->last_candidates;

  // logits_snapshot copy is intentional: fork's contract is "sample different
  // tokens from the same logit distribution." Without the copy, the child can't
  // sample without a redundant decode. Cost: n_vocab * 4 bytes (~512KB at 128k vocab).
  dst->logits_snapshot = src->logits_snapshot;
  dst->has_logits = src->has_logits;
  dst->logit_bias = src->logit_bias;

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
 * @param s Branch store
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
    BranchStore& s) {


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
 * @param s Branch store
 * @throws std::runtime_error if handle is invalid
 */
inline void clear_logit_bias(
    BranchHandle handle,
    BranchStore& s) {


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
 * @param s Branch store
 * @throws std::runtime_error if handle is invalid
 *
 * @warning Callback may capture references — ensure their lifetime exceeds branch usage.
 *
 * @example Action deduplication in tree search
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
    BranchStore& s) {


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
 * @param s Branch store
 * @throws std::runtime_error if handle is invalid
 */
inline void clear_steer(
    BranchHandle handle,
    BranchStore& s) {


  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("clear_steer: invalid branch handle");
  }

  state->steer_fn = nullptr;

  LLOYAL_LOG_DEBUG("[branch::clear_steer] Cleared steer callback on handle=%u", handle);
}

/**
 * @brief Replace a branch's sampler chain with new parameters
 *
 * Memoized: if the new params match the cached snapshot, this is a no-op.
 * Otherwise frees the old chain and creates a new one.
 *
 * Primary use case: Entropy-based Dynamic Temperature (EDT), where temperature
 * changes per-token based on model uncertainty.
 *
 * @tparam P Any type satisfying the SamplingParamsLike concept
 * @param handle Branch to modify
 * @param params New sampling parameters
 * @param s Branch store
 * @throws std::runtime_error if handle is invalid
 */
template <SamplingParamsLike P>
inline void set_sampler_params(BranchHandle handle, const P& params, BranchStore& s) {
  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("set_sampler_params: invalid branch handle");
  }

  CachedSamplingParams new_params = snapshot_params(params);
  if (new_params == state->cached_params && state->sampler_chain != 0) {
    return;  // Memoized — no rebuild needed
  }

  // Free old chain
  if (state->sampler_chain != 0) {
    s.free_sampler(state->sampler_chain);
  }

  // Create new chain
  state->sampler_chain = s.create_sampler(params);
  state->cached_params = new_params;

  LLOYAL_LOG_DEBUG("[branch::set_sampler_params] Rebuilt chain on handle=%u temp=%.3f",
                   handle, new_params.temperature);
}

/**
 * @brief Replace a branch's grammar constraint
 *
 * Frees the old grammar (if any) and attaches a new one. Pass nullptr or
 * empty string to remove grammar constraints entirely.
 *
 * @param handle Branch to modify
 * @param model Llama model (for vocab)
 * @param grammar_str GBNF grammar string, or nullptr to remove
 * @param s Branch store
 * @throws std::runtime_error if handle is invalid
 */
inline void set_grammar(
    BranchHandle handle,
    const llama_model* model,
    const char* grammar_str,
    BranchStore& s) {
  BranchState* state = s.get(handle);
  if (!state) {
    throw std::runtime_error("set_grammar: invalid branch handle");
  }

  // Free old grammar
  if (state->grammar != 0) {
    s.free_grammar(state->grammar);
    state->grammar = 0;
  }

  // Create new grammar if provided
  if (grammar_str && grammar_str[0] != '\0') {
    state->grammar = s.create_grammar(model, grammar_str);
  }

  LLOYAL_LOG_DEBUG("[branch::set_grammar] %s grammar on handle=%u",
                   state->grammar != 0 ? "Set" : "Cleared", handle);
}

/**
 * @brief Prune a leaf branch (RESTRICT — throws if children exist)
 *
 * Evicts the KV lease and frees all resources via BranchStore::release().
 * If the branch has children, throws — use pruneSubtree() for CASCADE.
 *
 * @param handle Branch to prune (INVALID_HANDLE is a safe no-op)
 * @param s Branch store
 * @throws std::runtime_error if branch has children
 */
inline void prune(BranchHandle handle, BranchStore& s) {
  BranchState* state = s.get(handle);
  if (!state) return;
  if (!state->children.empty())
    throw std::runtime_error("prune: RESTRICT — branch has children. Use pruneSubtree() for CASCADE.");
  s.release(handle);
}

/**
 * @brief Prune a branch and all descendants (CASCADE — iterative post-order)
 *
 * Traverses the subtree rooted at h, collecting all descendants, then prunes
 * leaves-first so RESTRICT on prune() always passes.
 *
 * @param h Root of subtree to prune
 * @param s Branch store
 */
inline void pruneSubtree(BranchHandle h, BranchStore& s) {
  std::vector<BranchHandle> stack{h}, post_order;
  while (!stack.empty()) {
    BranchHandle cur = stack.back(); stack.pop_back();
    post_order.push_back(cur);
    BranchState* st = s.get(cur);
    if (st) for (auto child : st->children) stack.push_back(child);
  }
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it)
    prune(*it, s);
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
 * @param s Branch store
 * @throws std::runtime_error if handle is invalid or decode fails
 *
 * @note For single-token decode, prefer decode_one() (zero heap allocation).
 */
inline void decode_batch(
    BranchHandle handle,
    const llama_token* tokens,
    size_t n_tokens,
    BranchStore& s) {


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
 * @param s Branch store
 * @throws std::runtime_error if handle is invalid or decode fails
 */
inline void decode_one(
    BranchHandle handle,
    llama_token token,
    BranchStore& s) {


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
 * @param s Branch store
 * @throws std::runtime_error if handle is invalid, vocab size is zero,
 *         or no logits are available (no prior decode with logits enabled)
 *
 * @note Sets has_logits = true, enabling sample() and get_logits().
 */
inline void capture_logits(BranchHandle handle, BranchStore& s) {


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
 * @param s Branch store
 * @throws std::runtime_error if handle is invalid, decode fails,
 *         or logits capture fails
 *
 * @note For single-token decode, prefer decode_and_capture_one() (zero heap allocation).
 */
inline void decode_and_capture_batch(
    BranchHandle handle,
    const llama_token* tokens,
    size_t n_tokens,
    BranchStore& s) {


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
 * @param s Branch store
 * @throws std::runtime_error if handle is invalid, decode fails,
 *         or logits capture fails
 */
inline void decode_and_capture_one(
    BranchHandle handle,
    llama_token token,
    BranchStore& s) {


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
 * @param s Branch store
 * @return Pointer to n_vocab floats, or nullptr if no logits captured
 */
inline const float* get_logits(BranchHandle handle, BranchStore& s) {


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
 * @param s Branch store
 * @return Sampled token ID, or -1 if no logits captured or sampling fails
 *
 * @note Call accept_token() after sampling to advance grammar and penalty state.
 */
inline llama_token sample(BranchHandle handle, BranchStore& s) {


  BranchState* state = s.get(handle);
  llama_sampler* chain = state ? s.get_sampler_chain(state->sampler_chain) : nullptr;
  if (!state || !chain) {
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
  llama_sampler* gram = s.get_grammar_sampler(state->grammar);
  if (gram) {
    grammar::apply(gram, &cur_p);
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
  sampler::apply(chain, &cur_p);

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
 * @param s Branch store
 *
 * @note Safe to call with invalid handle (silent no-op).
 */
inline void accept_token(
    BranchHandle handle,
    llama_token token,
    BranchStore& s) {


  BranchState* state = s.get(handle);
  if (!state) return;

  // Accept in grammar (via anti-corruption layer)
  llama_sampler* gram = s.get_grammar_sampler(state->grammar);
  if (gram) {
    grammar::accept(gram, token);
  }

  // Accept in sampler chain for penalty tracking (via anti-corruption layer)
  llama_sampler* chain = s.get_sampler_chain(state->sampler_chain);
  if (chain) {
    sampler::accept(chain, token);
  }

  // Update model-level perplexity (from raw logits)
  // Guard on has_logits to avoid computing surprisal from zero-filled buffer
  if (state->metrics != 0 && state->has_logits) {
    float ms = metrics::model_surprisal(
        state->logits_snapshot.data(), state->n_vocab, token);
    s.add_model_surprisal(state->metrics, ms);
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
    float ss = metrics::sampling_surprisal(
        candidate_logits.data(),
        candidate_ids.data(),
        static_cast<int>(candidate_logits.size()),
        token
    );
    s.add_sampling_surprisal(state->metrics, ss);
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
 * @param s Branch store
 *
 * @note No-op if handle is invalid or branch has no grammar attached.
 */
inline void apply_grammar(
    BranchHandle handle,
    float* logits,
    int n_vocab,
    BranchStore& s) {


  BranchState* state = s.get(handle);
  llama_sampler* gram = state ? s.get_grammar_sampler(state->grammar) : nullptr;
  if (!state || !gram) return;

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

  grammar::apply(gram, &cur_p);

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
 * Essential for policy priors in tree search: priors must only cover legal moves.
 *
 * @param handle Branch with captured logits and optional grammar
 * @param s Branch store
 * @return Vector of (token_id, probability) pairs, empty if no logits or no legal tokens
 *
 * @note If no grammar is attached, all tokens with finite logits are included.
 */
inline std::vector<std::pair<llama_token, float>> get_legal_priors(
    BranchHandle handle,
    BranchStore& s) {


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
  llama_sampler* gram = s.get_grammar_sampler(state->grammar);
  if (gram) {
    grammar::apply(gram, &cur_p);
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
 * @param s Branch store
 * @return Log-sum-exp value, or -INFINITY if no legal tokens or invalid state
 *
 * @see get_token_prior_assume_legal() for O(1) per-token prior using this value
 */
inline float get_legal_logsumexp(BranchHandle handle, BranchStore& s) {


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
  llama_sampler* gram = s.get_grammar_sampler(state->grammar);
  if (gram) {
    grammar::apply(gram, &cur_p);
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
 * @param s Branch store
 * @return true if token is legal (or no grammar attached), false if illegal
 */
inline bool is_token_legal(
    BranchHandle handle,
    llama_token token,
    BranchStore& s) {


  BranchState* state = s.get(handle);
  if (!state || token < 0 || token >= state->n_vocab) {
    return false;
  }

  // No grammar = all tokens legal
  llama_sampler* gram = s.get_grammar_sampler(state->grammar);
  if (!gram) {
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
  grammar::apply(gram, &cur_p);

  return std::isfinite(single_candidate.logit);
}

/**
 * @brief Compute prior probability for a token known to be grammar-legal
 *
 * O(1) operation — use in search inner loops where sample() already enforced grammar.
 * Does NOT validate grammar legality; caller must ensure token is legal.
 *
 * @param handle Branch with captured logits
 * @param token Token ID (must be legal under grammar)
 * @param logsumexp Pre-computed value from get_legal_logsumexp()
 * @param s Branch store
 * @return Probability in [0, 1], or 0 if state is invalid
 *
 * @see get_token_prior() for a safe version that checks grammar legality
 */
inline float get_token_prior_assume_legal(
    BranchHandle handle,
    llama_token token,
    float logsumexp,
    BranchStore& s) {


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
 * @param s Branch store
 * @return Probability in [0, 1], or 0 if token is illegal
 *
 * @note For search inner loops, prefer get_token_prior_assume_legal() since
 *       sample() already enforces grammar constraints.
 */
inline float get_token_prior(
    BranchHandle handle,
    llama_token token,
    float logsumexp,
    BranchStore& s) {
  if (!is_token_legal(handle, token, s)) {
    return 0.0f;
  }
  return get_token_prior_assume_legal(handle, token, logsumexp, s);
}

// ===== STATE ACCESSORS =====


/**
 * @brief Get the branch's current decode position
 * @param handle Branch handle
 * @param s Branch store
 * @return Token position, or -1 if handle is invalid
 */
inline llama_pos get_position(BranchHandle handle, BranchStore& s) {

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
 * @param s Branch store
 * @return Model perplexity, or INFINITY if no tokens accepted
 */
inline float get_perplexity(BranchHandle handle, BranchStore& s) {

  const BranchState* state = s.get(handle);
  if (!state || state->metrics == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return s.get_model_ppl(state->metrics);
}

/**
 * @brief Get sampling-level perplexity (from filtered distribution)
 *
 * Returns perplexity from the distribution actually sampled from
 * (after top-k/p/temp/penalties). Useful for policy priors and
 * monitoring sampler chain impact.
 *
 * @param handle Branch handle
 * @param s Branch store
 * @return Sampling-level perplexity, or INFINITY if no tokens accepted
 */
inline float get_sampling_perplexity(BranchHandle handle, BranchStore& s) {

  const BranchState* state = s.get(handle);
  if (!state || state->metrics == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return s.get_sampling_ppl(state->metrics);
}

/**
 * @brief Get the last sampled token's prior from the filtered distribution
 *
 * Returns P(token) from the post-filter sampling distribution.
 * This is the correct prior for UCT-family algorithms since it matches what was actually sampled.
 *
 * @param handle Branch handle
 * @param s Branch store
 * @return Probability of last sampled token in [0, 1], or 0 if unavailable
 */
inline float get_last_sampling_prior(BranchHandle handle, BranchStore& s) {

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
 * @param s Branch store
 * @return Vocabulary size, or 0 if handle is invalid
 */
inline int get_n_vocab(BranchHandle handle, BranchStore& s) {

  const BranchState* state = s.get(handle);
  return state ? state->n_vocab : 0;
}


// ===== RAII WRAPPER =====

/**
 * @brief RAII wrapper around BranchHandle for automatic resource management
 *
 * Move-only value type. Destructor calls pruneSubtree() (CASCADE).
 * Use prune() for RESTRICT (leaf-only) cleanup.
 *
 * @example Basic usage
 * @code
 *   store.init_tenancy(ctx);
 *   Branch root = Branch::create(ctx, model, store, pos, params);
 *   Branch child = root.fork();
 *
 *   child.decode_and_capture_one(token);
 *   auto next = child.sample();
 *   child.accept(next);
 * @endcode
 */
class Branch {
public:
  Branch() : store_(nullptr), handle_(INVALID_HANDLE) {}

  Branch(BranchStore* store, BranchHandle handle)
      : store_(store), handle_(handle) {}

  /// Destructor — CASCADE prunes entire subtree
  ~Branch() {
    if (handle_ != INVALID_HANDLE && store_) {
      branch::pruneSubtree(handle_, *store_);
    }
  }

  Branch(Branch&& other) noexcept
      : store_(other.store_), handle_(other.handle_) {
    other.handle_ = INVALID_HANDLE;
  }

  Branch& operator=(Branch&& other) noexcept {
    if (this != &other) {
      if (handle_ != INVALID_HANDLE && store_) {
        branch::pruneSubtree(handle_, *store_);
      }
      store_ = other.store_;
      handle_ = other.handle_;
      other.handle_ = INVALID_HANDLE;
    }
    return *this;
  }

  Branch(const Branch&) = delete;
  Branch& operator=(const Branch&) = delete;

  /// Factory: allocates slot + lease from store
  template <SamplingParamsLike P>
  static Branch create(
      llama_context* ctx,
      const llama_model* model,
      BranchStore& store,
      llama_pos start_pos,
      const P& params,
      int n_batch = DEFAULT_N_BATCH,
      const char* grammar_str = nullptr,
      boundaries::BoundaryTracker* boundary_tracker = nullptr) {
    BranchHandle h = branch::create(ctx, model, store, start_pos, params, n_batch, grammar_str, boundary_tracker);
    return Branch(&store, h);
  }

  /// Fork: allocates slot + lease, records topology edge
  Branch fork() {
    BranchHandle h = branch::fork(handle_, *store_);
    return Branch(store_, h);
  }

  /// RESTRICT prune (throws if children exist)
  void prune() {
    branch::prune(handle_, *store_);
    handle_ = INVALID_HANDLE;
  }

  /// CASCADE prune — removes entire subtree
  void pruneSubtree() {
    branch::pruneSubtree(handle_, *store_);
    handle_ = INVALID_HANDLE;
  }

  /// @brief Capture current context logits into this branch's snapshot
  /// @copydetails branch::capture_logits()
  void capture_logits() {
    branch::capture_logits(handle_, *store_);
  }

  /// @brief Decode multiple tokens into this branch's KV cache
  /// @param tokens Array of token IDs
  /// @param n Number of tokens
  /// @copydetails branch::decode_batch()
  void decode_batch(const llama_token* tokens, size_t n) {
    branch::decode_batch(handle_, tokens, n, *store_);
  }

  /// @brief Decode a single token (zero per-call allocation)
  /// @param token Token ID to decode
  void decode_one(llama_token token) {
    branch::decode_one(handle_, token, *store_);
  }

  /// @brief Decode multiple tokens and capture logits atomically
  /// @param tokens Array of token IDs
  /// @param n Number of tokens
  void decode_and_capture_batch(const llama_token* tokens, size_t n) {
    branch::decode_and_capture_batch(handle_, tokens, n, *store_);
  }

  /// @brief Decode one token and capture logits (zero per-call allocation)
  /// @param token Token ID to decode
  void decode_and_capture_one(llama_token token) {
    branch::decode_and_capture_one(handle_, token, *store_);
  }

  /// @brief Get the branch's captured logits snapshot
  /// @return Pointer to n_vocab floats, or nullptr if no logits captured
  const float* logits() const {
    return branch::get_logits(handle_, *store_);
  }

  /// @brief Sample a token from captured logits
  /// @return Sampled token ID, or -1 if no logits captured
  /// @see accept() to advance state after sampling
  llama_token sample() {
    return branch::sample(handle_, *store_);
  }

  /// @brief Accept a token — advance grammar, penalty window, and metrics
  /// @param token Token to accept (from sample())
  void accept(llama_token token) {
    branch::accept_token(handle_, token, *store_);
  }

  /// @brief Check if a token is end-of-generation for this branch's model
  bool is_eog(llama_token token) const {
    const BranchState* st = store_ ? store_->get(handle_) : nullptr;
    return st && st->model ? tokenizer::is_eog(st->model, token) : false;
  }

  /// @brief Replace sampler chain with new parameters (memoized)
  /// @see branch::set_sampler_params()
  template <SamplingParamsLike P>
  void setSamplerParams(const P& params) {
    branch::set_sampler_params(handle_, params, *store_);
  }

  /// @brief Replace grammar constraint (nullptr/empty to remove)
  /// @see branch::set_grammar()
  void setGrammar(const char* grammar_str) {
    const BranchState* st = store_ ? store_->get(handle_) : nullptr;
    branch::set_grammar(handle_, st ? st->model : nullptr, grammar_str, *store_);
  }

  // ===== ACCESSORS =====

  /// @brief Current decode position (token count)
  llama_pos position() const { return branch::get_position(handle_, *store_); }
  /// @brief Model-level perplexity (from raw logits, pre-filter)
  float perplexity() const { return branch::get_perplexity(handle_, *store_); }
  /// @brief Vocabulary size
  int n_vocab() const { return branch::get_n_vocab(handle_, *store_); }
  /// @brief True if this Branch holds a valid handle
  bool valid() const { return handle_ != INVALID_HANDLE; }
  /// @brief Underlying opaque handle (for interop with free functions)
  BranchHandle handle() const { return handle_; }

  // ===== TOPOLOGY =====

  /// @brief Parent branch handle, or INVALID_HANDLE if root
  BranchHandle parentHandle() const { return store_ ? store_->parent(handle_) : INVALID_HANDLE; }
  /// @brief Child branch handles (empty if leaf)
  const std::vector<BranchHandle>& childHandles() const {
    static const std::vector<BranchHandle> empty;
    return store_ ? store_->children(handle_) : empty;
  }
  /// @brief True if this branch has no children
  bool isLeaf() const { return store_ ? store_->isLeaf(handle_) : true; }
  /// @brief True if this branch holds a KV lease
  bool isActive() const { return store_ ? store_->isActive(handle_) : false; }

private:
  BranchStore* store_;
  BranchHandle handle_;
};

}  // namespace lloyal::branch
