#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

#include "common.hpp"
#include <common.h>  // llama.cpp common library: common_batch_clear, common_batch_add
#include <algorithm>
#include <cstdint>
#include <llama/llama.h>
#include <stdexcept>
#include <vector>

/**
 * LLOYAL_STACK_BATCH - Controls llama_batch construction strategy
 *
 * When 1 (default): Use zero-allocation stack-constructed batch in decode::one()
 *   - Fastest: no heap allocation per decode
 *   - Risk: breaks if llama_batch struct layout changes
 *
 * When 0: Use thread_local batch via llama_batch_init()
 *   - Slightly slower: one-time init per thread
 *   - Safe: uses llama.cpp's own initializer, handles new fields
 *
 * If build breaks after llama.cpp update due to llama_batch changes:
 *   1. Set LLOYAL_STACK_BATCH=0 to unblock immediately
 *   2. Update decode::one() to match new struct layout
 *   3. Update ABI stability test assertions
 *   4. Re-enable LLOYAL_STACK_BATCH=1
 */
#ifndef LLOYAL_STACK_BATCH
#define LLOYAL_STACK_BATCH 1
#endif

/**
 * @file decode.hpp
 * @brief Batch Decoding Operations
 *
 * Wraps llama.cpp decode APIs with batch management, chunking logic, and
 * orchestration primitives. Provides both batched and single-token decode operations.
 *
 * API naming follows this grid:
 *
 *                     Single Sequence       Multi Sequence
 *                    ┌─────────────────┬─────────────────┐
 *    Single Token    │  decode::one    │  decode::each   │
 *                    ├─────────────────┼─────────────────┤
 *    Multi Token     │  decode::many   │  decode::scatter│
 *                    └─────────────────┴─────────────────┘
 *
 * Uses batch utilities from llama.cpp common (common_batch_clear, common_batch_add).
 */

namespace lloyal::decode {

/**
 * Process tokens through model to update KV cache
 *
 * Orchestration logic:
 * 1. Initializes batch with RAII cleanup
 * 2. Chunks tokens into n_batch-sized pieces
 * 3. For each chunk: clear batch, add tokens, call llama_decode
 * 4. Automatic batch cleanup via RAII guard
 *
 * ## Sequence ID Parameter
 *
 * The `seq_id` parameter specifies which KV cache sequence to update.
 * Default is 0 (single-sequence mode, backward compatible).
 *
 * Use different seq_ids for:
 * - Parallel generations (multiple steppers, each with own seq_id)
 * - Branching/tree search (System 2)
 * - Shared prefix optimization (decode prefix to seq_id=0, copy to others)
 *
 * ## IMPORTANT: n_seq_max Clarification
 *
 * There are TWO different n_seq_max parameters - don't confuse them:
 *
 * 1. `llama_batch_init(n_tokens, embd, n_seq_max)`
 *    - Controls how many sequences A SINGLE TOKEN can belong to
 *    - Keep at 1 for normal decode (one token → one sequence)
 *    - Only increase for beam search where one token updates multiple branches
 *
 * 2. `llama_context_params.n_seq_max`
 *    - Controls max TOTAL sequences (distinct KV cache states)
 *    - Increase for parallel generations or tree search
 *
 * Example: 4 parallel steppers, each decoding its own branch
 *   - Context n_seq_max: 4 (four distinct sequences)
 *   - Batch n_seq_max: 1 (each token belongs to one sequence)
 *   - Call: decode::many(ctx, tokens, n, pos, batch, seq_id=stepper_id)
 *
 * @param ctx Llama context (must be initialized)
 * @param tokens Token array to decode
 * @param n_tokens Number of tokens in array
 * @param n_past Position to start decoding from (KV cache position)
 * @param n_batch Batch size for chunking
 * @param seq_id Sequence ID to update in KV cache (default: 0)
 * @return 0 on success, non-zero on decode failure
 * @throws std::runtime_error if ctx is NULL or tokens are invalid (validation errors)
 *
 * CRITICAL: Call kv::remove_range() BEFORE this function, never after.
 */
[[nodiscard]] inline int many(llama_context *ctx, const llama_token *tokens,
                               int32_t n_tokens, int32_t n_past, int32_t n_batch,
                               llama_seq_id seq_id = 0) {
  LLOYAL_LOG_DEBUG(
      "[decode::many] Processing %d tokens at position %d", n_tokens,
      n_past);

  if (!ctx) {
    LLOYAL_LOG_DEBUG("[decode::many] ERROR: NULL context");
    throw std::runtime_error("decode::many - NULL context");
  }

  if (!tokens || n_tokens <= 0) {
    LLOYAL_LOG_DEBUG("[decode::many] ERROR: Invalid token array");
    throw std::runtime_error("decode::many - Invalid token array");
  }

  // Initialize batch with RAII cleanup
  // Single-sequence batch (n_seq_max = 1)
  llama_batch batch = llama_batch_init(n_batch, 0, 1);

  // RAII guard for automatic batch cleanup
  struct BatchGuard {
    llama_batch &batch;
    explicit BatchGuard(llama_batch &b) : batch(b) {}
    ~BatchGuard() { llama_batch_free(batch); }
  } batch_guard(batch);

  // Process tokens in chunks
  int32_t processed = 0;
  while (processed < n_tokens) {
    const int32_t n_eval = std::min(n_tokens - processed, n_batch);

    // Clear batch using llama.cpp common library
    common_batch_clear(batch);

    // Add tokens one by one, mark logits=true on LAST token only
    for (int32_t i = 0; i < n_eval; ++i) {
      const int32_t pos = n_past + i;
      const bool want_logits = (i == n_eval - 1);

      // Add token to specified sequence using llama.cpp common library
      common_batch_add(batch, tokens[processed + i], pos, {seq_id}, want_logits);
    }

    // Decode chunk (updates KV cache)
    if (llama_decode(ctx, batch) != 0) {
      LLOYAL_LOG_DEBUG(
          "[decode::many] ERROR: llama_decode failed at position %d",
          n_past);
      return -1;
    }

    n_past += n_eval;
    processed += n_eval;

    LLOYAL_LOG_DEBUG("[decode::many] Processed %d/%d tokens",
                     processed, n_tokens);
  }

  LLOYAL_LOG_DEBUG("[decode::many] Decode complete");
  return 0;
}

/**
 * Convenience overload for std::vector<llama_token>
 */
[[nodiscard]] inline int many(llama_context *ctx,
                               const std::vector<llama_token> &tokens,
                               int32_t n_past, int32_t n_batch,
                               llama_seq_id seq_id = 0) {
  return many(ctx, tokens.data(), static_cast<int32_t>(tokens.size()), n_past,
              n_batch, seq_id);
}

/**
 * Decode a single token with zero heap allocation (when LLOYAL_STACK_BATCH=1)
 *
 * Uses stack-allocated llama_batch to avoid llama_batch_init() overhead.
 * This is the fast path for single-token expansion.
 *
 * If LLOYAL_STACK_BATCH=0, uses thread_local batch for ABI safety.
 *
 * @param ctx Llama context
 * @param tok Token to decode
 * @param pos Position in KV cache
 * @param seq_id Sequence ID (default: 0)
 * @param want_logits Request logits for this token (default: true)
 * @return 0 on success, non-zero on decode failure
 * @throws std::runtime_error if ctx is NULL
 */
[[nodiscard]] inline int one(llama_context *ctx, llama_token tok, llama_pos pos,
                              llama_seq_id seq_id = 0, bool want_logits = true) {
  if (!ctx) {
    throw std::runtime_error("decode::one - NULL context");
  }

#if LLOYAL_STACK_BATCH
  // Fast path: zero-allocation stack-constructed batch
  // WARNING: ABI-fragile - breaks if llama_batch struct layout changes
  llama_token tok_arr[1] = {tok};
  llama_pos pos_arr[1] = {pos};
  int32_t n_seq_id_arr[1] = {1};
  llama_seq_id seq_arr[1] = {seq_id};
  llama_seq_id *seq_ptrs[1] = {seq_arr};
  int8_t logits_arr[1] = {static_cast<int8_t>(want_logits)};

  llama_batch batch{};
  batch.n_tokens = 1;
  batch.token = tok_arr;
  batch.embd = nullptr;
  batch.pos = pos_arr;
  batch.n_seq_id = n_seq_id_arr;
  batch.seq_id = seq_ptrs;
  batch.logits = logits_arr;
#else
  // Safe path: thread_local batch via llama.cpp's own initializer
  // Handles any new fields with defaults, survives ABI changes
  thread_local llama_batch batch = llama_batch_init(1, 0, 1);

  batch.n_tokens = 1;
  batch.token[0] = tok;
  batch.pos[0] = pos;
  batch.n_seq_id[0] = 1;
  batch.seq_id[0][0] = seq_id;
  batch.logits[0] = static_cast<int8_t>(want_logits);
#endif

  return llama_decode(ctx, batch);
}

// ============================================================================
// Multi-Sequence Decode
// ============================================================================

/**
 * @brief Input item for decode::each: one token for one sequence
 */
struct EachItem {
  llama_token token;
  llama_pos pos;
  llama_seq_id seq_id;
  bool output_logits = false;
};

/**
 * @brief Input item for decode::scatter: multiple tokens for one sequence
 */
struct ScatterItem {
  const llama_token* tokens;
  int32_t n_tokens;
  llama_pos start_pos;
  llama_seq_id seq_id;
  bool output_logits_last_only = false;
};

/**
 * @brief Reusable scratch buffers for multi-seq batch construction
 */
struct Scratch {
  std::vector<llama_token> tokens_;
  std::vector<llama_pos> pos_;
  std::vector<int32_t> n_seq_id_;
  std::vector<llama_seq_id> seq_id_single_;
  std::vector<llama_seq_id*> seq_id_ptrs_;
  std::vector<int8_t> logits_;

  void resize(int32_t n) {
    tokens_.resize(n);
    pos_.resize(n);
    n_seq_id_.resize(n);
    seq_id_single_.resize(n);
    seq_id_ptrs_.resize(n);
    logits_.resize(n);
  }

  llama_batch as_batch(int32_t n_tokens) {
    llama_batch batch{};
    batch.n_tokens = n_tokens;
    batch.token = tokens_.data();
    batch.embd = nullptr;
    batch.pos = pos_.data();
    batch.n_seq_id = n_seq_id_.data();
    batch.seq_id = seq_id_ptrs_.data();
    batch.logits = logits_.data();
    return batch;
  }
};

/**
 * @brief Decode one token per sequence in a single llama_decode() call
 *
 * "each" = each sequence gets one token.
 * Packs N tokens (each targeting a different seq_id) into one llama_batch.
 * Amortizes GPU dispatch overhead across N sequences.
 *
 * @param ctx Llama context (must not be null)
 * @param items Array of (token, pos, seq_id, output_logits) tuples
 * @param n Number of items
 * @param scratch Reusable scratch buffers
 * @return 0 on success, non-zero on failure
 * @throws std::runtime_error if ctx is NULL
 */
[[nodiscard]] inline int each(llama_context* ctx,
                               const EachItem* items,
                               int32_t n,
                               Scratch& scratch) {
  if (!ctx) {
    throw std::runtime_error("decode::each - NULL context");
  }
  if (n <= 0) return 0;

  scratch.resize(n);

  for (int32_t i = 0; i < n; ++i) {
    scratch.tokens_[i] = items[i].token;
    scratch.pos_[i] = items[i].pos;
    scratch.n_seq_id_[i] = 1;
    scratch.seq_id_single_[i] = items[i].seq_id;
    scratch.seq_id_ptrs_[i] = &scratch.seq_id_single_[i];
    scratch.logits_[i] = items[i].output_logits ? int8_t{1} : int8_t{0};
  }

  llama_batch batch = scratch.as_batch(n);

  LLOYAL_LOG_DEBUG("[decode::each] Submitting %d tokens across %d sequences", n, n);

  return llama_decode(ctx, batch);
}

// Vector overload
[[nodiscard]] inline int each(llama_context* ctx,
                               const std::vector<EachItem>& items,
                               Scratch& scratch) {
  return each(ctx, items.data(), static_cast<int32_t>(items.size()), scratch);
}

/**
 * @brief Decode multiple tokens per sequence in a single llama_decode() call
 *
 * Packs token runs from multiple sequences into one llama_batch.
 *
 * @param ctx Llama context (must not be null)
 * @param items Array of (tokens, n_tokens, start_pos, seq_id) tuples
 * @param n Number of items
 * @param scratch Reusable scratch buffers
 * @return 0 on success, non-zero on failure
 * @throws std::runtime_error if ctx is NULL or items are invalid
 *
 * @note Does NOT auto-chunk. Total tokens must fit in n_batch.
 */
[[nodiscard]] inline int scatter(llama_context* ctx,
                                  const ScatterItem* items,
                                  int32_t n,
                                  Scratch& scratch) {
  if (!ctx) {
    throw std::runtime_error("decode::scatter - NULL context");
  }

  int32_t total = 0;
  for (int32_t i = 0; i < n; ++i) {
    if (items[i].n_tokens < 0) {
      throw std::runtime_error("decode::scatter - negative n_tokens");
    }
    if (items[i].n_tokens > 0 && items[i].tokens == nullptr) {
      throw std::runtime_error("decode::scatter - null tokens pointer");
    }
    total += items[i].n_tokens;
  }
  if (total == 0) return 0;

  scratch.resize(total);

  int32_t cursor = 0;
  for (int32_t i = 0; i < n; ++i) {
    const auto& item = items[i];
    const llama_pos base_pos = item.start_pos;

    for (int32_t j = 0; j < item.n_tokens; ++j) {
      scratch.tokens_[cursor] = item.tokens[j];
      scratch.pos_[cursor] = base_pos + j;
      scratch.n_seq_id_[cursor] = 1;
      scratch.seq_id_single_[cursor] = item.seq_id;
      scratch.seq_id_ptrs_[cursor] = &scratch.seq_id_single_[cursor];

      const bool want_logits =
          item.output_logits_last_only ? (j == item.n_tokens - 1) : false;
      scratch.logits_[cursor] = want_logits ? int8_t{1} : int8_t{0};

      ++cursor;
    }
  }

  llama_batch batch = scratch.as_batch(total);

  LLOYAL_LOG_DEBUG("[decode::scatter] Submitting %d total tokens across %d sequences", total, n);

  return llama_decode(ctx, batch);
}

// Vector overload
[[nodiscard]] inline int scatter(llama_context* ctx,
                                  const std::vector<ScatterItem>& items,
                                  Scratch& scratch) {
  return scatter(ctx, items.data(), static_cast<int32_t>(items.size()), scratch);
}

} // namespace lloyal::decode

// ============================================================================
// Backward Compatibility Aliases (deprecated)
// ============================================================================
// These aliases provide backward compatibility during migration.
// They will be removed in a future release.

namespace lloyal::decoder {

using MultiSeqItem = decode::EachItem;
using ScatterItem = decode::ScatterItem;
using MultiSeqScratch = decode::Scratch;

// decode_tokens -> decode::many (throws on error for backward compat)
inline void decode_tokens(llama_context *ctx, const llama_token *tokens,
                          int32_t n_tokens, int32_t n_past, int32_t n_batch,
                          llama_seq_id seq_id = 0) {
  int result = decode::many(ctx, tokens, n_tokens, n_past, n_batch, seq_id);
  if (result != 0) {
    throw std::runtime_error("decoder::decode_tokens - llama_decode failed");
  }
}

inline void decode_tokens(llama_context *ctx,
                          const std::vector<llama_token> &tokens,
                          int32_t n_past, int32_t n_batch,
                          llama_seq_id seq_id = 0) {
  decode_tokens(ctx, tokens.data(), static_cast<int32_t>(tokens.size()), n_past,
                n_batch, seq_id);
}

// decode_one -> decode::one (throws on error for backward compat)
inline void decode_one(llama_context *ctx, llama_token tok, llama_pos pos,
                       llama_seq_id seq_id = 0, bool want_logits = true) {
  int result = decode::one(ctx, tok, pos, seq_id, want_logits);
  if (result != 0) {
    throw std::runtime_error("decoder::decode_one - llama_decode failed");
  }
}

// decode_multiseq -> decode::each
[[nodiscard]] inline int decode_multiseq(llama_context* ctx,
                                         const MultiSeqItem* items,
                                         int32_t n,
                                         MultiSeqScratch& scratch) {
  return decode::each(ctx, items, n, scratch);
}

[[nodiscard]] inline int decode_multiseq(llama_context* ctx,
                                         const std::vector<MultiSeqItem>& items,
                                         MultiSeqScratch& scratch) {
  return decode::each(ctx, items, scratch);
}

// decode_scatter -> decode::scatter
[[nodiscard]] inline int decode_scatter(llama_context* ctx,
                                        const ScatterItem* items,
                                        int32_t n,
                                        MultiSeqScratch& scratch) {
  return decode::scatter(ctx, items, n, scratch);
}

[[nodiscard]] inline int decode_scatter(llama_context* ctx,
                                        const std::vector<ScatterItem>& items,
                                        MultiSeqScratch& scratch) {
  return decode::scatter(ctx, items, scratch);
}

} // namespace lloyal::decoder
