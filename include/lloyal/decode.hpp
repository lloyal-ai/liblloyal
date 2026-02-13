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
 * @brief Decode multiple tokens into the KV cache with auto-chunking
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
 *
 * @see one() for single-token decode (autoregressive generation)
 * @see scatter() for multi-token decode across multiple sequences
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

/// @overload
[[nodiscard]] inline int many(llama_context *ctx,
                               const std::vector<llama_token> &tokens,
                               int32_t n_past, int32_t n_batch,
                               llama_seq_id seq_id = 0) {
  return many(ctx, tokens.data(), static_cast<int32_t>(tokens.size()), n_past,
              n_batch, seq_id);
}

/**
 * @brief Decode a single token into the KV cache
 *
 * Fast path for autoregressive generation. Uses a thread_local batch
 * (one-time init per thread) so repeated calls avoid allocation entirely.
 *
 * Typical usage in a generation loop:
 * @code
 *   llama_token tok = sampler::sample(ctx, vocab);
 *   if (decode::one(ctx, tok, n_past++) != 0) { handle error }
 * @endcode
 *
 * @param ctx    Llama context (must not be null)
 * @param tok    Token to decode
 * @param pos    KV cache position for this token
 * @param seq_id Sequence ID to update (default: 0)
 * @param want_logits Whether to compute logits after this token (default: true).
 *                    Set to false when prefilling tokens that don't need sampling.
 * @return 0 on success, non-zero on decode failure
 * @throws std::runtime_error if ctx is NULL
 *
 * @see many() for batched multi-token decode with auto-chunking
 * @see each() for single-token decode across multiple sequences
 */
[[nodiscard]] inline int one(llama_context *ctx, llama_token tok, llama_pos pos,
                              llama_seq_id seq_id = 0, bool want_logits = true) {
  if (!ctx) {
    throw std::runtime_error("decode::one - NULL context");
  }

  thread_local llama_batch batch = llama_batch_init(1, 0, 1);

  batch.n_tokens = 1;
  batch.token[0] = tok;
  batch.pos[0] = pos;
  batch.n_seq_id[0] = 1;
  batch.seq_id[0][0] = seq_id;
  batch.logits[0] = static_cast<int8_t>(want_logits);

  return llama_decode(ctx, batch);
}

// ============================================================================
// Multi-Sequence Decode
// ============================================================================

/**
 * @brief Input item for decode::each — one token for one sequence
 */
struct EachItem {
  llama_token token;            ///< Token to decode
  llama_pos pos;                ///< KV cache position for this token
  llama_seq_id seq_id;          ///< Target sequence ID
  bool output_logits = false;   ///< Whether to compute logits after this token
};

/**
 * @brief Input item for decode::scatter — multiple tokens for one sequence
 */
struct ScatterItem {
  const llama_token* tokens;              ///< Token array (not owned)
  int32_t n_tokens;                       ///< Number of tokens in array
  llama_pos start_pos;                    ///< KV cache position for first token
  llama_seq_id seq_id;                    ///< Target sequence ID
  bool output_logits_last_only = false;   ///< Compute logits only for last token
};

/**
 * @brief Reusable scratch buffers for multi-sequence batch construction
 *
 * Holds pre-allocated vectors that back the llama_batch pointers.
 * Reuse a single Scratch across calls to avoid per-decode allocation.
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
 *
 * @see one() for single-sequence single-token decode
 * @see scatter() for multi-token-per-sequence variant
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

/// @overload
[[nodiscard]] inline int each(llama_context* ctx,
                               const std::vector<EachItem>& items,
                               Scratch& scratch) {
  return each(ctx, items.data(), static_cast<int32_t>(items.size()), scratch);
}

/**
 * @brief Decode multiple tokens per sequence in a single llama_decode() call
 *
 * Single-batch primitive: packs token runs from multiple sequences into one
 * llama_batch. Does NOT auto-chunk — total tokens must fit in n_batch.
 *
 * @param ctx Llama context (must not be null)
 * @param items Array of (tokens, n_tokens, start_pos, seq_id) tuples
 * @param n Number of items
 * @param scratch Reusable scratch buffers
 * @return 0 on success, non-zero on failure
 * @throws std::runtime_error if ctx is NULL or items are invalid
 *
 * @note Does NOT auto-chunk. Total tokens must fit in n_batch.
 *
 * @see many() for single-sequence multi-token decode with auto-chunking
 * @see each() for single-token-per-sequence variant
 * @see BranchStore::decode_scatter for auto-chunking branch-level variant
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

/// @overload
[[nodiscard]] inline int scatter(llama_context* ctx,
                                        const std::vector<ScatterItem>& items,
                                        Scratch& scratch) {
  return scatter(ctx, items.data(), static_cast<int32_t>(items.size()), scratch);
}

} // namespace lloyal::decode
