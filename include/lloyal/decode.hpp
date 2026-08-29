#pragma once

// SPDX-License-Identifier: LicenseRef-FSL-1.1-Apache-2.0
// Copyright 2026 Lloyal Labs


#include "common.hpp"
#include <common.h>  // llama.cpp common library: common_batch_clear, common_batch_add
#include <algorithm>
#include <cstdint>
#include <llama/llama.h>
#include <span>
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
 *
 * ## Logit Indexing: How llama_get_logits_ith() Maps to Batch Positions
 *
 * llama.cpp packs logits into a dense output buffer — only tokens with
 * `batch.logits[i] = true` get logits computed. The internal `output_ids`
 * vector translates batch positions to packed rows:
 *
 * @code
 *   Batch:      [tok0, tok1, tok2, tok3, tok4, tok5, tok6, tok7]
 *   logits[]:   [  0,    0,    0,    0,    1,    0,    0,    1 ]
 *
 *   output_ids: [ -1,   -1,   -1,   -1,    0,   -1,   -1,    1]
 *                                          ^                 ^
 *                                       row 0             row 1
 *
 *   llama_get_logits_ith(ctx, 4)  → output_ids[4] = 0  → logits + 0*n_vocab  ✓
 *   llama_get_logits_ith(ctx, 7)  → output_ids[7] = 1  → logits + 1*n_vocab  ✓
 *   llama_get_logits_ith(ctx, 0)  → output_ids[0] = -1 → throws (no logits)
 *   llama_get_logits_ith(ctx, -1) → n_outputs - 1 = 1  → logits + 1*n_vocab  (last output)
 * @endcode
 *
 * Callers always pass **batch positions**, not packed indices. The
 * `output_ids` indirection handles the translation. Negative indices
 * bypass `output_ids` entirely: `-1` means the last output row,
 * `-2` the second-to-last, etc.
 *
 * This matters for logit capture in BranchStore:
 *
 * | Decode pattern  | logits flag                    | Access index                        |
 * |-----------------|--------------------------------|-------------------------------------|
 * | decode::one     | Last token only                | `-1` (sole output)                  |
 * | decode::many    | Last token of final chunk only | `-1` (sole output of last dispatch) |
 * | decode::each    | All items (1:1 with branches)  | `i` (batch pos = item index)        |
 * | decode::scatter | Last token per item            | `cursor + n_tokens[k] - 1`          |
 *
 * For `decode::many`, each chunk is a separate `llama_decode()` call that
 * resets the output buffer. Only the final chunk's last token has logits,
 * so after the last dispatch `n_outputs = 1` and `-1` yields row 0.
 */

namespace lloyal::decode {

/**
 * @brief Decode multiple tokens into the KV cache with auto-chunking
 *
 * Orchestration logic:
 * 1. Uses a thread_local batch (heap-allocated once per thread, grows on demand)
 * 2. Chunks tokens into n_batch-sized pieces
 * 3. For each chunk: clear batch, add tokens, call llama_decode
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

  if (n_batch <= 0) {
    throw std::runtime_error("decode::many - n_batch must be positive");
  }

  // Thread-local batch avoids per-call allocation. Grows if needed, never shrinks.
  struct ThreadLocalBatch {
    llama_batch batch{};
    int32_t capacity = 0;

    void ensure(int32_t n) {
      if (n <= capacity) return;
      if (capacity > 0) llama_batch_free(batch);
      batch = llama_batch_init(n, 0, 1);
      capacity = n;
    }

    ~ThreadLocalBatch() {
      if (capacity > 0) llama_batch_free(batch);
    }
  };
  thread_local ThreadLocalBatch tl;
  tl.ensure(n_batch);
  llama_batch& batch = tl.batch;

  // Process tokens in chunks
  int32_t processed = 0;
  while (processed < n_tokens) {
    const int32_t n_eval = std::min(n_tokens - processed, n_batch);

    // Clear batch using llama.cpp common library
    common_batch_clear(batch);

    // Add tokens one by one, mark logits=true only on the final chunk's last token
    const bool is_last_chunk = (processed + n_eval >= n_tokens);
    for (int32_t i = 0; i < n_eval; ++i) {
      const int32_t pos = n_past + i;
      const bool want_logits = is_last_chunk && (i == n_eval - 1);

      // Add token via llama.cpp common library (function-call ABI).
      // {seq_id} constructs a temporary vector per token — acceptable cost
      // vs direct field writes which create struct-layout ABI coupling.
      common_batch_add(batch, tokens[processed + i], pos, {seq_id}, want_logits);
    }

    // Decode chunk (updates KV cache)
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
      LLOYAL_LOG_DEBUG(
          "[decode::many] ERROR: llama_decode failed at position %d (rc=%d)",
          n_past, rc);
      return rc;
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

  struct ThreadLocalBatch {
    llama_batch batch = llama_batch_init(1, 0, 1);
    ~ThreadLocalBatch() { llama_batch_free(batch); }
  };
  thread_local ThreadLocalBatch tl;

  common_batch_clear(tl.batch);
  common_batch_add(tl.batch, tok, pos, {seq_id}, want_logits);

  return llama_decode(ctx, tl.batch);
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
 *
 * Uses std::span for a non-owning view of the token array. The span
 * carries both pointer and length, eliminating raw-pointer + count
 * mismatch bugs. An empty span (size 0) is valid and skipped by scatter().
 */
struct ScatterItem {
  std::span<const llama_token> tokens;    ///< Token array (non-owning view)
  llama_pos start_pos;                    ///< KV cache position for first token
  llama_seq_id seq_id;                    ///< Target sequence ID
  bool output_logits = false;             ///< When true, compute logits for last token in this run
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

  /// ABI-sensitive: writes llama_batch fields directly (no common_batch_* wrapper
  /// exists for external-buffer batches). Audit on llama.cpp submodule bumps.
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
  if (n < 0) {
    throw std::runtime_error("decode::each - negative item count");
  }
  if (n == 0) return 0;

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
 * @param items Array of (tokens_span, start_pos, seq_id) tuples
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
  if (n < 0) {
    throw std::runtime_error("decode::scatter - negative item count");
  }

  int32_t total = 0;
  for (int32_t i = 0; i < n; ++i) {
    total += static_cast<int32_t>(items[i].tokens.size());
  }
  if (total == 0) return 0;

  scratch.resize(total);

  int32_t cursor = 0;
  for (int32_t i = 0; i < n; ++i) {
    const auto& item = items[i];
    const llama_pos base_pos = item.start_pos;
    const int32_t item_n = static_cast<int32_t>(item.tokens.size());

    for (int32_t j = 0; j < item_n; ++j) {
      scratch.tokens_[cursor] = item.tokens[j];
      scratch.pos_[cursor] = base_pos + j;
      scratch.n_seq_id_[cursor] = 1;
      scratch.seq_id_single_[cursor] = item.seq_id;
      scratch.seq_id_ptrs_[cursor] = &scratch.seq_id_single_[cursor];

      const bool want_logits =
          item.output_logits ? (j == item_n - 1) : false;
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

// ============================================================================
// Embedding-Row Decode (multimodal ingress)
// ============================================================================

/**
 * @brief Decode pre-computed embedding rows into one sequence's KV cache
 *
 * The embedding rail beside the token rail: rows (e.g. a clip-encoded image)
 * enter the KV via `batch.embd` with `batch.token = nullptr`. A llama_batch
 * is token-XOR-embd, so embedding rows can NEVER share a dispatch with
 * tokens — this is always its own llama_decode, distinct from scatter().
 *
 * Positions are SECTION-MAJOR: `n_tokens * n_pos_per_embd` entries laid out
 * `[s0...][s1...][s2...][s3...]`. For M-RoPE image rows the caller packs
 * t/y/x/z per mtmd's decoder-pos convention (y in section 1, x in section 2);
 * for plain-position models `n_pos_per_embd == 1` and `pos` is ordinary.
 *
 * Sub-chunks internally by n_batch — the MAIN path, not an edge: image row
 * counts routinely exceed the default batch size. Each view re-packs the pos
 * sections for the visible rows (the decode_embd_batch::get_view semantics).
 *
 * Rows are an interior prefix and request no logits; `want_last_logits`
 * marks the final row of the final view for the rows-terminal case (a
 * subsequent llama_decode resets the output buffer, so a prefill whose LAST
 * dispatch is this one must request logits here or `logits::get(ctx, -1)`
 * has no output row).
 *
 * `non_causal` brackets the whole loop in llama_set_causal_attn(false/true),
 * restoring on error (Gemma-class projectors; requires n_ubatch >= n_tokens).
 *
 * Buffers are allocated per call: embedding decodes are low-frequency
 * relative to the per-tick token paths, so allocation cost is noise next to
 * the encode itself (no Scratch/thread_local reuse needed).
 *
 * ABI-sensitive: writes llama_batch fields directly, same as
 * Scratch::as_batch above. Audit on llama.cpp submodule bumps.
 *
 * @param ctx            Llama context (must not be null)
 * @param rows           Embedding rows, n_tokens x n_embd_inp floats
 * @param n_tokens       Number of rows
 * @param n_embd_inp     Row width — llama_model_n_embd_inp(model)
 * @param pos            Section-major positions, n_tokens * n_pos_per_embd
 * @param n_pos_per_embd 1 (plain) or 4 (M-RoPE)
 * @param seq_id         Target sequence ID
 * @param n_batch        Max rows per llama_decode (sub-chunk bound)
 * @param non_causal     Bracket the loop in non-causal attention
 * @param want_last_logits Request logits on the final row (rows-terminal)
 * @return 0 on success, non-zero llama_decode rc on failure
 * @throws std::runtime_error if ctx/rows/pos are null or sizes non-positive
 *
 * @see scatter() for the multi-sequence token-rail primitive
 * @see BranchStore::prefill_embd for the branch-level wrapper (bookkeeping)
 */
[[nodiscard]] inline int embd(llama_context* ctx,
                              const float* rows,
                              int32_t n_tokens,
                              int32_t n_embd_inp,
                              const llama_pos* pos,
                              int32_t n_pos_per_embd,
                              llama_seq_id seq_id,
                              int32_t n_batch,
                              bool non_causal,
                              bool want_last_logits) {
  if (!ctx) {
    throw std::runtime_error("decode::embd - NULL context");
  }
  if (!rows || n_tokens <= 0 || n_embd_inp <= 0) {
    throw std::runtime_error("decode::embd - invalid rows");
  }
  if (!pos || (n_pos_per_embd != 1 && n_pos_per_embd != 4)) {
    throw std::runtime_error("decode::embd - invalid positions");
  }
  if (n_batch <= 0) {
    throw std::runtime_error("decode::embd - n_batch must be positive");
  }

  // Per-row metadata (full length; views index into these with an offset)
  std::vector<int32_t> n_seq_id(n_tokens, 1);
  llama_seq_id seq_id_single = seq_id;
  std::vector<llama_seq_id*> seq_id_ptrs(
      static_cast<size_t>(n_tokens) + 1, &seq_id_single);
  seq_id_ptrs[n_tokens] = nullptr;
  std::vector<int8_t> logits(n_tokens, 0);
  if (want_last_logits) logits[n_tokens - 1] = 1;

  // Per-view position buffer (re-packed section-major for the view)
  std::vector<llama_pos> pos_view;

  int32_t processed = 0;
  if (non_causal) llama_set_causal_attn(ctx, false);

  while (processed < n_tokens) {
    const int32_t n_view = std::min(n_tokens - processed, n_batch);

    const llama_pos* view_pos;
    if (n_pos_per_embd > 1) {
      // Section-major re-pack: source section s spans
      // pos[s*n_tokens + processed .. +n_view); view wants them contiguous
      // per section over n_view rows.
      pos_view.resize(static_cast<size_t>(n_view) * n_pos_per_embd);
      for (int32_t s = 0; s < n_pos_per_embd; ++s) {
        std::copy(pos + static_cast<size_t>(s) * n_tokens + processed,
                  pos + static_cast<size_t>(s) * n_tokens + processed + n_view,
                  pos_view.begin() + static_cast<size_t>(s) * n_view);
      }
      view_pos = pos_view.data();
    } else {
      view_pos = pos + processed;
    }

    llama_batch batch{};
    batch.n_tokens = n_view;
    batch.token = nullptr;
    batch.embd = const_cast<float*>(rows) +
                 static_cast<size_t>(processed) * n_embd_inp;
    batch.pos = const_cast<llama_pos*>(view_pos);
    batch.n_seq_id = n_seq_id.data() + processed;
    batch.seq_id = seq_id_ptrs.data() + processed;
    batch.logits = logits.data() + processed;

    LLOYAL_LOG_DEBUG("[decode::embd] Submitting %d/%d rows (seq %d)",
                     processed + n_view, n_tokens, seq_id);

    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
      if (non_causal) llama_set_causal_attn(ctx, true);
      LLOYAL_LOG_DEBUG("[decode::embd] ERROR: llama_decode failed (rc=%d)", rc);
      return rc;
    }

    processed += n_view;
  }

  if (non_causal) llama_set_causal_attn(ctx, true);

  LLOYAL_LOG_DEBUG("[decode::embd] Decode complete (%d rows)", n_tokens);
  return 0;
}

// ============================================================================
// Bin-Packing Utility
// ============================================================================

/**
 * @brief A chunk of item indices produced by bin_pack()
 *
 * Normal chunks contain items whose total tokens fit in n_batch.
 * Oversized chunks contain a single item whose tokens exceed n_batch
 * (caller must dispatch via decode::many with auto-chunking).
 */
struct PackedChunk {
  std::vector<int32_t> indices;   ///< Indices into the original items array
  bool oversized = false;         ///< True → single item exceeding n_batch
};

/**
 * @brief Greedy first-fit bin-packing of token spans into n_batch-sized chunks
 *
 * Pure packing algorithm — no decoding, no logit capture, no context.
 * Callers use the returned chunks to drive their own dispatch logic
 * (decode::scatter for normal chunks, decode::many for oversized).
 *
 * Empty spans (size 0) are skipped. Items exceeding n_batch get a
 * solo oversized chunk.
 *
 * @param items   Array of token spans (only .size() is inspected)
 * @param n       Number of items
 * @param n_batch Maximum total tokens per normal chunk
 * @return Vector of PackedChunks with indices into the input array
 */
inline std::vector<PackedChunk> bin_pack(
    const std::span<const llama_token>* items,
    int32_t n,
    int32_t n_batch) {

  std::vector<PackedChunk> chunks;
  int32_t chunk_total = 0;

  for (int32_t i = 0; i < n; ++i) {
    int32_t tc = static_cast<int32_t>(items[i].size());
    if (tc == 0) continue;

    if (tc > n_batch) {
      chunks.push_back({{i}, true});
      continue;
    }

    if (chunks.empty() || chunks.back().oversized ||
        chunk_total + tc > n_batch) {
      chunks.push_back({{i}, false});
      chunk_total = tc;
    } else {
      chunks.back().indices.push_back(i);
      chunk_total += tc;
    }
  }

  return chunks;
}

} // namespace lloyal::decode
