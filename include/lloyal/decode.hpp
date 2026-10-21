#pragma once

// SPDX-License-Identifier: LicenseRef-FSL-1.1-Apache-2.0
// Copyright 2026 Lloyal Labs


#include "common.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <llama/llama.h>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
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
 *                    ├─────────────────┼─────────────────┤
 *    Embedding Rows  │  decode::embd   │        —        │
 *                    └─────────────────┴─────────────────┘
 *
 * Single-sequence primitives auto-chunk internally (many, embd); the
 * multi-sequence ones do not — packing across sequences is a policy decision
 * that lives in BranchStore (see bin_pack).
 *
 * Batches allocated by llama_batch_init() are filled by detail::batch_clear/batch_add.
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

// Scratch::as_batch and as_embd_batch write llama_batch field by field; a new
// upstream field would arrive as zero without a word. Fail the build instead:
// the size on 64-bit, and the member count, which padding cannot hide.
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(llama_batch) == 56, "llama_batch changed shape: audit Scratch::as_batch and as_embd_batch");
#endif
namespace shape_guard {
inline void llama_batch_has_seven_members() {
  [[maybe_unused]] auto [n_tokens, token, embd, pos, n_seq_id, seq_id, logits] = llama_batch{};
}
}  // namespace shape_guard

namespace detail {
/// The two helpers llama.cpp's common library used to provide for a
/// llama_batch_init()-allocated batch: reset the count, append one token
/// targeting one sequence.
inline void batch_clear(llama_batch& batch) { batch.n_tokens = 0; }
inline void batch_add(llama_batch& batch, llama_token id, llama_pos pos,
                      llama_seq_id seq_id, bool logits) {
  const int32_t i = batch.n_tokens;
  batch.token[i] = id;
  batch.pos[i] = pos;
  batch.n_seq_id[i] = 1;
  batch.seq_id[i][0] = seq_id;
  batch.logits[i] = logits ? int8_t{1} : int8_t{0};
  batch.n_tokens++;
}
} // namespace detail

/**
 * @brief llama_decode failure carrying the raw return code
 *
 * Two facts travel as DATA, because the caller acts on them and can infer
 * neither from the message:
 *
 * - `rc` classifies the FAILING CALL (llama.h): `1` = no KV slot and `-1` =
 *   invalid batch both leave the memory as it was before that call; `2` =
 *   aborted and `< -1` = fatal leave partial ubatches behind.
 * - `partial` says whether EARLIER calls of the same operation landed. Every
 *   chunked path (many, embd, BranchStore::decode_scatter, decode_segments)
 *   may have committed chunks before the one that failed, and the branch's
 *   books never move on failure.
 *
 * The rule, for an operation on one branch: the branch is INTACT iff the
 * failing call restored state (rc is 1 or -1) and nothing before it landed
 * (!partial). Intact with rc 1 is a capacity wait — retry when the KV has
 * room; intact with rc -1 is the caller's input — do not resend the same
 * batch. Anything else ⇒ prune the branch and replay onto a fresh one.
 *
 * BranchStore::decode_scatter spans branches: its `partial` says an earlier
 * chunk landed, for some branch, whose books moved with it. Whether a given
 * branch is intact is its own `poisoned` flag (branch::is_poisoned()).
 *
 * The binding catches this type in C++ and forwards both fields structurally;
 * the exception itself never crosses N-API.
 *
 * Visibility caveat: safe while liblloyal is header-only (thrower and
 * catcher compile into one TU). If liblloyal ever becomes a separate shared
 * library, typed catches can miss under -fvisibility=hidden — attach the rc
 * some other way before making that move.
 *
 * The message also carries `rc=N` (and `partial` when set) for humans reading
 * logs; nothing parses it.
 */
/// The rule above as a predicate: whether a failed call left its branch intact.
inline bool leaves_branch_intact(int32_t rc, bool partial) {
  return (rc == 1 || rc == -1) && !partial;
}

struct DecodeError : std::runtime_error {
  int32_t rc;
  bool partial;
  DecodeError(int32_t rc_, bool partial_, const std::string& msg)
      : std::runtime_error(msg + " (rc=" + std::to_string(rc_) +
                           (partial_ ? ", partial)" : ")")),
        rc(rc_), partial(partial_) {}
};

/**
 * The context's batch is a hard limit: the pinned llama.cpp asserts
 * `n_tokens <= n_batch` inside llama_decode — an abort, not an rc. Chunk
 * sizes come from the caller (a branch's n_batch, a free function's
 * argument), so every primitive below checks the batch it is about to build
 * against the context first. An oversize chunk is a clear error; it is never
 * a dead process.
 */
inline void require_fits_batch(llama_context* ctx, int32_t n_tokens, const char* who) {
  const int32_t limit = static_cast<int32_t>(llama_n_batch(ctx));
  if (n_tokens > limit) {
    throw std::runtime_error(std::string(who) + " - a batch of " + std::to_string(n_tokens) +
                             " tokens exceeds the context's n_batch (" + std::to_string(limit) + ")");
  }
}

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
 * @param n_committed Optional out: tokens landed before return — `n_tokens`
 *        on success, fewer when a later chunk failed (see DecodeError::partial)
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
                               llama_seq_id seq_id = 0,
                               int32_t* n_committed = nullptr) {
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
  require_fits_batch(ctx, std::min(n_batch, n_tokens), "decode::many");

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

    detail::batch_clear(batch);

    // Add tokens one by one, mark logits=true only on the final chunk's last token
    const bool is_last_chunk = (processed + n_eval >= n_tokens);
    for (int32_t i = 0; i < n_eval; ++i) {
      const int32_t pos = n_past + i;
      const bool want_logits = is_last_chunk && (i == n_eval - 1);

      detail::batch_add(batch, tokens[processed + i], pos, seq_id, want_logits);
    }

    // Decode chunk (updates KV cache)
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
      LLOYAL_LOG_DEBUG(
          "[decode::many] ERROR: llama_decode failed at position %d (rc=%d)",
          n_past, rc);
      if (n_committed) *n_committed = processed;
      return rc;
    }

    n_past += n_eval;
    processed += n_eval;

    LLOYAL_LOG_DEBUG("[decode::many] Processed %d/%d tokens",
                     processed, n_tokens);
  }

  LLOYAL_LOG_DEBUG("[decode::many] Decode complete");
  if (n_committed) *n_committed = n_tokens;
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

  detail::batch_clear(tl.batch);
  detail::batch_add(tl.batch, tok, pos, seq_id, want_logits);

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

  /// @param n Tokens (or embedding rows) in the batch
  /// @param n_pos_per_embd Positions per entry — 1 for token batches, 4 for
  ///        M-RoPE embedding batches, where `pos_` is section-major and holds
  ///        `n * n_pos_per_embd` entries.
  void resize(int32_t n, int32_t n_pos_per_embd = 1) {
    tokens_.resize(n);
    pos_.resize(static_cast<size_t>(n) * n_pos_per_embd);
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

  /// The embedding-rail twin of as_batch(): `embd` points at CALLER-owned
  /// rows (never copied — that is the point), `token` is null. A llama_batch
  /// is token-XOR-embd. `tokens_` is unused on this path.
  ///
  /// Same ABI sensitivity as as_batch() — audit together.
  llama_batch as_embd_batch(int32_t n_tokens, const float* rows) {
    llama_batch batch{};
    batch.n_tokens = n_tokens;
    batch.token = nullptr;
    batch.embd = const_cast<float*>(rows);
    batch.pos = pos_.data();
    batch.n_seq_id = n_seq_id_.data();
    batch.seq_id = seq_id_ptrs_.data();
    batch.logits = logits_.data();
    return batch;
  }
};

/// Rows must be exactly the model's input width: a wrong-but-positive width
/// starts later rows mid-row and reads past the caller's allocation, with
/// nothing to signal it.
inline void require_row_width(llama_context* ctx, int32_t n_embd_inp, const char* who) {
  if (const llama_model* m = llama_get_model(ctx)) {
    const int32_t expected = llama_model_n_embd_inp(m);
    if (expected > 0 && n_embd_inp != expected) {
      throw std::runtime_error(std::string(who) + " - n_embd_inp " + std::to_string(n_embd_inp) +
                               " does not match the model's input width " + std::to_string(expected));
    }
  }
}

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
  require_fits_batch(ctx, n, "decode::each");

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
 * @note Does NOT auto-chunk. A total beyond the context's n_batch is refused
 *       (require_fits_batch), never dispatched.
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
  require_fits_batch(ctx, total, "decode::scatter");

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
 * @brief Input item for decode::embd — embedding rows for one sequence
 *
 * The embedding-rail counterpart of ScatterItem. `rows` is CALLER-owned and
 * never copied: an encoder's output buffer is pointed at directly.
 */
struct EmbdItem {
  /// n_rows x n_embd_inp floats, caller-owned, valid for the call's duration
  const float* rows = nullptr;
  int32_t n_rows = 0;
  /// Row width — llama_model_n_embd_inp(model), the INPUT dim (not n_embd)
  int32_t n_embd_inp = 0;
  /// Section-major positions: n_rows * n_pos_per_embd entries, laid out
  /// [s0...][s1...]... Under M-RoPE the producer decides what each section
  /// means; this layer only slices them.
  const llama_pos* pos = nullptr;
  /// 1 (plain positions) or 4 (M-RoPE)
  int32_t n_pos_per_embd = 1;
  llama_seq_id seq_id = 0;
  /// Bracket the whole decode in non-causal attention (Gemma-class
  /// projectors; requires n_ubatch >= n_rows since the block cannot split)
  bool non_causal = false;
  /// When true, compute logits for the LAST row of the last sub-chunk.
  /// Rows are an interior prefix otherwise — a subsequent llama_decode
  /// resets the output buffer, so only a rows-terminal prefill needs this.
  bool output_logits = false;
};

/**
 * @brief One run of input in a heterogeneous prefill
 *
 * A prefill is a sequence of segments, placed in order by `segments()`. How
 * they are dispatched is the context's to decide (see StreamReading): each its
 * own dispatch, TEXT through `many()` and EMBD through `embd()`, or the whole
 * stream as entries of one mixed batch.
 *
 * The store is deliberately blind to what produced an EMBD segment: rows are
 * rows, whether they came from a vision projector, an audio encoder, or a
 * cached embedding.
 */
struct Segment {
  enum class Kind { Text, Embd };
  Kind kind = Kind::Text;

  /// Text: ready token ids (never re-tokenized by the store)
  std::span<const llama_token> tokens;

  /// Embd: n_rows x n_embd_inp floats. Valid only until the source's next
  /// `at()` call — see SegmentSource's in-order contract.
  const float* rows = nullptr;
  int32_t n_rows = 0;
  int32_t n_embd_inp = 0;
  /// Position advance this segment costs (may be < n_rows under M-RoPE)
  llama_pos n_pos = 0;
  /// 1 (plain positions) or 4 (M-RoPE)
  int32_t n_pos_per_embd = 1;
  /// Bracket this segment's decode in non-causal attention
  bool non_causal = false;
};

/**
 * @brief Supplies the segment sequence for one branch's prefill
 *
 * The caller owns *placement* (which rail, at what position, which segment
 * captures logits); the source owns *production* (decoding bytes, encoding
 * rows, and the model-specific position geometry). Nothing about the
 * producing format crosses this interface — an implementation may pull in
 * llama.cpp's mtmd, a platform encoder, or nothing at all.
 *
 * **In-order contract.** Segments are requested strictly in order, and each is
 * done with before the next is requested: dispatched, or copied into a batch
 * that dispatches once the stream is whole. An implementation may therefore
 * hand out a pointer into a buffer it reuses (mtmd's encode output is one),
 * invalidating segment i-1 when `at(i)` is called. Callers must not hold a
 * segment's rows across an `at()` call, prefetch, or revisit.
 */
struct SegmentSource {
  virtual ~SegmentSource() = default;

  /// Number of segments in this prefill.
  virtual size_t size() = 0;

  /**
   * KV cells this whole sequence will consume, known BEFORE anything decodes.
   *
   * Exists so a caller can decide ADMISSION before touching a branch.
   * `decode_segments` is not atomic: once the first segment dispatches the
   * branch is mutated, so discovering mid-walk that the sequence does not fit
   * costs the branch. Refusing up front costs nothing.
   *
   * Text can already be measured by tokenizing it, but a source that encodes —
   * a vision projector, an audio tower — knows a row count its caller cannot
   * derive from the bytes it holds. Without this, such input is the one thing
   * that bypasses a context-pressure gate.
   *
   * Cells, not positions: a KV budget is spent in cells, and under M-RoPE a
   * segment costs far more cells than it advances position. The unit matches
   * `DecodeSegmentsResult::cells`, so a caller can compare what it was quoted
   * against what it was charged.
   *
   * Must equal the sum over `at(0 .. size()-1)` of `tokens.size()` for TEXT
   * and `n_rows` for EMBD. A source that cannot know the count before
   * encoding must not estimate: this is a budget promise, and an under-quote
   * is spent out of someone else's budget.
   */
  virtual size_t cells() const = 0;

  /// Segment `i`. Invalidates any previously returned segment.
  virtual Segment at(size_t i) = 0;

  /**
   * Fill positions for an EMBD segment, given the absolute base the store
   * chose. `out` has `n_rows * n_pos_per_embd` entries, section-major:
   * `[s0...][s1...]...`. Never called for a TEXT segment.
   *
   * The base is passed rather than exposed, because how it applies is
   * model-specific — M-RoPE freezes one section and leaves another at zero,
   * so a caller-side rebase would need the producer's model taxonomy.
   */
  virtual void positions(size_t i, llama_pos base, llama_pos* out) = 0;

  /// The input segment `i` belongs to (an image, a video, a recording), or -1
  /// for prompt text that belongs to none. A source without inputs has none.
  virtual int32_t input_of(size_t /*i*/) const { return -1; }

  /// KV cells input `input` takes, summed over its segments: what progress on
  /// it counts towards. Known before anything decodes, like cells().
  virtual size_t total_of(int32_t /*input*/) const { return 0; }
};

/**
 * @brief A token span as a stream of one TEXT segment, so text is placed by the
 * rail like any other input
 */
class TokenSource final : public SegmentSource {
public:
  explicit TokenSource(std::span<const llama_token> tokens) : tokens_(tokens) {}
  size_t size() override { return 1; }
  size_t cells() const override { return tokens_.size(); }
  Segment at(size_t) override {
    Segment seg;
    seg.tokens = tokens_;
    return seg;
  }
  void positions(size_t, llama_pos, llama_pos*) override {}

private:
  std::span<const llama_token> tokens_;
};

/**
 * @brief How far one input of a stream has landed: a reading of progress
 */
struct SegmentProgress {
  /// The input, as SegmentSource::input_of names it
  int32_t input = -1;
  /// Its cells in the KV so far
  size_t landed = 0;
  /// Its cells in all, as SegmentSource::total_of quotes them
  size_t total = 0;
};

/**
 * @brief Decode pre-computed embedding rows into one sequence's KV cache
 *
 * The embedding rail beside the token rail: rows enter via `batch.embd` with
 * `batch.token = nullptr`. A llama_batch is token-XOR-embd, so rows never
 * share a dispatch with tokens — this is always its own llama_decode,
 * distinct from scatter().
 *
 * Auto-chunks by n_batch like many() — and this is the MAIN path, not an
 * edge: `image_min_tokens` metadata commonly puts an image above the default
 * batch size. Each sub-chunk re-packs its positions section-major into the
 * scratch buffers, so no view/slice buffer is needed.
 *
 * @param ctx     Llama context (must not be null)
 * @param item    Rows, positions, sequence and flags
 * @param n_batch Max rows per llama_decode (sub-chunk bound)
 * @param scratch Reusable scratch buffers (shared with each()/scatter())
 * @param n_committed Optional out: rows landed before return — `n_rows` on
 *        success, fewer when a later chunk failed (see DecodeError::partial)
 * @return 0 on success, non-zero llama_decode rc on failure
 * @throws std::runtime_error on null ctx or malformed item
 *
 * @see scatter() for the multi-sequence token-rail primitive
 * @see BranchStore::decode_embd for the branch-level wrapper (bookkeeping)
 */
[[nodiscard]] inline int embd(llama_context* ctx,
                              const EmbdItem& item,
                              int32_t n_batch,
                              Scratch& scratch,
                              int32_t* n_committed = nullptr) {
  if (!ctx) {
    throw std::runtime_error("decode::embd - NULL context");
  }
  if (!item.rows || item.n_rows <= 0 || item.n_embd_inp <= 0) {
    throw std::runtime_error("decode::embd - invalid rows");
  }
  if (!item.pos || (item.n_pos_per_embd != 1 && item.n_pos_per_embd != 4)) {
    throw std::runtime_error("decode::embd - invalid positions");
  }
  if (n_batch <= 0) {
    throw std::runtime_error("decode::embd - n_batch must be positive");
  }
  // The row width must match the RESIDENT model, not merely be positive.
  // llama_batch carries no width metadata: llama_decode consumes rows at the
  // model's own input width while the chunk loop below strides by this one.
  // A wrong-but-positive width therefore starts later chunks mid-row and reads
  // past the caller's allocation — corrupt vision state, or an out-of-bounds
  // native read, with nothing to signal it.
  require_row_width(ctx, item.n_embd_inp, "decode::embd");

  const int32_t n    = item.n_rows;
  const int32_t nppe = item.n_pos_per_embd;
  require_fits_batch(ctx, std::min(n_batch, n), "decode::embd");

  // A non-causal block is bidirectional: every row must be able to attend to
  // every other row, which only holds if they share one forward pass. The
  // chunk loop below would split an oversized block across separate
  // llama_decode calls, and rows in an earlier call cannot see later ones —
  // the block silently stops being bidirectional and the vision state is
  // wrong with no error anywhere. Refuse the configuration instead.
  if (item.non_causal) {
    const int32_t n_ubatch = static_cast<int32_t>(llama_n_ubatch(ctx));
    if (n > n_batch || n > n_ubatch) {
      throw std::runtime_error(
          "decode::embd - non-causal block of " + std::to_string(n) +
          " rows exceeds n_batch (" + std::to_string(n_batch) +
          ") or n_ubatch (" + std::to_string(n_ubatch) +
          "); a bidirectional image must decode in a single dispatch");
    }
  }

  /// Restores causal attention on every exit path, including a throw.
  ///
  /// Causal mode is CONTEXT-WIDE, not per-batch: leaving it off would make
  /// every subsequent TEXT decode on this context non-causal, so the damage
  /// outlives this call. `scratch.resize()` can throw between the disable
  /// below and the end of the loop, which is why restoring at the return
  /// points is not enough.
  struct CausalGuard {
    llama_context* ctx = nullptr;
    /// False for causal items — the guard is then inert
    bool engaged = false;
    ~CausalGuard() { if (engaged) llama_set_causal_attn(ctx, true); }
  } causal_guard{ctx, item.non_causal};

  if (item.non_causal) llama_set_causal_attn(ctx, false);

  int32_t processed = 0;
  while (processed < n) {
    const int32_t n_view = std::min(n - processed, n_batch);
    scratch.resize(n_view, nppe);

    // Section-major re-pack for this view: source section s spans
    // [s*n + processed, +n_view); destination is contiguous per section.
    for (int32_t s = 0; s < nppe; ++s) {
      const llama_pos* src = item.pos + static_cast<size_t>(s) * n + processed;
      std::copy(src, src + n_view,
                scratch.pos_.begin() + static_cast<size_t>(s) * n_view);
    }

    for (int32_t i = 0; i < n_view; ++i) {
      scratch.n_seq_id_[i]     = 1;
      scratch.seq_id_single_[i] = item.seq_id;
      scratch.seq_id_ptrs_[i]  = &scratch.seq_id_single_[i];
      scratch.logits_[i]       = 0;
    }
    const bool is_last_view = (processed + n_view >= n);
    if (item.output_logits && is_last_view) {
      scratch.logits_[n_view - 1] = 1;
    }

    llama_batch batch = scratch.as_embd_batch(
        n_view, item.rows + static_cast<size_t>(processed) * item.n_embd_inp);

    LLOYAL_LOG_DEBUG("[decode::embd] Submitting %d/%d rows (seq %d)",
                     processed + n_view, n, item.seq_id);

    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
      LLOYAL_LOG_DEBUG("[decode::embd] ERROR: llama_decode failed (rc=%d)", rc);
      if (n_committed) *n_committed = processed;
      return rc;
    }

    processed += n_view;
  }

  LLOYAL_LOG_DEBUG("[decode::embd] Decode complete (%d rows)", n);
  if (n_committed) *n_committed = n;
  return 0;
}

// ============================================================================
// The segments rail
// ============================================================================

/**
 * @brief A segment of a stream once it is in the KV: what moved, for the
 * caller's books
 */
struct LandedSegment {
  /// Its index in the source
  size_t index = 0;
  Segment::Kind kind = Segment::Kind::Text;
  /// KV cells it took: tokens for TEXT, rows for EMBD
  int32_t cells = 0;
  /// Position it advanced: the token count for TEXT, `n_pos` for EMBD
  llama_pos n_pos = 0;
  /// Whether its last entry computed logits. Placed one segment per dispatch,
  /// every TEXT segment does, and an EMBD segment only when it ends the stream;
  /// placed whole, only the segment that ends the stream does.
  bool has_logits = false;
};

/**
 * @brief How a context reads what is placed into it: the one read of its traits
 * that placement decides from
 */
struct StreamReading {
  /// It keeps what it was given (a KV cache), so a later dispatch attends to an earlier one
  bool keeps_memory = true;
  /// Its attention is causal
  bool causal = true;
  /// It pools a sequence into one output
  bool pooled = false;

  /// With no memory, attention that is not causal, or pooling, a dispatch sees
  /// only its own entries, so a stream placed in pieces would be read in pieces.
  /// Any other context (a reasoning model's: memory, causal, no pooling) is
  /// placed a segment at a time.
  bool whole() const { return !keeps_memory || !causal || pooled; }

  /// Whether a bidirectional block can ride inside one mixed batch. llama.cpp
  /// encodes a context without memory with causal attention off for the whole
  /// batch; a context that keeps memory and attends causally cannot, since its
  /// causal mode is context-wide.
  bool takes_bidirectional() const { return !keeps_memory || !causal; }
};

inline StreamReading reading_of(llama_context* ctx) {
  return StreamReading{llama_get_memory(ctx) != nullptr, llama_get_causal_attn(ctx),
                       llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE};
}

namespace detail {

/// A segment the source yielded, checked before anything is sized from it or
/// handed back to the source: the source is a public extension point, so its
/// segments are untrusted input.
inline void require_segment(const Segment& seg, size_t i) {
  if (seg.kind == Segment::Kind::Text) {
    // Terminality comes from the index, so an empty trailing segment would make
    // the real final one non-terminal: [EMBD, empty TEXT] would decode the image
    // without logits and hand back nothing to sample.
    if (seg.tokens.empty()) {
      throw std::runtime_error("decode::segments - empty TEXT segment at " + std::to_string(i) +
                               "; a source must not yield empty segments");
    }
    return;
  }
  if (!seg.rows || seg.n_rows <= 0) {
    throw std::runtime_error("decode::segments - empty embedding segment at " + std::to_string(i));
  }
  // n_pos_per_embd of -1 wraps a size_t multiply into an enormous allocation,
  // and 0 hands positions() a zero-length buffer to write into.
  if (seg.n_pos_per_embd != 1 && seg.n_pos_per_embd != 4) {
    throw std::runtime_error("decode::segments - segment " + std::to_string(i) + " has n_pos_per_embd " +
                             std::to_string(seg.n_pos_per_embd) + " (expected 1 or 4)");
  }
  if (seg.n_embd_inp <= 0) {
    throw std::runtime_error("decode::segments - segment " + std::to_string(i) + " has non-positive n_embd_inp");
  }
  if (seg.n_pos <= 0 || seg.n_pos > seg.n_rows) {
    throw std::runtime_error("decode::segments - segment " + std::to_string(i) + " has n_pos " +
                             std::to_string(seg.n_pos) + " outside (0, n_rows=" + std::to_string(seg.n_rows) + "]");
  }
}

/// Each segment its own dispatch: the shape of a prefill into a sequence that
/// keeps what it was given.
template <typename OnLanded>
void segments_each(llama_context* ctx, llama_seq_id seq, llama_pos base, SegmentSource& source,
                   int32_t rows_n_batch, Scratch& scratch, std::vector<llama_pos>& positions,
                   OnLanded&& landed) {
  const size_t n = source.size();
  const int32_t text_n_batch = static_cast<int32_t>(llama_n_batch(ctx));
  llama_pos at = base;
  bool any_landed = false;

  const auto failed = [&](int rc, int32_t committed, const std::string& what) {
    return DecodeError(rc, any_landed || committed > 0, "decode::segments - " + what);
  };

  for (size_t i = 0; i < n; ++i) {
    const Segment seg = source.at(i);
    const bool is_last = (i + 1 == n);
    require_segment(seg, i);

    if (seg.kind == Segment::Kind::Text) {
      const int32_t n_tokens = static_cast<int32_t>(seg.tokens.size());
      int32_t committed = 0;
      if (const int rc = many(ctx, seg.tokens.data(), n_tokens, at, text_n_batch, seq, &committed); rc != 0) {
        throw failed(rc, committed, "TEXT segment " + std::to_string(i) + " failed");
      }
      at += n_tokens;
      any_landed = true;
      landed(LandedSegment{i, Segment::Kind::Text, n_tokens, n_tokens, true});
      continue;
    }

    // The base is handed to the source, which applies its own model's rules.
    positions.assign(static_cast<size_t>(seg.n_rows) * seg.n_pos_per_embd, 0);
    source.positions(i, at, positions.data());

    EmbdItem item;
    item.rows           = seg.rows;
    item.n_rows         = seg.n_rows;
    item.n_embd_inp     = seg.n_embd_inp;
    item.pos            = positions.data();
    item.n_pos_per_embd = seg.n_pos_per_embd;
    item.seq_id         = seq;
    item.non_causal     = seg.non_causal;
    item.output_logits  = is_last;
    int32_t committed = 0;
    if (const int rc = embd(ctx, item, rows_n_batch, scratch, &committed); rc != 0) {
      throw failed(rc, committed, "EMBD segment " + std::to_string(i) + " failed");
    }
    at += seg.n_pos;
    any_landed = true;
    landed(LandedSegment{i, Segment::Kind::Embd, seg.n_rows, seg.n_pos, is_last});
  }
}

/// The whole stream in one dispatch: one llama_batch_ext, tokens and rows as
/// entries in order, one llama_process. The batch keeps its own copy of every
/// row, so a source may reuse its row buffer while the stream is gathered.
template <typename OnLanded>
void segments_whole(llama_context* ctx, const StreamReading& reading, llama_seq_id seq, llama_pos base,
                    SegmentSource& source, std::vector<llama_pos>& positions, OnLanded&& landed) {
  // An oversized dispatch on this context aborts the process inside llama.cpp
  // rather than returning an error. The source's quote refuses one early, before
  // anything is encoded; the entries actually built are counted against the same
  // bound below, since the source is untrusted.
  const size_t n_fit = std::min(llama_n_batch(ctx), llama_n_ubatch(ctx));
  const auto too_large = [&](size_t n_entries) {
    return std::runtime_error("decode::segments - a stream of " + std::to_string(n_entries) +
                              " entries must go in one dispatch on this context, which holds " +
                              std::to_string(n_fit));
  };
  if (source.cells() > n_fit) throw too_large(source.cells());

  std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx),
                                                                          llama_batch_ext_free);
  if (!batch) {
    throw std::runtime_error("decode::segments - llama_batch_ext_init failed");
  }
  // A stream placed whole is read whole: every entry's output is on, pooled or
  // per entry, as llama.cpp's encoder computes them for such a context anyway.
  const auto entry = [&](int32_t idx, const llama_pos* pos) {
    if (idx >= 0 && static_cast<size_t>(idx) >= n_fit) throw too_large(static_cast<size_t>(idx) + 1);
    if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, pos) ||
        !llama_batch_ext_set_output_embd(batch.get(), idx, true)) {
      throw std::runtime_error("decode::segments - the batch refused an entry (" + std::to_string(idx) + ")");
    }
    return idx;
  };

  const size_t n = source.size();
  std::vector<LandedSegment> placed;
  placed.reserve(n);
  llama_pos at = base;
  int32_t last = -1;

  for (size_t i = 0; i < n; ++i) {
    const Segment seg = source.at(i);
    require_segment(seg, i);

    if (seg.kind == Segment::Kind::Text) {
      for (size_t k = 0; k < seg.tokens.size(); ++k) {
        const llama_pos pos = at + static_cast<llama_pos>(k);
        last = entry(llama_batch_ext_add_token(batch.get(), seq, seg.tokens[k]), &pos);
      }
      const int32_t n_tokens = static_cast<int32_t>(seg.tokens.size());
      placed.push_back(LandedSegment{i, Segment::Kind::Text, n_tokens, n_tokens, false});
      at += n_tokens;
      continue;
    }

    require_row_width(ctx, seg.n_embd_inp, "decode::segments");
    if (seg.non_causal && !reading.takes_bidirectional()) {
      throw std::runtime_error("decode::segments - segment " + std::to_string(i) +
                               " is bidirectional, and this context attends causally");
    }
    positions.assign(static_cast<size_t>(seg.n_rows) * seg.n_pos_per_embd, 0);
    source.positions(i, at, positions.data());
    // positions() is section-major; an entry takes its own [t, y, x, z].
    std::array<llama_pos, 4> pos{};
    for (int32_t r = 0; r < seg.n_rows; ++r) {
      for (int32_t k = 0; k < seg.n_pos_per_embd; ++k) {
        pos[k] = positions[static_cast<size_t>(k) * seg.n_rows + r];
      }
      const llama_embd row{seg.rows + static_cast<size_t>(r) * seg.n_embd_inp, 1,
                           static_cast<size_t>(seg.n_embd_inp)};
      last = entry(llama_batch_ext_add_embd(batch.get(), seq, row), pos.data());
    }
    placed.push_back(LandedSegment{i, Segment::Kind::Embd, seg.n_rows, seg.n_pos, false});
    at += seg.n_pos;
  }

  if (last < 0) return;
  // The stream's last entry carries its logits, as a per-segment placement's does.
  llama_batch_ext_set_output_logits(batch.get(), last, true);
  placed.back().has_logits = true;

  // A model that cannot mix tokens and rows in one batch answers -1 here.
  if (const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()); rc != 0) {
    throw DecodeError(rc, false, "decode::segments - the stream's one dispatch failed");
  }
  for (const auto& seg : placed) landed(seg);
}

}  // namespace detail


/**
 * @brief Place a source's segments into one sequence, in order
 *
 * The one owner of placement for a heterogeneous stream. The context decides
 * how, never the caller:
 *
 * - **One dispatch per segment** for a context that keeps what it was given
 *   and attends causally, a reasoning model's: TEXT through many() at the
 *   context's batch, EMBD through embd() at `rows_n_batch` (a branch's own
 *   batch), so a prefill keeps exactly the dispatch shape it has always had.
 * - **The whole stream in one dispatch** for a context that reads it only
 *   whole (see StreamReading), an embedding model's: tokens and rows as
 *   entries of one llama_batch_ext, one llama_process. It is refused before
 *   anything runs when it would not fit one ubatch, because llama.cpp aborts
 *   on an oversized one rather than returning an error.
 *
 * The source is untrusted input: every segment is checked before it sizes a
 * buffer or reaches positions(). A segment is placed at the position the
 * previous one ended, and `landed` is told once it is in the KV, so a caller
 * moves its books per segment and a later failure finds them true to what
 * landed.
 *
 * Moves nothing of the caller's itself. A failed dispatch throws DecodeError,
 * `partial` when an earlier segment landed or a chunk of this one did: the
 * stream is one operation to its caller. A throw from the source passes
 * through unchanged.
 *
 * @param ctx          Llama context (must not be null)
 * @param seq          Sequence the stream lands in
 * @param base         Position of the stream's first entry
 * @param source       Yields the segments; owns production, not placement
 * @param rows_n_batch Most rows per dispatch for an EMBD segment placed alone
 * @param scratch      Reusable batch buffers (shared with each()/scatter()/embd())
 * @param positions    Reusable buffer for an EMBD segment's positions
 * @param landed       Called with a LandedSegment after each segment lands
 */
template <typename OnLanded>
inline void segments(llama_context* ctx, llama_seq_id seq, llama_pos base,
                     SegmentSource& source, int32_t rows_n_batch,
                     Scratch& scratch, std::vector<llama_pos>& positions,
                     OnLanded&& landed) {
  if (!ctx) {
    throw std::runtime_error("decode::segments - NULL context");
  }
  if (const StreamReading reading = reading_of(ctx); reading.whole()) {
    detail::segments_whole(ctx, reading, seq, base, source, positions, landed);
    return;
  }
  detail::segments_each(ctx, seq, base, source, rows_n_batch, scratch, positions, landed);
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
