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
 *                    ├─────────────────┼─────────────────┤
 *    Embedding Rows  │  decode::embd   │        —        │
 *                    └─────────────────┴─────────────────┘
 *
 * Single-sequence primitives auto-chunk internally (many, embd); the
 * multi-sequence ones do not — packing across sequences is a policy decision
 * that lives in BranchStore (see bin_pack).
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
 * @brief llama_decode failure carrying the raw return code
 *
 * Two facts travel as DATA, because the caller acts on them and can infer
 * neither from the message:
 *
 * - `rc` classifies the FAILING CALL (llama.h): `1` = no KV slot, state
 *   restored for that call; `-1` = invalid batch, state restored; `2` =
 *   aborted and `< -1` = fatal — partial ubatches remain.
 * - `partial` says whether EARLIER calls of the same operation landed. Every
 *   chunked path (many, embd, BranchStore::decode_scatter) may have committed
 *   chunks before the one that failed; llama_decode restores only the call it
 *   rejected, and the branch's books never move on failure.
 *
 * The rule, true at every throw site: the branch is intact iff
 * `rc == 1 && !partial` — retry once the KV has room. Anything else ⇒ prune
 * the branch and replay onto a fresh one.
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
struct DecodeError : std::runtime_error {
  int32_t rc;
  bool partial;
  DecodeError(int32_t rc_, bool partial_, const std::string& msg)
      : std::runtime_error(msg + " (rc=" + std::to_string(rc_) +
                           (partial_ ? ", partial)" : ")")),
        rc(rc_), partial(partial_), msg_(msg) {}

  /// The same failure as an ENCLOSING operation reports it once work before
  /// the failing call has landed — decode_segments over its segments.
  DecodeError as_partial() const { return DecodeError(rc, true, msg_); }

private:
  std::string msg_;
};

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
 * A prefill is a sequence of segments, each entering the KV on one of two
 * rails: TEXT via the token rail (`decode_scatter`), EMBD via the embedding
 * rail (`decode_embd`). A `llama_batch` is token-XOR-embd, so the rails
 * never share a dispatch — the segment sequence is what interleaves them.
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
 * **In-order contract.** Segments are consumed strictly in order, and each is
 * dispatched before the next is requested. An implementation may therefore
 * hand out a pointer into a buffer it reuses (mtmd's encode output is one),
 * invalidating segment i-1 when `at(i)` is called. Callers must not hold a
 * segment across an `at()` call, prefetch, or revisit.
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
  if (const llama_model* m = llama_get_model(ctx)) {
    const int32_t expected = llama_model_n_embd_inp(m);
    if (expected > 0 && item.n_embd_inp != expected) {
      throw std::runtime_error(
          "decode::embd - n_embd_inp " + std::to_string(item.n_embd_inp) +
          " does not match the model's input width " +
          std::to_string(expected));
    }
  }

  const int32_t n    = item.n_rows;
  const int32_t nppe = item.n_pos_per_embd;

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
