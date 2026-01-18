#pragma once

#include "common.hpp"
#include "helpers.hpp"
#include <algorithm>
#include <cstdint>
#include <llama/llama.h>
#include <stdexcept>
#include <vector>

/**
 * LLOYAL_STACK_BATCH - Controls llama_batch construction strategy
 *
 * When 1 (default): Use zero-allocation stack-constructed batch in decode_one()
 *   - Fastest: no heap allocation per decode
 *   - Risk: breaks if llama_batch struct layout changes
 *
 * When 0: Use thread_local batch via llama_batch_init()
 *   - Slightly slower: one-time init per thread
 *   - Safe: uses llama.cpp's own initializer, handles new fields
 *
 * If build breaks after llama.cpp update due to llama_batch changes:
 *   1. Set LLOYAL_STACK_BATCH=0 to unblock immediately
 *   2. Update decode_one() to match new struct layout
 *   3. Update ABI stability test assertions
 *   4. Re-enable LLOYAL_STACK_BATCH=1
 */
#ifndef LLOYAL_STACK_BATCH
#define LLOYAL_STACK_BATCH 1
#endif

/**
 * Decoder Anti-Corruption Layer (Header-Only)
 *
 * Purpose: Single point of contact with llama.cpp decode APIs to isolate batch
 * management complexity, chunking logic, and decode operation orchestration.
 *
 * Calls helpers.hpp batch utilities (batch_clear, batch_add).
 */

namespace lloyal::detail {
/**
 * RAII guard for automatic batch cleanup
 * Ensures llama_batch_free is called even if exceptions occur
 */
struct BatchGuard {
  llama_batch &batch;
  explicit BatchGuard(llama_batch &b) : batch(b) {}
  ~BatchGuard() { llama_batch_free(batch); }
};

/**
 * Add tokens to batch with position info
 */
inline void add_tokens_to_batch(llama_batch &batch, const llama_token *tokens,
                                int32_t start_idx, int32_t n_eval,
                                int32_t n_past, int32_t capacity,
                                llama_seq_id seq_id = 0) {
  // Clear batch using helpers.hpp function
  lloyal::batch_clear(batch);

  // Add tokens one by one, mark logits=true on LAST token only
  for (int32_t i = 0; i < n_eval; ++i) {
    const int32_t pos = n_past + i;
    const bool want_logits = (i == n_eval - 1);

    // Add token to specified sequence
    lloyal::batch_add(batch, tokens[start_idx + i], pos, {seq_id}, want_logits,
                      capacity);
  }
}
} // namespace lloyal::detail

namespace lloyal::decoder {

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
 *   - Call: decode_tokens(ctx, tokens, n, pos, batch, seq_id=stepper_id)
 *
 * @param ctx Llama context (must be initialized)
 * @param tokens Token array to decode
 * @param n_tokens Number of tokens in array
 * @param n_past Position to start decoding from (KV cache position)
 * @param n_batch Batch size for chunking
 * @param seq_id Sequence ID to update in KV cache (default: 0)
 * @throws std::runtime_error if decode fails
 *
 * CRITICAL: Call kv::remove_range() BEFORE this function, never after.
 */
inline void decode_tokens(llama_context *ctx, const llama_token *tokens,
                          int32_t n_tokens, int32_t n_past, int32_t n_batch,
                          llama_seq_id seq_id = 0) {
  LLOYAL_LOG_DEBUG(
      "[decoder::decode_tokens] Processing %d tokens at position %d", n_tokens,
      n_past);

  if (!ctx) {
    LLOYAL_LOG_DEBUG("[decoder::decode_tokens] ERROR: NULL context");
    throw std::runtime_error("decoder::decode_tokens - NULL context");
  }

  if (!tokens || n_tokens <= 0) {
    LLOYAL_LOG_DEBUG("[decoder::decode_tokens] ERROR: Invalid token array");
    throw std::runtime_error("decoder::decode_tokens - Invalid token array");
  }

  // Initialize batch with RAII cleanup
  // Single-sequence batch (n_seq_max = 1)
  llama_batch batch = llama_batch_init(n_batch, 0, 1);
  detail::BatchGuard batch_guard(batch);

  // Process tokens in chunks
  int32_t processed = 0;
  while (processed < n_tokens) {
    const int32_t n_eval = std::min(n_tokens - processed, n_batch);

    // Add chunk to batch
    detail::add_tokens_to_batch(batch, tokens, processed, n_eval, n_past,
                                n_batch, seq_id);

    // Decode chunk (updates KV cache)
    if (llama_decode(ctx, batch) != 0) {
      LLOYAL_LOG_DEBUG(
          "[decoder::decode_tokens] ERROR: llama_decode failed at position %d",
          n_past);
      throw std::runtime_error("decoder::decode_tokens - llama_decode failed");
    }

    n_past += n_eval;
    processed += n_eval;

    LLOYAL_LOG_DEBUG("[decoder::decode_tokens] Processed %d/%d tokens",
                     processed, n_tokens);
  }

  LLOYAL_LOG_DEBUG("[decoder::decode_tokens] Decode complete");
}

/**
 * Convenience overload for std::vector<llama_token>
 */
inline void decode_tokens(llama_context *ctx,
                          const std::vector<llama_token> &tokens,
                          int32_t n_past, int32_t n_batch,
                          llama_seq_id seq_id = 0) {
  decode_tokens(ctx, tokens.data(), static_cast<int32_t>(tokens.size()), n_past,
                n_batch, seq_id);
}

/**
 * Decode a single token with zero heap allocation (when LLOYAL_STACK_BATCH=1)
 *
 * Uses stack-allocated llama_batch to avoid llama_batch_init() overhead.
 * This is the fast path for MCTS single-token expansion.
 *
 * If LLOYAL_STACK_BATCH=0, uses thread_local batch for ABI safety.
 *
 * @param ctx Llama context
 * @param tok Token to decode
 * @param pos Position in KV cache
 * @param seq_id Sequence ID (default: 0)
 * @param want_logits Request logits for this token (default: true)
 * @throws std::runtime_error if decode fails
 */
inline void decode_one(llama_context *ctx, llama_token tok, llama_pos pos,
                       llama_seq_id seq_id = 0, bool want_logits = true) {
  if (!ctx) {
    throw std::runtime_error("decoder::decode_one - NULL context");
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

  if (llama_decode(ctx, batch) != 0) {
    throw std::runtime_error("decoder::decode_one - llama_decode failed");
  }
}

/**
 * Encode tokens for embedding extraction
 *
 * Unlike decode_tokens(), this marks ALL tokens with logits=true which is
 * required for embedding extraction.
 *
 * NOTE: Use this with a dedicated embedding context (embeddings=true, pooling
 * enabled). Clear KV between texts with kv::clear_all():
 *
 *   // Create dedicated embedding context
 *   ctx_params.embeddings = true;
 *   ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
 *   auto embed_ctx = llama_init_from_model(model, ctx_params);
 *
 *   // Embed each text
 *   kv::clear_all(embed_ctx);
 *   decoder::encode(embed_ctx, tokens, 512);
 *   auto emb = embedding::get(embed_ctx);
 *
 * @param ctx Llama context (must have embeddings=true and pooling enabled)
 * @param tokens Token array to encode
 * @param n_tokens Number of tokens in array
 * @param n_batch Batch size
 * @throws std::runtime_error if encode fails
 */
inline void encode(llama_context *ctx, const llama_token *tokens,
                   int32_t n_tokens, int32_t n_batch) {
  LLOYAL_LOG_DEBUG("[decoder::encode] Encoding %d tokens for embeddings",
                   n_tokens);

  if (!ctx) {
    LLOYAL_LOG_DEBUG("[decoder::encode] ERROR: NULL context");
    throw std::runtime_error("decoder::encode - NULL context");
  }

  if (!tokens || n_tokens <= 0) {
    LLOYAL_LOG_DEBUG("[decoder::encode] ERROR: Invalid token array");
    throw std::runtime_error("decoder::encode - Invalid token array");
  }

  if (n_tokens > n_batch) {
    LLOYAL_LOG_DEBUG("[decoder::encode] ERROR: n_tokens (%d) > n_batch (%d)",
                     n_tokens, n_batch);
    throw std::runtime_error(
        "decoder::encode - token count exceeds batch size (truncation not "
        "supported, increase n_batch or reduce input length)");
  }

  // Initialize batch - single sequence
  llama_batch batch = llama_batch_init(n_batch, 0, 1);
  detail::BatchGuard batch_guard(batch);

  // Clear batch
  lloyal::batch_clear(batch);

  // Add ALL tokens with logits=true (required for embedding extraction)
  for (int32_t i = 0; i < n_tokens; ++i) {
    lloyal::batch_add(batch, tokens[i], i, {0}, true, n_batch);
  }

  // Decode/encode the batch (llama.cpp handles encoder vs decoder internally)
  if (llama_decode(ctx, batch) != 0) {
    LLOYAL_LOG_DEBUG("[decoder::encode] ERROR: llama_decode failed");
    throw std::runtime_error("decoder::encode - llama_decode failed");
  }

  LLOYAL_LOG_DEBUG("[decoder::encode] Encode complete");
}

/**
 * Convenience overload for std::vector<llama_token>
 */
inline void encode(llama_context *ctx, const std::vector<llama_token> &tokens,
                   int32_t n_batch) {
  encode(ctx, tokens.data(), static_cast<int32_t>(tokens.size()), n_batch);
}

} // namespace lloyal::decoder
