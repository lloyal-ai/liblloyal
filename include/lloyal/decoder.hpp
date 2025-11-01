#pragma once

#include "common.hpp"
#include "helpers.hpp"
#include <llama/llama.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <stdexcept>

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
    llama_batch& batch;
    explicit BatchGuard(llama_batch& b) : batch(b) {}
    ~BatchGuard() { llama_batch_free(batch); }
  };

  /**
   * Add tokens to batch with position info
   */
  inline void add_tokens_to_batch(
    llama_batch& batch,
    const llama_token* tokens,
    int32_t start_idx,
    int32_t n_eval,
    int32_t n_past,
    int32_t capacity
  ) {
    // Clear batch using helpers.hpp function
    lloyal::batch_clear(batch);

    // Add tokens one by one, mark logits=true on LAST token only
    for (int32_t i = 0; i < n_eval; ++i) {
      const int32_t pos = n_past + i;
      const bool want_logits = (i == n_eval - 1);

      // Add token to sequence 0 (single-sequence design)
      lloyal::batch_add(batch, tokens[start_idx + i], pos, {0}, want_logits, capacity);
    }
  }
}

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
 * @param ctx Llama context (must be initialized)
 * @param tokens Token array to decode
 * @param n_tokens Number of tokens in array
 * @param n_past Position to start decoding from (KV cache position)
 * @param n_batch Batch size for chunking
 * @throws std::runtime_error if decode fails
 *
 * CRITICAL: Call kv::remove_range() BEFORE this function, never after.
 */
inline void decode_tokens(
  llama_context* ctx,
  const llama_token* tokens,
  int32_t n_tokens,
  int32_t n_past,
  int32_t n_batch
) {
  LLOYAL_LOG_DEBUG("[decoder::decode_tokens] Processing %d tokens at position %d", n_tokens, n_past);

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
    detail::add_tokens_to_batch(batch, tokens, processed, n_eval, n_past, n_batch);

    // Decode chunk (updates KV cache)
    if (llama_decode(ctx, batch) != 0) {
      LLOYAL_LOG_DEBUG("[decoder::decode_tokens] ERROR: llama_decode failed at position %d", n_past);
      throw std::runtime_error("decoder::decode_tokens - llama_decode failed");
    }

    n_past += n_eval;
    processed += n_eval;

    LLOYAL_LOG_DEBUG("[decoder::decode_tokens] Processed %d/%d tokens", processed, n_tokens);
  }

  LLOYAL_LOG_DEBUG("[decoder::decode_tokens] Decode complete");
}

/**
 * Convenience overload for std::vector<llama_token>
 */
inline void decode_tokens(
  llama_context* ctx,
  const std::vector<llama_token>& tokens,
  int32_t n_past,
  int32_t n_batch
) {
  decode_tokens(ctx, tokens.data(), static_cast<int32_t>(tokens.size()), n_past, n_batch);
}

} // namespace lloyal::decoder
