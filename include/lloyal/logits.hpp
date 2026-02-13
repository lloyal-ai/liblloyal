#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

/**
 * @file logits.hpp
 * @brief Zero-copy logits access with clear lifetime semantics
 *
 * Provides safe wrapper around llama_get_logits_ith() with:
 * - Null checking and error handling
 * - Clear documentation of pointer lifetime
 * - Consistent error messages
 *
 * LIFETIME CONTRACT:
 * The returned pointer is valid ONLY until the next decode()/encode() call.
 * Shells are responsible for implementing their own safety mechanisms
 * (e.g., buffer detachment, reference tracking) to prevent use-after-invalidation.
 *
 * USAGE:
 *   float* logits = lloyal::logits::get(ctx);
 *   int n_vocab = lloyal::tokenizer::vocab_size(model);
 *   // Use logits[0..n_vocab-1] synchronously
 *   // DO NOT store across decode() calls
 */

#include <llama/llama.h>
#include <stdexcept>
#include <string>

namespace lloyal::logits {

/**
 * @brief Zero-copy access to logits from the last decode call
 *
 * Returns a pointer to the internal llama.cpp logits buffer.
 * This is a zero-copy operation — no data is copied.
 *
 * @param ctx Llama context (must not be null)
 * @param index Batch index passed to llama_get_logits_ith():
 *              - `-1` — last token (default; single-sequence decode)
 *              - `i`  — batch slot `i` from decode::each()
 *              - flattened cursor position from decode::scatter()
 * @returns Pointer to float array of size n_vocab
 * @throws std::runtime_error if ctx is null or logits unavailable
 *
 * @warning Pointer lifetime: valid only until the next
 *          decode()/encode()/dispose() call. Points to llama.cpp
 *          internal memory — do NOT free. Requires decode() was
 *          called with logits=true for this index.
 *
 * @example
 * @code
 *   // Single-sequence (last token):
 *   float* logits = lloyal::logits::get(ctx);
 *
 *   // Batched (decode::each slot i):
 *   float* logits = lloyal::logits::get(ctx, i);
 *
 *   // Batched (decode::scatter flattened cursor):
 *   int32_t cursor = sum_of_previous_token_counts + n_tokens_k - 1;
 *   float* logits = lloyal::logits::get(ctx, cursor);
 * @endcode
 *
 * @see BranchStore::decode_each() for batch-slot index usage
 * @see BranchStore::decode_scatter() for flattened-cursor index usage
 */
inline float* get(llama_context* ctx, int32_t index = -1) {
    if (!ctx) {
        throw std::runtime_error("logits::get - NULL context");
    }

    float* ptr = llama_get_logits_ith(ctx, index);
    if (!ptr) {
        throw std::runtime_error(
            "logits::get - Failed to get logits at index " +
            std::to_string(index) + ". "
            "Ensure decode() was called with logits=true for this index."
        );
    }

    return ptr;
}

} // namespace lloyal::logits
