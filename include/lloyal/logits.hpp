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

namespace lloyal::logits {

/**
 * Get raw logits pointer (zero-copy)
 *
 * Returns a pointer to the internal llama.cpp logits buffer.
 * This is a zero-copy operation - no data is copied.
 *
 * @param ctx Llama context (must not be null)
 * @param step Step index: -1 for last step (default), or specific step index
 * @returns Pointer to float array of size vocab_size
 * @throws std::runtime_error if ctx is null or logits unavailable
 *
 * IMPORTANT - Pointer Lifetime:
 * - Valid only until next decode()/encode()/dispose() call
 * - Points to llama.cpp internal memory (do NOT free)
 * - Requires decode() was called with logits=true for the step
 *
 * EXAMPLE:
 *   // After decode with logits=true
 *   float* logits = lloyal::logits::get(ctx);
 *   int n_vocab = lloyal::tokenizer::vocab_size(model);
 *
 *   // Compute entropy, sample, etc. - all synchronous
 *   float max_logit = *std::max_element(logits, logits + n_vocab);
 *
 *   // After next decode(), logits pointer is INVALID
 *   await ctx.decode(next_tokens);
 *   // logits now points to different/stale data!
 */
inline float* get(llama_context* ctx, int32_t step = -1) {
    if (!ctx) {
        throw std::runtime_error("logits::get - NULL context");
    }

    float* ptr = llama_get_logits_ith(ctx, step);
    if (!ptr) {
        throw std::runtime_error(
            "logits::get - Failed to get logits. "
            "Ensure decode() was called with logits=true for this step."
        );
    }

    return ptr;
}

} // namespace lloyal::logits
