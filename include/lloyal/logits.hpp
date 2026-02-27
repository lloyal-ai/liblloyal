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
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "decode.hpp"
#include "kv.hpp"

namespace lloyal::logits {

/**
 * @brief Zero-copy access to logits from the last decode call
 *
 * Returns a pointer to the internal llama.cpp logits buffer.
 * This is a zero-copy operation — no data is copied.
 *
 * @param ctx Llama context (must not be null)
 * @param index **Batch position** passed to llama_get_logits_ith().
 *              llama.cpp translates this to a packed output row via its
 *              internal `output_ids` table — callers never need the packed
 *              index directly. See decode.hpp file docs for the full
 *              indexing explanation. Valid values:
 *              - `-1` — last output (default; works for any decode pattern)
 *              - `i`  — batch position `i` (must have had `batch.logits[i] = true`)
 * @returns Pointer to float array of size n_vocab
 * @throws std::runtime_error if ctx is null, or logits were not requested
 *         for this batch position (`batch.logits[index]` was false)
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

/**
 * @brief Process arbitrary number of complete prompts for logit extraction
 *
 * Handles prompt counts exceeding n_seq_max by processing in groups.
 * Within each group, prompts are bin-packed via decode::bin_pack() into
 * n_batch-sized chunks, then dispatched via scatter/many. After each
 * group, used seq_ids are evicted from KV to make room for the next.
 *
 * @param ctx       Llama context (caller must ensure exclusive access)
 * @param prompts   Complete token arrays (any count — groups by n_seq_max)
 * @param output    Pre-allocated float buffers, one per prompt, each n_vocab
 * @param n_vocab   Vocabulary size (for memcpy sizing)
 */
inline void process_chunks(
    llama_context* ctx,
    const std::vector<std::span<const llama_token>>& prompts,
    std::vector<float*>& output,
    int32_t n_vocab) {

    if (!ctx) throw std::runtime_error("logits::process_chunks - NULL context");
    if (prompts.size() != output.size())
        throw std::runtime_error("logits::process_chunks - prompts/output size mismatch");
    if (prompts.empty()) return;

    const int32_t seq_max = static_cast<int32_t>(llama_n_seq_max(ctx));
    const int32_t batch_limit = static_cast<int32_t>(llama_n_batch(ctx));
    const int32_t n = static_cast<int32_t>(prompts.size());
    thread_local decode::Scratch scratch;

    for (int32_t group_start = 0; group_start < n; group_start += seq_max) {
        int32_t group_size = std::min(seq_max, n - group_start);

        // bin_pack skips empties internally — pass group slice directly
        auto chunks = decode::bin_pack(&prompts[group_start], group_size, batch_limit);
        if (chunks.empty()) continue;

        for (const auto& chunk : chunks) {
            if (chunk.oversized) {
                int32_t gi = group_start + chunk.indices[0];
                llama_seq_id seq = static_cast<llama_seq_id>(chunk.indices[0]);

                if (decode::many(ctx, prompts[gi].data(),
                                 static_cast<int32_t>(prompts[gi].size()),
                                 0, batch_limit, seq) != 0)
                    throw std::runtime_error("logits::process_chunks - decode::many failed");

                std::memcpy(output[gi], get(ctx, -1), n_vocab * sizeof(float));
                continue;
            }

            // Normal chunk — build ScatterItems
            std::vector<decode::ScatterItem> scatter_items(chunk.indices.size());
            for (size_t k = 0; k < chunk.indices.size(); ++k) {
                int32_t gi = group_start + chunk.indices[k];
                scatter_items[k].tokens = prompts[gi];
                scatter_items[k].start_pos = 0;
                scatter_items[k].seq_id = static_cast<llama_seq_id>(chunk.indices[k]);
                scatter_items[k].output_logits = true;
            }

            if (decode::scatter(ctx, scatter_items.data(),
                                static_cast<int32_t>(scatter_items.size()),
                                scratch) != 0)
                throw std::runtime_error("logits::process_chunks - decode::scatter failed");

            // Capture logits
            int32_t cursor = 0;
            for (size_t k = 0; k < scatter_items.size(); ++k) {
                int32_t gi = group_start + chunk.indices[k];
                int32_t item_n = static_cast<int32_t>(scatter_items[k].tokens.size());
                std::memcpy(output[gi], get(ctx, cursor + item_n - 1), n_vocab * sizeof(float));
                cursor += item_n;
            }
        }

        // Evict KV for this group's seq_ids (no-op on unused ids)
        for (int32_t s = 0; s < group_size; ++s) {
            kv::remove_range(ctx, static_cast<llama_seq_id>(s), 0, -1);
        }
    }
}

} // namespace lloyal::logits
