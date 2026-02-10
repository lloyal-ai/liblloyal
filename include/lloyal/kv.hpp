#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

/**
 * @file kv.hpp
 * @brief KV Cache Management
 *
 * Core primitives for KV cache operations in LLM applications:
 * - Multi-sequence management: independent recurrent states per seq_id
 * - Cache lifecycle: clear, remove, copy, keep operations
 * - State persistence: save/load with fragmentation fallback
 * - Cache reconstruction: clear_and_reseed for context compression strategies
 * - File I/O: session save/resume for app lifecycle management
 *
 * These primitives compose into diverse inference patterns including:
 * - Context window management (streaming, compression, eviction)
 * - Session persistence (save/resume across app restarts)
 * - Multi-sequence orchestration (parallel logical states)
 * - Specialized search and sampling strategies
 *
 * Memory management for llama.cpp primitives:
 * - llama_memory_* for cache lifecycle and multi-sequence ops
 * - llama_state_* for serialization with fragmentation fallback
 * - Adds null-safety, error handling, and defensive programming
 */

#include "common.hpp"
#include "decode.hpp"
#include <cstdint>
#include <llama/llama.h>
#include <vector>

namespace lloyal::kv {

// ===== KV SEQUENCE OPERATIONS =====

/**
 * @brief Remove token range from KV cache sequence
 *
 * Removes tokens in the range [p0, p1) from the specified sequence's KV cache.
 * Used for selective eviction in context window management.
 *
 * @param ctx Llama context (must not be null)
 * @param seq Sequence ID (use 0 for single-sequence mode)
 * @param p0 Start position (inclusive)
 * @param p1 End position (exclusive), use -1 for "to end"
 * @return true if successful, false if context is null or operation failed
 *
 * @warning CRITICAL: Call this BEFORE next llama_decode(), not after.
 *          Calling after decode may cause undefined behavior.
 */
inline bool remove_range(llama_context *ctx, llama_seq_id seq, llama_pos p0,
                         llama_pos p1) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[kv::remove_range] ERROR: null context");
    return false;
  }

  llama_memory_t mem = llama_get_memory(ctx);
  bool success = llama_memory_seq_rm(mem, seq, p0, p1);

  if (!success) {
    LLOYAL_LOG_DEBUG("[kv::remove_range] FAILED: seq=%d, p0=%d, p1=%d", seq, p0,
                     p1);
    LLOYAL_LOG_DEBUG("[kv::remove_range] Guard-rail reminder: Ensure "
                     "remove_range called BEFORE next llama_decode()");
  } else {
    LLOYAL_LOG_DEBUG("[kv::remove_range] OK: seq=%d, removed tokens [%d, %d)",
                     seq, p0, p1);
  }

  return success;
}

/**
 * @brief Get maximum position in KV cache sequence
 *
 * Returns the highest token position in the specified sequence's KV cache.
 * For a sequence with N tokens, this returns N-1 (zero-indexed).
 *
 * @param ctx Llama context (must not be null)
 * @param seq Sequence ID
 * @return Maximum position (number of tokens - 1), or -1 if empty or context is null
 */
inline llama_pos pos_max(llama_context *ctx, llama_seq_id seq) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[kv::pos_max] ERROR: null context");
    return -1;
  }

  llama_memory_t mem = llama_get_memory(ctx);
  llama_pos max_pos = llama_memory_seq_pos_max(mem, seq);

  LLOYAL_LOG_DEBUG("[kv::pos_max] seq=%d, max_pos=%d", seq, max_pos);
  return max_pos;
}

/**
 * @brief Copy KV cache from one sequence to another
 *
 * Copies KV cache state from source to destination sequence, enabling
 * efficient branching without duplicating model weights.
 *
 * @param ctx Llama context (must not be null)
 * @param src Source sequence ID
 * @param dst Destination sequence ID
 * @param p0 Start position (inclusive), default 0
 * @param p1 End position (exclusive), default -1 for "to end"
 *
 * @note Use case: Multi-sequence search (fork from trunk without copying model weights)
 */
inline void seq_cp(llama_context *ctx, llama_seq_id src, llama_seq_id dst,
                   llama_pos p0 = 0, llama_pos p1 = -1) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[kv::seq_cp] ERROR: null context");
    return;
  }

  llama_memory_t mem = llama_get_memory(ctx);
  llama_memory_seq_cp(mem, src, dst, p0, p1);

  LLOYAL_LOG_DEBUG("[kv::seq_cp] Copied seq %d → %d [%d, %d)", src, dst, p0, p1);
}

/**
 * @brief Keep only one sequence, removing all others
 *
 * Removes all sequences except the specified one from the KV cache.
 * Efficient way to prune unused branches.
 *
 * @param ctx Llama context (must not be null)
 * @param seq Sequence ID to keep
 *
 * @note Use case: After selection, prune all alternatives except chosen path
 */
inline void seq_keep(llama_context *ctx, llama_seq_id seq) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[kv::seq_keep] ERROR: null context");
    return;
  }

  llama_memory_t mem = llama_get_memory(ctx);
  llama_memory_seq_keep(mem, seq);

  LLOYAL_LOG_DEBUG("[kv::seq_keep] Kept only seq %d", seq);
}

// ===== STATE SNAPSHOT OPERATIONS =====

/**
 * @brief Get size needed to serialize sequence state
 *
 * Returns buffer size required to save the sequence's KV cache state.
 * Automatically falls back to global state size if per-sequence query fails
 * (may occur with fragmented caches).
 *
 * @param ctx Llama context (must not be null)
 * @param seq Sequence ID
 * @return Required buffer size in bytes, or 0 if empty/failed
 *
 * @note Fallback strategy: per-sequence → global state (handles fragmentation)
 */
inline size_t state_size(llama_context *ctx, llama_seq_id seq) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[kv::state_size] ERROR: null context");
    return 0;
  }

  llama_memory_t mem = llama_get_memory(ctx);
  llama_pos max_pos = llama_memory_seq_pos_max(mem, seq);
  if (max_pos < 0) {
    LLOYAL_LOG_DEBUG("[kv::state_size] WARNING: KV cache is empty (max_pos=%d) "
                     "- returning 0",
                     max_pos);
    return 0;
  }

  size_t size = llama_state_seq_get_size(ctx, seq);

  if (size == 0) {
    LLOYAL_LOG_DEBUG(
        "[kv::state_size] Per-sequence size query failed for seq=%d", seq);
    LLOYAL_LOG_DEBUG(
        "[kv::state_size] Attempting global state size (fallback)");
    size = llama_state_get_size(ctx);

    if (size > 0) {
      LLOYAL_LOG_DEBUG("[kv::state_size] Global fallback size: %zu bytes",
                       size);
    } else {
      LLOYAL_LOG_DEBUG("[kv::state_size] ERROR: Both per-sequence and global "
                       "size queries failed");
    }
  } else {
    LLOYAL_LOG_DEBUG(
        "[kv::state_size] Per-sequence size for seq=%d: %zu bytes (%.1f MB)",
        seq, size, size / 1024.0 / 1024.0);
  }

  return size;
}

/**
 * @brief Save sequence state to buffer
 *
 * Serializes the sequence's KV cache state into the provided buffer.
 * Automatically falls back to global state save if per-sequence save fails
 * (may occur with fragmented caches).
 *
 * @param ctx Llama context (must not be null)
 * @param seq Sequence ID
 * @param dst Destination buffer (must not be null)
 * @param size Buffer size in bytes
 * @return Bytes written, or 0 on failure
 *
 * @note Fallback strategy: per-sequence → global state (handles fragmentation)
 */
inline size_t state_save(llama_context *ctx, llama_seq_id seq, uint8_t *dst,
                         size_t size) {
  if (!ctx || !dst || size == 0) {
    LLOYAL_LOG_DEBUG(
        "[kv::state_save] ERROR: invalid parameters (ctx=%p, dst=%p, size=%zu)",
        ctx, dst, size);
    return 0;
  }

  llama_memory_t mem = llama_get_memory(ctx);
  llama_pos max_pos = llama_memory_seq_pos_max(mem, seq);
  if (max_pos < 0) {
    LLOYAL_LOG_DEBUG("[kv::state_save] WARNING: KV cache is empty (max_pos=%d) "
                     "- skipping save",
                     max_pos);
    return 0;
  }

  size_t written = llama_state_seq_get_data(ctx, dst, size, seq);

  if (written == 0) {
    LLOYAL_LOG_DEBUG("[kv::state_save] Per-sequence save failed for seq=%d "
                     "(possible KV fragmentation)",
                     seq);
    LLOYAL_LOG_DEBUG(
        "[kv::state_save] Attempting global state save (fallback)");
    written = llama_state_get_data(ctx, dst, size);

    if (written > 0) {
      LLOYAL_LOG_DEBUG(
          "[kv::state_save] Global fallback succeeded: %zu bytes (%.1f MB)",
          written, written / 1024.0 / 1024.0);
    } else {
      LLOYAL_LOG_DEBUG(
          "[kv::state_save] ERROR: Both per-sequence and global save failed");
    }
  } else {
    LLOYAL_LOG_DEBUG(
        "[kv::state_save] Per-sequence saved %zu bytes (%.1f MB) for seq=%d",
        written, written / 1024.0 / 1024.0, seq);
  }

  return written;
}

/**
 * @brief Restore sequence state from buffer
 *
 * Deserializes KV cache state from buffer and restores it to the sequence.
 * Automatically falls back to global state restore if per-sequence restore fails
 * (may occur with fragmented caches).
 *
 * @param ctx Llama context (must not be null)
 * @param seq Sequence ID
 * @param src Source buffer (must not be null)
 * @param size Buffer size in bytes
 * @return Bytes read, or 0 on failure
 *
 * @warning May crash on recurrent models if KV cache is empty during load
 * @note Fallback strategy: per-sequence → global state (handles fragmentation)
 */
inline size_t state_load(llama_context *ctx, llama_seq_id seq,
                         const uint8_t *src, size_t size) {
  if (!ctx || !src || size == 0) {
    LLOYAL_LOG_DEBUG(
        "[kv::state_load] ERROR: invalid parameters (ctx=%p, src=%p, size=%zu)",
        ctx, src, size);
    return 0;
  }

  llama_memory_t mem = llama_get_memory(ctx);
  llama_pos max_pos = llama_memory_seq_pos_max(mem, seq);
  if (max_pos < 0) {
    LLOYAL_LOG_DEBUG("[kv::state_load] WARNING: KV cache is empty (max_pos=%d) "
                     "- loading may crash on recurrent models",
                     max_pos);
  }

  size_t read = llama_state_seq_set_data(ctx, src, size, seq);

  if (read == 0) {
    LLOYAL_LOG_DEBUG("[kv::state_load] Per-sequence restore failed for seq=%d "
                     "(possible fragmentation)",
                     seq);
    LLOYAL_LOG_DEBUG(
        "[kv::state_load] Attempting global state restore (fallback)");
    read = llama_state_set_data(ctx, src, size);

    if (read > 0) {
      LLOYAL_LOG_DEBUG(
          "[kv::state_load] Global fallback succeeded: %zu bytes (%.1f MB)",
          read, read / 1024.0 / 1024.0);
    } else {
      LLOYAL_LOG_DEBUG("[kv::state_load] ERROR: Both per-sequence and global "
                       "restore failed");
    }
  } else {
    LLOYAL_LOG_DEBUG(
        "[kv::state_load] Per-sequence loaded %zu bytes (%.1f MB) for seq=%d",
        read, read / 1024.0 / 1024.0, seq);
  }

  return read;
}

// ===== GLOBAL STATE OPERATIONS =====

/**
 * @brief Get size needed to serialize global state
 *
 * Returns buffer size required to save the entire context's state.
 * Use when per-sequence serialization is not needed.
 *
 * @param ctx Llama context (must not be null)
 * @return Required buffer size in bytes, or 0 if context is null
 */
inline size_t global_state_size(llama_context *ctx) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[kv::global_state_size] ERROR: null context");
    return 0;
  }

  size_t size = llama_state_get_size(ctx);
  LLOYAL_LOG_DEBUG("[kv::global_state_size] %zu bytes (%.1f MB)", size,
                   size / 1024.0 / 1024.0);
  return size;
}

/**
 * @brief Save global state to buffer
 *
 * Serializes the entire context's state into the provided buffer.
 *
 * @param ctx Llama context (must not be null)
 * @param dst Destination buffer (must not be null)
 * @param size Buffer size in bytes
 * @return Bytes written, or 0 on failure
 */
inline size_t global_state_save(llama_context *ctx, uint8_t *dst, size_t size) {
  if (!ctx || !dst || size == 0) {
    LLOYAL_LOG_DEBUG("[kv::global_state_save] ERROR: invalid parameters");
    return 0;
  }

  size_t written = llama_state_get_data(ctx, dst, size);
  LLOYAL_LOG_DEBUG("[kv::global_state_save] %zu bytes written (%.1f MB)",
                   written, written / 1024.0 / 1024.0);
  return written;
}

/**
 * @brief Restore global state from buffer
 *
 * Deserializes and restores the entire context's state from buffer.
 *
 * @param ctx Llama context (must not be null)
 * @param src Source buffer (must not be null)
 * @param size Buffer size in bytes
 * @return Bytes read, or 0 on failure
 */
inline size_t global_state_load(llama_context *ctx, const uint8_t *src,
                                size_t size) {
  if (!ctx || !src || size == 0) {
    LLOYAL_LOG_DEBUG("[kv::global_state_load] ERROR: invalid parameters");
    return 0;
  }

  size_t read = llama_state_set_data(ctx, src, size);
  LLOYAL_LOG_DEBUG("[kv::global_state_load] %zu bytes read (%.1f MB)", read,
                   read / 1024.0 / 1024.0);
  return read;
}

// ===== DIAGNOSTICS =====

/**
 * @brief Log KV cache build info and current state
 *
 * Outputs debug information about the KV cache configuration and current state.
 * Useful for debugging and understanding cache behavior.
 *
 * @param ctx Llama context (can be null; limits output if null)
 *
 * @note Only produces output when DEBUG logging is enabled
 */
inline void log_build_info(llama_context *ctx) {
  LLOYAL_LOG_DEBUG(
      "[kv::build_info] ============================================");
  LLOYAL_LOG_DEBUG(
      "[kv::build_info] llama.cpp KV Sequence Operations Configuration");
  LLOYAL_LOG_DEBUG(
      "[kv::build_info] ============================================");
  LLOYAL_LOG_DEBUG("[kv::build_info] Version: b6870");
  LLOYAL_LOG_DEBUG("[kv::build_info] API naming: llama_memory_seq_*");
  LLOYAL_LOG_DEBUG(
      "[kv::build_info] Current MVP: n_seq_max=1 (single sequence only)");

  if (ctx) {
    llama_pos max_pos = pos_max(ctx, 0);
    if (max_pos >= 0) {
      LLOYAL_LOG_DEBUG("[kv::build_info] Current KV cursor (seq 0): %d tokens",
                       max_pos);
    } else {
      LLOYAL_LOG_DEBUG("[kv::build_info] KV cache empty (seq 0)");
    }

    size_t snapshot_size = state_size(ctx, 0);
    if (snapshot_size > 0) {
      LLOYAL_LOG_DEBUG(
          "[kv::build_info] Estimated snapshot size: %zu bytes (%.1f MB)",
          snapshot_size, snapshot_size / 1024.0 / 1024.0);
    }
  }

  LLOYAL_LOG_DEBUG(
      "[kv::build_info] Fragmentation fallback: per-sequence → global state");
  LLOYAL_LOG_DEBUG(
      "[kv::build_info] Critical: Call remove_range() BEFORE llama_decode()");
  LLOYAL_LOG_DEBUG(
      "[kv::build_info] ============================================");
}

// ===== CACHE CLEARING =====

/**
 * @brief Clear all KV cache (complete reset)
 *
 * Clears both metadata and data buffers for a complete cache reset.
 * Use when starting a new conversation or session.
 *
 * @param ctx Llama context (must not be null)
 * @throws std::runtime_error if ctx is null
 *
 * @note Uses llama_memory_clear(mem, true) which:
 *       - Clears metadata (cell positions, sequence heads)
 *       - Zeroes K/V tensor data buffers
 *       - Complete reset (slower than clear_metadata())
 *
 * @see clear_metadata() for faster metadata-only clearing
 */
inline void clear_all(llama_context *ctx) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[kv::clear_all] ERROR: NULL context");
    throw std::runtime_error("kv::clear_all - NULL context");
  }

  LLOYAL_LOG_DEBUG("[kv::clear_all] Clearing KV cache (metadata + data)");
  llama_memory_clear(llama_get_memory(ctx), true);  // true = clear data buffers too
  LLOYAL_LOG_DEBUG("[kv::clear_all] KV cache cleared");
}

/**
 * @brief Clear KV cache metadata only (fast reset)
 *
 * Clears logical structure but keeps buffer allocations.
 * Faster than clear_all() for compression patterns.
 *
 * @param ctx Llama context (must not be null)
 * @throws std::runtime_error if ctx is null
 *
 * @note Performance: Faster than clear_all() (no buffer zeroing)
 *       Use when immediately re-decoding; buffer reuse reduces overhead
 *
 * @see clear_all() for complete reset including data buffers
 */
inline void clear_metadata(llama_context *ctx) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[kv::clear_metadata] ERROR: NULL context");
    throw std::runtime_error("kv::clear_metadata - NULL context");
  }

  LLOYAL_LOG_DEBUG("[kv::clear_metadata] Clearing KV cache metadata only");
  llama_memory_clear(llama_get_memory(ctx), false);  // false = keep data buffers
  LLOYAL_LOG_DEBUG("[kv::clear_metadata] KV cache metadata cleared");
}

// ===== CONTEXT COMPRESSION =====

/**
 * @brief Clear KV cache and reconstruct with anchor + tail tokens
 *
 * Reconstructs KV cache with contiguous positions by:
 * 1. Clearing entire KV cache
 * 2. Re-decoding original_sinks (anchor tokens) at position 0
 * 3. Re-decoding tail (recent tokens) at position sinks.size()
 *
 * This maintains contiguous positions [0,1,2,...] which is simpler and more
 * reliable than selective removal with position gaps.
 *
 * @param ctx Llama context (must not be null)
 * @param original_sinks Anchor tokens from sequence start (typically 4)
 * @param tail Recent tokens to preserve (typically 252, total 256 with sinks)
 * @param n_batch Batch size for re-decoding chunks
 * @throws std::runtime_error if parameters are invalid or re-decode fails
 *
 * @warning CRITICAL: original_sinks MUST be the ORIGINAL first N tokens from
 *          sequence start. Reusing different "first N" tokens on each reseed
 *          will degrade quality for attention-sink patterns.
 *
 * @note After calling, KV cache position = sinks.size() + tail.size()
 *       Continue generation with n_past = static_cast<int32_t>(sinks.size() + tail.size())
 *
 * @example
 *   // Capture original anchor tokens once
 *   std::vector<llama_token> SINKS(tokens.begin(), tokens.begin() + 4);
 *
 *   // Each compression: reuse SAME anchors with current tail
 *   auto tail = std::vector<llama_token>(tokens.end() - 252, tokens.end());
 *   kv::clear_and_reseed(ctx, SINKS, tail, n_batch);
 */
inline void clear_and_reseed(llama_context *ctx,
                             const std::vector<llama_token> &original_sinks,
                             const std::vector<llama_token> &tail,
                             int32_t n_batch) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[kv::clear_and_reseed] ERROR: null context");
    throw std::runtime_error("kv::clear_and_reseed - NULL context");
  }

  if (original_sinks.empty() && tail.empty()) {
    LLOYAL_LOG_DEBUG("[kv::clear_and_reseed] ERROR: both sinks and tail are empty");
    throw std::runtime_error("kv::clear_and_reseed - no tokens to reseed");
  }

  LLOYAL_LOG_DEBUG("[kv::clear_and_reseed] Starting reseed: %zu sinks + %zu tail = %zu total",
                   original_sinks.size(), tail.size(), original_sinks.size() + tail.size());

  // Get memory handle
  llama_memory_t mem = llama_get_memory(ctx);

  // Log state before clear
  llama_pos max_pos_before = llama_memory_seq_pos_max(mem, 0);
  LLOYAL_LOG_DEBUG("[kv::clear_and_reseed] Before clear: KV cache max_pos=%d", max_pos_before);

  // Clear entire KV cache (simple and reliable)
  llama_memory_clear(mem, true);

  llama_pos max_pos_after_clear = llama_memory_seq_pos_max(mem, 0);
  if (max_pos_after_clear != -1) {
    LLOYAL_LOG_DEBUG("[kv::clear_and_reseed] WARNING: KV cache not empty after clear (max_pos=%d)",
                     max_pos_after_clear);
  }

  // Re-decode sinks at position 0
  if (!original_sinks.empty()) {
    LLOYAL_LOG_DEBUG("[kv::clear_and_reseed] Re-decoding %zu sinks at position 0", original_sinks.size());
    lloyal::decoder::decode_tokens(ctx, original_sinks, 0, n_batch);
  }

  // Re-decode tail at position sinks.size()
  if (!tail.empty()) {
    int32_t tail_start_pos = static_cast<int32_t>(original_sinks.size());
    LLOYAL_LOG_DEBUG("[kv::clear_and_reseed] Re-decoding %zu tail tokens at position %d",
                     tail.size(), tail_start_pos);
    lloyal::decoder::decode_tokens(ctx, tail, tail_start_pos, n_batch);
  }

  // Verify final state
  llama_pos max_pos_after = llama_memory_seq_pos_max(mem, 0);
  int32_t expected_pos = static_cast<int32_t>(original_sinks.size() + tail.size()) - 1;

  LLOYAL_LOG_DEBUG("[kv::clear_and_reseed] After reseed: KV cache max_pos=%d (expected %d)",
                   max_pos_after, expected_pos);

  if (max_pos_after != expected_pos) {
    LLOYAL_LOG_DEBUG("[kv::clear_and_reseed] WARNING: Unexpected final position (got %d, expected %d)",
                     max_pos_after, expected_pos);
  }

  LLOYAL_LOG_DEBUG("[kv::clear_and_reseed] Reseed complete");
}

// ===== FILE PERSISTENCE =====

/**
 * @brief Data structure returned by read_file
 *
 * Contains tokens and metadata restored from KV cache file.
 */
struct FileData {
  std::vector<llama_token> tokens; ///< Tokens restored from file
  size_t bytes_read;               ///< Total bytes read from file
};

/**
 * @brief Write KV state to file with self-describing format
 *
 * Serializes KV cache state to file using llama.cpp's standard format:
 * - Magic + Version (validation)
 * - Token count + Token array
 * - KV state data (cache + logits + embeddings)
 *
 * @param ctx Llama context (must not be null)
 * @param seq Sequence ID (use 0 for single-sequence mode)
 * @param filepath Destination file path (must not be empty)
 * @param tokens Token IDs to include in file
 * @return Bytes written, or 0 on failure
 *
 * @note Use cases:
 *       - Exit/resume: Save app state across restarts
 *       - Persistent sessions: Multiple save points per conversation
 *       - Context sharing: Serialize → upload → share
 *
 * @warning Skips write if KV cache is empty (returns 0)
 */
inline size_t write_file(llama_context *ctx, llama_seq_id seq,
                         const std::string &filepath,
                         const std::vector<llama_token> &tokens) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[kv::write_file] ERROR: null context");
    return 0;
  }

  if (filepath.empty()) {
    LLOYAL_LOG_DEBUG("[kv::write_file] ERROR: empty filepath");
    return 0;
  }

  // Guard: Don't write if KV cache is empty
  llama_memory_t mem = llama_get_memory(ctx);
  llama_pos max_pos = llama_memory_seq_pos_max(mem, seq);
  if (max_pos < 0) {
    LLOYAL_LOG_DEBUG(
        "[kv::write_file] WARNING: KV cache is empty - skipping write");
    return 0;
  }

  // Delegate to llama.cpp's session file writer
  // Note: llama.cpp signature is (ctx, filepath, seq_id, tokens, n_tokens)
  size_t bytes = llama_state_seq_save_file(ctx, filepath.c_str(), seq,
                                            tokens.data(), tokens.size());

  if (bytes > 0) {
    LLOYAL_LOG_DEBUG("[kv::write_file] Wrote %s: %zu bytes (%.1f MB), %zu "
                     "tokens",
                     filepath.c_str(), bytes, bytes / 1024.0 / 1024.0,
                     tokens.size());
  } else {
    LLOYAL_LOG_DEBUG("[kv::write_file] FAILED to write %s", filepath.c_str());
  }

  return bytes;
}

/**
 * @brief Read KV state from file
 *
 * Deserializes KV cache state from file and restores it to the sequence.
 * Validates magic + version automatically. Returns structured data with
 * restored tokens and metadata.
 *
 * @param ctx Llama context (must not be null)
 * @param seq Sequence ID (use 0 for single-sequence mode)
 * @param filepath Source file path (must not be empty)
 * @return FileData with tokens and bytes_read
 * @throws std::runtime_error if validation fails or file doesn't exist
 *
 * @note KV cache is automatically restored during load
 *       Use data.tokens for reconstruction/validation
 *
 * @example
 *   auto data = lloyal::kv::read_file(ctx, 0, "app_state.llama");
 *   // KV cache restored, tokens available in data.tokens
 */
inline FileData read_file(llama_context *ctx, llama_seq_id seq,
                          const std::string &filepath) {
  if (!ctx) {
    throw std::runtime_error("[kv::read_file] null context");
  }

  if (filepath.empty()) {
    throw std::runtime_error("[kv::read_file] empty filepath");
  }

  // Get model's n_ctx to allocate token buffer
  const uint32_t n_ctx = llama_n_ctx(ctx);

  std::vector<llama_token> tokens;
  tokens.resize(n_ctx); // Allocate buffer with capacity

  size_t token_count = 0;
  // Note: llama.cpp signature is (ctx, filepath, seq_id, tokens_out, capacity, count_out)
  size_t bytes =
      llama_state_seq_load_file(ctx, filepath.c_str(), seq, tokens.data(),
                                 tokens.size(), &token_count);

  if (bytes == 0) {
    throw std::runtime_error("[kv::read_file] failed to load from " +
                             filepath);
  }

  tokens.resize(token_count);

  LLOYAL_LOG_DEBUG("[kv::read_file] Loaded %s: %zu bytes (%.1f MB), %zu tokens",
                   filepath.c_str(), bytes, bytes / 1024.0 / 1024.0,
                   token_count);

  return FileData{std::move(tokens), bytes};
}

} // namespace lloyal::kv
