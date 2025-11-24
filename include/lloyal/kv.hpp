#pragma once

#include "common.hpp"
#include "decoder.hpp"
#include <cstdint>
#include <llama/llama.h>
#include <vector>

/**
 * KV Cache Anti-Corruption Layer (Header-Only)
 *
 * Purpose: Handles API name churn across llama.cpp versions.
 * Pinned version: commit b6870 (llama_memory_seq_* API naming)
 */

namespace lloyal::kv {

// ===== KV SEQUENCE OPERATIONS =====

/**
 * Remove token range from KV cache sequence.
 *
 * @param ctx llama context
 * @param seq sequence ID (use 0 for single-sequence mode)
 * @param p0 start position (inclusive)
 * @param p1 end position (exclusive), use -1 for "to end"
 * @return true if successful, false otherwise
 *
 * CRITICAL: Call this BEFORE next llama_decode(), not after.
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
 * Get maximum position in KV cache sequence.
 *
 * @param ctx llama context
 * @param seq sequence ID
 * @return maximum position (number of tokens - 1), or -1 if empty
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

// ===== STATE SNAPSHOT OPERATIONS (with fragmentation fallback) =====

/**
 * Get size needed to serialize sequence state.
 * Automatically falls back to global state size if per-sequence fails.
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
 * Save sequence state to buffer.
 * Automatically falls back to global state save if per-sequence fails.
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
 * Restore sequence state from buffer.
 * Automatically falls back to global state restore if per-sequence fails.
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

// ===== GLOBAL STATE FALLBACKS (explicit) =====

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

// ===== CACHE CLEARING (PHASE 3) =====

/**
 * Clear all KV cache (complete reset)
 *
 * Wrapper around llama_memory_clear() with:
 * - Null checking
 * - Error logging
 * - Clears both metadata and data buffers
 *
 * @param ctx Llama context (must be initialized)
 * @throws std::runtime_error if ctx is NULL
 *
 * USAGE:
 *   lloyal::kv::clear_all(ctx);  // Fresh start for new conversation
 *
 * IMPLEMENTATION NOTE:
 * Uses llama_memory_clear(mem, true) which:
 * - Clears metadata (cell positions, sequence heads)
 * - Zeroes K/V tensor data buffers
 * - Full reset for new conversation
 *
 * Compare with clear_metadata():
 * - clear_metadata() clears only metadata (keeps allocations, faster)
 * - clear_all() clears both metadata and data (complete reset)
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
 * Clear KV cache metadata only (fast reset)
 *
 * Clears logical structure but keeps buffer allocations.
 * Faster than clear_all() for StreamingLLM pattern.
 *
 * @param ctx Llama context (must be initialized)
 * @throws std::runtime_error if ctx is NULL
 *
 * USAGE:
 *   lloyal::kv::clear_metadata(ctx);  // Fast reset for reseed
 *
 * PERFORMANCE:
 * - Faster than clear_all() (no buffer zeroing)
 * - Use for StreamingLLM when immediately re-decoding
 * - Buffer reuse reduces allocation overhead
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

// ===== STREAMINGLLM SUPPORT (PHASE 3) =====

/**
 * StreamingLLM state for managing original sinks across reseeds
 *
 * StreamingLLM pattern requires ALWAYS reusing the ORIGINAL first 4 tokens
 * from conversation start as "attention sinks". This struct helps track them.
 *
 * NOTE: This is provided for convenience. Callers can also track original
 * sinks themselves and pass directly to clear_and_reseed().
 */
struct StreamingLlmState {
  std::vector<llama_token> original_sinks;  // First N tokens from conversation start
  size_t tail_size;                          // Number of recent tokens to keep (usually 252)
};

/**
 * Clear KV cache and re-decode sinks + tail (StreamingLLM pattern)
 *
 * Implements the "CLEAR" strategy validated in integration tests:
 * 1. Clear entire KV cache using llama_memory_clear()
 * 2. Re-decode original_sinks (first N tokens) at position 0
 * 3. Re-decode tail (last M tokens) at position sinks.size()
 *
 * This is SIMPLER and MORE RELIABLE than selective removal (llama_memory_seq_rm)
 * which has known bugs with position handling in some llama.cpp versions.
 *
 * ⚠️  CRITICAL: original_sinks MUST be the FIRST tokens from conversation start!
 *
 * StreamingLLM relies on attention sinks at fixed positions. Using different
 * "first 4" tokens after each reseed will violate the learned positional bias
 * and destroy perplexity preservation.
 *
 * CORRECT usage:
 *   // First time: Capture original sinks
 *   std::vector<llama_token> ORIGINAL_SINKS(conversation.begin(), conversation.begin() + 4);
 *   // Store ORIGINAL_SINKS for entire session
 *
 *   // Each reseed: Reuse SAME original sinks
 *   auto tail = std::vector<llama_token>(conversation.end() - 252, conversation.end());
 *   kv::clear_and_reseed(ctx, ORIGINAL_SINKS, tail, n_batch);
 *
 * WRONG usage:
 *   auto current_window = get_current_tokens();
 *   auto sinks = std::vector<llama_token>(current_window.begin(), current_window.begin() + 4);
 *   kv::clear_and_reseed(ctx, sinks, tail, n_batch);  // ❌ NOT original! Will degrade!
 *
 * @param ctx Llama context (must be initialized)
 * @param original_sinks MUST be first N tokens from conversation start (typically 4)
 * @param tail Recent M tokens to preserve (typically 252, total 256 with sinks)
 * @param n_batch Batch size for re-decoding chunks
 * @throws std::runtime_error if parameters are invalid or re-decode fails
 *
 * Empirical validation: Preserves perplexity within 10% (StreamingLLM paper: 3.7%)
 * See tests/integration/clear_and_reseed_validation.cpp for full validation.
 *
 * IMPORTANT: After calling, KV cache position = sinks.size() + tail.size()
 * Continue generation with n_past = static_cast<int32_t>(sinks.size() + tail.size())
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

// ===== FILE PERSISTENCE OPERATIONS =====

/**
 * FileData structure returned by read_file
 * Contains tokens and metadata from file
 */
struct FileData {
  std::vector<llama_token> tokens; // Tokens restored from file
  size_t bytes_read;               // Total bytes read from file
};

/**
 * Write KV state to file with self-describing format
 *
 * File format (llama.cpp standard):
 * - Magic + Version (validation)
 * - Token count + Token array
 * - KV state data (cache + logits + embeddings)
 *
 * @param ctx llama context
 * @param seq sequence ID (use 0 for single-sequence mode)
 * @param filepath Destination file path
 * @param tokens Token IDs to include in file
 * @return bytes written, or 0 on failure
 *
 * Use cases:
 * - Exit/resume app: kv::write_file(ctx, 0, "app_state.llama", tokens)
 * - Persistent pages: kv::write_file(ctx, 0, "fork_42.llama", fork_tokens)
 * - Context sharing: Write → upload to S3 → share signed URL
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
 * Read KV state from file
 *
 * Validates magic + version automatically.
 * Returns structured data (no output parameters).
 *
 * @param ctx llama context
 * @param seq sequence ID (use 0 for single-sequence mode)
 * @param filepath Source file path
 * @return FileData with tokens and bytes_read
 * @throws std::runtime_error if validation fails or file doesn't exist
 *
 * Example usage:
 * ```cpp
 * auto data = lloyal::kv::read_file(ctx, 0, "app_state.llama");
 * // Use data.tokens for reconstruction/validation
 * // KV cache is automatically restored
 * ```
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
