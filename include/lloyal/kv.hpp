#pragma once

#include "common.hpp"
#include <cstdint>
#include <llama/llama.h>

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

} // namespace lloyal::kv
