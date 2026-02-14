#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

#include "common.hpp"
#include <common.h>  // llama.cpp common library: common_batch_clear, common_batch_add
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <llama/llama.h>
#include <stdexcept>
#include <vector>

/**
 * @file embedding.hpp
 * @brief Embedding Extraction and Normalization
 *
 * Wraps llama.cpp embedding APIs with pooling mode management and L2 normalization.
 * Provides both context-bound extraction and model capability checks.
 *
 * Architecture:
 * - Context-bound primitives for embedding extraction
 * - Model-accepting overloads for capability checks
 * - Built-in L2 normalization for cosine similarity
 *
 * @example
 *   // Check model supports embeddings
 *   if (embedding::has_embeddings(model)) {
 *     int32_t dim = embedding::dimension(model);
 *
 *     // Decode tokens with pooling enabled
 *     decode::many(ctx, tokens, 0, 512);
 *
 *     // Extract normalized embeddings
 *     auto vec = embedding::get(ctx, embedding::Normalize::L2);
 *   }
 */

namespace lloyal::embedding {

// ===== NORMALIZATION MODES =====

/**
 * Normalization modes for embedding vectors
 */
enum class Normalize : int32_t {
  None = 0, // No normalization (raw embeddings)
  L2 = 1,   // L2 normalization (unit length, required for cosine similarity)
};

// ===== MODEL CAPABILITY CHECKS =====

/**
 * Check if model supports embeddings
 *
 * @param model Llama model
 * @return true if model has non-zero embedding dimension
 *
 * NOTE: This checks dimension only. For proper embeddings, the context
 * must also be created with pooling enabled (LLAMA_POOLING_TYPE_MEAN, etc.)
 */
inline bool has_embeddings(const llama_model *model) {
  if (!model) {
    LLOYAL_LOG_DEBUG("[embedding::has_embeddings] ERROR: model is null");
    return false;
  }

  int32_t n_embd = llama_model_n_embd(model);
  return n_embd > 0;
}

/**
 * Get embedding dimension for model
 *
 * @param model Llama model
 * @return Embedding dimension (e.g., 384, 768, 1024, 4096)
 */
inline int32_t dimension(const llama_model *model) {
  if (!model) {
    LLOYAL_LOG_DEBUG("[embedding::dimension] ERROR: model is null");
    return 0;
  }

  return llama_model_n_embd(model);
}

// ===== CONTEXT CAPABILITY CHECKS =====

/**
 * Check if context has pooling enabled
 *
 * @param ctx Llama context
 * @return true if pooling is enabled (required for embeddings)
 *
 * NOTE: Context must be created with pooling type != LLAMA_POOLING_TYPE_NONE
 * for embeddings to work correctly.
 */
inline bool has_pooling(llama_context *ctx) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[embedding::has_pooling] ERROR: ctx is null");
    return false;
  }

  return llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE;
}

/**
 * Get pooling type for context
 *
 * @param ctx Llama context
 * @return Pooling type enum value
 *
 * Types:
 * - LLAMA_POOLING_TYPE_NONE (0): No pooling
 * - LLAMA_POOLING_TYPE_MEAN (1): Mean pooling (most common)
 * - LLAMA_POOLING_TYPE_CLS (2): CLS token pooling
 * - LLAMA_POOLING_TYPE_LAST (3): Last token pooling
 */
inline int32_t pooling_type(llama_context *ctx) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[embedding::pooling_type] ERROR: ctx is null");
    return LLAMA_POOLING_TYPE_NONE;
  }

  return llama_pooling_type(ctx);
}

// ===== INTERNAL HELPERS =====

namespace detail {

/**
 * Apply L2 normalization to embedding vector (in-place)
 *
 * L2 normalization produces unit vectors required for cosine similarity:
 *   cosine_sim(a, b) = dot(normalize(a), normalize(b))
 */
inline void apply_l2_normalize(std::vector<float> &vec) {
  if (vec.empty())
    return;

  float norm_sq = 0.0f;
  for (float v : vec) {
    norm_sq += v * v;
  }

  float norm = std::sqrt(norm_sq);
  if (norm > 1e-8f) { // Avoid division by zero
    for (float &v : vec) {
      v /= norm;
    }
  } else {
    LLOYAL_LOG_DEBUG(
        "[embedding::detail::apply_l2_normalize] WARNING: near-zero norm");
  }
}

} // namespace detail

// ===== RAII GUARD FOR BATCH CLEANUP =====

namespace detail {
/**
 * RAII guard for automatic batch cleanup
 * Ensures llama_batch_free is called even if exceptions occur
 */
struct BatchGuard {
  llama_batch &batch;
  explicit BatchGuard(llama_batch &b) : batch(b) {}
  ~BatchGuard() { llama_batch_free(batch); }
};
} // namespace detail

// ===== ENCODING (FORWARD PASS FOR EMBEDDINGS) =====

/**
 * Encode tokens for embedding extraction
 *
 * Unlike decode::many(), this marks ALL tokens with logits=true which is
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
 *   embedding::encode(embed_ctx, tokens, 512);
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
  LLOYAL_LOG_DEBUG("[embedding::encode] Encoding %d tokens for embeddings",
                   n_tokens);

  if (!ctx) {
    LLOYAL_LOG_DEBUG("[embedding::encode] ERROR: NULL context");
    throw std::runtime_error("embedding::encode - NULL context");
  }

  if (!tokens || n_tokens <= 0) {
    LLOYAL_LOG_DEBUG("[embedding::encode] ERROR: Invalid token array");
    throw std::runtime_error("embedding::encode - Invalid token array");
  }

  if (n_tokens > n_batch) {
    LLOYAL_LOG_DEBUG("[embedding::encode] ERROR: n_tokens (%d) > n_batch (%d)",
                     n_tokens, n_batch);
    throw std::runtime_error(
        "embedding::encode - token count exceeds batch size (truncation not "
        "supported, increase n_batch or reduce input length)");
  }

  // Initialize batch - single sequence
  llama_batch batch = llama_batch_init(n_batch, 0, 1);
  detail::BatchGuard batch_guard(batch);

  // Clear batch using llama.cpp common library
  common_batch_clear(batch);

  // Add ALL tokens with logits=true (required for embedding extraction)
  for (int32_t i = 0; i < n_tokens; ++i) {
    common_batch_add(batch, tokens[i], i, {0}, true);
  }

  // Decode/encode the batch (llama.cpp handles encoder vs decoder internally)
  if (llama_decode(ctx, batch) != 0) {
    LLOYAL_LOG_DEBUG("[embedding::encode] ERROR: llama_decode failed");
    throw std::runtime_error("embedding::encode - llama_decode failed");
  }

  LLOYAL_LOG_DEBUG("[embedding::encode] Encode complete");
}

/**
 * Convenience overload for std::vector<llama_token>
 */
inline void encode(llama_context *ctx, const std::vector<llama_token> &tokens,
                   int32_t n_batch) {
  encode(ctx, tokens.data(), static_cast<int32_t>(tokens.size()), n_batch);
}

// ===== EMBEDDING EXTRACTION =====

/**
 * Get embeddings for last decoded batch
 *
 * @param ctx Llama context (must have pooling enabled)
 * @param normalize Normalization mode (default: L2 for cosine similarity)
 * @return Embedding vector (size = embedding dimension)
 * @throws std::runtime_error if extraction fails
 *
 * REQUIRES: Previous llama_decode() call with tokens
 *
 * EXAMPLE:
 *   auto tokens = tokenizer::tokenize(model, "Hello world");
 *   decode::many(ctx, tokens, 0, 512);
 *   auto embedding = embedding::get(ctx, Normalize::L2);
 */
inline std::vector<float> get(llama_context *ctx,
                              Normalize normalize = Normalize::L2) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[embedding::get] ERROR: ctx is null");
    throw std::invalid_argument("embedding::get: ctx is null");
  }

  // Get model to determine embedding dimension
  const llama_model *model = llama_get_model(ctx);
  if (!model) {
    LLOYAL_LOG_DEBUG("[embedding::get] ERROR: failed to get model from context");
    throw std::runtime_error("embedding::get: failed to get model");
  }

  // Warn if pooling not enabled (embeddings may be invalid)
  if (!has_pooling(ctx)) {
    LLOYAL_LOG_DEBUG(
        "[embedding::get] WARNING: pooling not enabled, embeddings may be "
        "invalid. Create context with pooling_type != NONE");
  }

  // Get embeddings pointer from llama.cpp
  // For pooled embeddings, use sequence-specific API (sequence 0)
  const float *embd_ptr = nullptr;
  if (has_pooling(ctx)) {
    embd_ptr = llama_get_embeddings_seq(ctx, 0);
    LLOYAL_LOG_DEBUG("[embedding::get] Using llama_get_embeddings_seq for pooled "
                     "embeddings");
  } else {
    embd_ptr = llama_get_embeddings(ctx);
    LLOYAL_LOG_DEBUG("[embedding::get] Using llama_get_embeddings (no pooling)");
  }

  if (!embd_ptr) {
    LLOYAL_LOG_DEBUG("[embedding::get] ERROR: embeddings pointer is null. "
                     "Ensure context was created with embeddings=true and "
                     "tokens were encoded with logits=true for all tokens.");
    throw std::runtime_error(
        "embedding::get: embeddings unavailable (ensure embeddings=true in "
        "context params and use encode_for_embeddings())");
  }

  // Copy to vector
  int32_t n_embd = llama_model_n_embd(model);
  std::vector<float> embeddings(embd_ptr, embd_ptr + n_embd);

  // Apply normalization
  if (normalize == Normalize::L2) {
    detail::apply_l2_normalize(embeddings);
  }

  LLOYAL_LOG_DEBUG("[embedding::get] Extracted embeddings (dim=%d, normalize=%d)",
                   n_embd, static_cast<int>(normalize));

  return embeddings;
}

/**
 * Get embeddings for specific sequence
 *
 * @param ctx Llama context
 * @param seq Sequence ID
 * @param normalize Normalization mode
 * @return Embedding vector
 * @throws std::runtime_error if extraction fails
 *
 * USE CASE: Multi-sequence embedding extraction (batch embedding different texts)
 *
 * NOTE: Falls back to get() for seq=0 if sequence-specific API unavailable
 */
inline std::vector<float> get_seq(llama_context *ctx, llama_seq_id seq,
                                  Normalize normalize = Normalize::L2) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[embedding::get_seq] ERROR: ctx is null");
    throw std::invalid_argument("embedding::get_seq: ctx is null");
  }

  const llama_model *model = llama_get_model(ctx);
  if (!model) {
    LLOYAL_LOG_DEBUG("[embedding::get_seq] ERROR: failed to get model");
    throw std::runtime_error("embedding::get_seq: failed to get model");
  }

  if (!has_pooling(ctx)) {
    LLOYAL_LOG_DEBUG("[embedding::get_seq] WARNING: pooling not enabled");
  }

  // Try sequence-specific API
  const float *embd_ptr = llama_get_embeddings_seq(ctx, seq);

  // Fallback to global embeddings for seq=0
  if (!embd_ptr) {
    if (seq == 0) {
      LLOYAL_LOG_DEBUG("[embedding::get_seq] Falling back to get() for seq=0");
      return get(ctx, normalize);
    }
    LLOYAL_LOG_DEBUG("[embedding::get_seq] ERROR: embeddings unavailable for "
                     "seq=%d",
                     seq);
    throw std::runtime_error("embedding::get_seq: embeddings unavailable");
  }

  int32_t n_embd = llama_model_n_embd(model);
  std::vector<float> embeddings(embd_ptr, embd_ptr + n_embd);

  if (normalize == Normalize::L2) {
    detail::apply_l2_normalize(embeddings);
  }

  LLOYAL_LOG_DEBUG("[embedding::get_seq] Extracted embeddings for seq=%d "
                   "(dim=%d)",
                   seq, n_embd);

  return embeddings;
}

/**
 * Get embeddings for specific token index in last batch
 *
 * @param ctx Llama context
 * @param idx Token index in batch
 * @param normalize Normalization mode
 * @return Embedding vector
 * @throws std::runtime_error if extraction fails
 *
 * USE CASE: Per-token embeddings for token-level analysis, kNN-LM
 *
 * NOTE: Per-token embeddings may work without pooling enabled
 */
inline std::vector<float> get_ith(llama_context *ctx, int32_t idx,
                                  Normalize normalize = Normalize::L2) {
  if (!ctx) {
    LLOYAL_LOG_DEBUG("[embedding::get_ith] ERROR: ctx is null");
    throw std::invalid_argument("embedding::get_ith: ctx is null");
  }

  const llama_model *model = llama_get_model(ctx);
  if (!model) {
    LLOYAL_LOG_DEBUG("[embedding::get_ith] ERROR: failed to get model");
    throw std::runtime_error("embedding::get_ith: failed to get model");
  }

  const float *embd_ptr = llama_get_embeddings_ith(ctx, idx);
  if (!embd_ptr) {
    LLOYAL_LOG_DEBUG("[embedding::get_ith] ERROR: embeddings unavailable for "
                     "idx=%d",
                     idx);
    throw std::runtime_error("embedding::get_ith: embeddings unavailable");
  }

  int32_t n_embd = llama_model_n_embd(model);
  std::vector<float> embeddings(embd_ptr, embd_ptr + n_embd);

  if (normalize == Normalize::L2) {
    detail::apply_l2_normalize(embeddings);
  }

  LLOYAL_LOG_DEBUG("[embedding::get_ith] Extracted embeddings for idx=%d "
                   "(dim=%d)",
                   idx, n_embd);

  return embeddings;
}

// ===== SIMILARITY =====

/**
 * Compute cosine similarity between two embedding vectors
 *
 * @param a First embedding vector (should be L2-normalized)
 * @param b Second embedding vector (should be L2-normalized)
 * @return Cosine similarity in range [-1, 1]
 *
 * NOTE: For normalized vectors, cosine similarity = dot product
 *
 * EXAMPLE:
 *   auto emb1 = embedding::get(ctx1, Normalize::L2);
 *   auto emb2 = embedding::get(ctx2, Normalize::L2);
 *   float sim = embedding::cosine_similarity(emb1, emb2);
 */
inline float cosine_similarity(const std::vector<float> &a,
                               const std::vector<float> &b) {
  if (a.size() != b.size()) {
    LLOYAL_LOG_DEBUG("[embedding::cosine_similarity] ERROR: dimension mismatch "
                     "(%zu vs %zu)",
                     a.size(), b.size());
    throw std::invalid_argument(
        "embedding::cosine_similarity: dimension mismatch");
  }

  if (a.empty()) {
    return 0.0f;
  }

  // For L2-normalized vectors, cosine similarity = dot product
  float dot = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
  }

  return dot;
}

} // namespace lloyal::embedding
