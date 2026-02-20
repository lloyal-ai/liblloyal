#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

/**
 * @file metrics.hpp
 * @brief Distribution Metrics for Test-Time Alignment
 *
 * Computes surprisal, entropy, and perplexity from logits (no attention needed).
 * All metrics derive from softmax(logits) with numerically stable log-sum-exp.
 *
 * Two measurement levels:
 * - Model metrics: Raw logits (before filters) - model's inherent belief
 * - Sampling metrics: Post-filter logits (after top-k/p/temp) - actual distribution sampled
 *
 * Use cases:
 * - KV eviction gates: High entropy -> trigger retrieval or cache pruning
 * - Adaptive sampling: Collapsed distribution -> widen search
 * - Quality monitoring: Track surprisal/perplexity for confidence estimates
 * - Dashboard signals: Real-time uncertainty visualization
 *
 * Perplexity tracking is instance-scoped via BranchStore registries —
 * no global static state. See branch.hpp for the handle-based CRUD.
 *
 * References:
 * - Shannon entropy: https://www.emblaustralia.org/wp-content/uploads/2023/11/information_theory.pdf
 * - Perplexity: https://huggingface.co/docs/transformers/perplexity
 *
 * Ported from tsampler/metrics.ts - identical algorithms, validated implementation.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace lloyal::metrics {

// ============================================================================
// Types
// ============================================================================

enum class Base { Nats, Bits };

// ============================================================================
// Internal helpers (ported from metrics.ts)
// ============================================================================

namespace detail {

constexpr float LN2 = 0.693147180559945309417232121458176568f;

/**
 * Find maximum finite value in array
 * Used for log-sum-exp shift to prevent overflow
 */
inline float max_finite(const float* a, int n) {
  float m = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i) {
    const float v = a[i];
    if (std::isfinite(v) && v > m) m = v;
  }
  return m;
}

/**
 * Numerically stable log-sum-exp
 * Computes log(Σ exp(aᵢ)) using shift trick to avoid overflow
 *
 * @param a Array of log-space values
 * @param n Array length
 * @param shift Max value for numerical stability
 * @returns log(Σ exp(aᵢ))
 */
inline float log_sum_exp(const float* a, int n, float shift) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float v = a[i];
    if (std::isfinite(v)) s += std::exp(v - shift);
  }
  if (s == 0.0f) return -std::numeric_limits<float>::infinity();
  return shift + std::log(s);
}

}  // namespace detail

// ============================================================================
// Perplexity tracking types (used by BranchStore registry)
// ============================================================================

/// Rolling NLL accumulator for perplexity computation
struct PerplexityState {
  float nll_sum_nats = 0.0f;
  int count = 0;
};

/// Unified model + sampling perplexity tracker
struct BranchMetricsState {
  PerplexityState model;    ///< Model-level (raw logits before filters)
  PerplexityState sampling; ///< Sampling-level (post top-k/p/temp)
};

// ============================================================================
// Model-level metrics (raw logits, before filters)
// ============================================================================

/**
 * Compute model-level surprisal for picked token
 *
 * Surprisal = -log p(tokenₜ | context) = uncertainty of the model's choice
 * Higher surprisal = more surprising token (lower probability)
 *
 * Use model logits (before temperature/top-k/p) to measure model's inherent uncertainty.
 *
 * @param logits Full vocabulary logits (before sampling filters)
 * @param n_vocab Vocabulary size
 * @param picked_id Token ID that was sampled
 * @param base Nats (natural log) or Bits (log₂)
 * @returns Surprisal in nats or bits (≥0, Infinity if invalid)
 *
 * @example
 *   float* logits = lloyal::logits::get(ctx);
 *   int n_vocab = llama_vocab_n_tokens(vocab);
 *   llama_token token = sample(logits);
 *   float s = metrics::model_surprisal(logits, n_vocab, token);
 *   if (s > 5.0f) {
 *     // High uncertainty - consider retrieval
 *   }
 */
inline float model_surprisal(
    const float* logits,
    int n_vocab,
    int picked_id,
    Base base = Base::Nats
) {
  if (!logits || n_vocab == 0) {
    return std::numeric_limits<float>::infinity();
  }
  if (picked_id < 0 || picked_id >= n_vocab) {
    return std::numeric_limits<float>::infinity();
  }

  const float picked = logits[picked_id];
  if (!std::isfinite(picked)) return std::numeric_limits<float>::infinity();

  const float m = detail::max_finite(logits, n_vocab);
  if (!std::isfinite(m)) return std::numeric_limits<float>::infinity();

  const float log_z = detail::log_sum_exp(logits, n_vocab, m);
  if (!std::isfinite(log_z)) return std::numeric_limits<float>::infinity();

  const float surprisal_nats = std::max(0.0f, -(picked - log_z));
  return base == Base::Bits ? surprisal_nats / detail::LN2 : surprisal_nats;
}

/**
 * Compute model-level entropy of distribution
 *
 * Entropy H = -Σₖ pₖ log pₖ = uncertainty of the next token
 * Higher entropy = flatter distribution (more uncertain)
 * Lower entropy = peaked distribution (more confident)
 *
 * Use model logits (before filters) for KV eviction gates, adaptive sampling triggers.
 *
 * @param logits Full vocabulary logits (before sampling filters)
 * @param n_vocab Vocabulary size
 * @param base Nats (natural log) or Bits (log₂)
 * @returns Entropy in nats or bits (≥0, Infinity if invalid)
 *
 * @example
 *   float* logits = lloyal::logits::get(ctx);
 *   float h = metrics::model_entropy(logits, n_vocab);
 *   if (h < 2.0f) {
 *     // Collapsed distribution -> widen search
 *   } else if (h > 5.0f) {
 *     // Too flat -> focus sampling
 *   }
 */
inline float model_entropy(
    const float* logits,
    int n_vocab,
    Base base = Base::Nats
) {
  if (!logits || n_vocab == 0) {
    return std::numeric_limits<float>::infinity();
  }

  const float m = detail::max_finite(logits, n_vocab);
  if (!std::isfinite(m)) return std::numeric_limits<float>::infinity();

  const float log_z = detail::log_sum_exp(logits, n_vocab, m);
  if (!std::isfinite(log_z)) return std::numeric_limits<float>::infinity();

  float ez = 0.0f;
  for (int i = 0; i < n_vocab; ++i) {
    const float z = logits[i];
    if (!std::isfinite(z)) continue;
    const float p = std::exp(z - log_z);
    ez += p * z;
  }

  const float h_nats = std::max(0.0f, log_z - ez);
  return base == Base::Bits ? h_nats / detail::LN2 : h_nats;
}

// ============================================================================
// Sampling-level metrics (post-filter logits, after top-k/p/temp)
// ============================================================================

/**
 * Compute sampling-level surprisal for picked token
 *
 * Measures uncertainty within the filtered candidate set (after top-k/p/temperature).
 * Lower than model surprisal if filters removed low-probability tokens.
 *
 * Use to monitor runtime hazard when grammar/constraints narrow the distribution.
 *
 * @param candidate_logits Logits of candidate tokens (post-filter)
 * @param candidate_ids Token IDs of candidates
 * @param n_candidates Number of candidates
 * @param picked_id Token ID that was sampled
 * @param base Nats (natural log) or Bits (log₂)
 * @returns Surprisal in nats or bits (≥0, Infinity if invalid)
 */
inline float sampling_surprisal(
    const float* candidate_logits,
    const int32_t* candidate_ids,
    int n_candidates,
    int picked_id,
    Base base = Base::Nats
) {
  if (!candidate_logits || !candidate_ids || n_candidates == 0) {
    return std::numeric_limits<float>::infinity();
  }

  // Find picked_id in candidates
  int local = -1;
  for (int i = 0; i < n_candidates; ++i) {
    if (candidate_ids[i] == picked_id) {
      local = i;
      break;
    }
  }
  if (local == -1) return std::numeric_limits<float>::infinity();
  if (n_candidates == 1) return 0.0f;

  const float picked = candidate_logits[local];
  if (!std::isfinite(picked)) return std::numeric_limits<float>::infinity();

  const float m = detail::max_finite(candidate_logits, n_candidates);
  if (!std::isfinite(m)) return std::numeric_limits<float>::infinity();

  const float log_z = detail::log_sum_exp(candidate_logits, n_candidates, m);
  if (!std::isfinite(log_z)) return std::numeric_limits<float>::infinity();

  const float surprisal_nats = std::max(0.0f, -(picked - log_z));
  return base == Base::Bits ? surprisal_nats / detail::LN2 : surprisal_nats;
}

/**
 * Compute sampling-level entropy of candidate distribution
 *
 * Measures uncertainty within the filtered candidate set (after top-k/p/temperature).
 * Use to monitor distribution health after grammar masks or constraints.
 *
 * @param candidate_logits Logits of candidate tokens (post-filter)
 * @param n_candidates Number of candidates
 * @param base Nats (natural log) or Bits (log₂)
 * @returns Entropy in nats or bits (≥0, Infinity if invalid)
 */
inline float sampling_entropy(
    const float* candidate_logits,
    int n_candidates,
    Base base = Base::Nats
) {
  if (!candidate_logits || n_candidates == 0) {
    return std::numeric_limits<float>::infinity();
  }
  if (n_candidates == 1) return 0.0f;

  const float m = detail::max_finite(candidate_logits, n_candidates);
  if (!std::isfinite(m)) return std::numeric_limits<float>::infinity();

  const float log_z = detail::log_sum_exp(candidate_logits, n_candidates, m);
  if (!std::isfinite(log_z)) return std::numeric_limits<float>::infinity();

  float ez = 0.0f;
  for (int i = 0; i < n_candidates; ++i) {
    const float z = candidate_logits[i];
    if (!std::isfinite(z)) continue;
    const float p = std::exp(z - log_z);
    ez += p * z;
  }

  const float h_nats = std::max(0.0f, log_z - ez);
  return base == Base::Bits ? h_nats / detail::LN2 : h_nats;
}

}  // namespace lloyal::metrics
