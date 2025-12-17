#pragma once

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
 * References:
 * - Shannon entropy: https://www.emblaustralia.org/wp-content/uploads/2023/11/information_theory.pdf
 * - Perplexity: https://huggingface.co/docs/transformers/perplexity
 *
 * Ported from tsampler/metrics.ts - identical algorithms, validated implementation.
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace lloyal::metrics {

// ============================================================================
// Types
// ============================================================================

enum class Base { Nats, Bits };

using PerplexityHandle = int32_t;

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

// Perplexity state for handle-based tracking
struct PerplexityState {
  float nll_sum_nats = 0.0f;
  int count = 0;
};

inline std::unordered_map<PerplexityHandle, PerplexityState>& get_registry() {
  static std::unordered_map<PerplexityHandle, PerplexityState> registry;
  return registry;
}

inline PerplexityHandle& get_next_handle() {
  static PerplexityHandle next = 1;
  return next;
}

}  // namespace detail

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

// ============================================================================
// Handle-based RollingPerplexity (supports clone for fork)
// ============================================================================

/**
 * Create a new rolling perplexity tracker
 *
 * Perplexity = exp(average surprisal) = geometric mean of inverse probabilities
 * Lower perplexity = model is more confident about the sequence
 * Higher perplexity = model is more uncertain
 *
 * Use to monitor sequence-level quality, detect distribution drift, or gate KV eviction.
 *
 * @returns Handle to the perplexity tracker
 *
 * @example
 *   auto ppl = metrics::create_perplexity();
 *   for (auto token : tokens) {
 *     float s = metrics::model_surprisal(logits, n_vocab, token);
 *     metrics::add_surprisal(ppl, s);
 *   }
 *   float perplexity = metrics::get_ppl(ppl);
 *   if (perplexity > 50.0f) {
 *     // High perplexity - consider retrieval
 *   }
 *   metrics::free_perplexity(ppl);
 */
inline PerplexityHandle create_perplexity() {
  PerplexityHandle h = detail::get_next_handle()++;
  detail::get_registry()[h] = detail::PerplexityState{};
  return h;
}

/**
 * Add token surprisal to running average
 *
 * @param handle Perplexity tracker handle
 * @param surprisal Token surprisal in nats (from model_surprisal)
 */
inline void add_surprisal(PerplexityHandle handle, float surprisal) {
  auto& registry = detail::get_registry();
  auto it = registry.find(handle);
  if (it == registry.end()) return;
  if (!std::isfinite(surprisal)) return;
  it->second.nll_sum_nats += std::max(0.0f, surprisal);
  it->second.count++;
}

/**
 * Get current perplexity
 *
 * @param handle Perplexity tracker handle
 * @returns exp(average surprisal), Infinity if no samples
 */
inline float get_ppl(PerplexityHandle handle) {
  auto& registry = detail::get_registry();
  auto it = registry.find(handle);
  if (it == registry.end() || it->second.count == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return std::exp(it->second.nll_sum_nats / static_cast<float>(it->second.count));
}

/**
 * Get number of tokens added to tracker
 *
 * @param handle Perplexity tracker handle
 * @returns Token count, or 0 if invalid handle
 */
inline int get_count(PerplexityHandle handle) {
  auto& registry = detail::get_registry();
  auto it = registry.find(handle);
  if (it == registry.end()) return 0;
  return it->second.count;
}

/**
 * Reset tracker to initial state (start new sequence)
 *
 * @param handle Perplexity tracker handle
 */
inline void reset_perplexity(PerplexityHandle handle) {
  auto& registry = detail::get_registry();
  auto it = registry.find(handle);
  if (it != registry.end()) {
    it->second = detail::PerplexityState{};
  }
}

/**
 * Clone perplexity tracker (for fork/branching)
 *
 * Creates a new tracker with identical state. Use when forking a branch
 * to preserve perplexity history.
 *
 * @param handle Source perplexity tracker handle
 * @returns New handle with cloned state, or 0 if invalid source
 *
 * @example
 *   // Fork branch
 *   kv::seq_cp(ctx, src_seq, dst_seq);
 *   auto new_grammar = grammar::clone_sampler(grammar_handle);
 *   auto new_ppl = metrics::clone_perplexity(ppl_handle);
 */
inline PerplexityHandle clone_perplexity(PerplexityHandle handle) {
  auto& registry = detail::get_registry();
  auto it = registry.find(handle);
  if (it == registry.end()) return 0;

  PerplexityHandle new_handle = detail::get_next_handle()++;
  registry[new_handle] = it->second;  // Copy state
  return new_handle;
}

/**
 * Free perplexity tracker
 *
 * @param handle Perplexity tracker handle
 */
inline void free_perplexity(PerplexityHandle handle) {
  detail::get_registry().erase(handle);
}

// ============================================================================
// Branch-Level Metrics (unified model + sampling tracking)
// ============================================================================

using BranchMetricsHandle = int32_t;

namespace detail {

struct BranchMetricsState {
  PerplexityState model;    // Model-level (raw logits before filters)
  PerplexityState sampling; // Sampling-level (post top-k/p/temp)
};

inline std::unordered_map<BranchMetricsHandle, BranchMetricsState>&
get_branch_metrics_registry() {
  static std::unordered_map<BranchMetricsHandle, BranchMetricsState> registry;
  return registry;
}

inline BranchMetricsHandle& get_next_branch_metrics_handle() {
  static BranchMetricsHandle next = 1;
  return next;
}

}  // namespace detail

/**
 * Create unified branch metrics tracker
 *
 * Tracks both model-level (raw logits) and sampling-level (filtered) perplexity
 * in a single handle for atomic clone/free operations.
 *
 * @returns Handle to the branch metrics tracker
 */
inline BranchMetricsHandle create_branch_metrics() {
  BranchMetricsHandle h = detail::get_next_branch_metrics_handle()++;
  detail::get_branch_metrics_registry()[h] = detail::BranchMetricsState{};
  return h;
}

/**
 * Free branch metrics tracker
 *
 * @param handle Branch metrics handle
 */
inline void free_branch_metrics(BranchMetricsHandle handle) {
  detail::get_branch_metrics_registry().erase(handle);
}

/**
 * Clone branch metrics tracker (for fork/branching)
 *
 * Creates a new tracker with identical state for both model and sampling levels.
 * Use when forking a branch to preserve metrics history.
 *
 * @param handle Source branch metrics handle
 * @returns New handle with cloned state, or 0 if invalid source
 */
inline BranchMetricsHandle clone_branch_metrics(BranchMetricsHandle handle) {
  auto& registry = detail::get_branch_metrics_registry();
  auto it = registry.find(handle);
  if (it == registry.end()) return 0;

  BranchMetricsHandle new_handle = detail::get_next_branch_metrics_handle()++;
  registry[new_handle] = it->second;  // Copy both model and sampling state
  return new_handle;
}

/**
 * Add model-level surprisal (from raw logits before filters)
 *
 * @param handle Branch metrics handle
 * @param surprisal Token surprisal in nats (from model_surprisal)
 */
inline void add_model_surprisal(BranchMetricsHandle handle, float surprisal) {
  auto& registry = detail::get_branch_metrics_registry();
  auto it = registry.find(handle);
  if (it == registry.end()) return;
  if (!std::isfinite(surprisal)) return;
  it->second.model.nll_sum_nats += std::max(0.0f, surprisal);
  it->second.model.count++;
}

/**
 * Get model-level perplexity (from raw logits)
 *
 * @param handle Branch metrics handle
 * @returns exp(average surprisal), Infinity if no samples
 */
inline float get_model_ppl(BranchMetricsHandle handle) {
  auto& registry = detail::get_branch_metrics_registry();
  auto it = registry.find(handle);
  if (it == registry.end() || it->second.model.count == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return std::exp(it->second.model.nll_sum_nats /
                  static_cast<float>(it->second.model.count));
}

/**
 * Add sampling-level surprisal (from filtered distribution)
 *
 * @param handle Branch metrics handle
 * @param surprisal Token surprisal in nats (from sampling_surprisal)
 */
inline void add_sampling_surprisal(BranchMetricsHandle handle, float surprisal) {
  auto& registry = detail::get_branch_metrics_registry();
  auto it = registry.find(handle);
  if (it == registry.end()) return;
  if (!std::isfinite(surprisal)) return;
  it->second.sampling.nll_sum_nats += std::max(0.0f, surprisal);
  it->second.sampling.count++;
}

/**
 * Get sampling-level perplexity (from filtered distribution)
 *
 * @param handle Branch metrics handle
 * @returns exp(average surprisal), Infinity if no samples
 */
inline float get_sampling_ppl(BranchMetricsHandle handle) {
  auto& registry = detail::get_branch_metrics_registry();
  auto it = registry.find(handle);
  if (it == registry.end() || it->second.sampling.count == 0) {
    return std::numeric_limits<float>::infinity();
  }
  return std::exp(it->second.sampling.nll_sum_nats /
                  static_cast<float>(it->second.sampling.count));
}

/**
 * Get number of tokens in model-level tracker
 */
inline int get_model_count(BranchMetricsHandle handle) {
  auto& registry = detail::get_branch_metrics_registry();
  auto it = registry.find(handle);
  if (it == registry.end()) return 0;
  return it->second.model.count;
}

/**
 * Get number of tokens in sampling-level tracker
 */
inline int get_sampling_count(BranchMetricsHandle handle) {
  auto& registry = detail::get_branch_metrics_registry();
  auto it = registry.find(handle);
  if (it == registry.end()) return 0;
  return it->second.sampling.count;
}

}  // namespace lloyal::metrics
