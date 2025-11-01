#pragma once

#include "common.hpp"
#include <llama/llama.h>
#include <cstdint>
#include <stdexcept>
#include <ctime>
#include <vector>
#include <optional>
#include <type_traits>

/**
 * Sampler Anti-Corruption Layer (Header-Only)
 *
 * Purpose: Single point of contact with llama.cpp sampling APIs to isolate
 * sampling strategy complexity and enable future extensions.
 *
 * Uses C++20 concept-constrained templates to accept any shell's Nitrogen-generated
 * SamplingParams type without requiring struct duplication or adapters.
 */

namespace lloyal::detail {

// ===== OPTIONAL VALUE EXTRACTION =====

/**
 * Type trait to detect std::optional<T>
 */
template<class T>
struct is_optional : std::false_type {};

template<class T>
struct is_optional<std::optional<T>> : std::true_type {};

/**
 * Extract value from either T or std::optional<T> with fallback
 *
 * Handles both:
 * - Direct values (T): cast to target type
 * - Optional values (std::optional<T>): unwrap with value_or(default)
 *
 * Used by sampler to accept Nitrogen-generated params (which use std::optional)
 */
template<class X, class T>
constexpr T as_value(const X& x, T def) {
  if constexpr (is_optional<X>::value) {
    return x.value_or(def);
  } else {
    return static_cast<T>(x);
  }
}

} // namespace lloyal::detail

namespace lloyal {

// ===== SAMPLING PARAMS CONCEPT =====

/**
 * C++20 concept: Any type with sampling parameter fields
 *
 * Allows template to accept any shell's Nitrogen-generated SamplingParams:
 * - margelo::nitro::calibratendk::SamplingParams
 * - margelo::nitro::nitrollama::SamplingParams
 * - Or any other conforming type
 *
 * Fields can be either T or std::optional<T>
 */
template<class P>
concept SamplingParamsLike = requires(const P& p) {
  p.temperature;
  p.top_k;
  p.top_p;
  p.typical_p;
  p.min_p;
  p.penalty_repeat;
  p.penalty_freq;
  p.penalty_present;
  p.penalty_last_n;
  p.seed;
  // Additional fields for future extensions:
  // p.mirostat, p.mirostat_tau, p.mirostat_eta
  // p.dry_multiplier, p.dry_base, p.dry_allowed_length, p.dry_penalty_last_n
  // p.xtc_probability, p.xtc_threshold
  // p.top_n_sigma
};

namespace sampler {

// ===== GREEDY SAMPLING =====

/**
 * Greedy sampling: Select token with highest probability
 *
 * Uses llama_get_logits_ith(-1) to get last-step logits (requires logits=true
 * in batch for that position). Performs argmax to find best token.
 *
 * @param ctx Llama context (must have decoded at least one token with logits=true)
 * @param vocab Vocabulary for size information
 * @return Token ID with highest probability
 * @throws std::runtime_error if logits retrieval fails
 *
 * IMPORTANT: Only works if decode batch had logits=true for last token.
 * Decoder layer automatically sets this correctly.
 */
inline llama_token greedy(llama_context* ctx, const llama_vocab* vocab) {
  LLOYAL_LOG_DEBUG("[sampler::greedy] Sampling next token");

  if (!ctx) {
    LLOYAL_LOG_DEBUG("[sampler::greedy] ERROR: NULL context");
    throw std::runtime_error("sampler::greedy - NULL context");
  }

  if (!vocab) {
    LLOYAL_LOG_DEBUG("[sampler::greedy] ERROR: NULL vocabulary");
    throw std::runtime_error("sampler::greedy - NULL vocabulary");
  }

  // Get last-step logits (index -1)
  // Per llama.cpp maintainers: only works if logits=true was set for that step in batch
  const float* logits = llama_get_logits_ith(ctx, -1);
  if (!logits) {
    LLOYAL_LOG_DEBUG("[sampler::greedy] ERROR: Failed to get logits (ensure batch had logits=true)");
    throw std::runtime_error("sampler::greedy - Failed to get logits");
  }

  // Get vocabulary size
  const int n_vocab = llama_vocab_n_tokens(vocab);
  if (n_vocab <= 0) {
    LLOYAL_LOG_DEBUG("[sampler::greedy] ERROR: Invalid vocabulary size: %d", n_vocab);
    throw std::runtime_error("sampler::greedy - Invalid vocabulary size");
  }

  // Argmax: Find token with highest probability
  int best_id = 0;
  float best_score = logits[0];
  for (int i = 1; i < n_vocab; ++i) {
    if (logits[i] > best_score) {
      best_score = logits[i];
      best_id = i;
    }
  }

  llama_token result = static_cast<llama_token>(best_id);
  LLOYAL_LOG_DEBUG("[sampler::greedy] Sampled token: %d (score: %.4f)", result, best_score);

  return result;
}

// ===== PARAMETERIZED SAMPLING =====

/**
 * Sample with configurable parameters (template accepts any SamplingParams type)
 *
 * Supports full range of llama.cpp sampling strategies:
 * - Temperature scaling
 * - Top-k, top-p, min-p filtering
 * - Repetition penalties (frequency, presence, repeat)
 * - Grammar constraints (via persistent grammar sampler)
 *
 * @param ctx Llama context (must have decoded at least one token with logits=true)
 * @param vocab Vocabulary for token information
 * @param params Sampling parameters (any type matching SamplingParamsLike concept)
 * @param grammarSampler Optional persistent grammar sampler (managed by caller)
 * @return Sampled token ID
 * @throws std::runtime_error if sampling fails
 *
 * TEMPLATE INSTANTIATION:
 * - calibrate-ndk: instantiates for margelo::nitro::calibratendk::SamplingParams
 * - nitro-llama: instantiates for margelo::nitro::nitrollama::SamplingParams
 * - No adapters needed, works via duck typing + concept constraint
 */
template<SamplingParamsLike P>
inline llama_token sample_with_params(
  llama_context* ctx,
  const llama_vocab* vocab,
  const P& params,
  llama_sampler* grammarSampler = nullptr
) {
  using detail::as_value;

  // Extract parameters with defaults (handles both T and std::optional<T>)
  uint32_t seed = as_value(params.seed, static_cast<uint32_t>(std::time(nullptr)));
  int32_t top_k = as_value(params.top_k, static_cast<int32_t>(40));
  float top_p = as_value(params.top_p, 0.95f);
  float min_p = as_value(params.min_p, 0.05f);
  float typical_p = as_value(params.typical_p, 1.0f);
  float temperature = as_value(params.temperature, 0.8f);
  int32_t penalty_last_n = as_value(params.penalty_last_n, static_cast<int32_t>(64));
  float penalty_repeat = as_value(params.penalty_repeat, 1.0f);
  float penalty_freq = as_value(params.penalty_freq, 0.0f);
  float penalty_present = as_value(params.penalty_present, 0.0f);

  LLOYAL_LOG_DEBUG("[sampler::sample_with_params] Building sampler");
  LLOYAL_LOG_DEBUG("[sampler::sample_with_params]   temperature=%.2f, top_k=%d, top_p=%.2f, min_p=%.2f",
           temperature, static_cast<int>(top_k), top_p, min_p);

  if (!ctx) {
    throw std::runtime_error("sampler::sample_with_params - NULL context");
  }
  if (!vocab) {
    throw std::runtime_error("sampler::sample_with_params - NULL vocabulary");
  }

  // ROUTING DECISION: Grammar present → use grammar-aware sampling
  //                   No grammar → use lightweight chain approach
  if (grammarSampler) {
    LLOYAL_LOG_DEBUG("[sampler::sample_with_params] Grammar sampler provided, using grammar-constrained sampling");

    // Get logits and build token data array
    const float* logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
      throw std::runtime_error("sampler::sample_with_params - Failed to get logits");
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
      throw std::runtime_error("sampler::sample_with_params - Invalid vocabulary size");
    }

    // Build candidate array from logits
    std::vector<llama_token_data> candidates(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
      candidates[i] = llama_token_data{
        static_cast<llama_token>(i),
        logits[i],
        0.0f
      };
    }

    llama_token_data_array cur_p = {
      candidates.data(),
      static_cast<size_t>(n_vocab),
      -1,     // selected (will be set by samplers)
      false   // sorted
    };

    // Build sampler chain (WITHOUT grammar - grammar applied separately)
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    chain_params.no_perf = true;
    auto* chain = llama_sampler_chain_init(chain_params);

    // Add samplers in order (penalties → top-k → typical-p → top-p → min-p → temperature → dist)
    if (penalty_repeat != 1.0f || penalty_freq != 0.0f || penalty_present != 0.0f) {
      llama_sampler_chain_add(chain, llama_sampler_init_penalties(
        penalty_last_n, penalty_repeat, penalty_freq, penalty_present
      ));
    }
    if (top_k > 0) {
      llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    }
    if (typical_p < 1.0f) {
      llama_sampler_chain_add(chain, llama_sampler_init_typical(typical_p, 1));
    }
    if (top_p < 1.0f) {
      llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 1));
    }
    if (min_p > 0.0f) {
      llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 1));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(seed));

    // Apply grammar constraint FIRST (uses persistent parser state from context)
    llama_sampler_apply(grammarSampler, &cur_p);

    // Then apply chain
    llama_sampler_apply(chain, &cur_p);

    // Get selected token
    if (cur_p.selected == -1) {
      llama_sampler_free(chain);
      throw std::runtime_error("No selected token during sampling - check sampling configuration");
    }
    llama_token result = cur_p.data[cur_p.selected].id;

    // Clean up chain (grammar sampler is managed by context, not freed here)
    llama_sampler_free(chain);

    LLOYAL_LOG_DEBUG("[sampler::sample_with_params] Grammar-sampled token: %d", result);
    return result;
  }

  // NO GRAMMAR: Use lightweight chain approach
  LLOYAL_LOG_DEBUG("[sampler::sample_with_params] No grammar, using lightweight chain approach");

  // Get logits
  const float* logits = llama_get_logits_ith(ctx, -1);
  if (!logits) {
    throw std::runtime_error("sampler::sample_with_params - Failed to get logits");
  }

  const int n_vocab = llama_vocab_n_tokens(vocab);
  if (n_vocab <= 0) {
    throw std::runtime_error("sampler::sample_with_params - Invalid vocabulary size");
  }

  // Create llama.cpp sampler chain
  llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
  chain_params.no_perf = true;

  auto* sampler_chain = llama_sampler_chain_init(chain_params);
  if (!sampler_chain) {
    throw std::runtime_error("sampler::sample_with_params - Failed to create sampler chain");
  }

  LLOYAL_LOG_DEBUG("[sampler::sample_with_params] Sampler chain created, adding samplers...");

  // 1. Repetition penalties (if enabled)
  if (penalty_repeat != 1.0f || penalty_freq != 0.0f || penalty_present != 0.0f) {
    LLOYAL_LOG_DEBUG("[sampler::sample_with_params]   + penalties (repeat=%.2f, freq=%.2f, present=%.2f, last_n=%d)",
             penalty_repeat, penalty_freq, penalty_present, penalty_last_n);
    llama_sampler_chain_add(
      sampler_chain,
      llama_sampler_init_penalties(
        penalty_last_n,
        penalty_repeat,
        penalty_freq,
        penalty_present
      )
    );
  }

  // 2. Top-K sampling (if enabled)
  if (top_k > 0) {
    LLOYAL_LOG_DEBUG("[sampler::sample_with_params]   + top_k (%d)", static_cast<int>(top_k));
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_k(top_k));
  }

  // 3. Typical-P sampling (if enabled)
  if (typical_p < 1.0f) {
    LLOYAL_LOG_DEBUG("[sampler::sample_with_params]   + typical_p (%.2f)", typical_p);
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_typical(typical_p, 1));
  }

  // 4. Top-P sampling (if enabled)
  if (top_p < 1.0f) {
    LLOYAL_LOG_DEBUG("[sampler::sample_with_params]   + top_p (%.2f)", top_p);
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_top_p(top_p, 1));
  }

  // 5. Min-P sampling (if enabled)
  if (min_p > 0.0f) {
    LLOYAL_LOG_DEBUG("[sampler::sample_with_params]   + min_p (%.2f)", min_p);
    llama_sampler_chain_add(sampler_chain, llama_sampler_init_min_p(min_p, 1));
  }

  // 6. Temperature scaling
  LLOYAL_LOG_DEBUG("[sampler::sample_with_params]   + temperature (%.2f)", temperature);
  llama_sampler_chain_add(sampler_chain, llama_sampler_init_temp(temperature));

  // 7. Final distribution sampler
  LLOYAL_LOG_DEBUG("[sampler::sample_with_params]   + dist (seed=%u)", seed);
  llama_sampler_chain_add(sampler_chain, llama_sampler_init_dist(seed));

  // Sample from the chain
  llama_token result = llama_sampler_sample(sampler_chain, ctx, -1);

  // Free the sampler chain
  llama_sampler_free(sampler_chain);

  LLOYAL_LOG_DEBUG("[sampler::sample_with_params] Sampled token: %d (temp=%.2f, top_k=%d, top_p=%.2f, min_p=%.2f)",
           result, temperature, static_cast<int>(top_k), top_p, min_p);

  return result;
}

} // namespace sampler
} // namespace lloyal
