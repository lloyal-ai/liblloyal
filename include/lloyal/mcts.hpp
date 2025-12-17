#pragma once

#include "branch.hpp"
#include "boundaries.hpp"
#include "metrics.hpp"
#include "tokenizer.hpp"
#include "common.hpp"

#include <llama/llama.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

/**
 * MCTS (Monte Carlo Tree Search) for LLM generation
 *
 * This header provides PUCT (Predictor + UCT) tree search with:
 * - Progressive widening for proper tree expansion
 * - Tree poisoning fix (failed expansion doesn't mark parent terminal)
 * - Value normalization for competitive exploration
 *
 * Based on AlphaZero-style MCTS with per-boundary forking.
 *
 * Example usage:
 *
 *   #include <lloyal/mcts.hpp>
 *
 *   auto oracle = [](const std::string& text) -> float {
 *     return evaluate_quality(text);  // Returns raw score
 *   };
 *
 *   lloyal::mcts::SearchConfig config;
 *   config.cpuct = 1.0f;
 *   config.widening_constant = 1.0f;
 *   config.value_normalizer = lloyal::mcts::normalizers::tanh_normalizer;
 *   config.normalizer_scale = 200.0f;
 *
 *   lloyal::mcts::PUCT puct(ctx, model, grammar, 32, 256, start_pos,
 *                           config, oracle);
 *   puct.search(100);
 *
 *   auto best_tokens = puct.get_best_sequence();
 */

namespace lloyal {
namespace mcts {

// Import branch types
using branch::BranchHandle;
using branch::BranchStore;
using branch::BranchState;
constexpr BranchHandle INVALID_HANDLE = branch::INVALID_HANDLE;

// ============================================================================
// Value Normalizers
// ============================================================================

/**
 * Value normalizer functions
 *
 * These functions map arbitrary oracle scores to bounded ranges suitable
 * for PUCT. This ensures the exploration term remains competitive with
 * the exploitation term.
 */
namespace normalizers {

/**
 * Tanh normalizer: maps (-inf, inf) → (-1, 1)
 *
 * Uses tanh(value / scale) for smooth sigmoid-like mapping.
 * Default scale=200.0f works well for oracle scores in [-1000, 1000].
 *
 * Properties:
 * - tanh(1000/200) = tanh(5) ≈ 0.9999 → +1
 * - tanh(-1000/200) = tanh(-5) ≈ -0.9999 → -1
 * - tanh(100/200) = tanh(0.5) ≈ 0.46
 * - Exploration term ~0.5 becomes competitive with normalized exploitation
 *
 * @param value Raw oracle value
 * @param scale Scaling factor (default 200.0f)
 * @return Normalized value in [-1, 1]
 */
inline float tanh_normalizer(float value, float scale = 200.0f) {
  if (scale <= 0.0f) scale = 1.0f;  // Safety
  return std::tanh(value / scale);
}

/**
 * Sigmoid normalizer: maps (-inf, inf) → (0, 1)
 *
 * Uses 1 / (1 + exp(-value / scale)) for asymmetric range.
 * Useful when negative values should be compressed near 0.
 *
 * @param value Raw oracle value
 * @param scale Scaling factor (default 200.0f)
 * @return Normalized value in (0, 1)
 */
inline float sigmoid_normalizer(float value, float scale = 200.0f) {
  if (scale <= 0.0f) scale = 1.0f;
  float x = value / scale;
  // Numerically stable sigmoid
  if (x >= 0) {
    float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  } else {
    float z = std::exp(x);
    return z / (1.0f + z);
  }
}

/**
 * Identity normalizer: no normalization
 *
 * Use when value function already returns normalized values in [-1, 1].
 *
 * @param value Raw oracle value
 * @param scale Unused
 * @return value (unchanged)
 */
inline float identity(float value, float scale = 1.0f) {
  (void)scale;  // Unused
  return value;
}

}  // namespace normalizers

// ============================================================================
// SeqIdPool - Sequence ID Management
// ============================================================================

/**
 * Pool of reusable sequence IDs
 *
 * Prevents hitting n_seq_max even after pruning branches. Uses a freelist
 * to recycle sequence IDs as branches are pruned.
 */
class SeqIdPool {
public:
  /**
   * Initialize pool with max_seq sequence IDs
   *
   * Seq 0 is reserved for root, IDs 1..max_seq-1 are available.
   *
   * @param max_seq Maximum number of sequences
   */
  explicit SeqIdPool(int max_seq) : max_seq_(max_seq) {
    // Seq 0 is reserved for root
    for (int i = max_seq - 1; i >= 1; --i) {
      freelist_.push_back(static_cast<llama_seq_id>(i));
    }
  }

  /**
   * Acquire a sequence ID from the pool
   *
   * @return Sequence ID, or -1 if pool is exhausted
   */
  llama_seq_id acquire() {
    if (freelist_.empty()) {
      return -1;  // No more seq IDs available
    }
    llama_seq_id id = freelist_.back();
    freelist_.pop_back();
    return id;
  }

  /**
   * Release a sequence ID back to the pool
   *
   * @param id Sequence ID to release
   */
  void release(llama_seq_id id) {
    if (id > 0 && id < max_seq_) {
      freelist_.push_back(id);
    }
  }

  /**
   * Check if pool is exhausted
   *
   * @return true if no seq IDs available
   */
  bool empty() const { return freelist_.empty(); }

private:
  int max_seq_;
  std::vector<llama_seq_id> freelist_;
};

// ============================================================================
// Search Configuration
// ============================================================================

/**
 * Configuration for PUCT search
 */
struct SearchConfig {
  /** PUCT exploration constant (default 1.0) */
  float cpuct = 1.0f;

  /** Progressive widening constant: W(N) = 1 + c*√N (default 1.0) */
  float widening_constant = 1.0f;

  /** Value normalization function */
  using ValueNormalizer = std::function<float(float, float)>;
  ValueNormalizer value_normalizer = normalizers::tanh_normalizer;

  /** Scale parameter for value normalizer (default 200.0) */
  float normalizer_scale = 200.0f;

  /** Max failed sampling attempts before giving up on expansion (default 3) */
  int max_expansion_attempts = 3;

  /** Safety limit for chunk size in tokens (default 100) */
  int max_chunk_tokens = 100;

  /** Track detailed metrics (default true) */
  bool track_metrics = true;

  /** Enable structural rollout for verification oracles (default false) */
  bool use_structural_rollout = false;
};

/**
 * Metrics collected during search
 */
struct SearchMetrics {
  /** Total successful expansions */
  size_t total_expansions = 0;

  /** Total tokens generated */
  size_t total_tokens = 0;

  /** Total boundaries detected */
  size_t total_boundaries = 0;

  /** Failed expansion attempts */
  size_t failed_expansions = 0;

  /** Average tokens per boundary */
  float tokens_per_boundary() const {
    return total_boundaries > 0
      ? static_cast<float>(total_tokens) / total_boundaries
      : 0.0f;
  }

  /** Fork reduction factor (per-token vs per-boundary) */
  float fork_reduction() const {
    return total_boundaries > 0
      ? static_cast<float>(total_tokens) / total_boundaries
      : 1.0f;
  }

  /** Expansion success rate */
  float expansion_success_rate() const {
    size_t total = total_expansions + failed_expansions;
    return total > 0
      ? static_cast<float>(total_expansions) / total
      : 1.0f;
  }
};

// ============================================================================
// PUCT Node
// ============================================================================

/**
 * PUCT tree node
 *
 * Each node represents a state after generating a chunk of tokens
 * up to a boundary.
 */
struct PUCTNode {
  /** Branch handle (consolidates KV, grammar, sampler, metrics) */
  BranchHandle branch = INVALID_HANDLE;

  /** Parent node index (-1 for root) */
  int parent_idx = -1;

  /** Child node indices */
  std::vector<int> children;

  /** Tokens generated to reach this node (chunk until boundary) */
  std::vector<llama_token> tokens;

  /** Boundary info (if detected at end of chunk) */
  std::optional<boundaries::BoundaryInfo> boundary;

  /** Visit count (N(s,a) in PUCT formula) */
  int visits = 0;

  /** Total value accumulated (sum of all backpropagated values) */
  float total_value = 0.0f;

  /** Policy prior P(a|s) from model */
  float prior = 1.0f;

  /** Domain-specific terminal state (EoG, win/loss) */
  bool is_terminal = false;

  /** Failed expansion attempts (for debugging) */
  int expansion_attempts = 0;

  /**
   * Q-value (average value)
   *
   * @return total_value / visits, or 0.0 if unvisited
   */
  float q_value() const {
    return visits > 0 ? total_value / visits : 0.0f;
  }

  /**
   * Maximum children for progressive widening
   *
   * W(N) = 1 + c * √N
   *
   * @param widening_constant Widening factor (default 1.0)
   * @return Max children this node can have
   */
  int max_children(float widening_constant = 1.0f) const {
    return 1 + static_cast<int>(widening_constant * std::sqrt(static_cast<float>(visits)));
  }

  /**
   * Check if node can be expanded
   *
   * FIX (Issue B): Prevent budget sinks from repeated failed expansions
   * If a node has failed too many expansion attempts, treat as exhausted.
   *
   * @param widening_constant Widening factor (default 1.0)
   * @param max_failed_attempts Threshold for giving up (default 10)
   * @return true if can add more children
   */
  bool can_expand(float widening_constant = 1.0f, int max_failed_attempts = 10) const {
    if (is_terminal) return false;

    // Quarantine nodes with excessive failed expansions
    if (expansion_attempts >= max_failed_attempts) return false;

    return static_cast<int>(children.size()) < max_children(widening_constant);
  }

  /**
   * Check if node is fully expanded
   *
   * @param widening_constant Widening factor (default 1.0)
   * @param max_failed_attempts Threshold for giving up (default 10)
   * @return true if at max children, terminal, or exhausted from failures
   */
  bool fully_expanded(float widening_constant = 1.0f, int max_failed_attempts = 10) const {
    return !can_expand(widening_constant, max_failed_attempts);
  }
};

// ============================================================================
// PUCT Tree Search
// ============================================================================

/**
 * PUCT (Predictor + UCT) tree search
 *
 * AlphaZero-style MCTS with:
 * - Progressive widening: W(N) = 1 + c*√N
 * - Tree poisoning fix: failed expansion doesn't mark parent terminal
 * - Value normalization: maps oracle scores to [-1, 1]
 *
 * @tparam ValueFn Value function type (const std::string&) -> float
 */
template <typename ValueFn = std::function<float(const std::string&)>>
class PUCT {
public:
  /**
   * Sampling parameters for branch creation
   */
  struct SamplingParams {
    float temperature = 0.7f;
    int32_t top_k = 40;
    float top_p = 0.9f;
    float min_p = 0.0f;
    float typical_p = 1.0f;
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    int32_t penalty_last_n = 0;
    uint32_t seed = 42;
  };

  /**
   * Construct PUCT search
   *
   * @param ctx llama_context for generation
   * @param model llama_model for tokenization
   * @param grammar_str GBNF grammar string (empty for no grammar)
   * @param max_seq Maximum sequence IDs (must be >= n_seq_max)
   * @param n_batch Batch size for decoding
   * @param start_pos Starting position after prefill
   * @param config Search configuration
   * @param value_fn Value function (nullptr for perplexity-based fallback)
   */
  PUCT(llama_context* ctx,
       const llama_model* model,
       const std::string& grammar_str,
       int max_seq,
       int n_batch,
       llama_pos start_pos,
       const SearchConfig& config = SearchConfig{},
       ValueFn value_fn = nullptr);

  /** Destructor (BranchStore handles cleanup automatically) */
  ~PUCT() = default;

  /**
   * Run PUCT search for n_iterations
   *
   * Each iteration:
   * 1. Selection: Find node to expand via PUCT formula
   * 2. Expansion: Create new child node
   * 3. Simulation: Evaluate node value
   * 4. Backpropagation: Update statistics up the tree
   *
   * @param n_iterations Number of search iterations
   */
  void search(int n_iterations);

  /**
   * Get best sequence by visit count
   *
   * Walks down tree following highest-visit children.
   *
   * @return Best token sequence
   */
  std::vector<llama_token> get_best_sequence() const;

  /**
   * Get index of best child (by visits)
   *
   * @return Child index, or -1 if no children
   */
  int best_child_index() const;

  /**
   * Get search metrics
   *
   * @return SearchMetrics with statistics
   */
  SearchMetrics get_metrics() const { return metrics_; }

  /**
   * Print metrics to stdout
   */
  void print_metrics() const;

  // Tree inspection

  /**
   * Get number of nodes in tree
   *
   * @return Node count
   */
  int get_node_count() const { return static_cast<int>(nodes_.size()); }

  /**
   * Get root visit count
   *
   * @return Root visits
   */
  int get_root_visits() const { return nodes_[0].visits; }

  /**
   * Get node by index
   *
   * @param idx Node index
   * @return Node reference
   */
  const PUCTNode& get_node(int idx) const { return nodes_[idx]; }

  /**
   * Get generated text for a node
   *
   * Walks from root to node, concatenating all tokens.
   *
   * @param node_idx Node index
   * @return Generated text
   */
  std::string get_node_text(int node_idx) const;

  /**
   * Get branch handle of best leaf
   *
   * @return Best branch handle
   */
  BranchHandle get_best_branch() const;

  /**
   * Get branch store (for advanced usage)
   *
   * @return BranchStore pointer
   */
  BranchStore* get_store() { return &store_; }

private:
  /**
   * Selection phase: find node to expand
   *
   * Uses PUCT formula with progressive widening.
   * Stops at partially-expanded nodes (not just leaves).
   *
   * PUCT(s,a) = Q(s,a) + cpuct * P(a|s) * √N(s) / (1 + N(s,a))
   *
   * @return Index of selected node
   */
  int select();

  /**
   * Expansion phase: create new child
   *
   * Generates tokens until boundary detected.
   * Handles sampling failures gracefully without poisoning parent.
   *
   * @param parent_idx Parent node index
   * @return Child node index, or parent_idx if expansion failed
   */
  int expand(int parent_idx);

  /**
   * Simulation phase: evaluate node value
   *
   * Uses value function if provided, otherwise perplexity.
   * Applies normalization for competitive exploration.
   *
   * @param node_idx Node index
   * @return Normalized value in [-1, 1]
   */
  float simulate(int node_idx);

  /**
   * Backpropagation phase: update statistics
   *
   * Walks up tree updating visit counts and values.
   *
   * @param node_idx Starting node index
   * @param value Value to backpropagate
   */
  void backpropagate(int node_idx, float value);

  // State
  llama_context* ctx_;
  const llama_model* model_;
  const llama_vocab* vocab_;
  int n_vocab_;
  int n_batch_;
  int max_seq_;

  SearchConfig config_;
  BranchStore store_;
  SeqIdPool seq_pool_;
  std::vector<PUCTNode> nodes_;
  std::function<float(const std::string&)> value_fn_;
  SearchMetrics metrics_;
};

// C++17 deduction guide for PUCT
template <typename ValueFn>
PUCT(llama_context*, const llama_model*, const std::string&, int, int, llama_pos,
     const SearchConfig&, ValueFn) -> PUCT<ValueFn>;

// ============================================================================
// PUCT Implementation
// ============================================================================

template <typename ValueFn>
PUCT<ValueFn>::PUCT(llama_context* ctx,
                     const llama_model* model,
                     const std::string& grammar_str,
                     int max_seq,
                     int n_batch,
                     llama_pos start_pos,
                     const SearchConfig& config,
                     ValueFn value_fn)
    : ctx_(ctx),
      model_(model),
      n_batch_(n_batch),
      max_seq_(max_seq),
      config_(config),
      store_(max_seq),
      seq_pool_(max_seq),
      value_fn_(value_fn) {

  vocab_ = llama_model_get_vocab(model_);
  n_vocab_ = llama_vocab_n_tokens(vocab_);

  // Create boundary tracker for per-boundary MCTS
  auto tracker = boundaries::createCommonMarkBoundaryTracker();

  // Create root branch (seq 0 is reserved for root)
  SamplingParams params;
  BranchHandle root_branch = branch::create(
      ctx_, model_, 0, start_pos, params, n_batch_,
      grammar_str.c_str(), tracker.release(), &store_);

  // Capture root logits after prefill
  branch::capture_logits(root_branch, &store_);

  // Create root node
  PUCTNode root;
  root.branch = root_branch;
  root.prior = 1.0f;  // Root has uniform prior
  nodes_.push_back(root);
}

template <typename ValueFn>
void PUCT<ValueFn>::search(int n_iterations) {
  for (int i = 0; i < n_iterations; ++i) {
    // 1. Selection: Find node to expand via PUCT
    int node_idx = select();

    // 2. Expansion: Create child if expandable and visited
    if (!nodes_[node_idx].is_terminal && nodes_[node_idx].visits > 0) {
      // Check if node can be expanded (progressive widening + quarantine)
      // FIX (Issue B): Wire config.max_expansion_attempts to quarantine threshold
      if (nodes_[node_idx].can_expand(config_.widening_constant, config_.max_expansion_attempts)) {
        node_idx = expand(node_idx);
      }
    }

    // 3. Simulation: Evaluate node value
    float value = simulate(node_idx);

    // 4. Backpropagation: Update statistics up tree
    backpropagate(node_idx, value);
  }
}

template <typename ValueFn>
int PUCT<ValueFn>::select() {
  int idx = 0;

  // ISSUE 1 FIX: Progressive widening
  // Stop at partially-expanded nodes (not just leaves)
  while (!nodes_[idx].children.empty() &&
         nodes_[idx].fully_expanded(config_.widening_constant, config_.max_expansion_attempts)) {
    int best_child = -1;
    float best_puct = -std::numeric_limits<float>::infinity();

    int parent_visits = nodes_[idx].visits;
    float sqrt_parent = std::sqrt(static_cast<float>(parent_visits));

    for (int child_idx : nodes_[idx].children) {
      const auto& child = nodes_[child_idx];

      float puct;
      if (child.visits == 0) {
        // Unvisited: use prior only with parent visits
        puct = config_.cpuct * child.prior * sqrt_parent;
      } else {
        float exploitation = child.q_value();
        float exploration = config_.cpuct * child.prior * sqrt_parent / (1 + child.visits);
        puct = exploitation + exploration;
      }

      if (puct > best_puct) {
        best_puct = puct;
        best_child = child_idx;
      }
    }

    if (best_child == -1) break;
    idx = best_child;
  }

  return idx;
}

template <typename ValueFn>
int PUCT<ValueFn>::expand(int parent_idx) {
  auto& parent = nodes_[parent_idx];

  // Use SeqIdPool for seq-id reuse
  llama_seq_id child_seq = seq_pool_.acquire();
  if (child_seq < 0) {
    return parent_idx;  // No more seq IDs available
  }

  // Fork branch: clones KV cache, grammar, metrics, logits, tracker atomically
  BranchHandle child_branch = branch::fork(parent.branch, child_seq, &store_);

  if (child_branch == INVALID_HANDLE) {
    seq_pool_.release(child_seq);
    return parent_idx;
  }

  // Get tracker from branch state for boundary detection
  BranchState* state = store_.get(child_branch);

  // FIX (Issue A): Reseed sampler to prevent duplicate children
  // Derive unique seed from parent index + child count
  // This ensures siblings explore different branches
  // SAFETY: Only reseed stochastic chains (has_dist_sampler=true)
  // Reseeding greedy chains would corrupt them by replacing greedy with dist
  if (state && state->sampler_chain && state->has_dist_sampler) {
    uint32_t base_seed = 42;  // Match default in SamplingParams
    uint32_t unique_seed = base_seed +
                           (parent_idx * 1000) +   // Parent contribution
                           parent.children.size();  // Child index contribution
    sampler::reseed_chain(state->sampler_chain, unique_seed);
  }

  // Generate chunk UNTIL BOUNDARY (per-boundary MCTS)
  std::vector<llama_token> chunk;
  std::optional<boundaries::BoundaryInfo> boundary;
  double log_prior_sum = 0.0;  // Accumulate in log-space
  int chunk_len = 0;
  int sample_failures = 0;

  while (!boundary) {
    // Sample next token (uses branch's grammar + sampler chain)
    llama_token token = branch::sample(child_branch, &store_);

    // ISSUE 2 FIX: Tree poisoning fix
    // Failed sampling doesn't mark parent terminal
    if (token < 0) {
      sample_failures++;

      if (sample_failures >= config_.max_expansion_attempts) {
        // Give up after max attempts
        branch::prune(child_branch, &store_);
        seq_pool_.release(child_seq);
        parent.expansion_attempts++;

        // CRITICAL: DO NOT mark parent terminal
        // This is sampling failure, not domain terminal state
        if (config_.track_metrics) {
          metrics_.failed_expansions++;
        }
        return parent_idx;  // Backprop will use neutral value
      }

      continue;  // Retry sampling
    }

    // Get prior from sampling distribution (Issue #6 solved via BranchMetrics)
    float prior = branch::get_last_sampling_prior(child_branch, &store_);

    // Accumulate in log-space to avoid underflow
    log_prior_sum += std::log(std::max(prior, 1e-30f));
    chunk_len++;

    // Detokenize for boundary detection
    std::string text = tokenizer::detokenize(model_, token);

    // Feed to boundary tracker if present
    if (state && state->boundary_tracker) {
      boundary = state->boundary_tracker->feed_draft(text);
      if (boundary) {
        state->boundary_tracker->commit_draft();
      }
    }

    // Accept token (updates grammar state and metrics)
    branch::accept_token(child_branch, token, &store_);

    // Decode token and capture logits for future expansion
    branch::decode_and_capture_one(child_branch, token, &store_);

    // Add to chunk
    chunk.push_back(token);

    // Check if terminal (end of generation)
    if (tokenizer::is_eog(vocab_, token)) {
      boundary = boundaries::BoundaryInfo{
        .kind = "eog",
        .pos = chunk.size(),
        .meta = {}
      };
      break;
    }

    // Safety: prevent infinite loops
    if (static_cast<int>(chunk.size()) >= config_.max_chunk_tokens) {
      // No boundary found - fail expansion gracefully
      branch::prune(child_branch, &store_);
      seq_pool_.release(child_seq);
      parent.expansion_attempts++;

      // CRITICAL: DO NOT mark parent terminal
      if (config_.track_metrics) {
        metrics_.failed_expansions++;
      }
      return parent_idx;
    }
  }

  // Check if terminal (end of generation)
  bool is_terminal = boundary && boundary->kind == "eog";

  // Update metrics
  if (config_.track_metrics) {
    metrics_.total_expansions++;
    metrics_.total_tokens += chunk.size();
    if (boundary) {
      metrics_.total_boundaries++;
    }
  }

  // Create child node with chunk
  PUCTNode child;
  child.branch = child_branch;
  child.parent_idx = parent_idx;
  child.tokens = chunk;
  child.boundary = boundary;
  // Length-normalized prior (geometric mean)
  child.prior = std::exp(log_prior_sum / std::max(1, chunk_len));
  child.is_terminal = is_terminal;

  int child_idx = static_cast<int>(nodes_.size());
  nodes_.push_back(child);
  nodes_[parent_idx].children.push_back(child_idx);

  return child_idx;
}

template <typename ValueFn>
float PUCT<ValueFn>::simulate(int node_idx) {
  const auto& node = nodes_[node_idx];

  // Terminal nodes get bonus
  if (node.is_terminal) {
    return 1.0f;
  }

  float raw_value;

  // If oracle is provided, use it
  if (value_fn_) {
    std::string text = get_node_text(node_idx);

    // Apply structural rollout if enabled for verification oracles
    // This completes open syntax containers deterministically (no LLM inference)
    // allowing compilers/SAST tools to parse incomplete code
    if (config_.use_structural_rollout) {
      BranchState* state = store_.get(node.branch);
      if (state && state->boundary_tracker) {
        // Try to cast to RemuxBoundaryTracker
        auto* remux_tracker = dynamic_cast<boundaries::RemuxBoundaryTracker*>(
            state->boundary_tracker
        );
        if (remux_tracker) {
          text = remux_tracker->structural_rollout(text);
        }
      }
    }

    raw_value = value_fn_(text);
  } else {
    // Fallback: Use inverse perplexity as value
    float ppl = branch::get_perplexity(node.branch, &store_);
    if (std::isinf(ppl) || ppl < 1.0f) {
      return 0.5f;  // Default for empty/invalid
    }
    raw_value = 1.0f / (1.0f + std::log(ppl));
  }

  // ISSUE 3 FIX: Value normalization
  // Apply normalization to map arbitrary scores to [-1, 1]
  // EXPERIMENT: Temporarily disabled to test hypothesis
  // if (config_.value_normalizer) {
  //   return config_.value_normalizer(raw_value, config_.normalizer_scale);
  // }

  return raw_value;
}

template <typename ValueFn>
void PUCT<ValueFn>::backpropagate(int node_idx, float value) {
  while (node_idx >= 0) {
    nodes_[node_idx].visits++;
    nodes_[node_idx].total_value += value;
    node_idx = nodes_[node_idx].parent_idx;
  }
}

template <typename ValueFn>
std::vector<llama_token> PUCT<ValueFn>::get_best_sequence() const {
  std::vector<llama_token> tokens;
  int idx = 0;

  while (!nodes_[idx].children.empty()) {
    // Pick child with highest Q-value (exploitation)
    // Matches LLM-MCTS paper: argmax_a Q(s,a) for final action selection
    int best_child = -1;
    float best_q = -std::numeric_limits<float>::infinity();
    for (int child_idx : nodes_[idx].children) {
      float q = nodes_[child_idx].q_value();
      if (q > best_q) {
        best_q = q;
        best_child = child_idx;
      }
    }
    if (best_child == -1) break;

    idx = best_child;

    // Append all tokens in this node's chunk
    tokens.insert(tokens.end(), nodes_[idx].tokens.begin(), nodes_[idx].tokens.end());
  }

  return tokens;
}

template <typename ValueFn>
int PUCT<ValueFn>::best_child_index() const {
  if (nodes_.empty() || nodes_[0].children.empty()) {
    return -1;
  }

  int best_child = -1;
  float best_q = -std::numeric_limits<float>::infinity();
  for (int child_idx : nodes_[0].children) {
    float q = nodes_[child_idx].q_value();
    if (q > best_q) {
      best_q = q;
      best_child = child_idx;
    }
  }

  return best_child;
}

template <typename ValueFn>
std::string PUCT<ValueFn>::get_node_text(int node_idx) const {
  // Collect path from root to node
  std::vector<int> path;
  int idx = node_idx;
  while (idx >= 0) {
    path.push_back(idx);
    idx = nodes_[idx].parent_idx;
  }

  // Reverse to go root → node
  std::reverse(path.begin(), path.end());

  // Collect all tokens along the path (skip root which has no tokens)
  std::vector<llama_token> tokens;
  for (size_t i = 1; i < path.size(); ++i) {
    const auto& node = nodes_[path[i]];
    tokens.insert(tokens.end(), node.tokens.begin(), node.tokens.end());
  }

  // Convert tokens to text
  return tokenizer::detokenize_batch(model_, tokens);
}

template <typename ValueFn>
BranchHandle PUCT<ValueFn>::get_best_branch() const {
  int idx = 0;
  while (!nodes_[idx].children.empty()) {
    int best_child = -1;
    int best_visits = -1;
    for (int child_idx : nodes_[idx].children) {
      if (nodes_[child_idx].visits > best_visits) {
        best_visits = nodes_[child_idx].visits;
        best_child = child_idx;
      }
    }
    if (best_child == -1) break;
    idx = best_child;
  }
  return nodes_[idx].branch;
}

template <typename ValueFn>
void PUCT<ValueFn>::print_metrics() const {
  std::cout << "\n=== MCTS Search Metrics ===" << std::endl;
  std::cout << "Total expansions: " << metrics_.total_expansions << std::endl;
  std::cout << "Failed expansions: " << metrics_.failed_expansions << std::endl;
  std::cout << "Total tokens: " << metrics_.total_tokens << std::endl;
  std::cout << "Total boundaries: " << metrics_.total_boundaries << std::endl;
  std::cout << "Tokens/boundary: " << metrics_.tokens_per_boundary() << std::endl;
  std::cout << "Fork reduction: " << metrics_.fork_reduction() << "x" << std::endl;
  std::cout << "Expansion success rate: "
            << (metrics_.expansion_success_rate() * 100.0f) << "%" << std::endl;

  std::cout << "\nTree statistics:" << std::endl;
  std::cout << "  Total nodes: " << nodes_.size() << std::endl;
  std::cout << "  Root visits: " << (nodes_.empty() ? 0 : nodes_[0].visits) << std::endl;
  if (!nodes_.empty() && nodes_[0].visits > 0) {
    float branching = static_cast<float>(nodes_.size()) / nodes_[0].visits;
    std::cout << "  Branching factor: " << branching << std::endl;
  }
}

}  // namespace mcts
}  // namespace lloyal
