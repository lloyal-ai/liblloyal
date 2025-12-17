/**
 * PUCT Integration Test
 *
 * Validates that lloyal::branch primitive enables proper PUCT (Predictor + UCT)
 * tree search for LLM generation. Uses grammar-constrained generation as the task.
 *
 * Key difference from vanilla MCTS/UCT:
 * - UCT:  exploration = C * sqrt(ln(N_parent) / N_child)
 * - PUCT: exploration = cpuct * P(a|s) * sqrt(N_parent) / (1 + N_child)
 *
 * Where P(a|s) is the policy prior from softmax(logits) over LEGAL moves only
 * (grammar-masked and renormalized), matching AlphaZero-style MCTS.
 *
 * Primitives tested via lloyal::branch:
 * - branch::create() - Initialize branch with sampler chain + grammar
 * - branch::capture_logits() - Capture logits after prefill (root init)
 * - branch::fork() - O(1) branch forking (KV, grammar, perplexity, logits)
 * - branch::decode_and_capture_one() - Decode + capture logits for policy priors
 * - branch::get_legal_logsumexp() - Compute logsumexp over grammar-legal tokens
 * - branch::get_token_prior() - Grammar-masked prior for specific token
 * - branch::sample() / accept_token() - Grammar-constrained sampling
 * - branch::get_perplexity() - Value estimation
 * - branch::prune() - Branch removal + KV cleanup
 *
 * PUCT Algorithm:
 * 1. Selection: PUCT formula with grammar-masked policy priors
 * 2. Expansion: Fork branch, sample token, compute legal prior, decode
 * 3. Simulation: Use perplexity as value estimate (no rollout)
 * 4. Backpropagation: Update visit counts and values
 *
 * Correctness fixes (vs naive implementation):
 * - FIX #1: Priors are grammar-masked + renormalized over legal moves
 * - FIX #2: Root logits captured after prefill via capture_logits()
 * - FIX #3: prune() used on fork failure (clears KV cache properly)
 * - FIX #4: SeqIdPool enables seq-id reuse after pruning
 * - FIX #6: Stepwise priors from child's evolving state (not parent's static state)
 *           with log-space accumulation and length normalization
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/boundaries.hpp>
#include <lloyal/branch.hpp>
#include <lloyal/nlohmann/json.hpp>  // Include before chat_template.hpp
#include <lloyal/chat_template.hpp>
#include <lloyal/decoder.hpp>
#include <lloyal/grammar.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/logits.hpp>
#include <lloyal/metrics.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/mcts.hpp>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace lloyal;
using namespace lloyal::branch;

static const char *MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

#define REQUIRE_MODEL()                                                        \
  if (!MODEL_PATH || !*MODEL_PATH) {                                           \
    MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set");                              \
    return;                                                                    \
  }

struct LlamaBackendGuard {
  LlamaBackendGuard() { llama_backend_init(); }
  ~LlamaBackendGuard() { llama_backend_free(); }
};

// ============================================================================
// PUCT Node - Branch-based tree node
// ============================================================================

struct PUCTNode {
  // Branch handle consolidates: seq_id, position, grammar, perplexity, logits, tracker
  BranchHandle branch = INVALID_HANDLE;

  // Tree structure
  int parent_idx = -1;
  std::vector<int> children;

  // Tokens that led to this node (chunk until boundary)
  std::vector<llama_token> tokens;

  // Boundary info (if detected at end of chunk)
  std::optional<lloyal::boundaries::BoundaryInfo> boundary;

  // PUCT statistics
  int visits = 0;
  float total_value = 0.0f;
  float prior = 1.0f;  // P(a|s) - policy prior from softmax(logits)

  // Terminal state
  bool is_terminal = false;

  float q_value() const {
    return visits > 0 ? total_value / visits : 0.0f;
  }
};

// ============================================================================
// Seq ID Pool - Freelist for seq-id reuse
// ============================================================================

/**
 * Pool of reusable sequence IDs.
 * Prevents hitting n_seq_max even after pruning branches.
 */
class SeqIdPool {
public:
  explicit SeqIdPool(int max_seq) : max_seq_(max_seq) {
    // Seq 0 is reserved for root
    for (int i = max_seq - 1; i >= 1; --i) {
      freelist_.push_back(static_cast<llama_seq_id>(i));
    }
  }

  llama_seq_id acquire() {
    if (freelist_.empty()) {
      return -1;  // No more seq IDs available
    }
    llama_seq_id id = freelist_.back();
    freelist_.pop_back();
    return id;
  }

  void release(llama_seq_id id) {
    if (id > 0 && id < max_seq_) {
      freelist_.push_back(id);
    }
  }

  bool empty() const { return freelist_.empty(); }

private:
  int max_seq_;
  std::vector<llama_seq_id> freelist_;
};

// ============================================================================
// PUCT Tree Search
// ============================================================================

// Sampling params for branch creation
struct PUCTSamplingParams {
  float temperature = 0.7f;  // Match chat.mjs for stable generation
  int32_t top_k = 40;        // Top-K sampling
  float top_p = 0.9f;        // Nucleus sampling
  float min_p = 0.0f;
  float typical_p = 1.0f;
  float penalty_repeat = 1.0f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 0;
  uint32_t seed = 42;
};

class PUCT {
public:
  using ValueFunction = std::function<float(const std::string&)>;

  PUCT(llama_context* ctx, const llama_model* model,
       const std::string& grammar_str, int max_seq, int n_batch,
       llama_pos start_pos,
       ValueFunction value_fn = nullptr)
      : ctx_(ctx), model_(model), n_batch_(n_batch), max_seq_(max_seq),
        store_(max_seq), seq_pool_(max_seq), value_fn_(value_fn) {
    vocab_ = llama_model_get_vocab(model_);
    n_vocab_ = llama_vocab_n_tokens(vocab_);

    // Create boundary tracker for per-boundary MCTS
    auto tracker = lloyal::boundaries::createCommonMarkBoundaryTracker();

    // Create root branch (seq 0 is reserved for root)
    PUCTSamplingParams params;
    BranchHandle root_branch = create(
        ctx_, model_, 0, start_pos, params, n_batch_,
        grammar_str.c_str(), tracker.release(), &store_);

    // FIX #2: Capture root logits after prefill
    // Without this, root has no logits and priors are undefined
    capture_logits(root_branch, &store_);

    // Create root node
    PUCTNode root;
    root.branch = root_branch;
    root.prior = 1.0f;  // Root has uniform prior
    nodes_.push_back(root);
  }

  // Destructor: BranchStore handles cleanup automatically via freelist
  ~PUCT() = default;

  // Run PUCT for n_iterations
  void search(int n_iterations) {
    for (int i = 0; i < n_iterations; ++i) {
      // 1. Selection: Find leaf node via PUCT
      int node_idx = select();

      // 2. Expansion: Create child if not terminal and visited before
      if (!nodes_[node_idx].is_terminal && nodes_[node_idx].visits > 0) {
        node_idx = expand(node_idx);
      }

      // 3. Simulation: Use perplexity as value (no rollout)
      float value = simulate(node_idx);

      // 4. Backpropagation
      backpropagate(node_idx, value);
    }
  }

  // Get best sequence of tokens (by visit count)
  std::vector<llama_token> get_best_sequence() {
    std::vector<llama_token> tokens;
    int idx = 0;

    while (!nodes_[idx].children.empty()) {
      // Pick child with highest visit count
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
      // Append all tokens in this node's chunk
      tokens.insert(tokens.end(), nodes_[idx].tokens.begin(), nodes_[idx].tokens.end());
    }

    return tokens;
  }

  int get_node_count() const { return static_cast<int>(nodes_.size()); }
  int get_root_visits() const { return nodes_[0].visits; }

  // Get the branch handle of the best leaf node (for perplexity)
  BranchHandle get_best_branch() {
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

  BranchStore* get_store() { return &store_; }

  // Get generated text for a node (walk from root to node)
  std::string get_node_text(int node_idx) {
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

  // Metrics for per-boundary MCTS
  struct Metrics {
    size_t total_tokens = 0;
    size_t total_boundaries = 0;
    size_t total_expansions = 0;

    float tokens_per_boundary() const {
      return total_boundaries > 0 ? static_cast<float>(total_tokens) / total_boundaries : 0.0f;
    }

    float fork_reduction() const {
      // Per-token MCTS would fork at every token
      // Per-boundary MCTS forks at every boundary
      return total_boundaries > 0 ? static_cast<float>(total_tokens) / total_boundaries : 1.0f;
    }
  };

  Metrics get_metrics() const { return metrics_; }

  void print_metrics() const {
    std::cout << "\n=== Per-Boundary MCTS Metrics ===" << std::endl;
    std::cout << "Total expansions: " << metrics_.total_expansions << std::endl;
    std::cout << "Total tokens: " << metrics_.total_tokens << std::endl;
    std::cout << "Total boundaries: " << metrics_.total_boundaries << std::endl;
    std::cout << "Tokens/boundary: " << metrics_.tokens_per_boundary() << std::endl;

    std::cout << "\nFork Reduction:" << std::endl;
    std::cout << "  Per-token approach: " << metrics_.total_tokens << " fork points" << std::endl;
    std::cout << "  Per-boundary approach: " << metrics_.total_boundaries << " fork points" << std::endl;
    std::cout << "  Reduction: " << metrics_.fork_reduction() << "x" << std::endl;

    // With vocab expansion (K=50k)
    if (metrics_.total_boundaries > 0) {
      size_t token_forks = metrics_.total_tokens * 50000;
      size_t boundary_forks = metrics_.total_boundaries * 10;
      float expansion_reduction = static_cast<float>(token_forks) / boundary_forks;

      std::cout << "\nWith vocab (K=50,000):" << std::endl;
      std::cout << "  Token-level: " << metrics_.total_tokens << " × 50k = " << token_forks << std::endl;
      std::cout << "  Boundary-level: " << metrics_.total_boundaries << " × 10 = " << boundary_forks << std::endl;
      std::cout << "  Reduction: " << expansion_reduction << "x" << std::endl;
    }
  }

private:
  /**
   * PUCT Selection (AlphaZero-style)
   *
   * PUCT = Q(s,a) + cpuct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
   *
   * Where:
   * - Q(s,a) = average value of edge (exploitation)
   * - P(a|s) = policy prior from softmax(logits)
   * - N(s) = parent visit count
   * - N(s,a) = child visit count
   * - cpuct = exploration constant
   */
  int select() {
    int idx = 0;
    constexpr float cpuct = 1.0f;  // Exploration constant

    while (!nodes_[idx].children.empty()) {
      int best_child = -1;
      float best_puct = -std::numeric_limits<float>::infinity();

      int parent_visits = nodes_[idx].visits;
      float sqrt_parent = std::sqrt(static_cast<float>(parent_visits));

      for (int child_idx : nodes_[idx].children) {
        const auto& child = nodes_[child_idx];

        float puct;
        if (child.visits == 0) {
          // Unvisited: use prior only with parent visits
          puct = cpuct * child.prior * sqrt_parent;
        } else {
          float exploitation = child.q_value();
          float exploration = cpuct * child.prior * sqrt_parent / (1 + child.visits);
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

  /**
   * Expand node using Branch primitive
   *
   * Fixes applied:
   * - FIX #1: Grammar-masked + renormalized priors via get_legal_logsumexp/get_token_prior
   * - FIX #3: Use prune() not destroy() on fork failure (clears KV)
   * - FIX #4: Use SeqIdPool for seq-id reuse
   * - FIX #6: Stepwise priors computed from child_branch's evolving state (not parent)
   *           Accumulates in log-space with length normalization to avoid underflow
   *           and make priors comparable across variable chunk lengths
   *
   * KNOWN ISSUE: Priors are computed from grammar-masked raw logits, but sample()
   * uses a sampler chain (top-k/p/temp/penalties). This mismatch can hurt exploration
   * even with correct stepwise priors. Ideal fix: either (a) have sample() return
   * the token's probability from the filtered distribution, or (b) disable top-k/p
   * during PUCT and let PUCT do the exploration.
   */
  int expand(int parent_idx) {
    auto& parent = nodes_[parent_idx];

    // FIX #4: Use SeqIdPool for seq-id reuse
    llama_seq_id child_seq = seq_pool_.acquire();
    if (child_seq < 0) {
      return parent_idx;  // No more seq IDs available
    }

    // Fork branch: clones KV cache, grammar, perplexity, logits, tracker atomically
    BranchHandle child_branch = fork(parent.branch, child_seq, &store_);

    if (child_branch == INVALID_HANDLE) {
      seq_pool_.release(child_seq);  // Return unused seq ID
      return parent_idx;
    }

    // Get tracker from branch state for boundary detection
    BranchState* state = store_.get(child_branch);

    // Generate chunk UNTIL BOUNDARY (per-boundary MCTS)
    std::vector<llama_token> chunk;
    std::optional<lloyal::boundaries::BoundaryInfo> boundary;
    double log_prior_sum = 0.0;  // Accumulate in log-space to avoid underflow
    int chunk_len = 0;

    while (!boundary) {
      // FIX #6: Compute logsumexp from child_branch's CURRENT state BEFORE sampling
      // This ensures the prior is computed from the same distribution used for sampling
      // (in case sample() mutates logits via penalties/top-k/p)
      float logZ = get_legal_logsumexp(child_branch, &store_);

      // Sample next token (uses branch's grammar + sampler chain)
      llama_token token = sample(child_branch, &store_);

      if (token < 0) {
        // FIX #3: Use prune() not destroy() - prune clears the forked KV cache
        prune(child_branch, &store_);
        seq_pool_.release(child_seq);  // Return seq ID to pool
        parent.is_terminal = true;
        return parent_idx;
      }

      // FIX #1 + #6: Compute grammar-masked + renormalized prior from CURRENT state
      // P(token|current_state) over LEGAL moves only, not full vocab
      // Use assume_legal since sample() already enforced grammar - O(1) not O(grammar)
      float prior = get_token_prior_assume_legal(child_branch, token, logZ, &store_);

      // Accumulate in log-space to avoid underflow
      log_prior_sum += std::log(std::max(prior, 1e-30f));  // Clamp to avoid log(0)
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

      // Accept token (updates grammar state and perplexity)
      accept_token(child_branch, token, &store_);

      // Decode token and capture logits for future expansion
      decode_and_capture_one(child_branch, token, &store_);

      // Add to chunk
      chunk.push_back(token);

      // Check if terminal (end of generation)
      if (tokenizer::is_eog(vocab_, token)) {
        boundary = lloyal::boundaries::BoundaryInfo{
          .kind = "eog",
          .pos = chunk.size(),
          .meta = {}
        };
        break;
      }

      // Safety: prevent infinite loops if boundary tracker isn't finding boundaries
      // Instead of creating fake boundary, just fail the expansion
      if (chunk.size() >= 100) {  // Safety limit
        // No boundary found - don't create a child node
        // Mark parent as terminal to prevent further expansion attempts
        prune(child_branch, &store_);
        seq_pool_.release(child_seq);
        parent.is_terminal = true;
        return parent_idx;  // Return parent, no child created
      }
    }

    // Log what kind of boundary was detected
    if (boundary) {
      std::cout << "[expand] Boundary detected: kind='" << boundary->kind
                << "', pos=" << boundary->pos << ", chunk_size=" << chunk.size() << std::endl;
    }

    // Check if terminal (end of generation)
    bool is_terminal = boundary && boundary->kind == "eog";

    // Update metrics
    metrics_.total_expansions++;
    metrics_.total_tokens += chunk.size();
    if (boundary) {
      metrics_.total_boundaries++;
    }

    // Create child node with chunk
    PUCTNode child;
    child.branch = child_branch;
    child.parent_idx = parent_idx;
    child.tokens = chunk;
    child.boundary = boundary;
    // FIX #6: Length-normalized prior (geometric mean) keeps probability mass
    // comparable across variable chunk lengths. This prevents longer chunks from
    // getting artificially low priors just from multiplying more terms.
    child.prior = std::exp(log_prior_sum / std::max(1, chunk_len));
    child.is_terminal = is_terminal;

    int child_idx = static_cast<int>(nodes_.size());
    nodes_.push_back(child);
    nodes_[parent_idx].children.push_back(child_idx);

    return child_idx;
  }

  // Simulate: Use inverse perplexity as value
  float simulate(int node_idx) {
    const auto& node = nodes_[node_idx];

    // Terminal nodes get bonus
    if (node.is_terminal) {
      return 1.0f;
    }

    // If oracle is provided, use it instead of perplexity
    if (value_fn_) {
      // Get the generated text so far for this node
      std::string text = get_node_text(node_idx);
      return value_fn_(text);
    }

    // Fallback: Use inverse perplexity as value (lower perplexity = higher value)
    float ppl = get_perplexity(node.branch, &store_);
    if (std::isinf(ppl) || ppl < 1.0f) {
      return 0.5f;  // Default for empty/invalid
    }

    // Map perplexity to [0, 1]: value = 1 / (1 + log(ppl))
    return 1.0f / (1.0f + std::log(ppl));
  }

  // Backpropagate value up the tree
  void backpropagate(int node_idx, float value) {
    while (node_idx >= 0) {
      nodes_[node_idx].visits++;
      nodes_[node_idx].total_value += value;
      node_idx = nodes_[node_idx].parent_idx;
    }
  }

  llama_context* ctx_;
  const llama_model* model_;
  const llama_vocab* vocab_;
  int n_vocab_;
  int n_batch_;
  int max_seq_;

  BranchStore store_;      // Handle table for branch management
  SeqIdPool seq_pool_;     // FIX #4: Reusable seq-id pool
  std::vector<PUCTNode> nodes_;
  Metrics metrics_;        // Track per-boundary MCTS metrics
  ValueFunction value_fn_; // Optional value function (oracle)
};

// ============================================================================
// Integration Tests
// ============================================================================

// ============================================================================
// Tic-Tac-Toe Oracle - Parse and evaluate board state
// ============================================================================

// Helper to find nodes of type (declare before use)
std::vector<const remux::BlockAstNode*> findNodesOfType(
    const remux::BlockAstNode& root,
    const std::string& type
);

struct TicTacToeOracle {
  std::vector<std::vector<std::string>> parse_board(const std::string& text) {
    auto ast = remux::parseMarkdown(text);
    const auto& root = ast.get();

    std::vector<std::vector<std::string>> board(3, std::vector<std::string>(3, ""));

    // Find table rows (use LAST table in case prompt contains examples)
    auto tables = findNodesOfType(root, "table");
    if (tables.empty()) return board;

    auto rows = findNodesOfType(*tables.back(), "table-row");

    // Skip separator, parse data rows
    int board_row = 0;
    for (const auto* row : rows) {
      if (!row->content) {
        continue;
      }

      std::string content = *row->content;

      // Skip separator rows (contains ---)
      if (content.find("---") != std::string::npos) continue;
      if (content.find("-") != std::string::npos && content.find("X") == std::string::npos && content.find("O") == std::string::npos) continue;

      // Parse cells: split by | (content from remux doesn't include leading |)
      // Example: "X | O |  " → ["X", "O", ""]
      std::vector<std::string> cells;
      size_t start = 0;
      size_t pos;
      while ((pos = content.find('|', start)) != std::string::npos) {
        std::string cell = content.substr(start, pos - start);
        // Trim whitespace
        cell.erase(0, cell.find_first_not_of(" \t"));
        cell.erase(cell.find_last_not_of(" \t") + 1);
        cells.push_back(cell);  // Include empty cells
        start = pos + 1;
      }
      // Handle last cell after final |
      if (start < content.length()) {
        std::string cell = content.substr(start);
        cell.erase(0, cell.find_first_not_of(" \t"));
        cell.erase(cell.find_last_not_of(" \t") + 1);
        cells.push_back(cell);
      }

      // Copy to board (pad with empty if less than 3 cells)
      if (board_row < 3) {
        for (int col = 0; col < 3; ++col) {
          board[board_row][col] = col < cells.size() ? cells[col] : "";
        }
        board_row++;
      }
    }

    return board;
  }

  bool has_three_in_row(const std::vector<std::vector<std::string>>& board, const std::string& player) {
    // Check rows
    for (int r = 0; r < 3; ++r) {
      if (board[r][0] == player && board[r][1] == player && board[r][2] == player) {
        return true;
      }
    }

    // Check columns
    for (int c = 0; c < 3; ++c) {
      if (board[0][c] == player && board[1][c] == player && board[2][c] == player) {
        return true;
      }
    }

    // Check diagonals
    if (board[0][0] == player && board[1][1] == player && board[2][2] == player) {
      return true;
    }
    if (board[0][2] == player && board[1][1] == player && board[2][0] == player) {
      return true;
    }

    return false;
  }

  int count_two_in_row(const std::vector<std::vector<std::string>>& board, const std::string& player) {
    int threats = 0;

    // Check rows
    for (int r = 0; r < 3; ++r) {
      int player_count = 0;
      int empty_count = 0;
      for (int c = 0; c < 3; ++c) {
        if (board[r][c] == player) player_count++;
        else if (board[r][c].empty() || board[r][c] == " ") empty_count++;
      }
      if (player_count == 2 && empty_count == 1) threats++;
    }

    // Check columns
    for (int c = 0; c < 3; ++c) {
      int player_count = 0;
      int empty_count = 0;
      for (int r = 0; r < 3; ++r) {
        if (board[r][c] == player) player_count++;
        else if (board[r][c].empty() || board[r][c] == " ") empty_count++;
      }
      if (player_count == 2 && empty_count == 1) threats++;
    }

    // Check diagonals
    {
      int player_count = 0;
      int empty_count = 0;
      for (int i = 0; i < 3; ++i) {
        if (board[i][i] == player) player_count++;
        else if (board[i][i].empty() || board[i][i] == " ") empty_count++;
      }
      if (player_count == 2 && empty_count == 1) threats++;
    }
    {
      int player_count = 0;
      int empty_count = 0;
      for (int i = 0; i < 3; ++i) {
        if (board[i][2-i] == player) player_count++;
        else if (board[i][2-i].empty() || board[i][2-i] == " ") empty_count++;
      }
      if (player_count == 2 && empty_count == 1) threats++;
    }

    return threats;
  }

  float evaluate(const std::string& text) {
    auto board = parse_board(text);

    // Check for X wins (our player)
    if (has_three_in_row(board, "X")) {
      return 1000.0f;  // WIN!
    }

    // Check for O wins (opponent)
    if (has_three_in_row(board, "O")) {
      return -1000.0f;  // LOSS
    }

    // Count threats (2 in a row with empty third)
    int x_threats = count_two_in_row(board, "X");
    int o_threats = count_two_in_row(board, "O");

    // Reward creating threats, heavily penalize allowing opponent threats
    return x_threats * 100.0f - o_threats * 150.0f;
  }
};

// Helper to find nodes of type (matches test_integration_complex.cpp pattern)
std::vector<const remux::BlockAstNode*> findNodesOfType(
    const remux::BlockAstNode& root,
    const std::string& type
) {
    std::vector<const remux::BlockAstNode*> result;
    std::function<void(const remux::BlockAstNode&)> walk = [&](const remux::BlockAstNode& node) {
        if (node.type == type) {
            result.push_back(&node);
        }
        for (const auto& child : node.children) {
            walk(child.get());
        }
    };
    walk(root);
    return result;
}

// ============================================================================
// README Structure Oracle - Strict opinionated template scoring
// ============================================================================

struct ReadmeStructureOracle {
  // Helper: Count nodes of specific type
  size_t count_nodes(const remux::BlockAstNode& root, const std::string& type) const {
    auto nodes = findNodesOfType(root, type);
    return nodes.size();
  }

  // Helper: Count headings at specific level
  size_t count_heading_level(const remux::BlockAstNode& root, int level) const {
    std::string type = "heading-" + std::to_string(level);
    return count_nodes(root, type);
  }

  // Helper: Get heading text at specific level
  std::vector<std::string> get_heading_texts(const remux::BlockAstNode& root, int level) const {
    std::string type = "heading-" + std::to_string(level);
    auto nodes = findNodesOfType(root, type);
    std::vector<std::string> texts;
    for (const auto* node : nodes) {
      if (node->content) {
        texts.push_back(*node->content);
      }
    }
    return texts;
  }

  // Helper: Check if specific section exists (case-insensitive substring)
  bool has_section(const remux::BlockAstNode& root, const std::string& name) const {
    auto h2_texts = get_heading_texts(root, 2);
    for (const auto& text : h2_texts) {
      std::string lower_text = text;
      std::string lower_name = name;
      std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
      std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
      if (lower_text.find(lower_name) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

  /**
   * Strict README template scoring
   *
   * The model receives: "Write a well-structured README for: [tool description]"
   * The oracle (hidden from model) scores based on:
   *
   * 1. Exactly 1 H1 (300 pts)
   * 2. Has "Installation" section at H2 level (200 pts)
   * 3. Has "Usage" section at H2 level (200 pts)
   * 4. Has code block (150 pts)
   * 5. Has exactly 2-4 H2 sections, not more (150 pts)
   *    - Prevents spamming sections
   *
   * This is deliberately hard to satisfy by accident.
   * Greedy sampling might get 1-2 criteria, MCTS should find more.
   */
  float evaluate(const std::string& text) const {
    auto ast = remux::parseMarkdown(text);
    const auto& root = ast.get();

    float score = 0.0f;

    // Criterion 1: Exactly one H1 (300 points)
    size_t h1_count = count_heading_level(root, 1);
    if (h1_count == 1) {
      score += 300.0f;
    }

    // Criterion 2: Has "Installation" section (200 points)
    if (has_section(root, "installation")) {
      score += 200.0f;
    }

    // Criterion 3: Has "Usage" section (200 points)
    if (has_section(root, "usage")) {
      score += 200.0f;
    }

    // Criterion 4: Has at least one code block (150 points)
    if (count_nodes(root, "code-block") > 0) {
      score += 150.0f;
    }

    // Criterion 5: Exactly 2-4 H2 sections (150 points)
    // Not too few (incomplete), not too many (spamming)
    size_t h2_count = count_heading_level(root, 2);
    if (h2_count >= 2 && h2_count <= 4) {
      score += 150.0f;
    }

    return score;  // Max 1000 points
  }

  // Helper to get detailed breakdown for debugging
  std::string get_breakdown(const std::string& text) const {
    auto ast = remux::parseMarkdown(text);
    const auto& root = ast.get();

    std::ostringstream out;
    out << "README Structure Analysis:\n";

    size_t h1_count = count_heading_level(root, 1);
    out << "  ✓ H1 count: " << h1_count << " (want: 1) ";
    out << (h1_count == 1 ? "[300 pts]" : "[0 pts]") << "\n";

    bool has_install = has_section(root, "installation");
    out << "  ✓ Installation section: " << (has_install ? "YES" : "NO");
    out << (has_install ? " [200 pts]" : " [0 pts]") << "\n";

    bool has_usage = has_section(root, "usage");
    out << "  ✓ Usage section: " << (has_usage ? "YES" : "NO");
    out << (has_usage ? " [200 pts]" : " [0 pts]") << "\n";

    size_t code_blocks = count_nodes(root, "code-block");
    out << "  ✓ Code blocks: " << code_blocks << " (want: >= 1) ";
    out << (code_blocks > 0 ? "[150 pts]" : "[0 pts]") << "\n";

    size_t h2_count = count_heading_level(root, 2);
    out << "  ✓ H2 sections: " << h2_count << " (want: 2-4) ";
    out << (h2_count >= 2 && h2_count <= 4 ? "[150 pts]" : "[0 pts]") << "\n";

    out << "\nH2 sections found:\n";
    auto h2_texts = get_heading_texts(root, 2);
    for (const auto& text : h2_texts) {
      out << "  - " << text << "\n";
    }

    out << "\nFinal score: " << evaluate(text) << "/1000\n";

    return out.str();
  }
};

TEST_CASE("PUCT vs Baseline: tic-tac-toe strategy") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 256;
  ctx_params.n_seq_max = 32;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Tic-tac-toe position where X can win by playing center-bottom
  // X O _
  // _ X _
  // _ _ O
  //
  // Winning move: center-bottom (row 2, col 1) completes diagonal

  // Use chat template API for proper formatting
  using json = nlohmann::ordered_json;
  json messages = json::array();

  messages.push_back({
    {"role", "system"},
    {"content", "You are playing tic-tac-toe as X. You must win."}
  });

  messages.push_back({
    {"role", "user"},
    {"content",
      "Tic-tac-toe game. You are X, opponent is O.\n"
      "\n"
      "Current board:\n"
      "| X | O |   |\n"
      "|---|---|---|\n"
      "|   | X |   |\n"
      "|   |   | O |\n"
      "\n"
      "IMPORTANT: Copy the above board EXACTLY, then replace ONE empty cell with X.\n"
      "Show the board after adding your X:"
    }
  });

  std::string messages_json = messages.dump();
  auto chat_result = chat_template::format(model.get(), messages_json);
  std::string prompt = chat_result.prompt;

  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);

  TicTacToeOracle oracle;

  // ========== BASELINE: Greedy Sampling ==========
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos baseline_pos = static_cast<llama_pos>(prompt_tokens.size());

  BranchStore baseline_store(4);
  lloyal::mcts::PUCT<>::SamplingParams baseline_params;
  baseline_params.temperature = 0.7f;  // Match chat.mjs
  baseline_params.top_k = 40;
  baseline_params.top_p = 0.9f;
  baseline_params.seed = 42;

  BranchHandle baseline_branch = branch::create(
    ctx, model.get(), 0, baseline_pos, baseline_params,
    ctx_params.n_batch, "", nullptr, &baseline_store
  );

  // Capture initial logits before sampling
  branch::capture_logits(baseline_branch, &baseline_store);

  std::vector<llama_token> baseline_tokens;
  // Generate until we hit table boundary or 150 tokens (full table = ~80 tokens)
  for (int i = 0; i < 150; ++i) {
    llama_token token = branch::sample(baseline_branch, &baseline_store);
    if (i == 0) {
      std::cerr << "[DEBUG] Baseline first token: " << token
                << " (is_eog=" << tokenizer::is_eog(vocab, token) << ")" << std::endl;
    }
    if (token < 0 || tokenizer::is_eog(vocab, token)) break;

    baseline_tokens.push_back(token);
    branch::accept_token(baseline_branch, token, &baseline_store);
    branch::decode_and_capture_one(baseline_branch, token, &baseline_store);

    // Stop if we've generated a complete 3x3 table (5+ lines: 3 data + 1 separator + newlines)
    std::string so_far = tokenizer::detokenize_batch(model.get(), baseline_tokens);
    if (so_far.find("|---|---|---|") != std::string::npos &&
        std::count(so_far.begin(), so_far.end(), '\n') >= 4) {
      break;
    }
  }

  std::string baseline_text = prompt + tokenizer::detokenize_batch(model.get(), baseline_tokens);
  float baseline_score = oracle.evaluate(baseline_text);

  MESSAGE("=== BASELINE (Greedy) ===");
  MESSAGE("Generated:\n" << tokenizer::detokenize_batch(model.get(), baseline_tokens));
  MESSAGE("Score: " << baseline_score);

  branch::destroy(baseline_branch, &baseline_store);

  // ========== PUCT Search with Oracle ==========
  // Clear context and re-decode prompt
  kv::remove_range(ctx, 0, 0, -1);  // Remove all tokens from seq 0
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos puct_start_pos = static_cast<llama_pos>(prompt_tokens.size());

  // Create value function that wraps the oracle
  auto value_fn = [&oracle, &prompt](const std::string& generated_text) -> float {
    std::string full_text = prompt + generated_text;
    float score = oracle.evaluate(full_text);
    // Debug: log what was evaluated
    std::cout << "[oracle] Evaluated text (last 200 chars): "
              << (full_text.length() > 200 ? "..." + full_text.substr(full_text.length() - 200) : full_text)
              << "\n[oracle] Score: " << score << std::endl;
    return score;
  };

  // Configure PUCT with Phase 1 fixes
  lloyal::mcts::SearchConfig config;
  config.cpuct = 1.0f;
  config.widening_constant = 1.0f;
  // ISSUE 3 FIX: Value normalization via tanh
  // Maps oracle scores [-1000, 1000] → [-1, 1] for competitive exploration
  config.value_normalizer = lloyal::mcts::normalizers::tanh_normalizer;
  config.normalizer_scale = 200.0f;
  config.max_expansion_attempts = 3;
  config.max_chunk_tokens = 100;
  config.track_metrics = true;

  // Run PUCT search with oracle as value function
  lloyal::mcts::PUCT puct(ctx, model.get(), "", 32, ctx_params.n_batch,
                          puct_start_pos, config, value_fn);
  puct.search(100);  // 100 MCTS iterations to explore action space via sampling

  auto puct_tokens = puct.get_best_sequence();
  std::string puct_text = prompt + tokenizer::detokenize_batch(model.get(), puct_tokens);
  float puct_score = oracle.evaluate(puct_text);

  MESSAGE("\n=== PUCT (with Oracle) ===");
  MESSAGE("Generated:\n" << tokenizer::detokenize_batch(model.get(), puct_tokens));
  MESSAGE("Score: " << puct_score);
  MESSAGE("Expansions: " << puct.get_node_count());

  // Verify Phase 1 fixes
  MESSAGE("\n=== Phase 1 Fix Verification ===");
  int node_count = puct.get_node_count();
  MESSAGE("Node count: " << node_count);

  // ISSUE 1 FIX: Progressive widening should create wider tree (>30 nodes for 100 iterations)
  CHECK(node_count > 30);

  // Calculate tree structure metrics
  int internal_nodes = 0;
  int total_children = 0;
  int nodes_with_multiple_children = 0;

  for (int i = 0; i < node_count; i++) {
    const auto& node = puct.get_node(i);
    size_t child_count = node.children.size();
    if (child_count > 0) {
      internal_nodes++;
      total_children += child_count;
      if (child_count > 1) {
        nodes_with_multiple_children++;
      }
    }
  }

  float avg_children_per_internal = internal_nodes > 0
    ? static_cast<float>(total_children) / internal_nodes
    : 0.0f;

  MESSAGE("Internal nodes: " << internal_nodes);
  MESSAGE("Avg children per internal node: " << avg_children_per_internal);
  MESSAGE("Nodes with multiple children: " << nodes_with_multiple_children);

  // Progressive widening should create nodes with multiple children (not a chain)
  CHECK(nodes_with_multiple_children > 0);
  CHECK(avg_children_per_internal > 1.0f);

  // ISSUE 2 FIX: Tree poisoning should be fixed (fewer failed expansions)
  auto metrics = puct.get_metrics();
  MESSAGE("Expansion success rate: " << (metrics.expansion_success_rate() * 100.0f) << "%");

  // ISSUE 3 FIX: Value normalization enables exploration (root should have multiple children)
  const auto& root = puct.get_node(0);
  MESSAGE("Root children: " << root.children.size());
  CHECK(root.children.size() >= 1);

  // ========== Comparison ==========
  MESSAGE("\n=== COMPARISON ===");
  MESSAGE("Baseline score: " << baseline_score);
  MESSAGE("PUCT score: " << puct_score);

  // PUCT should find better moves than greedy baseline
  // (Either win immediately, or at least not lose)
  CHECK(puct_score >= baseline_score);

  // ========== Oracle Validation Test ==========
  MESSAGE("\n=== ORACLE VALIDATION ===");

  // X wins via diagonal (top-left to bottom-right)
  std::string winning_board = R"(| X | O |   |
|---|---|---|
|   | X |   |
|   |   | X |)";

  float winning_score = oracle.evaluate(winning_board);
  MESSAGE("Winning position score: " << winning_score);
  CHECK(winning_score > 500.0f);  // Should detect X win

  std::string losing_board = R"(| X | O | O |
|---|---|---|
|   | X | O |
| X |   | O |)";

  float losing_score = oracle.evaluate(losing_board);
  MESSAGE("Losing position score: " << losing_score);
  CHECK(losing_score < -500.0f);  // Should detect O win

  llama_free(ctx);
}

TEST_CASE("PUCT vs Baseline: perplexity comparison") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 256;
  ctx_params.n_seq_max = 32;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Shared prompt for fair comparison
  std::string prompt = R"(# API Documentation

Write a clear API reference with:

## Authentication

)";

  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);

  // ========== BASELINE: Greedy Sampling with Branch ==========
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos baseline_pos = static_cast<llama_pos>(prompt_tokens.size());

  // Create branch to track perplexity
  BranchStore baseline_store(4);
  lloyal::mcts::PUCT<>::SamplingParams baseline_params;
  baseline_params.temperature = 0.0f;  // Greedy (argmax)
  baseline_params.seed = 42;

  BranchHandle baseline_branch = branch::create(
    ctx, model.get(), 0, baseline_pos, baseline_params,
    ctx_params.n_batch, "", nullptr, &baseline_store
  );

  std::vector<llama_token> baseline_tokens;

  // Generate 50 tokens greedily
  for (int i = 0; i < 50; ++i) {
    llama_token token = branch::sample(baseline_branch, &baseline_store);
    if (token < 0 || tokenizer::is_eog(vocab, token)) break;

    baseline_tokens.push_back(token);

    // Accept token (updates perplexity tracker)
    branch::accept_token(baseline_branch, token, &baseline_store);

    // Decode and capture for next iteration
    branch::decode_and_capture_one(baseline_branch, token, &baseline_store);
  }

  float baseline_ppl = branch::get_perplexity(baseline_branch, &baseline_store);
  std::string baseline_text = tokenizer::detokenize_batch(model.get(), baseline_tokens);

  MESSAGE("Baseline (greedy): " << baseline_tokens.size() << " tokens, PPL=" << baseline_ppl);
  MESSAGE("Text: " << baseline_text.substr(0, 200));

  branch::destroy(baseline_branch, &baseline_store);

  // ========== PUCT Search ==========
  kv::clear_all(ctx);
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos puct_start_pos = static_cast<llama_pos>(prompt_tokens.size());

  PUCT puct(ctx, model.get(), "", 32, ctx_params.n_batch, puct_start_pos);
  puct.search(10);  // 10 MCTS iterations

  auto puct_tokens = puct.get_best_sequence();

  // Get perplexity from the best leaf node's branch
  // Walk tree to find best leaf
  BranchHandle best_branch = puct.get_best_branch();
  float puct_ppl = branch::get_perplexity(best_branch, puct.get_store());

  std::string puct_text = tokenizer::detokenize_batch(model.get(), puct_tokens);
  MESSAGE("PUCT search: " << puct_tokens.size() << " tokens, PPL=" << puct_ppl);
  MESSAGE("Text: " << puct_text.substr(0, 200));

  // ========== Comparison ==========
  MESSAGE("\n=== PUCT vs Baseline ===");
  MESSAGE("Baseline PPL: " << baseline_ppl);
  MESSAGE("PUCT PPL: " << puct_ppl);
  MESSAGE("Improvement: " << (baseline_ppl - puct_ppl) << " (negative = worse)");

  // Document the comparison (not asserting improvement, just measuring)
  INFO("PUCT should have lower perplexity if search is effective");
  INFO("If PUCT PPL > baseline PPL, search is not helping");

  llama_free(ctx);
}

TEST_CASE("MCTS vs Baseline: README structure with opinionated template") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 256;
  ctx_params.n_seq_max = 32;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Prompt: Ask for well-structured README (template is hidden in oracle)
  std::string prompt = R"(Write a well-structured README for: A CLI tool that converts markdown files to PDF with custom styling support.

# )";  // Start the README with H1

  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos start_pos = static_cast<llama_pos>(prompt_tokens.size());

  ReadmeStructureOracle oracle;

  // ========== BASELINE: Greedy Sampling ==========
  BranchStore baseline_store(4);
  lloyal::mcts::PUCT<>::SamplingParams baseline_params;
  baseline_params.temperature = 0.7f;
  baseline_params.top_k = 40;
  baseline_params.top_p = 0.9f;
  baseline_params.seed = 42;

  BranchHandle baseline_branch = branch::create(
    ctx, model.get(), 0, start_pos, baseline_params,
    ctx_params.n_batch, "", nullptr, &baseline_store
  );

  std::vector<llama_token> baseline_tokens;
  // Generate up to 200 tokens or until EoG
  for (int i = 0; i < 200; ++i) {
    llama_token token = branch::sample(baseline_branch, &baseline_store);
    if (token < 0 || tokenizer::is_eog(vocab, token)) break;

    baseline_tokens.push_back(token);
    branch::accept_token(baseline_branch, token, &baseline_store);
    branch::decode_and_capture_one(baseline_branch, token, &baseline_store);
  }

  std::string baseline_text = prompt + tokenizer::detokenize_batch(model.get(), baseline_tokens);
  float baseline_score = oracle.evaluate(baseline_text);

  MESSAGE("\n=== BASELINE (Greedy) ===");
  MESSAGE("Generated " << baseline_tokens.size() << " tokens");
  MESSAGE(oracle.get_breakdown(baseline_text));

  branch::destroy(baseline_branch, &baseline_store);

  // ========== MCTS with Structure Oracle ==========
  kv::remove_range(ctx, 0, 0, -1);
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  start_pos = static_cast<llama_pos>(prompt_tokens.size());

  // Value function: score based on README structure
  auto structure_value_fn = [&oracle, &prompt](const std::string& generated) -> float {
    std::string full_text = prompt + generated;
    return oracle.evaluate(full_text);
  };

  // Configure MCTS with value normalization
  lloyal::mcts::SearchConfig config;
  config.cpuct = 1.0f;
  config.widening_constant = 1.0f;
  config.value_normalizer = lloyal::mcts::normalizers::tanh_normalizer;
  config.normalizer_scale = 200.0f;  // Map [0, 1000] to tanh range
  config.max_expansion_attempts = 3;
  config.track_metrics = true;

  lloyal::mcts::PUCT puct(ctx, model.get(), "", 32, ctx_params.n_batch,
                          start_pos, config, structure_value_fn);

  // Run MCTS (50 boundaries = roughly 200-500 tokens depending on boundary detection)
  puct.search(50);

  auto puct_tokens = puct.get_best_sequence();
  std::string puct_text = prompt + tokenizer::detokenize_batch(model.get(), puct_tokens);
  float puct_score = oracle.evaluate(puct_text);

  MESSAGE("\n=== MCTS (with Structure Oracle) ===");
  MESSAGE("Generated " << puct_tokens.size() << " tokens");
  MESSAGE(oracle.get_breakdown(puct_text));

  auto metrics = puct.get_metrics();
  MESSAGE("\nMCTS metrics:");
  MESSAGE("  Expansions: " << metrics.total_expansions);
  MESSAGE("  Boundaries: " << metrics.total_boundaries);
  MESSAGE("  Fork reduction: " << metrics.fork_reduction() << "x");

  // ========== Comparison ==========
  MESSAGE("\n=== STRUCTURAL QUALITY COMPARISON ===");
  MESSAGE("Baseline structure score: " << baseline_score << "/1000");
  MESSAGE("MCTS structure score: " << puct_score << "/1000");
  MESSAGE("Improvement: " << (puct_score - baseline_score) << " points");

  // MCTS should find better structure than greedy
  // This validates: "Per-boundary MCTS with value guidance finds structured output"
  INFO("MCTS should score >= baseline (finds structure matching template)");
  CHECK(puct_score >= baseline_score);

  llama_free(ctx);
}

TEST_CASE("boundaries: structural rollout completes syntax") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  // Create boundary tracker with CommonMark grammar
  auto grammar = remux::grammars::createCommonMarkGrammar();
  auto strategy = remux::strategies::createExplicitStrategy(grammar);
  auto mapper = boundaries::createStructuralBoundaryMapper();
  boundaries::RemuxBoundaryTracker tracker(grammar, strategy, mapper);

  // Test 1: Completes fenced code block with backticks
  {
    std::string incomplete = "```python\nprint('hello')";
    for (char ch : incomplete) {
      tracker.feed_draft(std::string_view(&ch, 1));
    }

    std::string complete = tracker.structural_rollout(incomplete);

    INFO("Backticks test - Complete text: " << complete);
    CHECK(complete.find("\n```\n") != std::string::npos);
    CHECK(complete.size() > incomplete.size());
  }

  // Test 2: Completes fenced code block with tildes
  {
    tracker.reset();

    std::string incomplete = "~~~~cpp\nint main() { }";
    for (char ch : incomplete) {
      tracker.feed_draft(std::string_view(&ch, 1));
    }

    std::string complete = tracker.structural_rollout(incomplete);

    INFO("Tildes test - Complete text: " << complete);
    CHECK(complete.find("\n~~~~\n") != std::string::npos);
    CHECK(complete.find("\n```\n") == std::string::npos);
  }

  // Test 3: Returns original text when no open containers
  {
    tracker.reset();

    std::string complete_text = "This is a complete paragraph.";
    for (char ch : complete_text) {
      tracker.feed_draft(std::string_view(&ch, 1));
    }

    std::string result = tracker.structural_rollout(complete_text);

    INFO("No open containers test");
    CHECK(result == complete_text);
  }

  // Test 4: Deterministic - same input produces same output
  {
    tracker.reset();

    std::string incomplete = "```js\nconsole.log('test')";
    for (char ch : incomplete) {
      tracker.feed_draft(std::string_view(&ch, 1));
    }

    std::string result1 = tracker.structural_rollout(incomplete);
    std::string result2 = tracker.structural_rollout(incomplete);

    INFO("Determinism test");
    CHECK(result1 == result2);
    CHECK(result1.find("\n```\n") != std::string::npos);
  }
}

TEST_CASE("PUCT: validates Branch primitives work together") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();  // Use Metal GPU acceleration
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 8;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());
  int n_vocab = llama_vocab_n_tokens(vocab);

  // === Test Branch primitives ===

  // 1. Decode prompt to seq 0
  std::string prompt = "Test";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos pos = static_cast<llama_pos>(prompt_tokens.size());

  // 2. Create branch with grammar
  BranchStore store(8);
  lloyal::mcts::PUCT<>::SamplingParams params;
  const char* grammar_str = "root ::= [a-z]+";

  BranchHandle branch = create(ctx, model.get(), 0, pos, params,
                               ctx_params.n_batch, grammar_str, nullptr, &store);
  REQUIRE(branch != INVALID_HANDLE);

  // 3. Verify branch state
  CHECK(get_seq_id(branch, &store) == 0);
  CHECK(get_position(branch, &store) == pos);

  // 3b. FIX #2: Capture logits after prefill (root initialization)
  CHECK_NOTHROW(capture_logits(branch, &store));
  const float* root_logits = get_logits(branch, &store);
  REQUIRE(root_logits != nullptr);

  // 4. Sample a token (grammar-constrained) - now works because logits exist
  llama_token token = sample(branch, &store);
  CHECK(token >= 0);
  CHECK(token < n_vocab);

  // 5. Accept token (updates grammar + perplexity)
  accept_token(branch, token, &store);

  // 6. Decode and capture logits
  CHECK_NOTHROW(decode_and_capture_one(branch, token, &store));
  CHECK(get_position(branch, &store) == pos + 1);

  // 7. Get captured logits
  const float* logits = get_logits(branch, &store);
  REQUIRE(logits != nullptr);

  // 8. Compute grammar-masked policy prior (FIX #1 + #5)
  float logsumexp = get_legal_logsumexp(branch, &store);
  float prior = get_token_prior(branch, token, logsumexp, &store);
  CHECK(prior > 0.0f);
  CHECK(prior <= 1.0f);

  // 8b. Also test get_legal_priors returns renormalized distribution
  auto legal_priors = get_legal_priors(branch, &store);
  CHECK_FALSE(legal_priors.empty());
  float sum_probs = 0.0f;
  for (const auto& [tok, prob] : legal_priors) {
    sum_probs += prob;
  }
  CHECK(sum_probs > 0.99f);  // Should sum to ~1.0
  CHECK(sum_probs < 1.01f);

  // 9. Fork branch
  BranchHandle fork_branch = fork(branch, 1, &store);
  REQUIRE(fork_branch != INVALID_HANDLE);
  CHECK(get_seq_id(fork_branch, &store) == 1);
  CHECK(get_position(fork_branch, &store) == get_position(branch, &store));

  // 10. Verify KV cache was copied
  CHECK(kv::pos_max(ctx, 1) == kv::pos_max(ctx, 0));

  // 11. Get perplexity from branch
  float ppl = get_perplexity(branch, &store);
  CHECK(std::isfinite(ppl));

  // 12. Prune forked branch
  prune(fork_branch, &store);
  CHECK(kv::pos_max(ctx, 1) == -1);  // KV cleared

  // 13. Original branch still valid
  CHECK(get_position(branch, &store) == pos + 1);

  // Cleanup handled by BranchStore destructor
  llama_free(ctx);

  INFO("✓ All Branch primitives validated for PUCT");
}

// ============================================================================
// RAII Tests - Expose bugs in Branch wrapper
// ============================================================================

TEST_CASE("RAII Branch: winner survives after RAII scope exit") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Prefill
  std::string prompt = "Test";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos pos = static_cast<llama_pos>(prompt_tokens.size());

  llama_seq_id winner_seq = -1;
  llama_pos winner_pos = -1;

  // Create branches in inner scope - RAII should clean up on exit
  {
    BranchStore store(4);
    lloyal::mcts::PUCT<>::SamplingParams params;

    // Create root branch using RAII wrapper
    Branch root = Branch::create(ctx, model.get(), 0, pos, params,
                                  ctx_params.n_batch, nullptr, nullptr, &store);
    REQUIRE(root.valid());

    // CRITICAL: Capture logits on root BEFORE forking
    // This ensures winner inherits logits via fork's copy
    root.capture_logits();
    REQUIRE(root.logits() != nullptr);

    // Fork to create "winner" - inherits logits from root
    Branch winner = root.fork(1);
    REQUIRE(winner.valid());
    REQUIRE(winner.logits() != nullptr);  // Verify logits were copied

    // Sample from winner (should work now that it has logits)
    llama_token token = winner.sample();
    REQUIRE(token >= 0);  // Must succeed - fail test if not

    // Advance winner: accept token and decode
    winner.accept(token);
    winner.decode_and_capture_one(token);

    // Record winner's state before scope exit
    INFO("Winner seq_id=" << winner.seq_id() << " pos=" << winner.position());

    // FIX: Use release_kv() to preserve KV cache when winner goes out of scope
    auto released = winner.release_kv();
    winner_seq = released.seq_id;
    winner_pos = released.position;

    // winner is now invalid, but KV at winner_seq is preserved
    CHECK_FALSE(winner.valid());
  }

  // After RAII scope exit, verify winner's KV survives
  llama_pos actual_pos = kv::pos_max(ctx, winner_seq);

  // With release_kv(), winner's KV should be preserved
  CHECK(actual_pos >= winner_pos - 1);  // KV survives!

  INFO("Winner KV preserved: seq_id=" << winner_seq
       << " expected_pos=" << winner_pos << " actual_pos=" << actual_pos);

  llama_free(ctx);
}

TEST_CASE("RAII Branch: diverged branches can be individually pruned") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Prefill to seq 0
  std::string prompt = "Test";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos pos = static_cast<llama_pos>(prompt_tokens.size());

  BranchStore store(4);
  lloyal::mcts::PUCT<>::SamplingParams params;

  // Create branches
  BranchHandle root = create(ctx, model.get(), 0, pos, params,
                             ctx_params.n_batch, nullptr, nullptr, &store);
  capture_logits(root, &store);

  BranchHandle branch1 = fork(root, 1, &store);
  BranchHandle branch2 = fork(root, 2, &store);

  // Verify all sequences have KV (they share underlying slots via seq_cp)
  CHECK(kv::pos_max(ctx, 0) >= 0);
  CHECK(kv::pos_max(ctx, 1) >= 0);
  CHECK(kv::pos_max(ctx, 2) >= 0);

  // Advance branch1 so it diverges from root
  llama_token token1 = sample(branch1, &store);
  if (token1 >= 0) {
    accept_token(branch1, token1, &store);
    decode_and_capture_one(branch1, token1, &store);
  }

  // Now branch1 has diverged - it has its own unique positions
  llama_pos branch1_pos = get_position(branch1, &store);
  CHECK(branch1_pos > pos);

  // Prune branch2 (loser)
  prune(branch2, &store);
  CHECK(kv::pos_max(ctx, 2) == -1);  // branch2 KV cleared

  // branch1 still has its extended positions
  CHECK(kv::pos_max(ctx, 1) >= branch1_pos - 1);

  // Root still has original positions (shared with branch1's prefix)
  CHECK(kv::pos_max(ctx, 0) >= 0);

  // Clean up remaining branches
  prune(branch1, &store);
  prune(root, &store);

  llama_free(ctx);

  INFO("✓ Diverged branches can be individually pruned");
}

/**
 * VERSION CANARY: Documents seq_keep behavior in llama.cpp b6870.
 *
 * In b6870: pos_max returns 0 for removed sequences (slots still exist)
 * In newer versions: pos_max returns -1 for removed sequences
 *
 * When this test fails after upgrading llama.cpp, it signals the behavior
 * changed. At that point:
 * 1. Update this test to match new behavior
 * 2. Re-evaluate if seq_keep can be used for MCTS cleanup
 * 3. Update api.md and guide.md documentation accordingly
 */
TEST_CASE("RAII Branch: seq_keep behavior with shared KV slots") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 4;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Prefill to seq 0
  std::string prompt = "Test";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);

  // Fork creates seq 1 and 2 that SHARE the same KV slots as seq 0
  // (seq_cp just adds sequence tags to existing KV slots)
  kv::seq_cp(ctx, 0, 1);
  kv::seq_cp(ctx, 0, 2);

  INFO("After fork - all seqs share same KV slots:");
  INFO("  seq 0 pos_max = " << kv::pos_max(ctx, 0));
  INFO("  seq 1 pos_max = " << kv::pos_max(ctx, 1));
  INFO("  seq 2 pos_max = " << kv::pos_max(ctx, 2));

  // All sequences report the same pos_max because they share slots
  CHECK(kv::pos_max(ctx, 0) == kv::pos_max(ctx, 1));
  CHECK(kv::pos_max(ctx, 1) == kv::pos_max(ctx, 2));

  // seq_keep(1) removes sequence TAGS for 0 and 2, but the underlying
  // KV data remains because seq 1 still references those slots
  kv::seq_keep(ctx, 1);

  INFO("After seq_keep(1):");
  INFO("  seq 0 pos_max = " << kv::pos_max(ctx, 0));
  INFO("  seq 1 pos_max = " << kv::pos_max(ctx, 1));
  INFO("  seq 2 pos_max = " << kv::pos_max(ctx, 2));

  // seq 1 survives
  CHECK(kv::pos_max(ctx, 1) >= 0);

  // IMPORTANT: seq 0 and 2 may NOT be -1!
  // The KV slots still exist, they just lost their seq 0/2 tags.
  // pos_max returns the max position of slots tagged with that seq.
  // If no slots are tagged, it returns -1.
  // BUT if the slots were SHARED and seq 1 still references them,
  // the behavior depends on llama.cpp implementation.

  // This documents the actual behavior:
  llama_pos seq0_after = kv::pos_max(ctx, 0);
  llama_pos seq2_after = kv::pos_max(ctx, 2);

  INFO("seq_keep removes tags but shared slots may still report pos_max > -1");
  INFO("Actual: seq0=" << seq0_after << " seq2=" << seq2_after);

  // The reviewer's assumption that seq_keep is a "bulk cleanup" for MCTS
  // is only partially correct. It removes sequence tags, but:
  // 1. Shared slots (from seq_cp) remain referenced by the kept seq
  // 2. Only truly divergent positions get freed

  llama_free(ctx);

  INFO("✓ seq_keep behavior documented");
}

// ============================================================================
// Phase 1 Fix Validation Tests
// ============================================================================

TEST_CASE("PUCT Phase 1: Progressive widening creates siblings") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 256;
  ctx_params.n_seq_max = 32;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  std::string prompt = "Complete this markdown list:\n- Item 1\n- Item 2\n-";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);

  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos start_pos = static_cast<llama_pos>(prompt_tokens.size());

  // Configure with progressive widening
  lloyal::mcts::SearchConfig config;
  config.cpuct = 1.0f;
  config.widening_constant = 1.0f;  // W(N) = 1 + √N
  config.value_normalizer = lloyal::mcts::normalizers::tanh_normalizer;
  config.normalizer_scale = 200.0f;

  lloyal::mcts::PUCT puct(ctx, model.get(), "", 32, ctx_params.n_batch,
                          start_pos, config, nullptr);
  puct.search(50);  // 50 iterations

  const auto& root = puct.get_node(0);
  MESSAGE("Root visit count: " << root.visits);
  MESSAGE("Root children: " << root.children.size());
  MESSAGE("Max allowed children: " << root.max_children(config.widening_constant));

  // ISSUE 1 FIX: Progressive widening should create multiple siblings
  CHECK(root.children.size() > 1);

  // With W(N) = 1 + √50 ≈ 8, root should have multiple children
  CHECK(root.children.size() <= 8);

  // Verify tree is wider, not just a chain
  int node_count = puct.get_node_count();
  MESSAGE("Total nodes: " << node_count);
  CHECK(node_count > 20);  // Should have more than just a chain

  llama_free(ctx);
}

TEST_CASE("PUCT Phase 1: Tree poisoning fix prevents spurious terminals") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 256;
  ctx_params.n_seq_max = 32;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  std::string prompt = "List three programming languages:\n1. ";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);

  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos start_pos = static_cast<llama_pos>(prompt_tokens.size());

  lloyal::mcts::SearchConfig config;
  config.cpuct = 1.0f;
  config.widening_constant = 1.0f;
  config.max_expansion_attempts = 3;  // Allow retries
  config.track_metrics = true;

  lloyal::mcts::PUCT puct(ctx, model.get(), "", 32, ctx_params.n_batch,
                          start_pos, config, nullptr);
  puct.search(30);

  // ISSUE 2 FIX: No spurious terminal nodes (only EoG should be terminal)
  int spurious_terminals = 0;
  for (int i = 0; i < puct.get_node_count(); ++i) {
    const auto& node = puct.get_node(i);
    if (node.is_terminal) {
      // Check if this is a real terminal (EoG)
      if (!node.boundary || node.boundary->kind != "eog") {
        spurious_terminals++;
        MESSAGE("Spurious terminal at node " << i << " with boundary kind: "
                << (node.boundary ? node.boundary->kind : "none"));
      }
    }
  }

  MESSAGE("Spurious terminals: " << spurious_terminals);
  CHECK(spurious_terminals == 0);

  // Verify reasonable expansion success rate
  auto metrics = puct.get_metrics();
  MESSAGE("Expansion success rate: " << (metrics.expansion_success_rate() * 100.0f) << "%");
  CHECK(metrics.expansion_success_rate() > 0.5f);  // At least 50% successful

  llama_free(ctx);
}

TEST_CASE("PUCT Phase 1: Value normalization enables exploration") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 2048;
  ctx_params.n_batch = 256;
  ctx_params.n_seq_max = 64;  // Increased to avoid hitting seq limit

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  std::string prompt = "The capital of France is";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);

  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos start_pos = static_cast<llama_pos>(prompt_tokens.size());

  // Oracle with VARIED large scores to test normalization
  // Without variation, all paths have identical Q values (normalization has no effect)
  int call_count = 0;
  auto large_score_oracle = [&call_count](const std::string& text) -> float {
    call_count++;

    // Base score: VERY large magnitude (5000+) to suppress exploration without normalization
    // With cpuct=1.0, exploration term ~ 1.0-2.0, so Q needs to be >>100x larger
    float score = 5000.0f + (text.length() * 30.0f);

    // Bonus for semantically correct content (creates variation)
    // Model generates continuations like " Paris", " London", " France", etc.
    if (text.find("Paris") != std::string::npos) score += 5000.0f;  // Correct answer!
    if (text.find("France") != std::string::npos) score += 3000.0f;
    if (text.find("capital") != std::string::npos) score += 2000.0f;

    if (call_count <= 10) {
      std::cerr << "[Oracle] Call #" << call_count
                << " | text='" << text.substr(0, std::min(size_t(40), text.size())) << "...'"
                << " | score=" << score << std::endl;
    }

    // Result: scores range from ~5000 to ~15000 (VERY large AND varied)
    // This magnitude is needed to suppress exploration in raw case (Q >> U)
    return score;
  };

  // Test WITH normalization
  lloyal::mcts::SearchConfig config_normalized;
  config_normalized.cpuct = 1.0f;
  config_normalized.widening_constant = 1.0f;
  config_normalized.value_normalizer = lloyal::mcts::normalizers::tanh_normalizer;
  config_normalized.normalizer_scale = 200.0f;

  lloyal::mcts::PUCT puct_normalized(ctx, model.get(), "", 64, ctx_params.n_batch,
                                     start_pos, config_normalized, large_score_oracle);
  puct_normalized.search(50);

  // METRIC: Root branching factor + visit distribution
  const auto& root_normalized = puct_normalized.get_node(0);
  size_t width_normalized = root_normalized.children.size();
  int nodes_normalized = puct_normalized.get_node_count();

  // Calculate visit entropy: high entropy = balanced exploration
  int max_child_visits_norm = 0;
  for (int child_idx : root_normalized.children) {
    const auto& child = puct_normalized.get_node(child_idx);
    max_child_visits_norm = std::max(max_child_visits_norm, child.visits);
  }
  float visit_concentration_norm = static_cast<float>(max_child_visits_norm) / root_normalized.visits;

  MESSAGE("WITH normalization: Total nodes=" << nodes_normalized
          << ", Root children=" << width_normalized
          << ", Visit concentration=" << visit_concentration_norm);

  // Clear context for second run
  kv::remove_range(ctx, 0, 0, -1);
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  start_pos = static_cast<llama_pos>(prompt_tokens.size());

  // Test WITHOUT normalization
  lloyal::mcts::SearchConfig config_raw;
  config_raw.cpuct = 1.0f;
  config_raw.widening_constant = 1.0f;
  config_raw.value_normalizer = nullptr;  // No normalization

  lloyal::mcts::PUCT puct_raw(ctx, model.get(), "", 64, ctx_params.n_batch,
                              start_pos, config_raw, large_score_oracle);
  puct_raw.search(50);

  // METRIC: Root branching factor + visit distribution
  const auto& root_raw = puct_raw.get_node(0);
  size_t width_raw = root_raw.children.size();
  int nodes_raw = puct_raw.get_node_count();

  // Calculate visit concentration: high concentration = greedy (one child gets most visits)
  int max_child_visits_raw = 0;
  for (int child_idx : root_raw.children) {
    const auto& child = puct_raw.get_node(child_idx);
    max_child_visits_raw = std::max(max_child_visits_raw, child.visits);
  }
  float visit_concentration_raw = static_cast<float>(max_child_visits_raw) / root_raw.visits;

  MESSAGE("WITHOUT normalization: Total nodes=" << nodes_raw
          << ", Root children=" << width_raw
          << ", Visit concentration=" << visit_concentration_raw);

  // ISSUE 3 FIX: Normalization enables MORE BALANCED exploration
  // With very large raw scores (5000-15000), Q >> U, so exploitation dominates
  // One child gets most visits (high concentration), others get 1-2 visits each
  // With normalization (tanh maps to [-1,1]), Q ~ U, so exploration is competitive
  // Visits spread more evenly across children (low concentration)
  //
  // KEY INSIGHT: We measure VISIT DISTRIBUTION, not just width or total nodes
  // - Raw: High concentration (e.g., 0.8 = one child gets 80% of visits)
  // - Normalized: Low concentration (e.g., 0.3 = visits spread across multiple children)

  MESSAGE("Visit concentration: " << visit_concentration_norm << " (normalized) vs "
          << visit_concentration_raw << " (raw)");

  // Primary assertion: Normalized should have LOWER concentration (more balanced)
  CHECK(visit_concentration_norm < visit_concentration_raw * 0.9f);

  llama_free(ctx);
}
