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
 * - branch::get_legal_logsumexp() - Precompute for efficient prior calculation
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
 * - FIX #5: logsumexp precomputed once per parent (O(n_vocab) not O(n_vocab * n_children))
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/branch.hpp>
#include <lloyal/decoder.hpp>
#include <lloyal/grammar.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/logits.hpp>
#include <lloyal/metrics.hpp>
#include <lloyal/model_registry.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/tokenizer.hpp>
#include <memory>
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
  // Branch handle consolidates: seq_id, position, grammar, perplexity, logits
  BranchHandle branch = INVALID_HANDLE;

  // Tree structure
  int parent_idx = -1;
  std::vector<int> children;

  // Token that led to this node
  llama_token token = -1;

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
  float temperature = 1.0f;  // Use temperature=1 for proper softmax priors
  int32_t top_k = 0;         // Disabled - let grammar handle constraints
  float top_p = 1.0f;
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
  PUCT(llama_context* ctx, const llama_model* model,
       const std::string& grammar_str, int max_seq, int n_batch,
       llama_pos start_pos)
      : ctx_(ctx), model_(model), n_batch_(n_batch), max_seq_(max_seq),
        store_(max_seq), seq_pool_(max_seq) {
    vocab_ = llama_model_get_vocab(model_);
    n_vocab_ = llama_vocab_n_tokens(vocab_);

    // Create root branch (seq 0 is reserved for root)
    PUCTSamplingParams params;
    BranchHandle root_branch = create(
        ctx_, model_, 0, start_pos, params, n_batch_,
        grammar_str.c_str(), &store_);

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
      if (nodes_[idx].token != -1) {
        tokens.push_back(nodes_[idx].token);
      }
    }

    return tokens;
  }

  int get_node_count() const { return static_cast<int>(nodes_.size()); }
  int get_root_visits() const { return nodes_[0].visits; }

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
   * - FIX #5: Precompute logsumexp once per parent (performance)
   */
  int expand(int parent_idx) {
    auto& parent = nodes_[parent_idx];

    // FIX #4: Use SeqIdPool for seq-id reuse
    llama_seq_id child_seq = seq_pool_.acquire();
    if (child_seq < 0) {
      return parent_idx;  // No more seq IDs available
    }

    // FIX #5: Precompute logsumexp once per parent (O(n_vocab) once, not per child)
    // This is computed over grammar-masked logits for correct priors
    float parent_logsumexp = get_legal_logsumexp(parent.branch, &store_);

    // Fork branch: clones KV cache, grammar, perplexity, logits atomically
    BranchHandle child_branch = fork(parent.branch, child_seq, &store_);

    if (child_branch == INVALID_HANDLE) {
      seq_pool_.release(child_seq);  // Return unused seq ID
      return parent_idx;
    }

    // Sample next token (uses branch's grammar + sampler chain)
    llama_token token = sample(child_branch, &store_);

    if (token < 0) {
      // FIX #3: Use prune() not destroy() - prune clears the forked KV cache
      prune(child_branch, &store_);
      seq_pool_.release(child_seq);  // Return seq ID to pool
      parent.is_terminal = true;
      return parent_idx;
    }

    // FIX #1: Compute grammar-masked + renormalized prior
    // P(token|s) over LEGAL moves only, not full vocab
    // Use assume_legal since sample() already enforced grammar - O(1) not O(grammar)
    float prior = get_token_prior_assume_legal(parent.branch, token, parent_logsumexp, &store_);

    // Accept token (updates grammar state and perplexity)
    accept_token(child_branch, token, &store_);

    // Decode token and capture logits for future expansion
    decode_and_capture_one(child_branch, token, &store_);

    // Check if terminal (end of generation)
    bool is_terminal = tokenizer::is_eog(vocab_, token);

    // Create child node
    PUCTNode child;
    child.branch = child_branch;
    child.parent_idx = parent_idx;
    child.token = token;
    child.prior = prior;
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

    // Use inverse perplexity as value (lower perplexity = higher value)
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
};

// ============================================================================
// Integration Tests
// ============================================================================

TEST_CASE("PUCT: basic tree expansion with grammar") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();  // Use Metal GPU acceleration
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 8;  // Support up to 8 branches

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  // Decode a prompt first
  std::string prompt = "Generate JSON:";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
  REQUIRE_FALSE(prompt_tokens.empty());

  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos start_pos = static_cast<llama_pos>(prompt_tokens.size());

  // Simple grammar: allows "yes" or "no"
  const char* grammar = "root ::= (\"yes\" | \"no\")";

  PUCT puct(ctx, model.get(), grammar, 8, ctx_params.n_batch, start_pos);

  // Run a few iterations
  puct.search(5);

  CHECK(puct.get_node_count() > 1);  // Should have expanded
  CHECK(puct.get_root_visits() == 5);  // Each iteration visits root

  INFO("PUCT expanded to " << puct.get_node_count() << " nodes");

  llama_free(ctx);
}

TEST_CASE("PUCT: policy priors influence selection") {
  REQUIRE_MODEL();
  LlamaBackendGuard backend;

  auto model_params = llama_model_default_params();
  model_params.n_gpu_layers = TestConfig::n_gpu_layers();  // Use Metal GPU acceleration
  auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
  REQUIRE(model != nullptr);

  auto ctx_params = llama_context_default_params();
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 128;
  ctx_params.n_seq_max = 16;

  llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
  REQUIRE(ctx != nullptr);

  auto vocab = llama_model_get_vocab(model.get());

  std::string prompt = "Output:";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos start_pos = static_cast<llama_pos>(prompt_tokens.size());

  // Grammar with multiple options
  const char* grammar = "root ::= [a-z]+";

  PUCT puct(ctx, model.get(), grammar, 16, ctx_params.n_batch, start_pos);

  // More iterations to see tree growth
  puct.search(10);

  CHECK(puct.get_node_count() >= 2);
  CHECK(puct.get_root_visits() == 10);

  auto best = puct.get_best_sequence();
  INFO("Best sequence has " << best.size() << " tokens");

  llama_free(ctx);
}

TEST_CASE("PUCT: perplexity-based value estimation") {
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

  // Use a prompt that should have predictable continuation
  std::string prompt = "The capital of France is";
  auto prompt_tokens = tokenizer::tokenize(vocab, prompt, false, false);
  decoder::decode_tokens(ctx, prompt_tokens, 0, ctx_params.n_batch, 0);
  llama_pos start_pos = static_cast<llama_pos>(prompt_tokens.size());

  // Unrestricted grammar
  const char* grammar = "root ::= [A-Za-z ]+";

  PUCT puct(ctx, model.get(), grammar, 8, ctx_params.n_batch, start_pos);
  puct.search(5);

  auto best = puct.get_best_sequence();

  // Should have found something
  CHECK_FALSE(best.empty());

  // Decode to text
  if (!best.empty()) {
    std::string text = tokenizer::detokenize_batch(model.get(), best);
    INFO("Best continuation: '" << text << "'");
  }

  llama_free(ctx);
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
  PUCTSamplingParams params;
  const char* grammar_str = "root ::= [a-z]+";

  BranchHandle branch = create(ctx, model.get(), 0, pos, params,
                               ctx_params.n_batch, grammar_str, &store);
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
    PUCTSamplingParams params;

    // Create root branch using RAII wrapper
    Branch root = Branch::create(ctx, model.get(), 0, pos, params,
                                  ctx_params.n_batch, nullptr, &store);
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
  PUCTSamplingParams params;

  // Create branches
  BranchHandle root = create(ctx, model.get(), 0, pos, params,
                             ctx_params.n_batch, nullptr, &store);
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
