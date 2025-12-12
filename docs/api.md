# liblloyal API Reference

**Version:** 1.0.0
**llama.cpp Compatibility:** commit b6870 (Dec 2024)

---

## Overview

**liblloyal** is a header-only C++ wrapper for [llama.cpp](https://github.com/ggerganov/llama.cpp) that provides production-ready abstractions over low-level inference primitives.

**Key Features:**
- Thread-safe model weight sharing with automatic reference counting
- Anti-corruption layers handling llama.cpp's two-pass sizing patterns and edge cases
- Zero-overhead inline functions (no linking required)
- Testable architecture with stub support for unit tests
- StreamingLLM cache management for fixed-memory long-context inference

**Design Philosophy:**
- Pure C++17 with no external dependencies beyond llama.cpp
- Domain-organized headers matching inference workflow (tokenizer → decoder → sampler → KV management)
- Convenience overloads accepting `llama_model*` eliminate boilerplate vocab extraction

---

## Table of Contents

- [Branch (Foundational Primitive)](#branch-foundational-primitive)
- [Model Registry](#model-registry)
- [Tokenizer](#tokenizer)
- [Decoder](#decoder)
- [Logits](#logits)
- [Sampler](#sampler)
- [KV Cache](#kv-cache)
- [KV Sequence Ops (System 2)](#kv-sequence-ops-system-2)
- [Chat Template](#chat-template)
- [Grammar](#grammar)
- [Metrics](#metrics)
- [Embedding](#embedding)
- [Helpers](#helpers)

---

## Branch (Foundational Primitive)

**Header:** `lloyal/branch.hpp`

The Branch primitive consolidates all forkable state (KV cache, grammar, sampler chain, metrics, logits) into a single handle-based API. This is the foundational primitive for MCTS/LATS tree search and multi-sequence generation.

### Design Overview

**Handle Format:** `BranchHandle = (generation << 16) | index`
- Upper 16 bits: generation counter (prevents ABA bugs on slot reuse)
- Lower 16 bits: slot index (max 65535 branches)
- Value 0 is invalid/null handle

**Fork Semantics:**
- `fork()` clones ALL state atomically: KV cache, grammar, sampler chain, metrics, logits
- No manual juggling of individual components
- Each branch is fully independent after fork

**Memory Management:**
- `BranchStore` provides handle table with freelist for efficient pooling
- Generation counters prevent use-after-free bugs
- RAII `Branch` wrapper available for automatic cleanup

---

### Types

#### `BranchHandle`
```cpp
using BranchHandle = uint32_t;
constexpr BranchHandle INVALID_HANDLE = 0;
```

Opaque handle to a branch. Always check against `INVALID_HANDLE` before use.

---

#### `BranchState`
```cpp
struct BranchState {
    llama_context* ctx;           // Context (not owned)
    const llama_model* model;     // Model (not owned)
    llama_seq_id seq_id;          // KV cache sequence ID
    llama_pos position;           // Current position in sequence
    llama_sampler* sampler_chain; // Penalties + temp + dist (owned)
    llama_sampler* grammar;       // Grammar sampler (owned, optional)
    metrics::PerplexityHandle ppl;// Perplexity tracker (owned)
    std::vector<float> logits_snapshot;  // Captured logits
    bool has_logits;              // True after capture_logits/decode_and_capture
    int n_batch;                  // Batch size for decode
    int n_vocab;                  // Vocabulary size
};
```

Internal state managed by handles. Access via `BranchStore::get()`.

---

#### `BranchStore`
```cpp
class BranchStore {
public:
    explicit BranchStore(size_t initial_capacity = 16);
    ~BranchStore();

    BranchHandle allocate();
    void release(BranchHandle handle);
    BranchState* get(BranchHandle handle);
    const BranchState* get(BranchHandle handle) const;
};
```

Handle table with generation counters and freelist. Thread safety requires external synchronization.

---

#### `shutdown_global_store()`
```cpp
void shutdown_global_store();
```

Explicitly destroy the global store before static destructors run. Call this during application shutdown to ensure deterministic cleanup order.

**When to Use:**
- At end of `main()` before `return`
- In shutdown handlers before other globals are destroyed

**Why:** Branches hold raw pointers to `llama_context` and `llama_model`. If the global store outlives llama.cpp cleanup (e.g., `llama_backend_free()`), branch destruction will access freed memory. This function ensures branches are destroyed while their dependencies are still valid.

**Example:**
```cpp
int main() {
    llama_backend_init();

    // ... use branches with global store ...

    // Cleanup in correct order
    branch::shutdown_global_store();  // First: destroy branches
    llama_backend_free();              // Then: destroy llama.cpp globals
    return 0;
}
```

---

### Core Functions

#### `create()`
```cpp
template <SamplingParamsLike P>
BranchHandle create(
    llama_context* ctx,
    const llama_model* model,
    llama_seq_id seq_id,
    llama_pos start_pos,
    const P& params,
    int n_batch = 512,
    const char* grammar_str = nullptr,
    BranchStore* store = nullptr
);
```

Create a new branch with initialized sampler chain, optional grammar, and perplexity tracker.

**Parameters:**
- `ctx` - Llama context (not owned, must outlive branch)
- `model` - Llama model for sampler/grammar initialization
- `seq_id` - Sequence ID for KV cache
- `start_pos` - Starting position (typically after prefill)
- `params` - Sampling parameters (temperature, top_k, penalties, etc.)
- `n_batch` - Batch size for decode operations
- `grammar_str` - Optional GBNF grammar string
- `store` - Optional store (nullptr = use global store)

**Returns:** Branch handle, or `INVALID_HANDLE` on failure

---

#### `fork()`
```cpp
BranchHandle fork(
    BranchHandle source,
    llama_seq_id new_seq_id,
    BranchStore* store = nullptr
);
```

Fork a branch to a new sequence. Clones all state atomically:
- KV cache (via `seq_cp` - O(1))
- Sampler chain (via `llama_sampler_clone`)
- Grammar sampler (via `grammar::clone_sampler`)
- Perplexity tracker (via `metrics::clone_perplexity`)
- Logits snapshot (deep copy)

**Parameters:**
- `source` - Source branch handle
- `new_seq_id` - Sequence ID for the new branch
- `store` - Optional store

**Returns:** New branch handle, or `INVALID_HANDLE` on failure

**Performance:** O(1) for KV cache (tag copy, not data copy)

---

#### `prune()`
```cpp
void prune(BranchHandle handle, BranchStore* store = nullptr);
```

Remove branch from KV cache and free all resources. Use when a branch is no longer needed (e.g., losing MCTS branch).

---

#### `destroy()`
```cpp
void destroy(BranchHandle handle, BranchStore* store = nullptr);
```

Free branch resources without removing from KV cache. Use only if KV cache cleanup is handled separately.

---

### Decode Operations

#### `decode_batch()`
```cpp
void decode_batch(
    BranchHandle handle,
    const llama_token* tokens,
    size_t n_tokens,
    BranchStore* store = nullptr
);
```

Decode multiple tokens into the branch's sequence. Updates position but does NOT capture logits.

---

#### `decode_one()`
```cpp
void decode_one(
    BranchHandle handle,
    llama_token token,
    BranchStore* store = nullptr
);
```

Decode a single token with zero heap allocation. Uses stack-allocated `llama_batch` internally. Optimal for MCTS inner loops where a single token is decoded per step.

**Performance:** No `llama_batch_init()`/`llama_batch_free()` overhead.

---

#### `capture_logits()`
```cpp
void capture_logits(BranchHandle handle, BranchStore* store = nullptr);
```

Capture current context logits into the branch's snapshot. Use after prefill to initialize root branch.

**Throws:** `std::runtime_error` if logits unavailable.

---

#### `decode_and_capture_batch()`
```cpp
void decode_and_capture_batch(
    BranchHandle handle,
    const llama_token* tokens,
    size_t n_tokens,
    BranchStore* store = nullptr
);
```

Decode multiple tokens AND capture logits atomically. Use for batch prefill with logits capture.

---

#### `decode_and_capture_one()`
```cpp
void decode_and_capture_one(
    BranchHandle handle,
    llama_token token,
    BranchStore* store = nullptr
);
```

Decode a single token AND capture logits with zero heap allocation. This is the primary decode function for MCTS tree search—optimal for the expand step where one token is decoded per branch.

---

### Sampling Operations

#### `sample()`
```cpp
llama_token sample(BranchHandle handle, BranchStore* store = nullptr);
```

Sample from the branch using its sampler chain (applies grammar first if present).

**Returns:** Sampled token, or `-1` if no logits captured or sampling fails

**Requires:** Prior call to `capture_logits()` or `decode_and_capture()`

---

#### `accept_token()`
```cpp
void accept_token(
    BranchHandle handle,
    llama_token token,
    BranchStore* store = nullptr
);
```

Accept a token to advance grammar and sampler chain state. Also updates perplexity tracker.

---

#### `apply_grammar()`
```cpp
void apply_grammar(
    BranchHandle handle,
    float* logits,
    int n_vocab,
    BranchStore* store = nullptr
);
```

Apply grammar constraints to an external logits buffer.

---

### Policy Prior Functions (PUCT Support)

#### `get_legal_priors()`
```cpp
std::vector<std::pair<llama_token, float>> get_legal_priors(
    BranchHandle handle,
    BranchStore* store = nullptr
);
```

Get grammar-masked candidate tokens with renormalized probabilities. Essential for proper PUCT: policy priors should only cover legal moves.

**Returns:** Vector of (token, probability) pairs for legal moves only. Probabilities sum to 1.0.

---

#### `get_legal_logsumexp()`
```cpp
float get_legal_logsumexp(BranchHandle handle, BranchStore* store = nullptr);
```

Compute logsumexp over legal (grammar-masked) logits. Used for efficient prior computation.

---

#### `is_token_legal()`
```cpp
bool is_token_legal(
    BranchHandle handle,
    llama_token token,
    BranchStore* store = nullptr
);
```

Check if a token is legal under the current grammar. O(grammar_complexity), not O(n_vocab).

**Parameters:**
- `token` - Token to check
- `store` - Optional store

**Returns:** `true` if token is legal (grammar allows it or no grammar), `false` otherwise

**Use Case:** Pre-filter candidate tokens before computing priors in MCTS selection.

---

#### `get_token_prior_assume_legal()`
```cpp
float get_token_prior_assume_legal(
    BranchHandle handle,
    llama_token token,
    float logsumexp,
    BranchStore* store = nullptr
);
```

Compute prior probability for a token **assuming it's already known to be legal**. O(1) lookup.

**Parameters:**
- `token` - Token to compute prior for (must be legal)
- `logsumexp` - Pre-computed value from `get_legal_logsumexp()`
- `store` - Optional store

**Returns:** Probability in [0,1]

**⚠️ WARNING:** If token is illegal, returns garbage (undefined behavior). Use `is_token_legal()` first or `get_token_prior()` for safe version.

**Performance:** O(1) - optimal for MCTS inner loops where tokens are pre-validated.

---

#### `get_token_prior()`
```cpp
float get_token_prior(
    BranchHandle handle,
    llama_token token,
    float logsumexp,
    BranchStore* store = nullptr
);
```

Compute prior probability for a specific token from grammar-masked logits (safe version).

**Parameters:**
- `token` - Token to compute prior for
- `logsumexp` - Pre-computed value from `get_legal_logsumexp()`

**Returns:** Probability in [0,1], or 0 if token is illegal

**Implementation:** Calls `is_token_legal()` internally, then `get_token_prior_assume_legal()`

---

### State Accessors

```cpp
llama_seq_id get_seq_id(BranchHandle handle, BranchStore* store = nullptr);
llama_pos get_position(BranchHandle handle, BranchStore* store = nullptr);
float get_perplexity(BranchHandle handle, BranchStore* store = nullptr);
int get_n_vocab(BranchHandle handle, BranchStore* store = nullptr);
const float* get_logits(BranchHandle handle, BranchStore* store = nullptr);
```

---

### RAII Wrapper

```cpp
class Branch {
public:
    // Factory
    template <SamplingParamsLike P>
    static Branch create(ctx, model, seq_id, start_pos, params, n_batch, grammar_str, store);

    // Operations
    Branch fork(llama_seq_id new_seq_id);
    void prune();
    void decode_batch(const llama_token* tokens, size_t n);
    void decode_one(llama_token token);
    void decode_and_capture_batch(const llama_token* tokens, size_t n);
    void decode_and_capture_one(llama_token token);
    llama_token sample();
    void accept(llama_token token);
    const float* logits() const;

    // Accessors
    llama_seq_id seq_id() const;
    llama_pos position() const;
    float perplexity() const;
    int n_vocab() const;
    bool valid() const;
    BranchHandle handle() const;
};
```

RAII wrapper that automatically calls `prune()` on destruction. Move-only (non-copyable).

---

### Complete Example

```cpp
#include <lloyal/branch.hpp>

// Setup
BranchStore store(32);  // Pool for up to 32 branches
SamplingParams params;
params.temperature = 0.8f;
params.top_k = 40;

// Create root after prefill
auto root = branch::create(ctx, model, 0, prefill_len, params, 512, nullptr, &store);
branch::capture_logits(root, &store);

// Fork for MCTS exploration
auto child1 = branch::fork(root, 1, &store);
auto child2 = branch::fork(root, 2, &store);

// Sample and expand child1
llama_token token1 = branch::sample(child1, &store);
branch::accept_token(child1, token1, &store);
branch::decode_and_capture_one(child1, token1, &store);

// Get policy priors for PUCT
auto priors = branch::get_legal_priors(child1, &store);
for (auto& [token, prob] : priors) {
    // prob is renormalized over legal moves
}

// Prune losing branch
branch::prune(child2, &store);

// Continue with winner
float ppl = branch::get_perplexity(child1, &store);
```

---

## Model Registry

**Header:** `lloyal/model_registry.hpp`

Thread-safe registry for sharing model weights across multiple contexts.

### `model_registry::acquire()`

```cpp
std::shared_ptr<llama_model> acquire(
    const std::string& path,
    const llama_model_params& params
);
```

**Parameters:**
- `path` - Filesystem path to GGUF model file
- `params` - Model load parameters (GPU layers, mmap, etc.)

**Returns:** `shared_ptr<llama_model>` with automatic cleanup

**Behavior:**
- First call: Loads model from disk, caches weak_ptr
- Subsequent calls: Returns existing model if still alive
- Thread-safe via internal mutex
- Model freed automatically when last context releases reference

**Example:**
```cpp
auto model = model_registry::acquire("model.gguf", llama_model_default_params());

// Multiple contexts share same model weights
llama_context* ctx1 = llama_init_from_model(model.get(), params);
llama_context* ctx2 = llama_init_from_model(model.get(), params);
```

---

## Tokenizer

**Header:** `lloyal/tokenizer.hpp`

Text ↔ token conversions with anti-corruption layer handling llama.cpp's two-pass sizing patterns.

### Primitives (vocab-accepting)

#### `tokenize()`
```cpp
std::vector<llama_token> tokenize(
    const llama_vocab* vocab,
    const std::string& text,
    bool add_special,
    bool parse_special
);
```

Tokenize text with explicit control over special token handling.

**Parameters:**
- `vocab` - Vocabulary from `llama_model_get_vocab()`
- `text` - Text to tokenize
- `add_special` - Add BOS/EOS based on model configuration
- `parse_special` - Parse special token strings like `<|endoftext|>`

---

#### `detokenize()`
```cpp
std::string detokenize(
    const llama_vocab* vocab,
    llama_token token,
    bool special = true
);
```

Convert single token to text (streaming use case).

---

#### `detokenize_batch()`
```cpp
std::string detokenize_batch(
    const llama_vocab* vocab,
    const llama_token* tokens,
    int32_t n_tokens,
    bool remove_special = false,
    bool unparse_special = false
);
```

Batch conversion of token array to text.

---

#### `is_eog()`
```cpp
bool is_eog(const llama_vocab* vocab, llama_token token);
```

Check if token marks end-of-generation (EOS, EOT).

---

#### `vocab_size()`
```cpp
int32_t vocab_size(const llama_vocab* vocab);
```

Get total number of tokens in vocabulary.

---

### Convenience Overloads (model-accepting)

All primitives have overloads accepting `const llama_model*` that automatically extract vocabulary and query metadata. These eliminate boilerplate in application code.

```cpp
// Model-accepting overloads handle vocab extraction internally
std::vector<llama_token> tokenize(const llama_model* model, const std::string& text);
std::string detokenize(const llama_model* model, llama_token token, bool special = true);
std::string detokenize_batch(const llama_model* model, const std::vector<llama_token>& tokens, ...);
bool is_eog(const llama_model* model, llama_token token);
int32_t vocab_size(const llama_model* model);
```

**Example:**
```cpp
// Convenience overload - one line
auto tokens = tokenizer::tokenize(model, "Hello world");

// Equivalent primitive call - three lines
const llama_vocab* vocab = tokenizer::get_vocab(model);
bool add_bos = llama_vocab_get_add_bos(vocab);
auto tokens = tokenizer::tokenize(vocab, "Hello world", add_bos, true);
```

---

## Decoder

**Header:** `lloyal/decoder.hpp`

Batch decoding operations for feeding tokens into the model.

### `decode_tokens()`

```cpp
void decode_tokens(
    llama_context* ctx,
    const std::vector<llama_token>& tokens,
    int32_t n_past,
    int32_t n_batch,
    llama_seq_id seq_id = 0
);
```

Process tokens through model to update KV cache.

**Parameters:**
- `ctx` - Llama context
- `tokens` - Tokens to process
- `n_past` - KV cache position to start at
- `n_batch` - Chunk size for batching (from `ctx_params.n_batch`)
- `seq_id` - Sequence ID to update (default: 0). Use different IDs for parallel/branched generation

**Behavior:**
- Automatically chunks tokens into `n_batch`-sized batches
- Sets `logits=true` on last token of each batch
- Updates KV cache at positions `[n_past, n_past + tokens.size())` for specified sequence

**Throws:** `std::runtime_error` on decode failure

---

### `decode_one()`

```cpp
void decode_one(
    llama_context* ctx,
    llama_token tok,
    llama_pos pos,
    llama_seq_id seq_id = 0,
    bool want_logits = true
);
```

Process a single token with zero heap allocation. Uses stack-allocated `llama_batch` internally.

**Parameters:**
- `ctx` - Llama context
- `tok` - Token to decode
- `pos` - Position in KV cache
- `seq_id` - Sequence ID (default: 0)
- `want_logits` - Whether to compute logits (default: true)

**Performance:** No `llama_batch_init()`/`llama_batch_free()` overhead—optimal for MCTS inner loops.

**Throws:** `std::runtime_error` on decode failure or NULL context

---

## Logits

**Header:** `lloyal/logits.hpp`

Zero-copy access to model logits with lifetime safety checks.

### `get()`

```cpp
float* get(llama_context* ctx, int idx = -1);
```

Get pointer to logits array for the specified batch index.

**Parameters:**
- `ctx` - Llama context
- `idx` - Batch index (-1 = last token in batch)

**Returns:** Pointer to logits array of size `n_vocab`, or `nullptr` if unavailable

**Lifetime:** Valid until next `llama_decode()` call. Copy if you need to persist.

**Example:**
```cpp
decoder::decode_tokens(ctx, tokens, 0, 512);
float* logits = logits::get(ctx);  // Get logits for last token
if (logits) {
    // Use logits before next decode
    float entropy = metrics::model_entropy(logits, n_vocab);
}
```

---

**System 2 (Multi-Sequence) Usage:**
```cpp
// Context must be created with n_seq_max > 1
ctx_params.n_seq_max = 4;  // Support up to 4 parallel sequences

// Decode to different sequences
decoder::decode_tokens(ctx, tokens_a, pos_a, 512, 0);  // seq 0
decoder::decode_tokens(ctx, tokens_b, pos_b, 512, 1);  // seq 1
```

**Example:**
```cpp
// Decode prompt (single sequence)
decoder::decode_tokens(ctx, prompt_tokens, 0, 512);

// Decode generated token
std::vector<llama_token> single = {next_token};
decoder::decode_tokens(ctx, single, n_past, 512);
n_past++;
```

---

## Sampler

**Header:** `lloyal/sampler.hpp`

Token sampling with support for greedy, temperature, top-k/p, and grammar-constrained generation.

### Primitives (vocab-accepting)

#### `greedy()`
```cpp
llama_token greedy(llama_context* ctx, const llama_vocab* vocab);
```

Select token with highest probability (argmax sampling).

**Requires:** Previous `decode_tokens()` call with `logits=true`
**Throws:** `std::runtime_error` if logits unavailable

---

#### `sample_with_params()`
```cpp
template<SamplingParamsLike P>
llama_token sample_with_params(
    llama_context* ctx,
    const llama_vocab* vocab,
    const P& params,
    llama_sampler* grammarSampler = nullptr
);
```

Sample with configurable parameters using C++20 concept-constrained template.

**Parameters:**
- `ctx` - Llama context
- `vocab` - Vocabulary
- `params` - Any type matching `SamplingParamsLike` concept
- `grammarSampler` - Optional grammar constraint

**Supported Parameters:**
```cpp
struct SamplingParams {
    std::optional<float> temperature;      // 0.0 = deterministic, >0 = random
    std::optional<int32_t> top_k;          // Keep top K tokens
    std::optional<float> top_p;            // Nucleus sampling threshold
    std::optional<float> min_p;            // Minimum probability threshold
    std::optional<float> typical_p;        // Typical sampling
    std::optional<uint32_t> seed;          // RNG seed
    std::optional<float> penalty_repeat;   // Repetition penalty
    std::optional<float> penalty_freq;     // Frequency penalty
    std::optional<float> penalty_present;  // Presence penalty
    std::optional<int32_t> penalty_last_n; // Apply penalty to last N tokens
};
```

---

### Convenience Overloads (model-accepting)

```cpp
llama_token greedy(llama_context* ctx, const llama_model* model);

template<SamplingParamsLike P>
llama_token sample_with_params(
    llama_context* ctx,
    const llama_model* model,
    const P& params,
    llama_sampler* grammarSampler = nullptr
);
```

**Example:**
```cpp
// Greedy sampling
llama_token next = sampler::greedy(ctx, model);

// Parameterized sampling
SamplingParams params;
params.temperature = 0.7f;
params.top_k = 40;
params.top_p = 0.9f;
llama_token next = sampler::sample_with_params(ctx, model, params);
```

---

## KV Cache

**Header:** `lloyal/kv.hpp`

Sequence-aware KV cache management with StreamingLLM support.

### Basic Operations

#### `pos_max()`
```cpp
llama_pos pos_max(llama_context* ctx, llama_seq_id seq);
```

Get maximum position in KV cache (returns `-1` if empty).

---

#### `remove_range()`
```cpp
bool remove_range(
    llama_context* ctx,
    llama_seq_id seq,
    llama_pos p0,
    llama_pos p1
);
```

Remove token range from KV cache.

**Parameters:**
- `p0` - Start position (inclusive)
- `p1` - End position (exclusive), or `-1` for end of cache

**Critical:** Call BEFORE next `llama_decode()`, not after.

---

### State Serialization

#### `state_size()`
```cpp
size_t state_size(llama_context* ctx, llama_seq_id seq);
```

Get buffer size needed for serialization. Returns 0 if cache empty.

---

#### `state_save()`
```cpp
size_t state_save(
    llama_context* ctx,
    llama_seq_id seq,
    uint8_t* dst,
    size_t size
);
```

Serialize sequence state to buffer.

---

#### `state_load()`
```cpp
size_t state_load(
    llama_context* ctx,
    llama_seq_id seq,
    const uint8_t* src,
    size_t size
);
```

Restore sequence state from buffer.

---

### File Persistence

#### `FileData`
```cpp
struct FileData {
    std::vector<llama_token> tokens;  // Tokens restored from file
    size_t bytes_read;                 // Total bytes read from file
};
```

Return type for `read_file()` containing both the restored tokens and file size.

---

#### `write_file()`
```cpp
size_t write_file(
    llama_context* ctx,
    llama_seq_id seq,
    const std::string& filepath,
    const std::vector<llama_token>& tokens
);
```

Persist KV cache and tokens to disk using llama.cpp's session file format.

**Parameters:**
- `ctx` - Llama context
- `seq` - Sequence ID
- `filepath` - Destination file path (e.g., "session.llama")
- `tokens` - Token sequence corresponding to KV state

**Returns:** Number of bytes written, or `0` on failure

**Guards:**
- Returns `0` if context is null, filepath is empty, or KV cache is empty
- File contains: magic header, version, token count, tokens, and KV state

**Use Cases:**
1. **Exit and resume:** Save conversation state before app exit, restore on next launch
2. **Forking conversations:** Save state at decision points, load to create alternate paths
3. **Sharing context:** Upload to S3, share signed URL for loading in other clients

**Example:**
```cpp
// Save conversation state
std::vector<llama_token> conversation_tokens = {1, 100, 200, 300};
decoder::decode_tokens(ctx, conversation_tokens, 0, 512);

size_t bytes = kv::write_file(ctx, 0, "saved_session.llama", conversation_tokens);
if (bytes > 0) {
    std::cout << "Saved " << bytes << " bytes" << std::endl;
}
```

---

#### `read_file()`
```cpp
FileData read_file(
    llama_context* ctx,
    llama_seq_id seq,
    const std::string& filepath
);
```

Restore KV cache and tokens from disk.

**Parameters:**
- `ctx` - Llama context
- `seq` - Destination sequence ID
- `filepath` - Source file path

**Returns:** `FileData` struct containing restored tokens and bytes read

**Throws:** `std::runtime_error` if:
- Context is null
- Filepath is empty
- File doesn't exist or has invalid format
- File read fails

**Example:**
```cpp
// Restore conversation state
auto data = kv::read_file(ctx, 0, "saved_session.llama");

std::cout << "Loaded " << data.bytes_read << " bytes" << std::endl;
std::cout << "Restored " << data.tokens.size() << " tokens" << std::endl;

// Verify KV state restored
llama_pos max_pos = kv::pos_max(ctx, 0);
assert(max_pos == static_cast<llama_pos>(data.tokens.size() - 1));

// Continue generation from restored state
llama_token next = sampler::greedy(ctx, model);
```

**File Format:** Uses llama.cpp's built-in session format (magic + version + tokens + KV state). Compatible with upstream llama.cpp session files.

---

### Cache Clearing

#### `clear_all()`
```cpp
void clear_all(llama_context* ctx);
```

Clear entire KV cache (metadata + data buffers). Use for starting new conversation.

---

#### `clear_metadata()`
```cpp
void clear_metadata(llama_context* ctx);
```

Clear metadata only, keeping buffer allocations. Faster than `clear_all()` when immediately re-decoding.

---

## KV Sequence Ops (System 2)

**Header:** `lloyal/kv.hpp`

Operations for multi-sequence KV cache management, enabling LATS/MCTS tree search.

**Prerequisites:**
- Context created with `ctx_params.n_seq_max > 1`
- Different sequence IDs (0, 1, 2, ...) for parallel branches

### `seq_cp()`

```cpp
void seq_cp(
    llama_context* ctx,
    llama_seq_id src,
    llama_seq_id dst,
    llama_pos p0 = 0,
    llama_pos p1 = -1
);
```

Copy KV cache from source sequence to destination (O(1) tag copy).

**Parameters:**
- `ctx` - Llama context
- `src` - Source sequence ID
- `dst` - Destination sequence ID
- `p0` - Start position (default: 0)
- `p1` - End position (default: -1 = to end)

**Use Case:** Fork a stepper for tree search without duplicating model weights or KV data.

**Performance:** O(1) - copies sequence tags, not actual K/V tensors.

**Example:**
```cpp
// Decode shared prefix to seq 0
decoder::decode_tokens(ctx, prefix_tokens, 0, 512, 0);

// Fork to sequences 1 and 2 for parallel exploration
kv::seq_cp(ctx, 0, 1);  // seq 0 → seq 1
kv::seq_cp(ctx, 0, 2);  // seq 0 → seq 2

// Each branch can now decode independently
decoder::decode_tokens(ctx, branch_a_tokens, prefix_len, 512, 1);
decoder::decode_tokens(ctx, branch_b_tokens, prefix_len, 512, 2);
```

---

### `seq_keep()`

```cpp
void seq_keep(llama_context* ctx, llama_seq_id seq);
```

Keep only the specified sequence, removing all others.

**Parameters:**
- `ctx` - Llama context
- `seq` - Sequence ID to keep (all others removed)

**⚠️ WARNING: Does NOT work for MCTS/tree search cleanup.**

When branches share KV slots via `seq_cp()`, `seq_keep()` only removes sequence tags—the underlying slots remain occupied. This means:
- `pos_max()` will still return positions for "removed" sequences
- Memory is NOT reclaimed
- The KV cache slots cannot be reused

**Correct MCTS cleanup pattern:** Use `remove_range()` on each losing branch individually via `branch::prune()` or the RAII `Branch` destructor.

**Valid Use Case:** Cleaning up truly independent sequences that were decoded separately (not forked via `seq_cp`).

**Example (independent sequences only):**
```cpp
// Decode completely separate sequences (NOT forked)
decoder::decode_tokens(ctx, tokens_a, 0, 512, 0);  // seq 0
decoder::decode_tokens(ctx, tokens_b, 0, 512, 1);  // seq 1 (independent)

// seq_keep works here because sequences don't share slots
kv::seq_keep(ctx, 0);  // Remove seq 1
```

**For MCTS/tree search, use Branch primitive instead:**
```cpp
// Correct: RAII handles cleanup via remove_range()
{
    Branch child = parent.fork(new_seq_id);
    // ... explore ...
} // ~Branch calls prune() which uses remove_range()

// Or for winner: release_kv() preserves KV, prevents prune
ReleasedKV winner = best_branch.release_kv();
// Other branches auto-cleanup via RAII
```

---

### `pos_max()`

```cpp
llama_pos pos_max(llama_context* ctx, llama_seq_id seq);
```

Get maximum position in KV cache for a sequence.

**Parameters:**
- `ctx` - Llama context
- `seq` - Sequence ID

**Returns:** Maximum position (number of tokens - 1), or `-1` if sequence is empty.

**Example:**
```cpp
llama_pos current_len = kv::pos_max(ctx, 0);
if (current_len >= n_ctx - 10) {
    // Approaching context limit, trigger reseed
    kv::clear_and_reseed(ctx, sinks, tail, n_batch);
}
```

---

### `remove_range()`

```cpp
bool remove_range(
    llama_context* ctx,
    llama_seq_id seq,
    llama_pos p0,
    llama_pos p1
);
```

Remove token range from KV cache sequence.

**Parameters:**
- `ctx` - Llama context
- `seq` - Sequence ID
- `p0` - Start position (inclusive)
- `p1` - End position (exclusive), or `-1` for end of cache

**Returns:** `true` on success, `false` on failure

**CRITICAL:** Call BEFORE `llama_decode()`, not after.

**Example:**
```cpp
// Remove tokens 100-200 from sequence 0
kv::remove_range(ctx, 0, 100, 200);

// Remove everything from position 50 onwards
kv::remove_range(ctx, 0, 50, -1);
```

---

### StreamingLLM Support

#### `StreamingLlmState`
```cpp
struct StreamingLlmState {
    std::vector<llama_token> original_sinks;  // First N tokens from conversation start
    size_t tail_size;                          // Number of recent tokens (typically 252)
};
```

Helper struct for tracking StreamingLLM state.

---

#### `clear_and_reseed()`
```cpp
void clear_and_reseed(
    llama_context* ctx,
    const std::vector<llama_token>& original_sinks,
    const std::vector<llama_token>& tail,
    int32_t n_batch
);
```

Implement StreamingLLM pattern: clear cache and re-decode sinks + tail.

**Parameters:**
- `ctx` - Llama context
- `original_sinks` - **MUST be first N tokens from conversation start** (typically 4)
- `tail` - Recent M tokens to preserve (typically 252)
- `n_batch` - Batch size for re-decoding

**⚠️  CRITICAL:** `original_sinks` must ALWAYS be the same first N tokens from conversation start. Using different "first 4" tokens after each reseed violates StreamingLLM's attention sink assumption and destroys perplexity preservation.

**Correct Usage:**
```cpp
// Capture sinks ONCE at conversation start
std::vector<llama_token> ORIGINAL_SINKS(conversation.begin(), conversation.begin() + 4);

// Each reseed: Reuse SAME original sinks
auto tail = std::vector<llama_token>(conversation.end() - 252, conversation.end());
kv::clear_and_reseed(ctx, ORIGINAL_SINKS, tail, n_batch);
```

**Incorrect Usage:**
```cpp
// ❌ WRONG: Using different sinks each time
auto sinks = std::vector<llama_token>(current_window.begin(), current_window.begin() + 4);
kv::clear_and_reseed(ctx, sinks, tail, n_batch);  // Will degrade perplexity!
```

**Implementation:** Uses `llama_memory_clear()` followed by re-decode. Simpler and more reliable than selective removal (`llama_memory_seq_rm`) which has position-handling bugs in some llama.cpp versions.

**Validation:** Preserves perplexity within 10% per empirical testing (StreamingLLM paper reported 3.7%). See `tests/integration/clear_and_reseed_validation.cpp`.

---

## Chat Template

**Header:** `lloyal/chat_template.hpp`

Chat format rendering using minja template engine (supports ChatML, Llama-2, etc.).

### `format()`

```cpp
struct FormatResult {
    std::string prompt;
    std::vector<std::string> additional_stops;
};

FormatResult format(
    const llama_model* model,
    const std::string& messages_json,
    const std::string& template_override = ""
);
```

Format chat messages using model's built-in template.

**Parameters:**
- `model` - Llama model (extracts template from GGUF)
- `messages_json` - JSON string with messages array in OpenAI format
- `template_override` - Optional custom Jinja2 template

**Message Format:**
```cpp
[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"}
]
```

**Returns:** `FormatResult` with formatted prompt and extracted stop tokens

**Fallback Hierarchy:**
1. `template_override` (if provided)
2. Model's built-in template from GGUF
3. ChatML template
4. Simple "role: content" format

---

### `validate()`

```cpp
bool validate(const std::string& template_str);
```

Validate Jinja2 template syntax. Returns `false` on error (never throws).

---

## Grammar

**Header:** `lloyal/grammar.hpp`

GBNF grammar-constrained generation for structured output. Includes System 2 primitives for two-phase sampling and sampler cloning.

### `from_json_schema()`

```cpp
std::string from_json_schema(const std::string& schema_json);
```

Convert JSON Schema to GBNF grammar compatible with `llama_sampler_init_grammar()`.

**Example:**
```cpp
std::string schema = R"({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "age"]
})";

std::string gbnf = grammar::from_json_schema(schema);

// Use with sampler
auto vocab = llama_model_get_vocab(model);
llama_sampler* grammar_sampler = llama_sampler_init_grammar(vocab, gbnf.c_str(), "root");
llama_token next = sampler::sample_with_params(ctx, model, params, grammar_sampler);
llama_sampler_free(grammar_sampler);
```

---

### `init_sampler()`

```cpp
llama_sampler* init_sampler(
    const llama_model* model,
    const std::string& grammar_str,
    const std::string& root_rule = "root"
);
```

Initialize a grammar sampler from GBNF grammar string.

**Parameters:**
- `model` - Llama model (for vocab extraction)
- `grammar_str` - GBNF grammar string
- `root_rule` - Root rule name (default: "root")

**Returns:** Grammar sampler, or `nullptr` on failure

**Ownership:** Caller owns returned sampler and must call `llama_sampler_free()`

**Example:**
```cpp
std::string gbnf = grammar::from_json_schema(schema);
llama_sampler* sampler = grammar::init_sampler(model, gbnf);

// Two-phase sampling:
// 1. Apply grammar mask (doesn't advance state)
llama_sampler_apply(sampler, &candidates);

// 2. Accept token (advances parser state)
llama_sampler_accept(sampler, chosen_token);

llama_sampler_free(sampler);
```

---

### `clone_sampler()`

```cpp
llama_sampler* clone_sampler(llama_sampler* smpl);
```

Clone a grammar sampler (deep copy including parser state).

**Parameters:**
- `smpl` - Source sampler to clone

**Returns:** New sampler with identical state, or `nullptr` if input was null

**Ownership:** Caller owns returned sampler and must call `llama_sampler_free()`

**Use Case:** Fork a stepper for tree search, preserving grammar position.

**Example:**
```cpp
// Create initial sampler
llama_sampler* trunk_sampler = grammar::init_sampler(model, gbnf);

// Generate some tokens, advancing grammar state
llama_sampler_accept(trunk_sampler, token1);
llama_sampler_accept(trunk_sampler, token2);

// Fork for branching
llama_sampler* branch_a = grammar::clone_sampler(trunk_sampler);
llama_sampler* branch_b = grammar::clone_sampler(trunk_sampler);

// Each branch can now diverge independently
llama_sampler_accept(branch_a, token_a);
llama_sampler_accept(branch_b, token_b);

// Cleanup
llama_sampler_free(trunk_sampler);
llama_sampler_free(branch_a);
llama_sampler_free(branch_b);
```

**System 2 Integration:**
```cpp
// When forking a stepper:
// 1. Copy KV cache: kv::seq_cp(ctx, src_seq, dst_seq)
// 2. Clone sampler: new_sampler = grammar::clone_sampler(sampler)
// 3. Clone other stepper state (PRNG, history, etc.)
```

---

## Metrics

**Header:** `lloyal/metrics.hpp`

Distribution metrics for test-time alignment: surprisal, entropy, and perplexity. Ported from `tsampler/metrics.ts` with identical validated algorithms.

### Stateless Functions

#### `model_surprisal()`

```cpp
float model_surprisal(
    const float* logits,
    int n_vocab,
    int picked_id,
    Base base = Base::Nats
);
```

Compute model-level surprisal for a picked token.

**Parameters:**
- `logits` - Full vocabulary logits (before sampling filters)
- `n_vocab` - Vocabulary size
- `picked_id` - Token ID that was sampled
- `base` - `Base::Nats` (natural log) or `Base::Bits` (log₂)

**Returns:** Surprisal ≥ 0, or `INFINITY` if invalid

**Interpretation:**
- Higher surprisal = more surprising token (lower probability)
- Use model logits (before temperature/top-k/p) to measure model's inherent uncertainty

**Example:**
```cpp
float* logits = lloyal::logits::get(ctx);
int n_vocab = llama_vocab_n_tokens(vocab);
llama_token token = sample(logits);

float s = metrics::model_surprisal(logits, n_vocab, token);
if (s > 5.0f) {
    // High uncertainty - consider retrieval
}
```

---

#### `model_entropy()`

```cpp
float model_entropy(
    const float* logits,
    int n_vocab,
    Base base = Base::Nats
);
```

Compute model-level entropy of distribution.

**Parameters:**
- `logits` - Full vocabulary logits (before sampling filters)
- `n_vocab` - Vocabulary size
- `base` - `Base::Nats` (natural log) or `Base::Bits` (log₂)

**Returns:** Entropy ≥ 0, or `INFINITY` if invalid

**Interpretation:**
- Higher entropy = flatter distribution (more uncertain)
- Lower entropy = peaked distribution (more confident)

**Example:**
```cpp
float h = metrics::model_entropy(logits, n_vocab);
if (h < 2.0f) {
    // Collapsed distribution -> widen search
} else if (h > 5.0f) {
    // Too flat -> focus sampling
}
```

---

#### `sampling_surprisal()`

```cpp
float sampling_surprisal(
    const float* candidate_logits,
    const int32_t* candidate_ids,
    int n_candidates,
    int picked_id,
    Base base = Base::Nats
);
```

Compute sampling-level surprisal within filtered candidate set (after top-k/p/temperature).

---

#### `sampling_entropy()`

```cpp
float sampling_entropy(
    const float* candidate_logits,
    int n_candidates,
    Base base = Base::Nats
);
```

Compute sampling-level entropy of candidate distribution.

---

### Handle-Based Perplexity Tracking

Handle-based API for RollingPerplexity that supports `clone()` for MCTS fork.

#### `create_perplexity()`

```cpp
PerplexityHandle create_perplexity();
```

Create a new rolling perplexity tracker.

**Returns:** Handle to the perplexity tracker

---

#### `add_surprisal()`

```cpp
void add_surprisal(PerplexityHandle handle, float surprisal);
```

Add token surprisal to running average. Non-finite values are ignored.

---

#### `get_ppl()`

```cpp
float get_ppl(PerplexityHandle handle);
```

Get current perplexity: `exp(average surprisal)`.

**Returns:** Perplexity, or `INFINITY` if no samples

---

#### `get_count()`

```cpp
int get_count(PerplexityHandle handle);
```

Get number of tokens added to tracker.

---

#### `reset_perplexity()`

```cpp
void reset_perplexity(PerplexityHandle handle);
```

Reset tracker to initial state (start new sequence).

---

#### `clone_perplexity()`

```cpp
PerplexityHandle clone_perplexity(PerplexityHandle handle);
```

Clone perplexity tracker for MCTS fork. Creates a new tracker with identical state.

**Returns:** New handle with cloned state, or 0 if invalid source

**Example:**
```cpp
// Fork branch - clone all state
kv::seq_cp(ctx, src_seq, dst_seq);
auto new_grammar = grammar::clone_sampler(grammar_handle);
auto new_ppl = metrics::clone_perplexity(ppl_handle);
```

---

#### `free_perplexity()`

```cpp
void free_perplexity(PerplexityHandle handle);
```

Free perplexity tracker.

---

### Complete Example

```cpp
#include <lloyal/metrics.hpp>
#include <lloyal/logits.hpp>

// Track perplexity during generation
auto ppl = metrics::create_perplexity();

for (int i = 0; i < max_tokens; ++i) {
    float* logits = lloyal::logits::get(ctx);

    // Sample token
    llama_token token = sample(logits);

    // Track surprisal
    float s = metrics::model_surprisal(logits, n_vocab, token);
    metrics::add_surprisal(ppl, s);

    // Optional: Check entropy for adaptive sampling
    float h = metrics::model_entropy(logits, n_vocab);
    if (h < 1.0f) {
        // Very confident - can reduce temperature
    }

    // Decode token...
}

float perplexity = metrics::get_ppl(ppl);
if (perplexity > 50.0f) {
    // High perplexity - consider retrieval augmentation
}

metrics::free_perplexity(ppl);
```

---

## Embedding

**Header:** `lloyal/embedding.hpp`

Extract and compare dense vector embeddings from models with pooling support.

### Model Capability Checks

#### `has_embeddings()`
```cpp
bool has_embeddings(const llama_model* model);
```

Check if model supports embedding extraction.

**Returns:** `true` if model has non-zero embedding dimension

---

#### `dimension()`
```cpp
int32_t dimension(const llama_model* model);
```

Get embedding vector dimension (model's hidden size).

**Returns:** Embedding dimension, or `0` if model is null

**Common Dimensions:**
- 768 (BERT-base, nomic-embed-text)
- 1024 (BERT-large)
- 2048 (SmolLM2)
- 4096 (Llama-7B)

---

### Context Capability Checks

#### `has_pooling()`
```cpp
bool has_pooling(llama_context* ctx);
```

Check if context has pooling enabled (required for meaningful embeddings).

**Returns:** `true` if pooling type is not `LLAMA_POOLING_TYPE_NONE`

---

#### `pooling_type()`
```cpp
int32_t pooling_type(llama_context* ctx);
```

Get context's pooling strategy.

**Returns:** One of:
- `LLAMA_POOLING_TYPE_NONE` (0) - No pooling
- `LLAMA_POOLING_TYPE_MEAN` (1) - Mean pooling (recommended)
- `LLAMA_POOLING_TYPE_CLS` (2) - CLS token pooling
- `LLAMA_POOLING_TYPE_LAST` (3) - Last token pooling

---

### Embedding Extraction

#### `Normalize` enum
```cpp
enum class Normalize : int32_t {
    None = 0,  // Raw embeddings
    L2 = 1,    // L2-normalized (unit length)
};
```

Normalization mode for extracted embeddings.

---

#### `get()`
```cpp
std::vector<float> get(
    llama_context* ctx,
    Normalize normalize = Normalize::L2
);
```

Extract pooled embedding from context (sequence 0).

**Parameters:**
- `ctx` - Llama context (must have decoded tokens)
- `normalize` - Normalization mode (default: L2)

**Returns:** Embedding vector of size `dimension(model)`

**Throws:**
- `std::invalid_argument` if context is null
- `std::runtime_error` if embeddings unavailable

**Example:**
```cpp
// Create context with pooling enabled
auto ctx_params = llama_context_default_params();
ctx_params.embeddings = true;
ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
llama_context* ctx = llama_init_from_model(model, ctx_params);

// Tokenize and decode
auto tokens = tokenizer::tokenize(model, "Hello world");
decoder::decode_tokens(ctx, tokens, 0, 512);

// Extract L2-normalized embedding
auto emb = embedding::get(ctx);  // Default: L2 normalized
```

---

#### `get_seq()`
```cpp
std::vector<float> get_seq(
    llama_context* ctx,
    llama_seq_id seq,
    Normalize normalize = Normalize::L2
);
```

Extract pooled embedding for specific sequence.

**Use Case:** Multi-sequence batch embedding where each sequence represents a different text.

---

#### `get_ith()`
```cpp
std::vector<float> get_ith(
    llama_context* ctx,
    int32_t token_idx,
    Normalize normalize = Normalize::L2
);
```

Extract embedding for specific token position (no pooling).

**Use Case:** Token-level analysis, attention visualization.

---

### Similarity Computation

#### `cosine_similarity()`
```cpp
float cosine_similarity(
    const std::vector<float>& a,
    const std::vector<float>& b
);
```

Compute cosine similarity between two embedding vectors.

**Parameters:**
- `a`, `b` - Embedding vectors (should be same dimension)

**Returns:** Similarity score in range `[-1.0, 1.0]`:
- `1.0` = identical direction
- `0.0` = orthogonal (unrelated)
- `-1.0` = opposite direction

**Throws:** `std::invalid_argument` if dimensions mismatch

**Performance Note:** For L2-normalized vectors, cosine similarity reduces to dot product (no sqrt needed).

**Example:**
```cpp
// Embed two sentences
auto tokens1 = tokenizer::tokenize(model, "The cat sat on the mat");
decoder::decode_tokens(ctx, tokens1, 0, 512);
auto emb1 = embedding::get(ctx);

kv::clear_all(ctx);

auto tokens2 = tokenizer::tokenize(model, "A cat rested on the rug");
decoder::decode_tokens(ctx, tokens2, 0, 512);
auto emb2 = embedding::get(ctx);

// Compare similarity
float sim = embedding::cosine_similarity(emb1, emb2);
// sim > 0.7 for semantically similar sentences
```

---

### Complete Embedding Workflow

```cpp
#include <lloyal/embedding.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/decoder.hpp>
#include <lloyal/kv.hpp>

// 1. Load model (any LLM works, but embedding models are better)
auto model = ModelRegistry::acquire("nomic-embed-text.gguf", params);

// 2. Check embedding support
if (!embedding::has_embeddings(model.get())) {
    throw std::runtime_error("Model doesn't support embeddings");
}
int32_t dim = embedding::dimension(model.get());

// 3. Create context with pooling
auto ctx_params = llama_context_default_params();
ctx_params.embeddings = true;
ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
llama_context* ctx = llama_init_from_model(model.get(), ctx_params);

// 4. Embed documents
std::vector<std::vector<float>> doc_embeddings;
for (const auto& doc : documents) {
    kv::clear_all(ctx);
    auto tokens = tokenizer::tokenize(model.get(), doc);
    decoder::decode_tokens(ctx, tokens, 0, 512);
    doc_embeddings.push_back(embedding::get(ctx));
}

// 5. Find most similar to query
auto query_tokens = tokenizer::tokenize(model.get(), query);
decoder::decode_tokens(ctx, query_tokens, 0, 512);
auto query_emb = embedding::get(ctx);

float best_sim = -1.0f;
size_t best_idx = 0;
for (size_t i = 0; i < doc_embeddings.size(); ++i) {
    float sim = embedding::cosine_similarity(query_emb, doc_embeddings[i]);
    if (sim > best_sim) {
        best_sim = sim;
        best_idx = i;
    }
}
```

---

## Helpers

**Header:** `lloyal/helpers.hpp`

Utility functions for parameter conversion, chat template processing, and string operations.

### Batch Utilities

```cpp
void batch_clear(llama_batch& batch);

void batch_add(
    llama_batch& batch,
    llama_token id,
    int32_t pos,
    const std::vector<llama_seq_id>& seq_ids,
    bool logits,
    int32_t capacity = -1
);
```

### Chat Template Processing

```cpp
struct ChatTemplateResult {
    std::string prompt;
    std::vector<std::string> additional_stops;
};

ChatTemplateResult format_chat_template_complete(
    const llama_model* model,
    const std::string& messages_json,
    const std::string& template_override = ""
);

bool validate_chat_template_helper(const std::string& template_str);
```

### KV Cache Type Conversion

```cpp
ggml_type kv_cache_type_from_str(const std::string& s);
const std::vector<ggml_type>& get_kv_cache_types();
```

### String Utilities

```cpp
std::string string_repeat(const std::string& str, size_t n);
std::string string_join(const std::vector<std::string>& values, const std::string& separator);
std::vector<std::string> string_split(const std::string& str, const std::string& delimiter);
```

---

## Advanced Topics

### Thread Safety

- `model_registry::acquire()` is thread-safe (uses internal mutex)
- All other functions are **NOT thread-safe** - use one context per thread
- Models (read-only after load) can be shared across threads

### Error Handling

All functions throw `std::runtime_error` on failure with descriptive messages:

```cpp
try {
    auto tokens = tokenizer::tokenize(model, text);
    decoder::decode_tokens(ctx, tokens, 0, 512);
} catch (const std::runtime_error& e) {
    // Handle error: e.what() contains details
}
```

### Testing Strategy

**Unit Tests:** Use stubs from `tests/stubs/llama_stubs.h` to test without loading real models:

```cpp
#include "llama_stubs.h"

resetStubConfig();
llamaStubConfig().tokenize_result = {100, 200, 300};
llamaStubConfig().tokenize_succeeds = true;

// Test code without real llama.cpp
```

**Integration Tests:** Use real models via `LLAMA_TEST_MODEL` environment variable.

---

## License

Apache 2.0 (same as llama.cpp)

## Support

- GitHub Issues: [liblloyal/issues](https://github.com/yourusername/liblloyal/issues)
- Documentation: This file + [guide.md](./guide.md)
- Examples: See `tests/integration/` for production usage patterns
