# liblloyal Usage Guide

Comprehensive guide for building LLM applications with liblloyal's composable primitives.

---

## Table of Contents

- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [Core Patterns](#core-patterns)
  - [Tokenization & Detokenization](#tokenization--detokenization)
  - [Decoding](#decoding)
  - [Sampling](#sampling)
  - [Chat Templates](#chat-templates)
- [Advanced Features](#advanced-features)
  - [Metrics (Entropy, Surprisal, Perplexity)](#metrics-entropy-surprisal-perplexity)
  - [Embeddings](#embeddings)
  - [Multi-Sequence Operations](#multi-sequence-operations)
  - [Handle-Based APIs](#handle-based-apis)
- [Cache Management Strategies](#cache-management-strategies)
  - [KV Cache Basics](#kv-cache-basics)
  - [State Persistence](#state-persistence)
  - [Context Compression with clear_and_reseed](#context-compression-with-clear_and_reseed)
  - [Multi-User Serving](#multi-user-serving)
- [Development & Testing](#development--testing)
  - [Running Unit Tests](#running-unit-tests)
  - [Running Integration Tests](#running-integration-tests)
  - [Updating llama.cpp Version](#updating-llamacpp-version)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Installation & Setup

liblloyal is a header-only C++ library providing composable primitives for llama.cpp inference.

### As Git Submodule

```bash
# Add liblloyal to your project
git submodule add https://github.com/lloyal-ai/liblloyal.git
git submodule update --init --recursive
```

### CMake Integration

#### Recommended: Using `add_subdirectory()` (v1.0.1+)

When you include llama.cpp and liblloyal via `add_subdirectory()`, liblloyal automatically sets up the required include paths. No manual configuration needed.

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_app)

set(CMAKE_CXX_STANDARD 20)

# 1. Add llama.cpp first (creates llama, ggml targets)
add_subdirectory(vendor/llama.cpp)

# 2. Add liblloyal (auto-configures include paths for llama/llama.h)
add_subdirectory(vendor/liblloyal)

# 3. Create your target and link
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE liblloyal::liblloyal)
```

**What happens automatically:**
- liblloyal detects the `llama` target and links to it
- Generates wrapper headers `llama/llama.h` and `llama/ggml.h` in the build directory (cross-platform compatible)
- Exports include paths via `liblloyal::liblloyal` target

**Your code can use:**
```cpp
#include <lloyal/tokenizer.hpp>   // liblloyal headers
#include <llama/llama.h>          // llama.cpp headers (auto-resolved)
```

#### For Tests: Override llama.cpp Path

When running liblloyal tests with a custom llama.cpp location:

```bash
cmake -B build \
  -DLLOYAL_BUILD_INTEGRATION_TESTS=ON \
  -DLLAMA_CPP_DIR=/path/to/your/llama.cpp
```

#### Legacy: Manual Include Path Setup

For consumers **not** using `add_subdirectory()` (e.g., pre-built llama.cpp), you'll need to set up include paths manually:

```cmake
# Include headers
target_include_directories(your_target PRIVATE 
    liblloyal/include
    llama.cpp/include       # For llama.h
    llama.cpp/ggml/include  # For ggml.h
)

# Link llama.cpp
target_link_libraries(your_target PRIVATE llama)
```

**Note:** liblloyal headers use `#include <llama/llama.h>` (with `llama/` prefix). If your llama.cpp has flat includes, you may need to create a wrapper directory structure.

### CocoaPods (iOS)

```ruby
s.header_dir = "lloyal"
s.source_files = "liblloyal/include/**/*.{hpp,h}"
```

---

## Quick Start

Minimal example demonstrating complete inference pipeline.

**From:** `liblloyal/tests/integration/multi_sequence_integration_test.cpp:46-94`

```cpp
#include <lloyal/model_registry.hpp>
#include <lloyal/tokenizer.hpp>
#include <lloyal/decode.hpp>
#include <lloyal/sampler.hpp>
#include <llama/llama.h>

int main() {
    // Initialize backend
    llama_backend_init();

    // Load model (shared across contexts)
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1;  // Full GPU offload
    auto model = lloyal::ModelRegistry::acquire("model.gguf", model_params);

    // Create inference context
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    llama_context* ctx = llama_init_from_model(model.get(), ctx_params);

    // Get vocabulary
    auto vocab = llama_model_get_vocab(model.get());

    // Tokenize input
    std::string prompt = "Explain quantum computing:";
    auto tokens = lloyal::tokenizer::tokenize(vocab, prompt, false, false);

    // Decode through model
    lloyal::decoder::decode_tokens(ctx, tokens, 0, ctx_params.n_batch);

    // Sample next token
    llama_token next = lloyal::sampler::greedy(ctx, vocab);

    // Convert to text
    std::string output = lloyal::tokenizer::detokenize(vocab, next);
    std::cout << output;

    // Cleanup
    llama_free(ctx);
    llama_backend_free();
}
```

**Key points:**
- `ModelRegistry::acquire()` enables model weight sharing (ref-counted)
- `decode_tokens()` processes prompt through model
- `greedy()` selects highest probability token
- Each context has independent KV cache (isolated sessions)

---

## Core Patterns

### Tokenization & Detokenization

**From:** `include/lloyal/tokenizer.hpp`

```cpp
#include <lloyal/tokenizer.hpp>

// Tokenize text to token IDs
auto vocab = llama_model_get_vocab(model);
auto tokens = lloyal::tokenizer::tokenize(vocab, "Hello world", false, false);
// tokens = [1, 15043, 3186]

// Detokenize single token
std::string text = lloyal::tokenizer::detokenize(vocab, tokens[0]);

// Detokenize batch
std::string full_text = lloyal::tokenizer::detokenize(vocab, tokens);

// Check for end-of-generation
bool is_done = lloyal::tokenizer::is_eog(vocab, token);

// Vocab size
int n_vocab = lloyal::tokenizer::vocab_size(vocab);
```

**Special tokens:**
```cpp
// Add BOS (beginning-of-sequence) token
bool add_bos = true;
auto tokens = lloyal::tokenizer::tokenize(vocab, prompt, add_bos, false);

// Parse special tokens (e.g., "<|im_start|>")
bool parse_special = true;
auto tokens = lloyal::tokenizer::tokenize(vocab, prompt, false, parse_special);
```

### Decoding

**From:** `include/lloyal/decode.hpp`

```cpp
#include <lloyal/decode.hpp>

// Decode token batch (initial prompt)
std::vector<llama_token> tokens = {1, 100, 200, 300};
lloyal::decoder::decode_tokens(ctx, tokens, 0, n_batch);

// Decode single token (generation loop)
int n_past = static_cast<int>(tokens.size());
llama_token next_token = sample_next();
lloyal::decoder::decode_tokens(ctx, {next_token}, n_past, n_batch);
n_past++;
```

**Multi-sequence decoding:**
```cpp
// Enable multi-sequence in context params
ctx_params.n_seq_max = 4;  // Support 4 parallel sequences

// Decode to different sequences
lloyal::decoder::decode_tokens(ctx, tokens, 0, n_batch, /*seq_id=*/0);
lloyal::decoder::decode_tokens(ctx, tokens, 0, n_batch, /*seq_id=*/1);
// Each sequence maintains independent KV state
```

### Sampling

**From:** `include/lloyal/sampler.hpp`

```cpp
#include <lloyal/sampler.hpp>

// Greedy sampling (argmax)
llama_token token = lloyal::sampler::greedy(ctx, vocab);

// Temperature sampling
auto params = llama_sampler_chain_default_params();
// ... configure params ...
llama_token token = lloyal::sampler::sample_with_params(ctx, vocab, params);

// Common parameter patterns
params.temp = 0.7f;           // Temperature (0.0 = greedy, 1.0 = neutral)
params.top_k = 40;            // Top-K filtering
params.top_p = 0.95f;         // Nucleus (top-p) sampling
params.min_p = 0.05f;         // Min-p filtering
params.typical_p = 1.0f;      // Typical sampling
params.penalty_repeat = 1.1f; // Repetition penalty
```

### Chat Templates

**From:** `include/lloyal/chat_template.hpp`

```cpp
#include <lloyal/chat_template.hpp>
#include <nlohmann/json.hpp>

// Build conversation
nlohmann::json messages = nlohmann::json::array();
messages.push_back({
    {"role", "user"},
    {"content", "What is the capital of France?"}
});

// Format with model's built-in template
auto result = lloyal::chat_template::format(
    model,
    messages.dump(),
    ""  // Empty string = use model's template
);

// result.prompt contains formatted text
// Example: "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

// Tokenize formatted prompt
auto tokens = lloyal::tokenizer::tokenize(vocab, result.prompt, true, true);
```

**Multi-turn conversation:**
```cpp
// Add assistant response
messages.push_back({
    {"role", "assistant"},
    {"content", "The capital of France is Paris."}
});

// Add follow-up question
messages.push_back({
    {"role", "user"},
    {"content", "What about Italy?"}
});

// Reformat entire conversation
auto result = lloyal::chat_template::format(model, messages.dump(), "");
```

---

## Advanced Features

### Metrics (Entropy, Surprisal, Perplexity)

**From:** `include/lloyal/metrics.hpp`

liblloyal provides dual-level uncertainty metrics for test-time alignment, adaptive sampling, and quality monitoring.

#### Two Measurement Levels

1. **Model metrics** - Raw logits (before filters) → model's inherent belief
2. **Sampling metrics** - Post-filter logits (after top-k/p/temp) → actual sampled distribution

#### Model-Level Entropy & Surprisal

```cpp
#include <lloyal/metrics.hpp>
#include <lloyal/logits.hpp>

// Get raw logits from model
float* logits = lloyal::logits::get(ctx);
int n_vocab = lloyal::tokenizer::vocab_size(vocab);

// Compute model entropy (uncertainty of next token)
float h = lloyal::metrics::model_entropy(logits, n_vocab);

// Use for routing decisions
if (h > 5.0f) {
    // High entropy → trigger retrieval or context expansion
}

// Compute surprisal for sampled token
llama_token token = sampler::greedy(ctx, vocab);
float s = lloyal::metrics::model_surprisal(logits, n_vocab, token);

if (s > 5.0f) {
    // High surprisal → model is uncertain about this token
}
```

#### Rolling Perplexity Tracking

**From:** `include/lloyal/metrics.hpp:327-361`

```cpp
// Create perplexity tracker
auto ppl_handle = lloyal::metrics::create_perplexity();

// Generation loop
for (int i = 0; i < max_tokens; i++) {
    float* logits = lloyal::logits::get(ctx);
    llama_token token = sample_next();

    // Compute and track surprisal
    float s = lloyal::metrics::model_surprisal(logits, n_vocab, token);
    lloyal::metrics::add_surprisal(ppl_handle, s);

    // Decode next token
    lloyal::decoder::decode_tokens(ctx, {token}, n_past++, n_batch);
}

// Get perplexity (exp of average surprisal)
float ppl = lloyal::metrics::get_ppl(ppl_handle);
int count = lloyal::metrics::get_count(ppl_handle);

std::cout << "Perplexity: " << ppl << " over " << count << " tokens\n";

if (ppl > 50.0f) {
    // High perplexity → consider retrieval or cache eviction
}

// Free tracker
lloyal::metrics::free_perplexity(ppl_handle);
```

#### Use Cases

- **KV eviction gates**: High entropy → trigger retrieval before cache pruning
- **Adaptive sampling**: Collapsed distribution → widen search parameters
- **Quality monitoring**: Track perplexity for confidence estimates
- **Branch comparison**: Compare perplexity across alternative continuations

---

### Embeddings

**From:** `liblloyal/tests/integration/embedding_integration_test.cpp:243-290`

Extract semantic embeddings for similarity search, semantic caching, or retrieval augmented generation.

#### Model Capability Check

```cpp
#include <lloyal/embedding.hpp>

// Check if model supports embeddings
if (lloyal::embedding::has_embeddings(model)) {
    int32_t dim = lloyal::embedding::dimension(model);
    std::cout << "Embedding dimension: " << dim << "\n";
}
```

#### Creating Embedding Context

```cpp
// Create dedicated context for embeddings
auto ctx_params = llama_context_default_params();
ctx_params.n_ctx = 512;
ctx_params.n_batch = 512;
ctx_params.embeddings = true;                      // Enable embeddings
ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN; // Mean pooling

llama_context* embed_ctx = llama_init_from_model(model.get(), ctx_params);
```

#### Extract Embeddings

**From:** `examples/embed/embed.mjs:63-77`

```cpp
// Tokenize text
std::string query = "What is machine learning?";
auto tokens = lloyal::tokenizer::tokenize(vocab, query, true, true);

// Clear KV cache (each text needs fresh context)
lloyal::kv::clear_all(embed_ctx);

// Encode for embeddings (marks all tokens with logits=true)
lloyal::embedding::encode(embed_ctx, tokens, n_batch);

// Extract L2-normalized embedding (unit length for cosine similarity)
auto embedding = lloyal::embedding::get(embed_ctx, lloyal::embedding::Normalize::L2);

// embedding.size() == dimension
```

#### Cosine Similarity

**From:** `liblloyal/tests/integration/embedding_integration_test.cpp:292-345`

```cpp
// Embed multiple texts
auto emb1 = get_embedding(embed_ctx, "The cat sat on the mat");
auto emb2 = get_embedding(embed_ctx, "A cat rested on the rug");
auto emb3 = get_embedding(embed_ctx, "Stock prices rose sharply");

// Compute similarity (for L2-normalized vectors, this is dot product)
float sim_similar = lloyal::embedding::cosine_similarity(emb1, emb2);
float sim_different = lloyal::embedding::cosine_similarity(emb1, emb3);

std::cout << "Similar sentences: " << sim_similar << "\n";      // ~0.8
std::cout << "Different sentences: " << sim_different << "\n";  // ~0.3
```

#### Use Cases

- **Semantic search**: Find similar documents/passages by embedding similarity
- **Semantic caching**: Cache responses by embedding distance thresholds
- **RAG pipelines**: Embed queries and documents for retrieval
- **Clustering**: Group similar texts by embedding proximity

**Note:** For meaningful semantic embeddings, use dedicated embedding models like `nomic-embed-text` or `bge-small-en`. Standard LLMs work but aren't optimized for this task.

---

### Multi-Sequence Operations

**From:** `liblloyal/tests/integration/multi_sequence_integration_test.cpp:46-94`

Enable parallel hypothesis exploration, speculative decoding, or A/B testing within a single context (shared model weights).

#### Enable Multi-Sequence

```cpp
// Configure context for multiple sequences
auto ctx_params = llama_context_default_params();
ctx_params.n_ctx = 512;
ctx_params.n_batch = 128;
ctx_params.n_seq_max = 4;  // Support 4 parallel sequences

llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
```

#### Decode to Different Sequences

```cpp
std::string prompt = "Once upon a time";
auto tokens = lloyal::tokenizer::tokenize(vocab, prompt, false, false);

// Decode to sequence 0
lloyal::decoder::decode_tokens(ctx, tokens, 0, n_batch, /*seq_id=*/0);

// Decode to sequence 1 (independent KV state)
lloyal::decoder::decode_tokens(ctx, tokens, 0, n_batch, /*seq_id=*/1);

// Check positions independently
llama_pos pos0 = lloyal::kv::pos_max(ctx, 0);
llama_pos pos1 = lloyal::kv::pos_max(ctx, 1);
// Both sequences have same number of tokens, but independent KV state
```

#### Copy and Branch

**From:** `include/lloyal/kv.hpp:114-125`

```cpp
// Fork sequence 0 to create sequence 1
lloyal::kv::seq_cp(ctx, /*src=*/0, /*dst=*/1);

// Now seq 1 has same KV state as seq 0
// Generate different continuations
llama_token token_a = sample_with_temperature(ctx, vocab, 0.7f, /*seq=*/0);
llama_token token_b = sample_with_temperature(ctx, vocab, 1.2f, /*seq=*/1);

// Continue each branch independently
lloyal::decoder::decode_tokens(ctx, {token_a}, n_past, n_batch, 0);
lloyal::decoder::decode_tokens(ctx, {token_b}, n_past, n_batch, 1);
```

#### Keep Best, Prune Others

**From:** `include/lloyal/kv.hpp:138-148`

```cpp
// After comparing multiple branches, keep only the best
int best_seq = compare_branches();  // Your selection logic

// Remove all sequences except best_seq
lloyal::kv::seq_keep(ctx, best_seq);

// Now only best_seq remains, continue generation
```

#### Clear Specific Sequence

**From:** `include/lloyal/kv.hpp:54-75`

```cpp
// Remove specific sequence without affecting others
lloyal::kv::remove_range(ctx, /*seq=*/1, /*p0=*/-1, /*p1=*/-1);

// Verify it's gone
llama_pos pos = lloyal::kv::pos_max(ctx, 1);
// pos == -1 (empty)
```

#### Use Cases

- **Parallel hypothesis exploration**: Fork prompt, explore multiple continuations
- **Speculative decoding**: Draft with small model on seq=0, verify with large model on seq=1
- **A/B testing**: Compare different sampling strategies on identical context
- **Beam search**: Maintain top-k sequences, prune low-probability branches

---

### Handle-Based APIs

liblloyal provides persistent handle-based APIs for efficient reuse of complex objects across generation loops.

#### Persistent Sampler Chains

**From:** `include/lloyal/sampler.hpp`

```cpp
#include <lloyal/sampler.hpp>

// Create reusable sampler chain (configure once, use many times)
auto params = llama_sampler_chain_default_params();
params.temp = 0.7f;
params.top_k = 40;
params.top_p = 0.95f;

auto chain = lloyal::sampler::create_chain(model, params);

// Reuse chain across generation loop (no repeated initialization)
for (int i = 0; i < max_tokens; i++) {
    // Apply filters (top-k, top-p, temperature)
    lloyal::sampler::apply(chain, ctx, vocab);

    // Sample token
    llama_token token = lloyal::sampler::sample(chain, ctx);

    // Decode and continue
    lloyal::decoder::decode_tokens(ctx, {token}, n_past++, n_batch);
}

// Free when done
lloyal::sampler::free(chain);
```

**Why handles?**
- **Efficiency**: Avoid repeated sampler initialization (expensive)
- **State management**: Grammar samplers maintain internal state across tokens
- **Reusability**: Same chain for entire generation, or clone for branches

#### Grammar Handles for Structured Output

**From:** `include/lloyal/grammar.hpp`

```cpp
#include <lloyal/grammar.hpp>

// Convert JSON schema to GBNF grammar
nlohmann::json schema = {
    {"type", "object"},
    {"properties", {
        {"name", {{"type", "string"}}},
        {"age", {{"type", "integer"}}}
    }},
    {"required", {"name", "age"}}
};

std::string gbnf = lloyal::grammar::from_json_schema(schema.dump());

// Create grammar sampler handle (maintains parse state)
auto grammar_handle = lloyal::grammar::init_sampler(model, gbnf);

// Use throughout generation (grammar state tracks valid tokens)
for (int i = 0; i < max_tokens; i++) {
    llama_token token = lloyal::grammar::sample(grammar_handle, ctx, vocab);
    lloyal::decoder::decode_tokens(ctx, {token}, n_past++, n_batch);
}

// Result is guaranteed valid JSON
lloyal::grammar::free(grammar_handle);
```

#### Cloneable Metrics for Branching

**From:** `include/lloyal/metrics.hpp:404-412`

```cpp
// Track perplexity on main branch
auto ppl_main = lloyal::metrics::create_perplexity();
// ... add tokens to ppl_main ...

// Fork branch and clone metrics (preserves history)
lloyal::kv::seq_cp(ctx, 0, 1);
auto ppl_alt = lloyal::metrics::clone_perplexity(ppl_main);

// Now both branches track perplexity independently
// Compare results
float ppl_1 = lloyal::metrics::get_ppl(ppl_main);
float ppl_2 = lloyal::metrics::get_ppl(ppl_alt);

// Free both
lloyal::metrics::free_perplexity(ppl_main);
lloyal::metrics::free_perplexity(ppl_alt);
```

---

## Cache Management Strategies

### KV Cache Basics

**From:** `include/lloyal/kv.hpp`

```cpp
#include <lloyal/kv.hpp>

// Clear entire cache (start new conversation)
lloyal::kv::clear_all(ctx);

// Check cache position
llama_pos pos = lloyal::kv::pos_max(ctx, 0);
// pos == -1 means empty, otherwise returns number of tokens - 1

// Remove range [p0, p1) from cache
lloyal::kv::remove_range(ctx, /*seq=*/0, /*p0=*/100, /*p1=*/200);
// Removes tokens at positions 100-199
```

### State Persistence

**From:** `liblloyal/tests/integration/kv_file_persistence_test.cpp:36-92`

Save and restore conversation state across app restarts, fork decision points, or share context.

#### Save State to File

```cpp
// Populate KV cache
std::vector<llama_token> conversation = {1, 100, 200, 300};
lloyal::decoder::decode_tokens(ctx, conversation, 0, n_batch);

// Save to file (includes KV state + tokens)
const std::string filepath = "session.llama";
size_t bytes = lloyal::kv::write_file(ctx, 0, filepath, conversation);

if (bytes > 0) {
    std::cout << "Saved " << bytes << " bytes\n";
}
```

#### Restore State from File

```cpp
// Clear cache first
lloyal::kv::clear_all(ctx);

// Load state from file
auto data = lloyal::kv::read_file(ctx, 0, filepath);

// data.tokens contains the tokens
// data.bytes_read contains file size
// KV cache is automatically restored

// Verify restoration
llama_pos max_pos = lloyal::kv::pos_max(ctx, 0);
assert(max_pos == static_cast<llama_pos>(data.tokens.size() - 1));

// Continue generation from restored state
llama_token next = sampler::greedy(ctx, vocab);
```

#### Use Cases

1. **Exit and Resume**: Save before app termination, restore on next launch
2. **Conversation Forking**: Save at decision points, load to explore alternatives
3. **Context Sharing**: Upload session file to cloud, share across devices

**Example: Forking Conversations**
```cpp
// Save state at decision point
lloyal::kv::write_file(ctx, 0, "fork_point.llama", tokens);

// Explore path A
generate_response("Option A prompt");
lloyal::kv::write_file(ctx, 0, "path_a.llama", tokens);

// Backtrack and explore path B
lloyal::kv::clear_all(ctx);
auto data = lloyal::kv::read_file(ctx, 0, "fork_point.llama");
generate_response("Option B prompt");
lloyal::kv::write_file(ctx, 0, "path_b.llama", tokens);
```

### Context Compression with clear_and_reseed

**From:** `liblloyal/tests/integration/clear_and_reseed_test.cpp:172-260`

One strategy for managing context limits: preserve anchor tokens (attention sinks) + recent tail, evict middle tokens via cache reconstruction.

#### The clear_and_reseed Pattern

**Problem:** Context window fills during long conversations (n_past → n_ctx).

**Solution:** Reconstruct cache with:
- **Anchor tokens** (original first N tokens, typically 4)
- **Recent tail** (last M tokens, typically 252)
- **Evict middle** (everything between anchors and tail)

This maintains contiguous positions `[0, 1, 2, ..., anchor_size + tail_size - 1]` instead of unbounded gaps.

**From:** `include/lloyal/kv.hpp:544-604`

```cpp
// CRITICAL: Capture anchor tokens ONCE at conversation start
std::vector<llama_token> ORIGINAL_ANCHORS;

void start_conversation(const std::string& initial_prompt) {
    auto tokens = lloyal::tokenizer::tokenize(vocab, initial_prompt, false, false);

    // Capture first 4 tokens as anchors (NEVER change these)
    const int ANCHOR_COUNT = 4;
    ORIGINAL_ANCHORS.assign(tokens.begin(), tokens.begin() + ANCHOR_COUNT);

    // Decode initial prompt
    lloyal::decoder::decode_tokens(ctx, tokens, 0, n_batch);
    n_past = static_cast<int>(tokens.size());
}

void compress_if_needed() {
    llama_pos current_pos = lloyal::kv::pos_max(ctx, 0);
    const int COMPRESSION_THRESHOLD = n_ctx - 10;

    if (current_pos >= COMPRESSION_THRESHOLD) {
        // Prepare tail (recent 252 tokens)
        const int TAIL_SIZE = 252;
        size_t tail_start = all_tokens.size() - TAIL_SIZE;
        std::vector<llama_token> tail(
            all_tokens.begin() + tail_start,
            all_tokens.end()
        );

        // Reconstruct with ORIGINAL anchors (not rolling "first 4")
        lloyal::kv::clear_and_reseed(ctx, ORIGINAL_ANCHORS, tail, n_batch);

        // Update position counter
        n_past = ANCHOR_COUNT + TAIL_SIZE;
    }
}
```

#### Critical Invariants

**MUST:** Use `ORIGINAL_ANCHORS` captured at conversation start
**MUST NOT:** Use rolling "first 4" tokens on each compression

**Incorrect (will degrade quality):**
```cpp
// ❌ WRONG: Reusing different anchors each time
auto sinks = std::vector<llama_token>(tokens.begin(), tokens.begin() + 4);
lloyal::kv::clear_and_reseed(ctx, sinks, tail, n_batch);
```

**Correct:**
```cpp
// ✅ RIGHT: Same ORIGINAL_ANCHORS every time
lloyal::kv::clear_and_reseed(ctx, ORIGINAL_ANCHORS, tail, n_batch);
```

#### Performance

**Theory:** Xiao et al. (2023) "Efficient Streaming Language Models with Attention Sinks" demonstrated that transformer attention develops stable "sinks" at initial positions. Maintaining these sinks + recent context preserves perplexity while enabling unbounded generation.

**Empirical:** <10% perplexity increase with 4 anchors + 252 tail (paper: 3.7%).

**See:** `liblloyal/tests/integration/clear_and_reseed_test.cpp` for validation tests.

#### When to Use This Pattern

**Use when:**
- Long conversations beyond context limit
- Bounded memory is critical
- Initial prompt establishes important context

**Don't use when:**
- Context never fills (most single-turn tasks)
- You need full conversation history (use larger model or RAG)
- Initial tokens aren't representative (quality degrades)

**Alternatives:**
- Increase n_ctx (if memory allows)
- Summarization + re-prompt (higher quality, slower)
- Sliding window (simpler, loses early context)
- Retrieval augmented generation (best quality, most complex)

### Multi-User Serving

**From:** Current guide.md:384-427 (validated API)

Share model weights across independent user sessions for memory efficiency.

```cpp
#include <lloyal/model_registry.hpp>

class InferenceService {
private:
    std::shared_ptr<llama_model> model_;
    std::unordered_map<std::string, llama_context*> contexts_;

public:
    InferenceService(const std::string& model_path) {
        // Single model load (4GB)
        model_ = lloyal::ModelRegistry::acquire(
            model_path,
            llama_model_default_params()
        );
    }

    ~InferenceService() {
        for (auto& [user_id, ctx] : contexts_) {
            llama_free(ctx);
        }
    }

    bool create_session(const std::string& user_id) {
        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;

        // Shares model_ weights, independent KV cache
        llama_context* ctx = llama_init_from_model(model_.get(), ctx_params);
        if (!ctx) return false;

        contexts_[user_id] = ctx;
        return true;
    }

    std::string infer(const std::string& user_id, const std::string& prompt) {
        auto it = contexts_.find(user_id);
        if (it == contexts_.end()) return "";

        // Per-user isolated inference
        lloyal::kv::clear_all(it->second);
        auto tokens = lloyal::tokenizer::tokenize(
            llama_model_get_vocab(model_.get()),
            prompt,
            false,
            false
        );
        lloyal::decoder::decode_tokens(it->second, tokens, 0, 512);

        llama_token next = lloyal::sampler::greedy(
            it->second,
            llama_model_get_vocab(model_.get())
        );
        return lloyal::tokenizer::detokenize(
            llama_model_get_vocab(model_.get()),
            next
        );
    }
};
```

**Memory efficiency:** 1 model (~4GB) + N KV caches (~200MB each) instead of N full models.

---

## Development & Testing

liblloyal has comprehensive test coverage with both stub-based unit tests and integration tests against real llama.cpp.

### Running Unit Tests

Stub-based tests validate API contracts without requiring real models (fast, no external dependencies).

```bash
cd liblloyal/tests
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/TestRunner --success
```

**What they test:**
- 84+ test cases covering all primitives
- API contracts (null safety, error handling)
- Edge cases (empty inputs, boundary conditions)
- No real models needed (uses stubs)

### Running Integration Tests

Integration tests use real llama.cpp to validate correctness with actual models.

**Setup llama.cpp:**
```bash
# Reads version from .llama-cpp-version
.github/scripts/setup-llama-cpp.sh

# Build llama.cpp (uses build-llama.sh script)
LLAMA_DIR=llama.cpp .github/scripts/build-llama.sh
```

**Build and run:**
```bash
cd tests
cmake -B build_integration \
  -DLLOYAL_BUILD_INTEGRATION_TESTS=ON \
  -DLLAMA_CPP_DIR=../llama.cpp \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_integration

# Run with test model (any GGUF model works)
LLAMA_TEST_MODEL=path/to/model.gguf ./build_integration/IntegrationRunner

# Some tests need dedicated embedding model
LLAMA_EMBED_MODEL=path/to/nomic-embed.gguf ./build_integration/IntegrationRunner
```

**What they test:**
- Multi-sequence operations (seq_cp, seq_keep)
- KV file persistence (write_file/read_file)
- Embeddings (encode, extract, cosine similarity)
- Context compression (clear_and_reseed position contiguity)
- Real model inference workflows

**CI Configuration:**
- Tests run on GitHub Actions
- Llama.cpp version pinned in `.llama-cpp-version`
- Build cached (keyed by llama.cpp version)
- Matrix: macOS (arm64), Linux (x64), sanitizers (ASan, UBSan, LeakSan)

### Updating llama.cpp Version

liblloyal pins llama.cpp version for reproducible builds. To update:

**Edit `.llama-cpp-version`:**
```bash
# Current content (example)
b8087

# Update to new version
echo "b7000" > .llama-cpp-version
```

**Test locally:**
```bash
# Setup will read new version
.github/scripts/setup-llama-cpp.sh

# Build llama.cpp
LLAMA_DIR=llama.cpp .github/scripts/build-llama.sh

# Run integration tests
cd tests
cmake -B build_integration \
  -DLLOYAL_BUILD_INTEGRATION_TESTS=ON \
  -DLLAMA_CPP_DIR=../llama.cpp
cmake --build build_integration
LLAMA_TEST_MODEL=path/to/model.gguf ./build_integration/IntegrationRunner
```

**Commit:**
```bash
git add .llama-cpp-version
git commit -m "chore: update llama.cpp to b7000"
git push
```

**CI automatically:**
- Reads `.llama-cpp-version`
- Clones llama.cpp at that commit
- Builds with `.github/scripts/build-llama.sh`
- Caches build (keyed by version)
- Runs integration tests
- Fails PR if tests break

**See:** `.github/workflows/tests.yml` for full CI configuration.

---

## Best Practices

### Memory Management

**Efficient patterns:**

```cpp
// ✅ Share models via ModelRegistry (ref-counted)
auto model = lloyal::ModelRegistry::acquire("model.gguf", params);
// Model shared across all contexts using same path+params

// ✅ Destroy idle contexts
llama_free(ctx);
ctx = nullptr;

// ✅ Use clear_and_reseed for unbounded conversations
if (n_past > n_ctx - 10) {
    lloyal::kv::clear_and_reseed(ctx, anchors, tail, n_batch);
}
```

**Inefficient patterns:**

```cpp
// ❌ Loading same model multiple times (wastes GB)
auto model1 = std::shared_ptr<llama_model>(
    llama_load_model_from_file("model.gguf", params),
    llama_free_model
);
auto model2 = std::shared_ptr<llama_model>(
    llama_load_model_from_file("model.gguf", params),  // Loads again!
    llama_free_model
);

// ❌ Keeping unused contexts alive (leaks KV memory)
// Don't keep contexts in map if user disconnected

// ❌ Unbounded cache growth
// Without compression, n_past → n_ctx → crash
```

### Performance Tuning

**Key parameters:**

```cpp
// Mobile-optimized (iPhone, iPad)
ctx_params.n_ctx = 1024;      // Smaller context = less memory
ctx_params.n_batch = 128;     // Smaller batch = less decode memory
ctx_params.n_threads = 2;     // Match efficiency cores
ctx_params.n_gpu_layers = -1; // Full Metal offload

// Server-optimized (Linux, high-end GPU)
ctx_params.n_ctx = 4096;      // Larger context for long conversations
ctx_params.n_batch = 512;     // Larger batch = faster prompt processing
ctx_params.n_threads = 8;     // Match physical cores
ctx_params.n_gpu_layers = -1; // Full GPU offload
```

**Parameter effects:**
- `n_ctx`: Larger = longer context, more KV memory (~200MB per 2048 ctx)
- `n_batch`: Larger = faster prompt processing, more decode memory
- `n_threads`: Match physical cores (diminishing returns beyond)
- `n_gpu_layers`: -1 for full offload (fastest), 0 for CPU-only

**Benchmarking:**
```cpp
auto start = std::chrono::high_resolution_clock::now();

// Your inference code here

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

std::cout << "Tokens/sec: " << (token_count * 1000.0 / duration.count()) << "\n";
```

### Error Handling

**Validate inputs and check return values:**

```cpp
#include <optional>

std::optional<std::string> safe_generate(
    llama_context* ctx,
    const llama_model* model,
    const std::string& prompt
) {
    try {
        // Validate inputs
        if (!ctx || !model || prompt.empty()) {
            return std::nullopt;
        }

        auto vocab = llama_model_get_vocab(model);
        auto tokens = lloyal::tokenizer::tokenize(vocab, prompt, false, false);
        if (tokens.empty()) {
            return std::nullopt;
        }

        // Check context capacity
        if (tokens.size() > static_cast<size_t>(llama_n_ctx(ctx))) {
            return std::nullopt;  // Prompt too long
        }

        // Decode and sample
        lloyal::decoder::decode_tokens(ctx, tokens, 0, 512);
        llama_token next = lloyal::sampler::greedy(ctx, vocab);

        return lloyal::tokenizer::detokenize(vocab, next);

    } catch (const std::exception& e) {
        std::cerr << "Generation error: " << e.what() << "\n";
        return std::nullopt;
    }
}
```

**Defensive programming:**
```cpp
// Check model loaded
if (!model) {
    std::cerr << "Failed to load model\n";
    return 1;
}

// Check context created
llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
if (!ctx) {
    std::cerr << "Failed to create context (out of memory?)\n";
    return 1;
}

// Check tokenization succeeded
auto tokens = lloyal::tokenizer::tokenize(vocab, prompt, false, false);
if (tokens.empty()) {
    std::cerr << "Tokenization failed\n";
    return 1;
}
```

---

## Troubleshooting

### Model Load Failure

**Symptoms:** `ModelRegistry::acquire()` returns `nullptr`

**Solutions:**
1. Verify file path is absolute and accessible
2. Confirm GGUF format (not GGML, PyTorch, safetensors, etc.)
3. Check available disk space (models are memory-mapped)
4. Try default params (disable GPU offload initially)

```cpp
auto model = lloyal::ModelRegistry::acquire("model.gguf", llama_model_default_params());
if (!model) {
    // Debug: Try absolute path
    model = lloyal::ModelRegistry::acquire(
        "/full/path/to/model.gguf",
        llama_model_default_params()
    );
}

if (!model) {
    std::cerr << "Check: file exists, is GGUF format, is readable\n";
}
```

### Out of Memory

**Symptoms:** `llama_init_from_model()` returns `nullptr`

**Solutions:**
1. Reduce `n_ctx` (context window)
2. Reduce `n_batch` (batch size)
3. Use smaller quantized model (Q4_K_M instead of F16)
4. Reduce `n_gpu_layers` if GPU memory constrained

```cpp
auto ctx_params = llama_context_default_params();
ctx_params.n_ctx = 512;    // Reduce from 2048
ctx_params.n_batch = 128;  // Reduce from 512

llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
if (!ctx) {
    std::cerr << "Still OOM? Try smaller model quantization\n";
}
```

### Context Length Exceeded

**Symptoms:** `decode_tokens()` throws exception with message about context capacity

**Cause:** `n_past + tokens.size() > n_ctx`

**Solutions:**
1. Use `clear_and_reseed()` for compression (see Cache Management)
2. Increase `n_ctx` when creating context
3. Truncate input prompt
4. Summarize conversation history before continuing

```cpp
llama_pos current_pos = lloyal::kv::pos_max(ctx, 0);
int n_ctx = llama_n_ctx(ctx);

if (current_pos + new_tokens.size() > n_ctx - 10) {
    // Option 1: Compress
    lloyal::kv::clear_and_reseed(ctx, anchors, tail, n_batch);

    // Option 2: Clear and start over
    lloyal::kv::clear_all(ctx);

    // Option 3: Increase n_ctx (requires new context)
    // llama_free(ctx);
    // ctx_params.n_ctx = 4096;
    // ctx = llama_init_from_model(model.get(), ctx_params);
}
```

### Perplexity Degradation After Compression

**Symptoms:** Output quality drops after `clear_and_reseed()`

**Likely cause:** Changing anchor tokens between compressions

**Solution:** Verify anchor immutability

```cpp
void compress() {
    static bool first_call = true;
    static std::vector<llama_token> EXPECTED_ANCHORS;

    if (first_call) {
        EXPECTED_ANCHORS = ORIGINAL_ANCHORS;
        first_call = false;
    } else {
        // Verify anchors haven't changed
        assert(ORIGINAL_ANCHORS == EXPECTED_ANCHORS);
    }

    auto tail = std::vector<llama_token>(
        all_tokens.end() - 252,
        all_tokens.end()
    );

    lloyal::kv::clear_and_reseed(ctx, ORIGINAL_ANCHORS, tail, n_batch);
}
```

### Decode Failure

**Symptoms:** `decode_tokens()` throws exception

**Common causes:**
1. Null context or empty token vector
2. Position overflow: `n_past + tokens.size() > n_ctx`
3. Batch size exceeds context limit: `tokens.size() > n_batch`
4. Invalid sequence ID (multi-sequence)

**Debug:**
```cpp
try {
    lloyal::decoder::decode_tokens(ctx, tokens, n_past, n_batch);
} catch (const std::exception& e) {
    std::cerr << "Decode error: " << e.what() << "\n";
    std::cerr << "  Context: " << (ctx ? "valid" : "null") << "\n";
    std::cerr << "  Tokens: " << tokens.size() << "\n";
    std::cerr << "  Position: " << n_past << "\n";
    std::cerr << "  n_ctx: " << llama_n_ctx(ctx) << "\n";
    std::cerr << "  n_batch: " << n_batch << "\n";

    if (n_past + tokens.size() > llama_n_ctx(ctx)) {
        std::cerr << "Position overflow - trigger compression\n";
    }
}
```

### Embedding Extraction Returns Null

**Symptoms:** `embedding::get()` throws "embeddings unavailable"

**Causes:**
1. Context created without `embeddings = true`
2. Pooling not enabled (`pooling_type = NONE`)
3. Tokens not encoded with `embedding::encode()` (need `logits=true` for all tokens)

**Solution:**
```cpp
// Verify context configuration
auto ctx_params = llama_context_default_params();
ctx_params.embeddings = true;                      // REQUIRED
ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN; // REQUIRED

llama_context* ctx = llama_init_from_model(model.get(), ctx_params);

// Verify pooling enabled
if (!lloyal::embedding::has_pooling(ctx)) {
    std::cerr << "Pooling not enabled!\n";
}

// Use embedding::encode (not decoder::decode_tokens)
lloyal::kv::clear_all(ctx);
lloyal::embedding::encode(ctx, tokens, n_batch);  // Marks all tokens with logits=true

// Now extraction should work
auto emb = lloyal::embedding::get(ctx, lloyal::embedding::Normalize::L2);
```

---

## Additional Resources

**Within this repository:**
- API headers: `include/lloyal/*.hpp` - Full API documentation in header comments
- Integration tests: `tests/integration/` - Real-world usage examples
- Unit tests: `tests/` - API contract validation

**External:**
- llama.cpp: https://github.com/ggml-org/llama.cpp
- StreamingLLM paper: Xiao et al. (2023) ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453)

---

**Note:** This guide documents liblloyal C++ primitives. For React Native bindings, see the parent lloyal.node project.
