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

- [Model Registry](#model-registry)
- [Tokenizer](#tokenizer)
- [Decoder](#decoder)
- [Sampler](#sampler)
- [KV Cache](#kv-cache)
- [Chat Template](#chat-template)
- [Grammar](#grammar)
- [Embedding](#embedding)
- [Helpers](#helpers)

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
    int32_t n_batch
);
```

Process tokens through model to update KV cache.

**Parameters:**
- `ctx` - Llama context
- `tokens` - Tokens to process
- `n_past` - KV cache position to start at
- `n_batch` - Chunk size for batching (from `ctx_params.n_batch`)

**Behavior:**
- Automatically chunks tokens into `n_batch`-sized batches
- Sets `logits=true` on last token of each batch
- Updates KV cache at positions `[n_past, n_past + tokens.size())`

**Throws:** `std::runtime_error` on decode failure

**Example:**
```cpp
// Decode prompt
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

GBNF grammar-constrained generation for structured output.

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
