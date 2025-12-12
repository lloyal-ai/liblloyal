# liblloyal

**Header-only llama.cpp-bound core library**

## Overview

`liblloyal` is a header-only C++ library that provides high-level abstractions over llama.cpp for local LLM inference. It implements anti-corruption layers for tokenization, sampling, KV cache management, chat templates, and grammar-constrained generation.

## Architecture

- **Header-only**: All implementations are inline in headers under `include/lloyal/`
- **llama.cpp-bound**: Designed to work with llama.cpp b6870 (pinned version)
- **Namespace**: `lloyal::` (portable, neutral naming)
- **Zero dependencies**: Only requires llama.cpp and standard library

## Components

### Core Abstractions

- **`branch.hpp`**: **Foundational primitive** for tree search (MCTS/LATS/PUCT) - consolidates KV, grammar, sampler, metrics, logits into single forkable handle
- **`tokenizer.hpp`**: Tokenization/detokenization with special token handling
- **`decoder.hpp`**: Token-to-text streaming decoding with multi-sequence support
- **`logits.hpp`**: Zero-copy logits access with lifetime safety checks
- **`sampler.hpp`**: Advanced sampling with 52 tunable parameters
- **`kv.hpp`**: KV cache management including sequence operations (copy, remove, pos_max)
- **`embedding.hpp`**: Embedding extraction with pooling and L2 normalization
- **`chat_template.hpp`**: Jinja2-based chat template formatting
- **`grammar.hpp`**: Grammar-constrained generation with JSON schema support + sampler cloning
- **`metrics.hpp`**: Distribution metrics (surprisal, entropy, perplexity) with handle-based tracking
- **`model_registry.hpp`**: Model metadata and feature detection
- **`helpers.hpp`**: Batch utilities, string operations, parameter conversions

### System 2 Support (LATS/MCTS/PUCT Tree Search)

**Recommended:** Use `branch.hpp` for tree search - it consolidates all forkable state:

```cpp
#include <lloyal/branch.hpp>

// Create root branch after prefill
auto root = branch::create(ctx, model, 0, prefill_len, params, 512, nullptr, &store);
branch::capture_logits(root, &store);

// Fork: clones KV, grammar, sampler, metrics, logits atomically
auto child = branch::fork(root, new_seq_id, &store);

// Get policy priors for PUCT (grammar-masked, renormalized)
auto priors = branch::get_legal_priors(child, &store);

// Sample and advance
llama_token token = branch::sample(child, &store);
branch::accept_token(child, token, &store);
branch::decode_and_capture(child, &token, 1, &store);

// Cleanup losing branches
branch::prune(child, &store);
```

**Lower-level primitives** (used internally by branch.hpp):
- **Multi-sequence decoding**: `decode_tokens()` accepts `seq_id` for parallel branches
- **KV sequence ops**: `seq_cp()` (O(1) fork), `seq_keep()`, `pos_max()`, `remove_range()`
- **Grammar cloning**: `clone_sampler()` for forking grammar state across branches
- **Metrics cloning**: `clone_perplexity()` for forking perplexity trackers across branches

### Vendored Dependencies

- **`minja/`**: Jinja2 template engine (from llama.cpp)
- **`nlohmann/json.hpp`**: JSON library (ordered_json for schema)

## Usage

```cpp
#include <lloyal/branch.hpp>      // Tree search primitive (recommended for MCTS/PUCT)
#include <lloyal/tokenizer.hpp>
#include <lloyal/logits.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/embedding.hpp>
#include <lloyal/decoder.hpp>
#include <lloyal/metrics.hpp>

// Tokenize text
auto tokens = lloyal::tokenizer::tokenize(vocab, "Hello world", true, false);

// Get logits with safety checks (zero-copy pointer, valid until next decode)
float* logits = lloyal::logits::get(ctx);  // throws if null/unavailable

// Sample next token
int token_id = lloyal::sampler::sample(ctx, vocab, params);

// Compute distribution metrics
float entropy = lloyal::metrics::model_entropy(logits, n_vocab);
float surprisal = lloyal::metrics::model_surprisal(logits, n_vocab, token_id);

// Track perplexity across generation (handle-based for fork support)
auto ppl = lloyal::metrics::create_perplexity();
lloyal::metrics::add_surprisal(ppl, surprisal);
float perplexity = lloyal::metrics::get_ppl(ppl);
lloyal::metrics::free_perplexity(ppl);

// Manage KV cache
lloyal::kv::clear_range(ctx, 0, 100, 512, 1024);

// Extract embeddings (requires context with embeddings=true, pooling enabled)
lloyal::kv::clear_all(embed_ctx);
lloyal::decoder::encode(embed_ctx, tokens, 512);
auto emb = lloyal::embedding::get(embed_ctx, lloyal::embedding::Normalize::L2);
float similarity = lloyal::embedding::cosine_similarity(emb1, emb2);
```

## Integration

### CMake (Android, Linux, etc.)

```cmake
add_subdirectory(liblloyal)
target_link_libraries(your_target PRIVATE lloyal llama)
```

### CocoaPods (iOS)

```ruby
s.header_dir = "lloyal"
s.source_files = "liblloyal/include/**/*.{hpp,h}"
s.vendored_frameworks = "llama.cpp/build-apple/llama.xcframework"
```

## Version Pinning

This library is built against **llama.cpp b6870**. The CMakeLists.txt includes build-time validation to ensure version compatibility.

## Testing

Smoke tests verify core functionality:
- `tests/test_logits_mask.cpp` - Validates logits biasing operations
- `tests/test_kv_range.cpp` - Validates KV cache range operations

Run tests: `ctest` (from build directory)

## Design Principles

1. **Anti-Corruption Layer**: Isolates shells from llama.cpp API churn
2. **Header-Only**: Simplifies linking, enables aggressive inlining
3. **Explicit Over Implicit**: Clear parameters, no hidden state
4. **Fail-Safe**: Validates inputs, returns empty/false on errors
5. **Two-Pass Algorithms**: Safe buffer sizing for tokenization/detokenization

## License

MIT (matches llama.cpp licensing)

## Shells

This library is used by:
- **calibrate-ndk**: Commercial React Native module (with TokenLedger)
- **nitro-llama**: Open-source React Native module
