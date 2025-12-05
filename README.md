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

- **`tokenizer.hpp`**: Tokenization/detokenization with special token handling
- **`decoder.hpp`**: Token-to-text streaming decoding and embedding encoding
- **`logits.hpp`**: Zero-copy logits access with lifetime safety checks
- **`sampler.hpp`**: Advanced sampling with 52 tunable parameters
- **`kv.hpp`**: KV cache management (operations, defragmentation, compression)
- **`embedding.hpp`**: Embedding extraction with pooling and L2 normalization
- **`chat_template.hpp`**: Jinja2-based chat template formatting
- **`grammar.hpp`**: Grammar-constrained generation with JSON schema support
- **`model_registry.hpp`**: Model metadata and feature detection
- **`helpers.hpp`**: Batch utilities, string operations, parameter conversions

### Vendored Dependencies

- **`minja/`**: Jinja2 template engine (from llama.cpp)
- **`nlohmann/json.hpp`**: JSON library (ordered_json for schema)

## Usage

```cpp
#include <lloyal/tokenizer.hpp>
#include <lloyal/logits.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/kv.hpp>
#include <lloyal/embedding.hpp>
#include <lloyal/decoder.hpp>

// Tokenize text
auto tokens = lloyal::tokenizer::tokenize(vocab, "Hello world", true, false);

// Get logits with safety checks (zero-copy pointer, valid until next decode)
float* logits = lloyal::logits::get(ctx);  // throws if null/unavailable

// Sample next token
int token_id = lloyal::sampler::sample(ctx, vocab, params);

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
