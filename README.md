# liblloyal

**Composable primitives for llama.cpp inference**

C++ primitives library for llama.cpp with composable building blocks (tokenization, sampling, embeddings, KV cache) and advanced patterns (handle-based APIs, shared model weights, multi-sequence management) enabling applications from simple streaming to complex inference orchestration.

## What it provides

### Core Primitives
- **Tokenization** - Two-pass safe buffer sizing, special token handling
- **Decoding** - Batch orchestration, sequence-aware operations
- **KV Cache** - Sequence operations, state snapshots, long-context patterns
- **Sampling** - Grammar-constrained, persistent chains, 52 parameters
- **Metrics** - Dual-level entropy/surprisal, rolling perplexity, cloneable state
- **Embeddings** - Pooled extraction, L2 normalization, similarity
- **Chat Templates** - Jinja2 formatting with fallbacks

### Advanced Patterns

**Handle-Based APIs** - Persistent, reusable objects for efficiency:
```cpp
// Create reusable sampler chain
auto chain = lloyal::sampler::create_chain(model, params);
lloyal::sampler::apply(chain, ctx, vocab);  // Reuse across tokens

// Grammar handle for structured output
auto grammar_handle = lloyal::grammar::init_sampler(model, schema);
```

**Shared Model Weights** - Multiple contexts share same loaded model:
```cpp
// ModelRegistry caches by (path, n_gpu_layers, use_mmap)
auto model1 = lloyal::ModelRegistry::acquire(path, params);
auto model2 = lloyal::ModelRegistry::acquire(path, params);  // Cache hit
// model1 and model2 share weights, independent KV caches
```

**Multi-Sequence Orchestration** - Independent execution paths per context:
```cpp
// Parallel hypothesis exploration
lloyal::kv::seq_cp(ctx, 0, 1);  // Branch to seq 1
lloyal::kv::seq_cp(ctx, 0, 2);  // Branch to seq 2
// Each sequence maintains independent recurrent state
```

### Sequence-Aware Operations
Every primitive supports sequence IDs (default seq=0 for single-path):
```cpp
// Copy KV state to new sequence
lloyal::kv::seq_cp(ctx, 0, 1);

// Sample from different sequences
lloyal::sampler::sample_with_params(ctx, vocab, params, /*seq=*/1);

// Remove tokens from specific sequence
lloyal::kv::remove_range(ctx, seq, p0, p1);
```

**Use case:** Speculative decoding - draft with small model on seq=0, verify with large model on seq=1, copy accepted prefix.

### Cloneable Metrics
Track metrics independently across execution paths:
```cpp
// Create baseline tracker
auto tracker1 = lloyal::metrics::create_perplexity(ctx);

// Clone for alternative
auto tracker2 = lloyal::metrics::clone_perplexity(ctx, tracker1);

// Compare results
float ppl1 = lloyal::metrics::get_ppl(ctx, tracker1);
float ppl2 = lloyal::metrics::get_ppl(ctx, tracker2);
```

**Use case:** A/B testing prompt variations - track quality metrics for each variant independently.

### Dual-Level Uncertainty
Monitor both model and sampling distributions:
```cpp
// Model's inherent uncertainty (raw logits)
float model_entropy = lloyal::metrics::model_entropy(ctx, vocab);

// Actual sampling distribution (post-filter)
float sampling_entropy = lloyal::metrics::sampling_entropy(ctx, vocab, params);
```

**Use case:** Routing decisions - high model entropy triggers retrieval, collapsed sampling distribution suggests overfitting.

### Long-Context Patterns
```cpp
// Preserve initial tokens + recent window, clear middle
lloyal::kv::clear_and_reseed(ctx, initial_tokens, recent_tail);
```

**Use case:** Chat applications beyond context limit - preserve conversation start + recent exchanges without full reprocessing.

### Constrained Generation
```cpp
// JSON schema → GBNF grammar
auto grammar = lloyal::grammar::from_json_schema(schema);
auto chain = lloyal::sampler::create_chain(model, grammar);
```

**Use case:** Structured API responses, data extraction, format enforcement.

## Architecture

- **Header-only** - All implementations inline in `include/lloyal/*.hpp`
- **Composable primitives** - Building blocks combine into diverse patterns
- **Handle-based APIs** - Persistent samplers, grammar chains for efficiency
- **Shared model weights** - Thread-safe registry enables multi-context with single model load
- **Multi-sequence support** - All primitives sequence-aware (default seq=0)
- **llama.cpp binding** - Compile-time dependency, validated by build system
- **Zero runtime dependencies** - Only requires C++20 standard library
- **Multi-binding** - C++20 concepts decouple from binding-specific types

## Integration

### CMake
```cmake
add_subdirectory(liblloyal)
target_link_libraries(your_target PRIVATE lloyal llama)
```

### CocoaPods (iOS)
```ruby
s.header_dir = "lloyal"
s.source_files = "liblloyal/include/**/*.{hpp,h}"
```

## Documentation

**Usage Guide:** [`docs/guide.md`](docs/guide.md) - Comprehensive patterns, examples, and best practices

**API Reference:** Auto-generated from inline header comments
- **Online:** https://lloyal-ai.github.io/liblloyal/ (auto-published on every commit)
- **Local:** Generate with `./scripts/generate-docs.sh` and open `docs/api/html/index.html`
- **Headers:** Browse `include/lloyal/*.hpp` directly - fully documented inline

**Publishing:** See [`docs/PUBLISHING.md`](docs/PUBLISHING.md) for GitHub Pages setup

## Common Patterns

### From Simple to Complex

**Simple** - Single-sequence streaming:
```cpp
lloyal::decoder::decode_tokens(ctx, prompt_tokens, 0);
while (!done) {
  auto token = lloyal::sampler::sample_with_params(ctx, vocab, params);
  lloyal::decoder::decode_one(ctx, token, n_past++);
}
```

**Intermediate** - Streaming with cache compression:
```cpp
// When approaching context limit
auto sinks = std::vector<llama_token>(tokens.begin(), tokens.begin() + 4);
auto tail = std::vector<llama_token>(tokens.end() - 252, tokens.end());
lloyal::kv::clear_and_reseed(ctx, sinks, tail, n_batch);
// Continue generation with bounded positions
```

**Advanced** - Multi-sequence search with shared weights:
```cpp
// Fork exploration paths on same model (shared weights)
lloyal::kv::seq_cp(ctx, 0, 1);
lloyal::kv::seq_cp(ctx, 0, 2);
// Decode alternatives in parallel, compare metrics, prune branches
lloyal::kv::seq_keep(ctx, best_seq);  // Keep winner, discard others
```

### Pattern Examples

**Speculative decoding:**
```cpp
// Draft on seq=0
lloyal::decoder::decode_one(draft_ctx, draft_token, pos, 0);

// Verify on seq=1 (copied from seq=0)
lloyal::kv::seq_cp(verify_ctx, 0, 1);
lloyal::decoder::decode_one(verify_ctx, draft_token, pos, 1);

// Accept or reject based on logits comparison
```

**Model comparison:**
```cpp
// Load same prompt into multiple contexts
for (auto& ctx : contexts) {
  lloyal::decoder::decode_tokens(ctx, prompt_tokens, 0);
  auto tracker = lloyal::metrics::create_perplexity(ctx);
  // Compare perplexities across checkpoints
}
```

**Prefix caching:**
```cpp
// Share common prefix across requests
lloyal::kv::seq_cp(ctx, 0, request_id);
// Continue from shared prefix without re-decode
```

## Testing

Comprehensive test suite with stubs:
- 84+ unit tests covering all primitives
- Integration tests with real llama.cpp
- Sanitizer validation (ASan, UBSan, LeakSan)

### Unit Tests (Stub-based)

```bash
cd tests
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/TestRunner --success
```

### Integration Tests (Real llama.cpp)

```bash
# Setup llama.cpp (reads version from .llama-cpp-version)
.github/scripts/setup-llama-cpp.sh

# Build llama.cpp
LLAMA_DIR=llama.cpp .github/scripts/build-llama.sh

# Build and run integration tests
cd tests
cmake -B build_integration \
  -DLLOYAL_BUILD_INTEGRATION_TESTS=ON \
  -DLLAMA_CPP_DIR=../llama.cpp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build_integration

# Run with test model
LLAMA_TEST_MODEL=path/to/model.gguf ./build_integration/IntegrationRunner
```

**llama.cpp version:** Pinned in `.llama-cpp-version` for reproducible testing

## Design Principles

1. **Primitives, not opinions** - Build your patterns, we provide the tools
2. **Explicit over implicit** - No hidden state, clear contracts
3. **Sequence-aware** - All operations support independent execution paths
4. **Testable** - No framework coupling, works standalone
5. **Version-isolated** - Absorbs llama.cpp API changes

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Security

For security issues, see [SECURITY.md](SECURITY.md) for our disclosure policy.

## License

Apache 2.0 - See LICENSE file for details

## Integrations

This library is used by multiple inference bindings including React Native modules, Node.js addons, and CLI applications.
