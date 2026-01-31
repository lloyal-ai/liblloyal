# liblloyal

[![Tests](https://github.com/lloyal-ai/liblloyal/actions/workflows/tests.yml/badge.svg)](https://github.com/lloyal-ai/liblloyal/actions/workflows/tests.yml)
[![Docs](https://github.com/lloyal-ai/liblloyal/actions/workflows/docs.yml/badge.svg)](https://lloyal-ai.github.io/liblloyal/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b6870-green.svg)](https://github.com/ggml-org/llama.cpp/releases/tag/b6870)

**Branched Inference for llama.cpp**

Composable C++ primitives library for llama.cpp with advanced patterns (handle-based APIs, shared model weights, multi-sequence management) enabling applications from simple streaming to complex inference orchestration.

## The Branch API

Branch is the high-level API that composes all of liblloyal's primitives into a single forkable handle. Each branch owns a KV cache sequence, sampler chain, grammar, metrics tracker, and logits snapshot — everything needed for independent generation. Fork a branch to explore alternatives, compare by perplexity, prune losers.

```cpp
using namespace lloyal::branch;

// Prefill prompt on seq 0
lloyal::decoder::decode_tokens(ctx, prompt, prompt_len, 0);

// Create root branch, capture logits from prefill
auto root = Branch::create(ctx, model, /*seq=*/0, prompt_len, params);
root.capture_logits();

// Fork N candidates, each on its own KV sequence
std::vector<Branch> candidates;
for (int i = 0; i < n; i++) {
    candidates.push_back(root.fork(/*seq=*/i + 1));
}

// Generate one token per candidate
for (auto& c : candidates) {
    auto tok = c.sample();
    c.accept(tok);
    c.decode_and_capture_one(tok);
}

// Select winner by perplexity
auto& winner = *std::min_element(candidates.begin(), candidates.end(),
    [](auto& a, auto& b) { return a.perplexity() < b.perplexity(); });

// Commit winner's KV cache, prune losers automatically (RAII)
auto released = winner.release_kv();
// released.seq_id and released.position ready for continued generation
```

**What `fork()` clones:**
- KV cache sequence (via `llama_memory_seq_cp`)
- Sampler chain (penalties, PRNG, top-k/p filters)
- Grammar constraints (GBNF parser state)
- Metrics (model + sampling perplexity trackers)
- Logits snapshot and logit bias

**Use cases:** Best-of-N sampling, speculative decoding (draft/verify), MCTS/LATS tree search, beam search, grammar-constrained exploration.

## Building Blocks

### Core Primitives

- **Tokenization** - Two-pass safe buffer sizing, special token handling
- **Decoding** - Batch orchestration, sequence-aware operations
- **KV Cache** - Sequence operations, state snapshots, long-context patterns
- **Sampling** - Grammar-constrained, persistent chains, 52 parameters
- **Metrics** - Dual-level entropy/surprisal, rolling perplexity, cloneable state
- **Embeddings** - Pooled extraction, L2 normalization, similarity
- **Chat Templates** - Jinja2 formatting with fallbacks

### Advanced Patterns

**Branch API** - The primary handle-based API (see above). Composes all primitives into forkable sessions with RAII cleanup.

**Lower-Level Handles** - Reusable sampler chains and grammar handles for fine-grained control:

```cpp
auto chain = lloyal::sampler::create_chain(params);          // Reusable sampler
auto grammar_handle = lloyal::grammar::init_sampler(model, schema);  // Grammar state
```

**Shared Model Weights** - Multiple contexts share same loaded model:

```cpp
auto model1 = lloyal::ModelRegistry::acquire(path, params);
auto model2 = lloyal::ModelRegistry::acquire(path, params);  // Cache hit
// model1 and model2 share weights, independent KV caches
```

**Multi-Sequence Orchestration** - Branch handles this automatically via `fork()`. For fine-grained control, raw KV operations are available:

```cpp
lloyal::kv::seq_cp(ctx, 0, 1);      // Copy KV state to new sequence
lloyal::kv::seq_keep(ctx, best_seq); // Keep winner, discard others
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

**Use case:** The Branch API manages sequences automatically. These low-level operations are available for custom patterns like speculative decoding or prefix caching.

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

### Git Submodule

```bash
# Pin to stable release (recommended)
git submodule add -b v0.1.0 https://github.com/lloyal-ai/liblloyal.git

# Or track main for latest (less stable)
git submodule add https://github.com/lloyal-ai/liblloyal.git
```

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

**Advanced** - Multi-sequence search with Branch API:

```cpp
using namespace lloyal::branch;

auto root = Branch::create(ctx, model, 0, prompt_len, params);
root.capture_logits();

auto child1 = root.fork(1);
auto child2 = root.fork(2);
// Each child has independent KV, sampler, grammar, metrics
// Generate, compare perplexities, prune losers (automatic on scope exit)
```

### Pattern Examples

**Speculative decoding:**

```cpp
using namespace lloyal::branch;

// Draft branch generates candidate tokens
auto draft = Branch::create(draft_ctx, draft_model, 0, pos, draft_params);
draft.decode_and_capture_one(token);
auto draft_token = draft.sample();

// Verify branch forks from shared prefix
auto verify = Branch::create(verify_ctx, verify_model, 0, pos, verify_params);
verify.decode_and_capture_one(draft_token);

// Accept or reject based on logits comparison
// Prune rejected branches automatically (RAII)
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
