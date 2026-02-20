# liblloyal

[![Tests](https://github.com/lloyal-ai/liblloyal/actions/workflows/tests.yml/badge.svg)](https://github.com/lloyal-ai/liblloyal/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-b8087-green.svg)](https://github.com/ggml-org/llama.cpp/releases/tag/b8087)

**Covalent Inference for llama.cpp**

Composable C++ primitives for forkable decode state and shared-prefix (KV) branching in llama.cpp. Fork a generation into a tree — branches share a prefix while keeping independent machinery (sampler chain, seed, grammar, logits snapshot, perplexity tracker) for controlled divergence at decode time.

## Continuous Tree Batching

Tree search with N branches means N calls to `llama_decode()` — each paying GPU dispatch overhead, memory barriers, and PCIe round-trips. Continuous tree batching eliminates this: BranchStore packs tokens from N branches — each at a different position, different seq_id, each needing independent logits captured — into a single `llama_batch` and dispatches once. N branches, 1 GPU call.

```cpp
// Tree search inner loop: all branches advance in one GPU dispatch
store.decode_each({{child1.handle(), tok1},
                   {child2.handle(), tok2},
                   {child3.handle(), tok3}});
// 3 branches × 3 positions × 3 seq_ids → 1 llama_decode()
// Per-branch logits captured, positions advanced
```

Two packing strategies for different access patterns:

```cpp
// commit: 1 token per branch — synchronous tree expansion
store.decode_each(items);

// prefill: variable tokens per branch — asymmetric injection
store.decode_scatter({
    {branchA.handle(), system_tokens},  // 200 tokens
    {branchB.handle(), query_tokens},   //  12 tokens
    {branchC.handle(), doc_tokens},     // 800 tokens
});
// Greedy bin-packed into ceil(total / n_batch) dispatches
// Oversized items auto-fallback to chunked single-sequence decode
```

The underlying decode grid (`decode.hpp`):

|                  | Single Sequence  | Multi Sequence   |
|------------------|------------------|------------------|
| **Single Token** | `decode::one`    | `decode::each`   |
| **Multi Token**  | `decode::many`   | `decode::scatter` |

## The Branch API

Each branch owns a KV cache lease, sampler chain, grammar, metrics tracker, and logits snapshot — everything needed for independent generation. `fork()` deep-clones all of it. Branches compose into best-of-N, speculative decoding, tree search, beam search.

```cpp
using namespace lloyal::branch;

BranchStore store;
store.init_tenancy(ctx);

// Shared prompt: "Explain quantum entanglement"
auto root = Branch::create(ctx, model, store, 0, params);
root.decode_and_capture_batch(prompt_tokens.data(), prompt_tokens.size());

// Fork 4 branches — each will get a different reasoning prefix
auto analogy  = root.fork();
auto formal   = root.fork();
auto socratic = root.fork();
auto visual   = root.fork();

// Scatter-prefill: inject divergent prefixes in one batched dispatch
// 4 branches × variable lengths → auto bin-packed into minimal GPU calls
store.decode_scatter({
    {analogy.handle(),  tokenize("Think of it like two coins...")},   // 12 tokens
    {formal.handle(),   tokenize("In quantum mechanics, the...")},    // 8 tokens
    {socratic.handle(), tokenize("What happens when you measure...")},// 10 tokens
    {visual.handle(),   tokenize("Imagine two particles...")},        // 7 tokens
});

// Generate — all 4 in lockstep, 1 GPU call per step
std::vector<Branch*> branches = {&analogy, &formal, &socratic, &visual};
while (!branches.empty()) {
    std::vector<decode::EachItem> items;
    for (auto* b : branches) {
        auto tok = b->sample();
        if (b->is_eog(tok)) { b->prune(); continue; }
        b->accept(tok);
        items.push_back({b->handle(), tok});
    }
    std::erase_if(branches, [](auto* b) { return !b->valid(); });
    if (!items.empty()) store.decode_each(items);
}

// Winner takes all — one seq_keep pass, losers vaporized
auto* winner = *std::min_element(branches.begin(), branches.end(),
    [](auto* a, auto* b) { return a->perplexity() < b->perplexity(); });
store.retainOnly(winner->handle());
```

**What `fork()` clones:** KV cache sequence, sampler chain handle (penalties, PRNG, filters), grammar handle (GBNF parser state), metrics handle (model + sampling perplexity), logits snapshot, logit bias, cached sampler params.

**What `fork()` does NOT clone:** steer callback (captures references, unsafe to copy).

## Hot-Swap Sampler & Grammar

Sampler chains, grammars, and metrics live in handle-based registries on BranchStore — instance-scoped, no global state. `set_sampler_params()` rebuilds the sampler chain with memoization (no-op if params unchanged). `set_grammar()` hot-swaps the grammar constraint.

```cpp
// EDT (Entropy-based Dynamic Temperature): adapt temperature per token
for (int i = 0; i < max_tokens; i++) {
    float entropy = metrics::model_entropy(root.logits(), root.n_vocab());
    float temp = T0 * std::pow(N, THETA / std::max(entropy, 0.1f));
    root.setSamplerParams(MyParams{.temperature = temp});  // memoized — no-op if temp unchanged
    auto tok = root.sample();
    if (root.is_eog(tok)) break;
    root.accept(tok);
    root.decode_and_capture_one(tok);
}

// Hot-swap grammar mid-generation
root.setGrammar(json_gbnf);     // constrain to JSON
auto tok = root.sample();       // grammar-legal token
root.setGrammar(nullptr);       // remove constraint
```

Handles are freed automatically on `prune()` — no manual cleanup. `fork()` deep-clones all registry entries.

## KV Tenancy

Two resources, two scales. Slots (65K) are how many branches can *exist* — cheap CPU state. Leases (256) are how many can *decode* — scarce KV cache residency. `kv::tenancy` manages the scarce resource as leases, acquired on `create()`/`fork()`, evicted on `prune()`, rebuilt on `retainOnly()`. No manual seq_id tracking, ever.

```cpp
store.available();              // leases remaining — use for width/depth budget
store.retainOnly(winner);       // nuclear: 1 seq_keep, rebuild vacancy
store.drain();                  // explicit teardown before llama_free(ctx)
```

The turn lifecycle: search is surgical (N × `prune()`), promotion is nuclear (1 × `retainOnly()`). Per turn, fork → expand → evaluate → prune losers → repeat. Between turns, promote winner → tree is gone → next turn starts fresh.

## Topology

Parent/child edges are always-on. Simple chat → best-of-N → deep search is one continuum — the library provides topology queries at every point on the spectrum.

```cpp
store.parent(handle);       // INVALID_HANDLE if root
store.children(handle);     // child handles
store.isLeaf(handle);       // no children?
store.isActive(handle);     // holds a KV lease?
```

| Method | FK analogy | Behavior |
|--------|-----------|----------|
| `prune()` | RESTRICT | Throws if children exist |
| `pruneSubtree()` | CASCADE | Iterative post-order traversal |

RAII `~Branch()` uses CASCADE — cleanup always succeeds, even with deep trees. Multi-tag KV cells ensure pruning a parent doesn't corrupt children's cache — a cell is freed only when ALL tags are removed.

## Primitives

The building blocks that compose into the above:

- **Tokenization** — Two-pass safe buffer sizing, special token handling
- **Decoding** — Continuous tree batching, cross-sequence dispatch packing
- **KV Cache** — Tenancy (vacancy manager), sequence ops, state snapshots, long-context compression
- **Sampling** — Grammar-constrained, persistent chains, hot-swap with memoization
- **Metrics** — Dual-level entropy/surprisal, rolling perplexity, cloneable state (BranchStore-scoped)
- **Embeddings** — Pooled extraction, L2 normalization, similarity
- **Chat Templates** — Jinja2 formatting with fallbacks

Lower-level handles for fine-grained control:

```cpp
auto chain = lloyal::sampler::create_chain(params);
auto grammar_handle = lloyal::grammar::init_sampler(model, schema);
```

Shared model weights — multiple contexts, one model load:

```cpp
auto model1 = lloyal::ModelRegistry::acquire(path, params);
auto model2 = lloyal::ModelRegistry::acquire(path, params);  // cache hit
```

## From Simple to Complex

**Single-sequence streaming** — the baseline everyone has:

```cpp
lloyal::decode::many(ctx, prompt_tokens.data(), prompt_tokens.size(), 0, n_batch);
while (!done) {
    auto token = lloyal::sampler::sample_with_params(ctx, vocab, params);
    lloyal::decode::one(ctx, token, n_past++);
}
```

**Best-of-N** — fork once, diverge everywhere, keep the best:

```cpp
using namespace lloyal::branch;

BranchStore store;
store.init_tenancy(ctx);

auto root = Branch::create(ctx, model, store, 0, params);
root.decode_and_capture_batch(prompt_tokens.data(), prompt_tokens.size());

// Fork 8 candidates — KV prefix shared, each gets unique PRNG
std::vector<Branch> candidates;
for (int i = 0; i < 8; i++) {
    candidates.push_back(root.fork());
    reseed_chain(candidates.back().handle(), store, 1000 + i);
}

// Generate 64 tokens — all 8 branches batched into single GPU calls
for (int t = 0; t < 64; t++) {
    std::vector<decode::EachItem> items;
    for (auto& c : candidates) {
        auto tok = c.sample();
        c.accept(tok);
        items.push_back({c.handle(), tok});
    }
    store.decode_each(items);  // 8 branches, 1 llama_decode()
}

// Winner takes all — one seq_keep pass, 7 branches vaporized
auto& winner = *std::min_element(candidates.begin(), candidates.end(),
    [](auto& a, auto& b) { return a.perplexity() < b.perplexity(); });
store.retainOnly(winner.handle());
// store.available() == n_seq_max - 1 — all leases recovered
```

**Tree search with continuous tree batching** — the full show:

```cpp
BranchStore store;
store.init_tenancy(ctx);

auto root = Branch::create(ctx, model, store, 0, params);
root.decode_and_capture_batch(prompt_tokens.data(), prompt_tokens.size());

for (int turn = 0; turn < max_turns; turn++) {
    // Expand: fork children, each samples a different continuation
    std::vector<Branch> leaves;
    int width = std::min((int)store.available(), max_width);
    for (int i = 0; i < width; i++) {
        auto leaf = root.fork();
        reseed_chain(leaf.handle(), store, turn * 1000 + i);
        leaves.push_back(std::move(leaf));
    }

    // Evaluate: generate depth tokens per leaf, batched across all branches
    for (int d = 0; d < depth; d++) {
        std::vector<decode::EachItem> items;
        for (auto& leaf : leaves) {
            auto tok = leaf.sample();
            leaf.accept(tok);
            items.push_back({leaf.handle(), tok});
        }
        store.decode_each(items);  // width branches × 1 GPU dispatch
    }

    // Score + surgical prune: RESTRICT evicts one lease per loser
    std::sort(leaves.begin(), leaves.end(),
        [](auto& a, auto& b) { return a.perplexity() < b.perplexity(); });
    for (size_t i = 1; i < leaves.size(); i++)
        leaves[i].prune();

    // Promote: nuclear — winner becomes the new trunk
    store.retainOnly(leaves[0].handle());
    root = std::move(leaves[0]);
}
```

## Architecture

- **Header-only** — All implementations inline in `include/lloyal/*.hpp`
- **Managed KV residency** — `kv::tenancy` tracks seq_id leases; consumers never see raw seq_ids
- **Handle-based APIs** — Generation counters prevent ABA bugs on slot reuse; sampler chains, grammars, and metrics live in BranchStore-scoped registries (no global state)
- **Shared model weights** — Thread-safe registry enables multi-context with single model load
- **Zero runtime dependencies** — Only requires C++20 standard library + llama.cpp
- **Multi-binding** — C++20 concepts decouple from binding-specific types (Node.js, React Native, CLI)

## Integration

### Git Submodule

```bash
git submodule add -b v0.1.0 https://github.com/lloyal-ai/liblloyal.git
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

**Usage Guide:** [`docs/guide.md`](docs/guide.md)

**API Reference:** Auto-generated from Doxygen-annotated headers

- **Online:** [lloyal-ai.github.io/liblloyal](https://lloyal-ai.github.io/liblloyal/)
- **Local:** `./scripts/generate-docs.sh` → `docs/api/html/index.html`
- **Headers:** `include/lloyal/*.hpp` — fully documented with Doxygen

## Testing

- 256 unit tests (tenancy, topology, continuous tree batching, RESTRICT/CASCADE, handle registries)
- 128 integration tests with real llama.cpp (multi-step generation, ABA prevention, batch error paths, retainOnly, hot-swap sampler/grammar)
- Sanitizer validation (ASan, UBSan, LeakSan)

```bash
# Unit tests (stub-based, no model required)
cd tests && cmake -B build && cmake --build build && ./build/TestRunner

# Integration tests (real llama.cpp)
cd tests && cmake -B build -DLLOYAL_BUILD_INTEGRATION_TESTS=ON \
  -DLLAMA_CPP_DIR=../llama.cpp && cmake --build build
LLAMA_TEST_MODEL=path/to/model.gguf ./build/IntegrationRunner
```

## Design Principles

1. **Primitives, not opinions** — Build your patterns, we provide the tools
2. **Managed scarcity** — KV leases are automatic; capacity is queryable
3. **Explicit over implicit** — No hidden state, clear contracts
4. **Testable** — No framework coupling, works standalone
5. **Version-isolated** — Absorbs llama.cpp API changes

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

Apache 2.0 — See LICENSE file for details
