# Changelog

All notable changes to liblloyal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release
- Header-only C++ library for llama.cpp primitives
- Core modules:
  - `tokenizer.hpp` - Tokenization/detokenization with special token handling
  - `decoder.hpp` - Batch decoding with sequence awareness
  - `kv.hpp` - KV cache management and state operations
  - `sampler.hpp` - Parameterized sampling with grammar support
  - `metrics.hpp` - Dual-level entropy and perplexity tracking
  - `embedding.hpp` - Pooled embedding extraction
  - `grammar.hpp` - Constrained generation primitives
  - `chat_template.hpp` - Jinja2 chat template formatting
  - `model_registry.hpp` - Model weight sharing
- Comprehensive test suite (84+ tests with stubs)
- CI/CD with sanitizers (ASan, UBSan, LeakSan)
- Apache 2.0 license with patent grant

### Design Highlights
- Sequence-aware operations for branching/forking
- Cloneable metric trackers for A/B testing
- Clear+reseed pattern for long-context chat
- C++20 concepts for parameter type flexibility
- Zero-copy logits and embeddings

## Release Process

Releases follow semantic versioning:
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible
