# Changelog

All notable changes to liblloyal will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 3.2.0 - 2026-10-26

- Multimodal embedding: `embedding::encode(ctx, source)` embeds a segment
  stream of text, images, video and audio, alone or interleaved as one input.
  On EmbeddingGemma 2 with its projector, every input kind agrees with
  llama.cpp's own embedding server to a cosine of 0.999 or better; the
  reference vectors are committed in `tests/fixtures/embedding` and
  regenerated offline by `tests/scripts/embedding_reference.sh`. Text embeds
  through the same path as a one-segment stream (`decode::TokenSource`), with
  the same signature as before. An empty input is refused. So is a context
  that keeps memory, attends causally and does not pool, since an embedding
  reads every entry of one dispatch.
- `embedding::dimension()` reads the width the model outputs
  (`llama_model_n_embd_out`), and every embedding reader takes the width from
  it: EmbeddingGemma 2's vectors are 768 wide, not its backbone's 512.
- `decode::segments` places a source's segments into one sequence, in order,
  in the shape the context needs (`StreamReading`). A context that keeps what
  it is given gets one dispatch per segment, so a reasoning prefill keeps its
  dispatch shape. A context that reads its input only whole (pooled, without
  memory, or with attention that is not causal) gets the whole stream as one
  `llama_batch_ext` and one `llama_process`; a stream larger than one ubatch is
  refused before anything runs. `BranchStore::decode_segments` moves the
  branch's books as each segment lands. A decode error after a segment landed,
  from the dispatch or from the source, is partial and poisons the branch.
- `MtmdSource` takes `add_special`, so the caller frames a standalone media
  input with the model's special tokens, as `tokenizer::tokenize` does for
  text. It defaults to false, so chat callers are unchanged.
- A video's timestamp intervals count from its first frame, so a window cut
  from the middle of a clip fuses its frames as the clip's start does.
- `BranchStore::decode_segments` takes an optional `on_progress`, told as each
  segment of an input lands how much of that input is in the KV
  (`SegmentProgress`). `SegmentSource` gains `input_of` and `total_of`,
  defaulted so a source without inputs is unchanged.
- `MtmdSource` maps no inputs for an empty stream, and its scan for an input's
  neighbouring segments never steps below the first.

## 3.1.1 — 2026-10-17

- An absent seed means the seed the chain has. A branch created without a seed
  is seeded from the clock once, and its cached params record that seed instead
  of 0; `set_sampler_params()` resolves an absent seed to the cached one before
  comparing, so a call that names only a temperature keeps the chain, its
  penalty window and its PRNG position when nothing else changed, and a rebuild
  draws on the same seed rather than a new clock seed. The chain is built from
  the resolved record in both places, so the cache says exactly what the chain
  draws with. `snapshot_params(p, absent_seed)` makes the caller say what an
  absent seed means.

## 3.1.0 — 2026-10-17

- `fast_forward(parent, child)`: the child's state becomes the parent's, which
  keeps its handle, place in the tree and steer callback (`git merge --ff-only`).
  Fork, work on the fork, then fast-forward the parent to it or prune it: a
  checkpoint any branch can keep or discard while siblings and descendants run
  on. Everything that follows the child's tokens moves, sampler and grammar
  included; no cells move, and `cells_used` is unchanged. Refused, with nothing
  changed, unless the child was forked from this parent, the parent has neither
  decoded nor changed since, and the child has no children and is not poisoned.
- `BranchState::revision` counts every change to the state a fast-forward moves,
  and `fork_revision` records the parent's at fork time.
- `BranchState::poisoned` (read with `is_poisoned()`) marks a branch whose
  failed decode left its KV ahead of its books. It is per branch: in a
  `decode_scatter` across branches, the error's `partial` is about the whole
  operation, and this flag says which branches were hurt. A fork inherits it.
  Prune and replay.

- Video input: `MediaInput::Kind::Video` carries the sampled frames of one clip
  (`VideoFrames`: encoded frames, their timestamps, the timestamp interval and
  optional `VideoNote`s). `MtmdSource` lays them out as llama.cpp's video helper
  does, through a lazy bitmap at the input's marker, so Qwen-VL projectors fuse
  adjacent frames in pairs and the tokens match the server's video path chunk for
  chunk. Frames must share one size; timestamps must increase. Notes join the
  timestamp text, leaving fusion unchanged. Video containers are refused: the
  caller decodes the clip. `MTMD_VIDEO` stays off.
- `tests/fixtures/video`: a generated four-second clip, its 16 frames at 4 fps,
  and `video-reference.json`, the helper's own tokenization, regenerated by
  `tests/scripts/video_reference.sh`.

- `branch::reseed_sampler(handle, seed)` reseeds a stochastic chain's PRNG (a
  greedy chain has none and is left alone). It counts as a change for
  `fast_forward`, and the branch's cached params record the new seed, so
  `set_sampler_params` compares against the chain as it is: asking for the
  seed the chain had before applies it, and asking for the seed it has now
  keeps the chain and its penalty window.
- `merge_logits` checks every expert before writing, so a refused merge changes
  nothing; a merge of no experts is not a change.

## 3.0.0 — 2026-10-13

- llama.cpp pinned to b11559 (from b9581, 2026-06-09): about two thousand
  upstream commits. Required by models whose architectures the old pin did not
  know (EmbeddingGemma 2) and by the upstream fix that makes `fork()` carry the
  repetition-penalty window (the four fork-fidelity integration cases, held
  since the defect was found, now pass).
- Breaking: `chat_in::FormatResult.parser` is a `common_peg_arena` (upstream
  dropped the parser's string form) and `thinking_end_tag` is
  `thinking_end_tags`; `chat_out::parse` takes the arena and no longer takes a
  reasoning format, which upstream now decides when the parser is built.
- Breaking: `sampler::create_chain` and `BranchStore::create_sampler` take the
  vocabulary size, which upstream's penalties sampler needs.
- `ModelRegistry` keys on `llama_model_params.load_mode` (upstream replaced
  `use_mmap`, `use_mlock` and `use_direct_io` with it).
- `decode.hpp` and `embedding.hpp` fill `llama_batch_init` batches themselves and
  no longer include llama.cpp's `common.h`; `llama_batch` and `mtmd_input_text`
  carry `static_assert` shape guards, and `MtmdSource` names the text struct's
  fields (upstream inserted `text_len`, which a positional initialiser compiled
  wrong without a word).
- `tests/bench`: the engine benchmark and the b9581 baseline taken on an M2 Pro;
  the integration CMake links against a plain llama.cpp build tree; the version
  check reads `.llama-cpp-version`.

## 2.0.0 — 2026-10-03

- **Relicensed from Apache 2.0 to FSL-1.1-Apache-2.0** (Functional Source
  License; each release converts to Apache 2.0 after two years). See LICENSE,
  NOTICE, and LICENSE-FAQ.md.
- llama.cpp pinned to b9581 (build-llama.sh gains LLAMA_BUILD_APP=OFF;
  test CMake adapts to the llama-common/common-base split).
- Branch natives: `branch::get_logits_at` (position logit reads) and
  `ForkOpts{clone_logits}` — the primitives behind the SDK's single-decode
  rerank scoring.
- +307-line branch integration test covering fork/logits semantics.

## [1.0.1-alpha] - 2025-01-25

### Added
- **Automatic include path setup**: When using `add_subdirectory(liblloyal)` after `add_subdirectory(llama.cpp)`, liblloyal now automatically creates the `llama/llama.h` and `llama/ggml.h` include structure. Consumers no longer need manual include path workarounds.

### Changed
- **Conditional install rules**: CMake install/export rules now only run when liblloyal is the top-level project. When used via `add_subdirectory()`, install rules are skipped to avoid conflicts with parent project builds.

### Migration Notes
- **No breaking changes**. Existing consumers using manual include path setup will continue to work.
- Consumers using `add_subdirectory()` now get automatic include path setup — no code changes required.

## [1.0.0-alpha] - Initial Release

### Added
- Header-only C++ library for llama.cpp primitives
- Core modules:
  - `tokenizer.hpp` - Tokenization/detokenization with special token handling
  - `decode.hpp` - Batch decoding with sequence awareness
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
