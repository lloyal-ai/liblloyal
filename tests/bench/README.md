# Engine benchmark

One binary, compiled unchanged against whichever llama.cpp the build directory holds, so a pin and its successor report the same items in the same units. It is the non-functional gate for an engine change: take the numbers before the change, take them after, compare.

## What it measures

Qwen3.5-4B with its projector, greedy sampling, one context (`n_ctx` 32768, `n_batch` 512, `n_seq_max` 10, all layers on the GPU). Each row is the p50 of its samples.

| item | what |
| -- | -- |
| `model_load`, `context_init` | the two loads |
| `chat_in_format_with_tool` | `chat_in::format` with a tool definition; also reports the grammar size |
| `prefill` | 881 prompt tokens through `branch::prefill` |
| `fork` | one `branch::fork` of that prefix |
| `decode_each_step_n{1,3,8}` | one batched decode step for 1, 3 and 8 branches |
| `sample_no_grammar_n{1,3,8}` | `branch::sample` per branch in that loop |
| `sample_grammar_mask_first`, `sample_grammar_sample_first` | `branch::sample` under the tool's grammar in each order |
| `image_prefill_total`, `image_decode_and_tokenize` | one image through `MtmdSource` and `decode_segments` |

## Run

```bash
cd tests && cmake -B build_bench -DLLOYAL_BUILD_BENCH=ON \
  -DLLAMA_CPP_DIR=../../llama.cpp -DLLAMA_CPP_BUILD_DIR=../../llama.cpp/build \
  && cmake --build build_bench --target Bench
./build_bench/Bench model.gguf mmproj.gguf fixtures/cat.jpg <label> > run.jsonl
python3 bench/compare.py bench/baselines/<baseline>.jsonl run.jsonl
```

Run it twice and keep the second: the first run pays for the page cache. Keep the machine otherwise idle. A baseline under `baselines/` names its machine, model and flags on its first line; take a new baseline on the same machine before an engine change, never compare across machines.

## Stop rule

A refactor or pin move stops when any `decode_*`, `prefill` or `image_*` row regresses more than 10 percent at p50 against the baseline taken before it. The other rows are informative.
