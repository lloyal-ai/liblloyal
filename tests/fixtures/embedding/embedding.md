# Embedding integration fixtures

These fixtures check that an input embedded through liblloyal, with text, an image, or both placed as one stream, lands where llama.cpp's own server puts it. The model is EmbeddingGemma 2 (Q8_0) with its BF16 projector, whose text and images share one 768-wide space.

## The images

The three images in this folder were provided by the project's maintainer as test inputs, and are used here only to exercise the model. The two that were WebP were converted to JPEG with `sips` at quality 90, because llama.cpp reads WebP only through ffmpeg, which liblloyal's builds leave out. `cat.jpg` is the existing fixture one folder up.

| File | Shows | Pixels | Bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| `../cat.jpg` | a cat | 631×896 | 180657 | 039e6c833f967f564da231da29a9b77c4e63db7529ce82a9ee29f6bbeb8b6321 |
| `f1.jpg` | an open-plan office with desks, screens and plants | 1295×809 | 945599 | a6d58128992b9483a097f9fe1d7726a0c1847bc193f15fb3ee22cb93735c8989 |
| `navvis-factory-planning2-image1.jpg` | robots assembling car bodies on a factory line | 1200×800 | 502249 | b1c655a0a529c794b9bd2d13b2c88c5439d08cf582f4490f9252f23d8e3f9d94 |
| `Global-X-Digital-Evolution-Series-Aug-25-Part-2.jpg` | an infographic of AI adoption by industry | 1200×2219 | 657298 | 46f7e5c9e06a1e27cb7cf5a8d2dda095aea51a237a44d6ab6968268f73d2d01b |

## The reference

`embeddinggemma2-reference.json` holds the vectors llama.cpp's server gives, at the pinned revision:

- each image alone;
- the clip `../video/red-square-moves.mp4`, read by llama.cpp's own video helper at its default 4 frames a second, which gives exactly the 16 frames committed beside it;
- a search query for each image and for the clip, with the model card's prefix (`task: search result | query: …`), naming the input it must find;
- one input with text around an image, which only reads right when the stream is placed whole.

The server tokenizes each input with the model's own special tokens (BOS and EOS), pools it whole, and normalises it to unit length. At the pin, each query's best match among the four images and the clip is its own: 5 of 5.

The file was made on the CPU, as CI runs these cases. The same server on Metal agrees with it to a cosine of 0.99988 or better on every entry, so the cases' bar of 0.999 holds on either backend.

`tests/scripts/embedding_reference.sh` regenerates the file. It builds llama.cpp's server in its own build directory, runs it on the CPU (`-c 8192 -b 8192 -ub 8192`, mean pooling), and asks it for every vector through `embedding_reference.py`. It is run by hand, offline: no test and no CI job builds or calls the server. Regenerate the file whenever the pin moves.

```sh
LLAMA_TEST_MODEL=/path/to/embeddinggemma-2-Q8_0.gguf \
LLAMA_MMPROJ_MODEL=/path/to/mmproj-BF16.gguf \
tests/scripts/embedding_reference.sh
```

## Running the cases

The `embedding multimodal:*` cases run with the model pair, in their own process:

```sh
LLOYAL_EMBED_MM_TEST=1 LLAMA_N_GPU_LAYERS=-1 \
LLAMA_TEST_MODEL=/path/to/embeddinggemma-2-Q8_0.gguf \
LLAMA_MMPROJ_MODEL=/path/to/mmproj-BF16.gguf \
tests/build_integration/IntegrationRunner --test-case='embedding multimodal:*'
```

The infographic and the clip are each over a thousand entries; on a CPU each takes minutes to encode, so the cases share one embedding of every input.
