# Audio integration fixtures

These fixtures exercise Qwen3-ASR through the existing real-model integration runner. They are synthetic test speech, not a quality benchmark or recordings of a person.

`asr-counting.wav` says:

> One, two, three, four, five. The meeting is on Thursday at nine thirty.

`asr-repeated.wav` concatenates that recording twice. Its 11.2855-second duration crosses the pinned Qwen3-ASR encoder's 800-mel-frame chunk window, so tests can detect lost or overwritten encoder output.

| File | Encoding | Samples | Duration | SHA-256 |
| --- | --- | ---: | ---: | --- |
| asr-counting.wav | Mono, 16 kHz, PCM16 WAV | 90,284 | 5.64275 s | b400cee35414b71dc72350fe3685996e6c823a3f81d60d49eba2c6db7e145e1e |
| asr-repeated.wav | Mono, 16 kHz, PCM16 WAV | 180,568 | 11.2855 s | ee1679d044380028fca1a49adf76caef7dbdf3321bf30b4adfb79d41341f3251 |

Generated locally on 10 October 2026 with macOS Samantha at rate 150. Regenerating with another macOS/voice version may change samples and hashes. Tests accept either “nine thirty” or “9:30” while requiring the counting sequence and meeting day.

Run from the liblloyal root after building `IntegrationRunner`, using a matched Qwen3-ASR decoder/projector pair:

```sh
LLOYAL_ASR_TEST=1 \
LLAMA_N_GPU_LAYERS=-1 \
LLAMA_TEST_MODEL=/path/to/Qwen3-ASR-0.6B-Q8_0.gguf \
LLAMA_MMPROJ_MODEL=/path/to/mmproj-Qwen3-ASR-0.6B-Q8_0.gguf \
tests/build_integration/IntegrationRunner --test-case='audio:*'
```

The explicit audio flag keeps these cases separate from the runner's text and image model selections. Missing model paths fail the selected audio tests. Run image regression tests in a separate process with their own matched model/projector.

CI uses [ggml-org/Qwen3-ASR-0.6B-GGUF at revision 928ab958557df9aa2ef1c93e0e83c7ad0933fae2](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF/tree/928ab958557df9aa2ef1c93e0e83c7ad0933fae2). Both artifacts are verified against [asr-models.sha256](asr-models.sha256), including when restored from cache. Configure the integration runner with `LLOYAL_ENABLE_UBSAN_INTEGRATION=ON` and set `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1` to reproduce its sanitizer checks. Use `LLAMA_N_GPU_LAYERS=0` for CPU execution, as CI does.

The cases cover prompt/tokenization parity, encode-once fan-out, inherited transcripts, exact cell accounting, winner retention, multiple encoder chunks and recordings, partial-failure reclamation, malformed inputs, explicit admission limits, resampling and caller-buffer lifetime. These fixtures establish native correctness; they do not establish accuracy on general speech or production latency.

`MediaInput::Kind::Audio` admits ordinary RIFF integer PCM WAV (8/16/24/32-bit), with optional INFO/JUNK metadata. The caller supplies aggregate byte and mono-sample budgets through `AudioLimits`; samples are counted at the projector's rate. Resampling reserves one extra sample for the upstream decoder's allocation bound. Sample conversion and resampling remain owned by llama.cpp's mtmd helper. Compressed audio, floating-point WAV, RF64 and WAVE_FORMAT_EXTENSIBLE are outside this admission contract.
