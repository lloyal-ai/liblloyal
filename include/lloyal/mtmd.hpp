#pragma once

// SPDX-License-Identifier: LicenseRef-FSL-1.1-Apache-2.0
// Copyright 2026 Lloyal Labs

#include "decode.hpp"

#include <mtmd.h>
#include <mtmd-helper.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file mtmd.hpp
 * @brief llama.cpp mtmd as a decode::SegmentSource (multimodal codec)
 *
 * **Opt-in header.** Including it takes a dependency on llama.cpp's
 * `tools/mtmd`; liblloyal's core does not. `branch.hpp` and `decode.hpp` see
 * only `decode::SegmentSource` and never an `mtmd_*` type — this file is the
 * one place the two vocabularies meet.
 *
 * **Build requirement (the consumer's, not liblloyal's).** liblloyal is
 * header-only and does not link anything on your behalf — linking is the
 * binding layer's job. A target that includes this header must itself link
 * llama.cpp's `mtmd` target, which PUBLIC-propagates the `tools/mtmd`
 * include path:
 *
 *     add_subdirectory(${LLAMA_CPP_DIR}/tools/mtmd mtmd EXCLUDE_FROM_ALL)
 *     target_link_libraries(your_target PRIVATE liblloyal::liblloyal mtmd)
 *
 * A consumer that never includes this header links neither, and carries no
 * multimodal dependency at all.
 *
 * The split it implements:
 *
 * - **Codec (here):** media decode, tokenization, encoder execution, and the
 *   per-model position geometry. Everything format-specific.
 * - **Kernel (`BranchStore::decode_segments`):** which rail each segment
 *   takes, at what position, which one captures logits, and the cell/slack
 *   bookkeeping. Everything KV-specific.
 *
 * A binding therefore constructs an MtmdSource and makes one kernel call.
 * Every `SessionContext` implementation (N-API, Nitro/JSI, …) shares this
 * file rather than reimplementing the walk — and a platform encoder (CoreML,
 * say) can supply a different SegmentSource without touching the kernel.
 */

namespace lloyal {

/** A borrowed admitted representation. Bytes are consumed during construction. */
struct MediaInput {
  enum class Kind { Image, Audio };
  Kind kind;
  std::span<const uint8_t> bytes;
};

/** Per-source audio budgets, across all inputs. Samples are mono at projector rate. */
struct AudioLimits {
  size_t max_bytes = 0;
  size_t max_samples = 0;
};

namespace mtmd_detail {

inline bool tag(std::span<const uint8_t> bytes, size_t offset, const char* value) {
  return offset <= bytes.size() && bytes.size() - offset >= 4 &&
         std::memcmp(bytes.data() + offset, value, 4) == 0;
}

inline uint16_t u16(std::span<const uint8_t> bytes, size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset < 2) {
    throw std::runtime_error("MtmdSource - truncated PCM WAV");
  }
  return static_cast<uint16_t>(bytes[offset] | (uint16_t(bytes[offset + 1]) << 8));
}

inline uint32_t u32(std::span<const uint8_t> bytes, size_t offset) {
  return uint32_t(u16(bytes, offset)) | (uint32_t(u16(bytes, offset + 2)) << 16);
}

// The pinned helper's audio sniff is private. Refuse its audio signatures on
// the image entry point before it can allocate a decoded recording.
inline bool encoded_audio(std::span<const uint8_t> bytes) {
  if (bytes.size() < 12) return false;
  return (tag(bytes, 0, "RIFF") && tag(bytes, 8, "WAVE")) ||
         tag(bytes, 0, "fLaC") || std::memcmp(bytes.data(), "ID3", 3) == 0 ||
         (bytes[0] == 0xff && (bytes[1] & 0xe0) == 0xe0);
}

// Inspect framing only; the existing mtmd/miniaudio helper still owns sample
// conversion and resampling. Restrict the admitted format to integer PCM WAV.
inline size_t wav_sample_bound(std::span<const uint8_t> bytes, uint32_t target_rate) {
  if (bytes.size() < 12 || !tag(bytes, 0, "RIFF") || !tag(bytes, 8, "WAVE") ||
      u32(bytes, 4) != bytes.size() - 8) {
    throw std::runtime_error("MtmdSource - invalid PCM WAV container");
  }
  std::span<const uint8_t> format;
  std::span<const uint8_t> data;
  bool found_data = false;
  for (size_t offset = 12; offset < bytes.size();) {
    if (bytes.size() - offset < 8) {
      throw std::runtime_error("MtmdSource - truncated WAV chunk");
    }
    const size_t length = u32(bytes, offset + 4);
    const size_t remaining = bytes.size() - offset - 8;
    const size_t padding = length & 1;
    if (length > remaining || padding > remaining - length) {
      throw std::runtime_error("MtmdSource - truncated WAV chunk");
    }
    const size_t padded = length + padding;
    const auto chunk = bytes.subspan(offset + 8, length);
    if (tag(bytes, offset, "fmt ")) {
      if (!format.empty() || (length != 16 && length != 18)) {
        throw std::runtime_error("MtmdSource - invalid WAV format chunk");
      }
      format = chunk;
    } else if (tag(bytes, offset, "data")) {
      if (format.empty() || found_data) {
        throw std::runtime_error("MtmdSource - invalid WAV data chunk");
      }
      data = chunk;
      found_data = true;
    } else if (!tag(bytes, offset, "JUNK") &&
               !(tag(bytes, offset, "LIST") && tag(chunk, 0, "INFO"))) {
      throw std::runtime_error("MtmdSource - unsupported WAV chunk");
    }
    offset += 8 + padded;
  }
  if (format.empty() || !found_data || u16(format, 0) != 1 ||
      (format.size() == 18 && u16(format, 16) != 0)) {
    throw std::runtime_error("MtmdSource - expected integer PCM WAV");
  }
  const uint32_t channels = u16(format, 2);
  const uint32_t rate = u32(format, 4);
  const uint32_t bits = u16(format, 14);
  const bool integer_pcm = bits == 8 || bits == 16 || bits == 24 || bits == 32;
  const uint32_t alignment = channels * (bits / 8);
  if (!channels || !rate || !integer_pcm || !alignment ||
      u16(format, 12) != alignment || u32(format, 8) != uint64_t(rate) * alignment ||
      data.size() % alignment != 0) {
    throw std::runtime_error("MtmdSource - inconsistent PCM WAV format");
  }
  const uint64_t frames = data.size() / alignment;
  // b9581's miniaudio may allocate floor(ratio * frames) + 1 after resampling.
  const uint64_t samples = rate == target_rate ? frames : frames * target_rate / rate + 1;
  if (frames < 2 || samples < 2 ||
      samples > std::numeric_limits<int32_t>::max() / sizeof(float)) {
    throw std::runtime_error("MtmdSource - unsupported audio sample count");
  }
  return static_cast<size_t>(samples);
}

inline std::vector<MediaInput> image_inputs(const std::vector<std::vector<uint8_t>>& images) {
  std::vector<MediaInput> inputs;
  inputs.reserve(images.size());
  for (const auto& bytes : images) inputs.push_back({MediaInput::Kind::Image, bytes});
  return inputs;
}

} // namespace mtmd_detail

/**
 * @brief Turns a marker'd prompt + admitted media into a decode segment stream
 *
 * The prompt carries one media marker (`mtmd_default_marker()`,
 * `"<__media__>"`) per input; mtmd splits it into interleaved text and media
 * chunks. Text chunks come back as ready token ids — never re-tokenized.
 *
 * **Tokenization flags are a parity contract, not a choice.**
 * `add_special = false` (no mid-conversation BOS — matches the text path's
 * `tokenizer::tokenize(vocab, text, false, true)`) and
 * `parse_special = true` (template specials are real tokens). Diverging here
 * puts a spurious BOS around the image and desynchronizes the KV from what
 * the model was trained to see.
 *
 * **Lifetime.** Owns its bitmaps and chunk list. The input bytes are decoded
 * into those bitmaps during construction and never retained, so the caller may
 * release them as soon as the constructor returns. `ctx` and `sep` ARE retained
 * and must outlive this object. Satisfies SegmentSource's in-order contract:
 * an Embd segment's `rows` point into mtmd's context-owned encode buffer,
 * which the *next* `at()` overwrites.
 *
 * @code
 *   MtmdSource src(mtmd, prompt, imageBytes, sepTokens, n_embd_inp);
 *   auto r = store.decode_segments(handle, src);
 * @endcode
 */
class MtmdSource final : public decode::SegmentSource {
public:
  /** Existing image entry point; audio requires typed inputs and explicit limits. */
  MtmdSource(mtmd_context* ctx,
             const std::string& prompt,
             const std::vector<std::vector<uint8_t>>& images,
             std::span<const llama_token> sep,
             int32_t n_embd_inp)
      : MtmdSource(ctx, prompt, mtmd_detail::image_inputs(images), sep, n_embd_inp, {}) {}

  /**
   * Inputs follow prompt marker order. Audio is integer PCM WAV, with optional
   * INFO/JUNK metadata. Limits apply before decoding or preprocessing; the
   * caller supplies policy. ctx and sep must outlive this source.
   */
  MtmdSource(mtmd_context* ctx,
             const std::string& prompt,
             std::span<const MediaInput> inputs,
             std::span<const llama_token> sep,
             int32_t n_embd_inp,
             AudioLimits limits)
      : ctx_(ctx), sep_(sep), n_embd_inp_(n_embd_inp) {
    if (!ctx_) throw std::runtime_error("MtmdSource - NULL mtmd context");
    if (n_embd_inp_ <= 0) throw std::runtime_error("MtmdSource - invalid embedding width");
    validate_markers(prompt, inputs.size());
    const auto sample_bounds = validate_inputs(inputs, limits);

    std::vector<const mtmd_bitmap*> ptrs;
    ptrs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      auto wrap = mtmd_helper_bitmap_init_from_buf(
          ctx_, input.bytes.data(), input.bytes.size(), /*placeholder*/ false);
      if (wrap.video_ctx) {
        mtmd_helper_video_free(wrap.video_ctx);
        if (wrap.bitmap) mtmd_bitmap_free(wrap.bitmap);
        throw std::runtime_error("MtmdSource - video input is not supported");
      }
      if (!wrap.bitmap) {
        throw std::runtime_error("MtmdSource - unsupported media bytes at input " + std::to_string(i));
      }
      bitmaps_.emplace_back(wrap.bitmap);
      const bool audio = input.kind == MediaInput::Kind::Audio;
      if (mtmd_bitmap_is_audio(wrap.bitmap) != audio) {
        throw std::runtime_error("MtmdSource - decoded media kind does not match input");
      }
      if (audio && (mtmd_bitmap_get_nx(wrap.bitmap) < 2 ||
                    mtmd_bitmap_get_nx(wrap.bitmap) > sample_bounds[i])) {
        throw std::runtime_error("MtmdSource - decoded audio exceeds validated sample bound");
      }
      ptrs.push_back(wrap.bitmap);
    }

    chunks_.reset(mtmd_input_chunks_init());
    mtmd_input_text txt{prompt.c_str(), /*add_special*/ false, /*parse_special*/ true};
    if (mtmd_tokenize(ctx_, chunks_.get(), &txt, ptrs.data(), ptrs.size()) != 0) {
      throw std::runtime_error("MtmdSource - media preprocessing failed");
    }
    mrope_ = mtmd_decode_use_mrope(ctx_);
    n_chunks_ = mtmd_input_chunks_size(chunks_.get());
    lead_ = sep_.empty() ? 0 : 1;
    cells_ = sep_.size();
    for (size_t k = 0; k < n_chunks_; ++k) {
      const auto* chunk = mtmd_input_chunks_get(chunks_.get(), k);
      const auto kind = mtmd_input_chunk_get_type(chunk);
      if (kind != MTMD_INPUT_CHUNK_TYPE_TEXT && kind != MTMD_INPUT_CHUNK_TYPE_IMAGE &&
          kind != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        throw std::runtime_error("MtmdSource - unsupported media chunk");
      }
      const size_t rows = mtmd_input_chunk_get_n_tokens(chunk);
      if (rows > std::numeric_limits<int32_t>::max() ||
          rows > std::numeric_limits<size_t>::max() - cells_) {
        throw std::runtime_error("MtmdSource - media token count overflow");
      }
      cells_ += rows;
    }
  }

  size_t size() override { return lead_ + n_chunks_; }

  /**
   * @brief KV cells this prefill will consume — see decode::SegmentSource::cells
   *
   * Knowable here, and never an estimate: counted during construction, after
   * `mtmd_tokenize` and before any encoder execution. Media row counts are fixed at
   * tokenize time, which is what the placeholder-bitmap counting flow in
   * `mtmd.h` relies on.
   */
  size_t cells() const override { return cells_; }

  decode::Segment at(size_t i) override {
    decode::Segment seg;

    if (lead_ != 0 && i == 0) {
      seg.kind   = decode::Segment::Kind::Text;
      seg.tokens = sep_;
      return seg;
    }

    const mtmd_input_chunk* chunk = chunk_at(i);
    switch (mtmd_input_chunk_get_type(chunk)) {
      case MTMD_INPUT_CHUNK_TYPE_TEXT: {
        size_t n_text = 0;
        const llama_token* toks =
            mtmd_input_chunk_get_tokens_text(chunk, &n_text);
        seg.kind   = decode::Segment::Kind::Text;
        seg.tokens = std::span<const llama_token>(toks, n_text);
        return seg;
      }

      case MTMD_INPUT_CHUNK_TYPE_IMAGE:
      case MTMD_INPUT_CHUNK_TYPE_AUDIO: {
        if (mtmd_encode_chunk(ctx_, chunk) != 0) {
          throw std::runtime_error("MtmdSource - media encode failed");
        }
        seg.kind = decode::Segment::Kind::Embd;
        // Context-owned buffer, reused by the next encode — the in-order
        // contract is what makes handing it out safe.
        seg.rows           = mtmd_get_output_embd(ctx_);
        seg.n_rows         = static_cast<int32_t>(
            mtmd_input_chunk_get_n_tokens(chunk));
        seg.n_embd_inp     = n_embd_inp_;
        seg.n_pos          = mtmd_input_chunk_get_n_pos(chunk);
        seg.n_pos_per_embd = mrope_ ? 4 : 1;
        seg.non_causal     = mtmd_decode_use_non_causal(ctx_, chunk);
        return seg;
      }

      default:
        throw std::runtime_error("MtmdSource - unsupported media chunk");
    }
  }

  void positions(size_t i, llama_pos base, llama_pos* out) override {
    const mtmd_input_chunk* chunk = chunk_at(i);
    const int32_t n =
        static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(chunk));

    if (!mrope_ || mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
      if (int64_t(base) + n > std::numeric_limits<llama_pos>::max()) {
        throw std::runtime_error("MtmdSource - media position overflow");
      }
      const size_t axes = mrope_ ? 4 : 1;
      for (size_t axis = 0; axis < axes; ++axis) {
        std::iota(out + axis * n, out + (axis + 1) * n, base);
      }
      return;
    }

    const mtmd_image_tokens* img = mtmd_input_chunk_get_tokens_image(chunk);
    if (!img) {
      throw std::runtime_error("MtmdSource - image tokens missing");
    }

    // The impl applies `base` itself (the header's "relative position" note
    // is stale) and the rules differ per model — M-RoPE freezes t and leaves
    // z at 0, HunyuanVL uses row/col — which is exactly why the base is
    // passed in rather than the geometry passed out.
    rel_.resize(static_cast<size_t>(n));
    mtmd_helper_image_get_decoder_pos(img, base, rel_.data());

    // Section-major, and note the order: mtmd's decoder convention puts
    // y in section 1 and x in section 2.
    for (int32_t k = 0; k < n; ++k) {
      out[k]                                  = static_cast<llama_pos>(rel_[k].t);
      out[k + n]                              = static_cast<llama_pos>(rel_[k].y);
      out[k + static_cast<size_t>(2) * n]     = static_cast<llama_pos>(rel_[k].x);
      out[k + static_cast<size_t>(3) * n]     = static_cast<llama_pos>(rel_[k].z);
    }
  }

private:
  void validate_markers(const std::string& prompt, size_t count) const {
    const std::string marker = mtmd_get_marker(ctx_);
    size_t markers = 0;
    if (!marker.empty()) {
      for (size_t p = prompt.find(marker); p != std::string::npos;
           p = prompt.find(marker, p + marker.size())) ++markers;
    }
    // Upstream folds a marker mismatch into a generic preprocessing error.
    if (markers != count) {
      throw std::runtime_error("MtmdSource - media marker count (" + std::to_string(markers) +
                               ") does not match media count (" + std::to_string(count) + ")");
    }
  }

  std::vector<size_t> validate_inputs(std::span<const MediaInput> inputs, AudioLimits limits) const {
    std::vector<size_t> samples(inputs.size(), 0);
    size_t bytes_left = limits.max_bytes;
    size_t samples_left = limits.max_samples;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      switch (input.kind) {
        case MediaInput::Kind::Image:
          if (mtmd_detail::encoded_audio(input.bytes)) {
            throw std::runtime_error("MtmdSource - audio bytes require a typed audio input");
          }
          if (!mtmd_support_vision(ctx_)) {
            throw std::runtime_error("MtmdSource - projector does not support image input");
          }
          break;
        case MediaInput::Kind::Audio:
          if (!mtmd_support_audio(ctx_) || mtmd_get_audio_sample_rate(ctx_) <= 0) {
            throw std::runtime_error("MtmdSource - projector does not support audio input");
          }
          if (!limits.max_bytes || !limits.max_samples) {
            throw std::runtime_error("MtmdSource - audio limits must be positive");
          }
          if (input.bytes.size() > bytes_left) {
            throw std::runtime_error("MtmdSource - audio byte limit exceeded");
          }
          bytes_left -= input.bytes.size();
          samples[i] = mtmd_detail::wav_sample_bound(input.bytes, mtmd_get_audio_sample_rate(ctx_));
          if (samples[i] > samples_left) {
            throw std::runtime_error("MtmdSource - audio sample limit exceeded");
          }
          samples_left -= samples[i];
          break;
        default:
          throw std::runtime_error("MtmdSource - unsupported input kind");
      }
    }
    return samples;
  }

  const mtmd_input_chunk* chunk_at(size_t i) const {
    return mtmd_input_chunks_get(chunks_.get(), i - lead_);
  }

  mtmd_context* ctx_ = nullptr;
  std::span<const llama_token> sep_;
  int32_t n_embd_inp_ = 0;

  std::vector<::mtmd::bitmap_ptr> bitmaps_;
  ::mtmd::input_chunks_ptr chunks_;
  std::vector<mtmd_decoder_pos> rel_;

  bool   mrope_    = false;
  size_t n_chunks_ = 0;
  size_t lead_     = 0;
  size_t cells_    = 0;
};

} // namespace lloyal
