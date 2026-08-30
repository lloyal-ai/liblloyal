#pragma once

// SPDX-License-Identifier: LicenseRef-FSL-1.1-Apache-2.0
// Copyright 2026 Lloyal Labs

#include "decode.hpp"

#include <mtmd.h>
#include <mtmd-helper.h>

#include <cstdint>
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
 * - **Codec (here):** pixel decode, tokenization, clip encode, and the
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

/**
 * @brief Turns a marker'd prompt + image bytes into a decode segment stream
 *
 * The prompt carries one media marker (`mtmd_default_marker()`,
 * `"<__media__>"`) per image; mtmd splits it into interleaved text and image
 * chunks. Text chunks come back as ready token ids — never re-tokenized.
 *
 * **Tokenization flags are a parity contract, not a choice.**
 * `add_special = false` (no mid-conversation BOS — matches the text path's
 * `tokenizer::tokenize(vocab, text, false, true)`) and
 * `parse_special = true` (template specials are real tokens). Diverging here
 * puts a spurious BOS around the image and desynchronizes the KV from what
 * the model was trained to see.
 *
 * **Lifetime.** Owns its bitmaps and chunk list. The image bytes are decoded
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
  /**
   * @param ctx        Loaded mtmd context (from `mtmd_init_from_file`)
   * @param prompt     Templated prompt containing one marker per image
   * @param images     Encoded image bytes (jpg/png/bmp/gif), marker order
   * @param sep        Optional leading token run, emitted as segment 0
   * @param n_embd_inp Row width — `llama_model_n_embd_inp(model)`
   * @throws std::runtime_error on a null context, undecodable bytes, audio
   *         or video input, or a marker/image count mismatch
   */
  MtmdSource(mtmd_context* ctx,
             const std::string& prompt,
             const std::vector<std::vector<uint8_t>>& images,
             std::span<const llama_token> sep,
             int32_t n_embd_inp)
      : ctx_(ctx), sep_(sep), n_embd_inp_(n_embd_inp) {
    if (!ctx_) {
      throw std::runtime_error("MtmdSource - NULL mtmd context");
    }

    // Count markers here rather than trusting mtmd_tokenize's return code.
    // Its header documents rc==1 for a marker/bitmap count mismatch, but that
    // check throws from mtmd_tokenizer's constructor and mtmd_tokenize's
    // catch-all reports it as rc==2 — indistinguishable from a genuine
    // preprocessing failure. Counting up front gives the caller the actual
    // diagnosis, and costs nothing next to decoding the bitmaps.
    const std::string marker = mtmd_get_marker(ctx_);
    size_t n_markers = 0;
    if (!marker.empty()) {
      for (size_t p = prompt.find(marker); p != std::string::npos;
           p = prompt.find(marker, p + marker.size())) {
        ++n_markers;
      }
    }
    if (n_markers != images.size()) {
      throw std::runtime_error(
          "MtmdSource - media marker count (" + std::to_string(n_markers) +
          ") does not match image count (" + std::to_string(images.size()) +
          ")");
    }

    // Bytes → bitmaps. Video fails to decode with MTMD_VIDEO off; audio is
    // sniffed by magic bytes and routed to an AUDIO chunk, rejected in the
    // scan after tokenization below. Both fail loud.
    std::vector<const mtmd_bitmap*> ptrs;
    ptrs.reserve(images.size());
    for (const auto& bytes : images) {
      auto wrap = mtmd_helper_bitmap_init_from_buf(
          ctx_, bytes.data(), bytes.size(), /*placeholder*/ false);
      if (wrap.video_ctx) {
        mtmd_helper_video_free(wrap.video_ctx);
        if (wrap.bitmap) mtmd_bitmap_free(wrap.bitmap);
        throw std::runtime_error("MtmdSource - video input is not supported");
      }
      if (!wrap.bitmap) {
        throw std::runtime_error(
            "MtmdSource - unsupported media bytes at image " +
            std::to_string(bitmaps_.size()) + " (expected jpg/png/bmp/gif)");
      }
      bitmaps_.emplace_back(wrap.bitmap);
      ptrs.push_back(wrap.bitmap);
    }

    chunks_.reset(mtmd_input_chunks_init());
    mtmd_input_text txt{prompt.c_str(), /*add_special*/ false,
                        /*parse_special*/ true};
    const int32_t rc =
        mtmd_tokenize(ctx_, chunks_.get(), &txt, ptrs.data(), ptrs.size());
    if (rc == 1) {
      throw std::runtime_error(
          "MtmdSource - media marker count does not match image count");
    }
    if (rc != 0) {
      throw std::runtime_error("MtmdSource - image preprocessing failed");
    }

    mrope_    = mtmd_decode_use_mrope(ctx_);
    n_chunks_ = mtmd_input_chunks_size(chunks_.get());
    lead_     = sep_.empty() ? 0 : 1;

    // Reject audio HERE, not in at(). The helper sniffs audio by magic bytes
    // and routes it automatically, so it arrives as an AUDIO chunk; deferring
    // the error means decode_segments may already have committed preceding
    // text or image chunks, handing the caller a failure with a partially
    // advanced branch.
    cells_ = sep_.size();
    for (size_t k = 0; k < n_chunks_; ++k) {
      const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks_.get(), k);
      const auto kind = mtmd_input_chunk_get_type(ch);
      if (kind != MTMD_INPUT_CHUNK_TYPE_TEXT &&
          kind != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        throw std::runtime_error("MtmdSource - audio input is not supported");
      }
      cells_ += mtmd_input_chunk_get_n_tokens(ch);
    }
  }

  size_t size() override { return lead_ + n_chunks_; }

  /**
   * @brief KV cells this prefill will consume — see decode::SegmentSource::cells
   *
   * Knowable here, and never an estimate: counted during construction, after
   * `mtmd_tokenize` and before any clip encode. Image row counts are fixed at
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

      case MTMD_INPUT_CHUNK_TYPE_IMAGE: {
        if (mtmd_encode_chunk(ctx_, chunk) != 0) {
          throw std::runtime_error("MtmdSource - image encode failed");
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
        // AUDIO (or anything new upstream adds). Never skip silently — the
        // caller asked for this content to be in the KV.
        throw std::runtime_error("MtmdSource - audio input is not supported");
    }
  }

  void positions(size_t i, llama_pos base, llama_pos* out) override {
    const mtmd_input_chunk* chunk = chunk_at(i);
    const int32_t n =
        static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(chunk));

    if (!mrope_) {
      for (int32_t k = 0; k < n; ++k) out[k] = base + k;
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
