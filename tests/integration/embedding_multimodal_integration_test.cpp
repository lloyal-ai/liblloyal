#include <cstdlib>
#include <cstring>
#include <doctest/doctest.h>
#include "media_fixtures.hpp"
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/mtmd.hpp>
#include <algorithm>
#include <vector>

using namespace lloyal;

/**
 * Media on an embedding context: EmbeddingGemma 2 with its projector, whose
 * images share one space with its text.
 *
 * A matched decoder and projector per process, as the vision and audio suites
 * run: LLAMA_TEST_MODEL is the EmbeddingGemma 2 decoder and LLAMA_MMPROJ_MODEL
 * its projector. Every case is skipped unless LLOYAL_EMBED_MM_TEST=1, so the
 * suite's other groups never load this pair. Text-only checks on the same model
 * run in the embedding suite, under LLAMA_EMBED_MODEL.
 */

static bool embedding_multimodal_disabled() {
  const char* flag = std::getenv("LLOYAL_EMBED_MM_TEST");
  return !flag || std::strcmp(flag, "1") != 0;
}

namespace {

struct LlamaBackendGuard {
  LlamaBackendGuard() { llama_backend_init(); }
  ~LlamaBackendGuard() { llama_backend_free(); }
};

}  // namespace

TEST_CASE("embedding multimodal: image input tokenizes as upstream does"
          * doctest::skip(embedding_multimodal_disabled())) {
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  auto* mtmd = TestConfig::acquire_mtmd(model.get());
  REQUIRE_MESSAGE(mtmd != nullptr, "LLAMA_MMPROJ_MODEL must name EmbeddingGemma 2's projector");
  REQUIRE(mtmd_support_vision(mtmd));

  const auto image = TestConfig::read_fixture("cat.jpg");
  REQUIRE_FALSE(image.empty());
  const std::string prompt = mtmd_default_marker();

  // Upstream: what llama-server tokenizes for an image embedding input.
  auto decoded = mtmd_helper_bitmap_init_from_buf(mtmd, image.data(), image.size(), false,
                                                  mtmd_helper_init_opt_default());
  REQUIRE(decoded.bitmap);
  ::mtmd::bitmap_ptr bitmap(decoded.bitmap);
  const mtmd_bitmap* ptr = bitmap.get();
  ::mtmd::input_chunks_ptr expected(mtmd_input_chunks_init());
  mtmd_input_text text{};
  text.text = prompt.c_str();
  text.text_len = prompt.size();
  text.add_special = true;
  text.parse_special = true;
  REQUIRE(mtmd_tokenize(mtmd, expected.get(), &text, &ptr, 1) == 0);

  const bool embedding_framing = true;  // add_special: a standalone input, framed as upstream frames it
  MtmdSource source(mtmd, prompt, {image}, {}, llama_model_n_embd_inp(model.get()), embedding_framing);

  const size_t n = mtmd_input_chunks_size(expected.get());
  REQUIRE(source.size() == n);
  size_t cells = 0;
  for (size_t i = 0; i < n; ++i) {
    const auto* chunk = mtmd_input_chunks_get(expected.get(), i);
    cells += mtmd_input_chunk_get_n_tokens(chunk);
    if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
    size_t n_tokens = 0;
    const auto* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
    const auto segment = source.at(i);
    REQUIRE(segment.kind == decode::Segment::Kind::Text);
    REQUIRE(segment.tokens.size() == n_tokens);
    CHECK(std::equal(tokens, tokens + n_tokens, segment.tokens.begin()));
  }
  CHECK(source.cells() == cells);
}
