#include <cstdlib>
#include <cstring>
#include <doctest/doctest.h>
#include "media_fixtures.hpp"
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/embedding.hpp>
#include <lloyal/mtmd.hpp>
#include <lloyal/tokenizer.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <memory>
#include <string>
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

using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;

/** An embedding context whose batch holds `n_batch` entries in one ubatch, as
 *  the reference server ran (-c 8192 -b 8192 -ub 8192, mean pooling). */
Context embedding_context(llama_model* model, uint32_t n_batch = 8192) {
  auto params = llama_context_default_params();
  params.n_ctx = 8192;
  params.n_batch = n_batch;
  params.n_ubatch = n_batch;
  params.embeddings = true;
  params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
  return Context(llama_init_from_model(model, params), llama_free);
}

/** The reference-sized context, one per process: the cache above belongs to it. */
llama_context* full_context(llama_model* model) {
  static Context ctx = embedding_context(model);
  return ctx.get();
}

/** The vectors llama.cpp's own server gave the fixtures (scripts/embedding_reference.sh). */
nlohmann::json reference() {
  const auto bytes = TestConfig::read_fixture("embedding/embeddinggemma2-reference.json");
  REQUIRE_FALSE(bytes.empty());
  return nlohmann::json::parse(bytes.begin(), bytes.end());
}

/** An input as the reference describes it: an image, the committed clip, text, or text around an image. */
std::vector<float> embed(llama_context* ctx, mtmd_context* mtmd, const llama_model* model,
                         const nlohmann::json& entry) {
  const std::string kind = entry["kind"];
  if (kind == "text") {
    auto tokens = tokenizer::tokenize(llama_model_get_vocab(model), entry["text"].get<std::string>(), true, true);
    embedding::encode(ctx, tokens, static_cast<int32_t>(llama_n_batch(ctx)));
    return embedding::get(ctx, embedding::Normalize::L2);
  }
  const bool embedding_framing = true;  // add_special: the model's own BOS and EOS
  if (kind == "video") {
    // The frames the server's video helper reads from the clip, at their sample times.
    const TestConfig::Clip clip;
    const std::array media{clip.input()};
    MtmdSource source(mtmd, mtmd_default_marker(), media, {}, llama_model_n_embd_inp(model), {}, embedding_framing);
    embedding::encode(ctx, source);
    return embedding::get(ctx, embedding::Normalize::L2);
  }
  const auto image = TestConfig::read_fixture(entry["image"].get<std::string>().c_str());
  REQUIRE_FALSE(image.empty());
  const std::string prompt = kind == "image"
      ? std::string(mtmd_default_marker())
      : entry["text_before"].get<std::string>() + mtmd_default_marker() + entry["text_after"].get<std::string>();
  MtmdSource source(mtmd, prompt, {image}, {}, llama_model_n_embd_inp(model), embedding_framing);
  embedding::encode(ctx, source);
  return embedding::get(ctx, embedding::Normalize::L2);
}

/** Each reference input embedded once per process on the full-size context:
 *  an image costs an encode, and on a CPU runner the largest takes minutes. */
const std::vector<float>& embedded(llama_context* ctx, mtmd_context* mtmd, const llama_model* model,
                                   const nlohmann::json& entry) {
  static std::map<std::string, std::vector<float>> cache;
  const std::string name = entry["name"];
  auto it = cache.find(name);
  if (it == cache.end()) it = cache.emplace(name, embed(ctx, mtmd, model, entry)).first;
  return it->second;
}

float dot(const std::vector<float>& a, const std::vector<float>& b) {
  float sum = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}

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

TEST_CASE("embedding multimodal: every fixture embeds as the reference server embeds it"
          * doctest::skip(embedding_multimodal_disabled())) {
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  auto* mtmd = TestConfig::acquire_mtmd(model.get());
  REQUIRE(mtmd != nullptr);
  auto* ctx = full_context(model.get());
  REQUIRE(ctx);

  const auto ref = reference();
  for (const auto& entry : ref["entries"]) {
    const auto expected = entry["vector"].get<std::vector<float>>();
    const auto& actual = embedded(ctx, mtmd, model.get(), entry);
    INFO(entry["name"].get<std::string>());
    REQUIRE(actual.size() == expected.size());
    CHECK(std::none_of(actual.begin(), actual.end(), [](float v) { return std::isnan(v); }));
    CHECK(std::sqrt(dot(actual, actual)) == doctest::Approx(1.0f).epsilon(0.001));
    CHECK(dot(actual, expected) >= 0.999f);
  }
}

TEST_CASE("embedding multimodal: each query finds its own image or clip"
          * doctest::skip(embedding_multimodal_disabled())) {
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  auto* mtmd = TestConfig::acquire_mtmd(model.get());
  REQUIRE(mtmd != nullptr);
  auto* ctx = full_context(model.get());
  REQUIRE(ctx);

  std::vector<std::pair<std::string, std::vector<float>>> media;
  std::vector<std::pair<std::string, std::vector<float>>> queries;  // with the input each must find
  const auto ref = reference();
  for (const auto& entry : ref["entries"]) {
    if (entry["kind"] == "image") media.emplace_back(entry["image"], embedded(ctx, mtmd, model.get(), entry));
    if (entry["kind"] == "video") media.emplace_back(entry["video"], embedded(ctx, mtmd, model.get(), entry));
    if (entry["kind"] == "text") queries.emplace_back(entry["finds"], embedded(ctx, mtmd, model.get(), entry));
  }
  REQUIRE(media.size() == 5);
  REQUIRE(queries.size() == 5);
  for (const auto& [finds, query] : queries) {
    const auto best = std::max_element(media.begin(), media.end(), [&](const auto& a, const auto& b) {
      return dot(query, a.second) < dot(query, b.second);
    });
    CHECK(best->first == finds);
  }
}

TEST_CASE("embedding multimodal: an input larger than the context's batch is refused untouched"
          * doctest::skip(embedding_multimodal_disabled())) {
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  auto* mtmd = TestConfig::acquire_mtmd(model.get());
  REQUIRE(mtmd != nullptr);
  // The infographic takes over a thousand entries; this batch holds 512.
  auto ctx = embedding_context(model.get(), 512);
  REQUIRE(ctx);

  const auto ref = reference();
  const auto& entries = ref["entries"];
  const auto infographic = std::find_if(entries.begin(), entries.end(), [](const auto& e) {
    return e["kind"] == "image" && e["image"].template get<std::string>().find("Global-X") != std::string::npos;
  });
  REQUIRE(infographic != entries.end());
  CHECK_THROWS_AS(embed(ctx.get(), mtmd, model.get(), *infographic), std::runtime_error);

  // Nothing ran, so the context still embeds as the reference does.
  const auto cat = std::find_if(entries.begin(), entries.end(), [](const auto& e) {
    return e["kind"] == "image" && e["image"] == "cat.jpg";
  });
  REQUIRE(cat != entries.end());
  CHECK(dot(embed(ctx.get(), mtmd, model.get(), *cat), (*cat)["vector"].get<std::vector<float>>()) >= 0.999f);
}
