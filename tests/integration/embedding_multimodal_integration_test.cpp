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
#include <optional>
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

/** Room for every committed recording. Admission limits are the audio suite's to test. */
constexpr AudioLimits ROOM_FOR_FIXTURES{64u << 20, 16000u * 600};

/** The parts an entry names, in order: a mixed entry's own, or its one image, clip or recording. */
nlohmann::json parts_of(const nlohmann::json& entry) {
  const std::string kind = entry["kind"];
  if (kind == "mixed") return entry["parts"];
  return nlohmann::json::array({{{kind, entry[kind]}}});
}

/**
 * Ordered parts as the native input takes them: text as written, each media
 * part behind a marker, its input in the same order. The bytes and the clip
 * live here, so the inputs' views stay valid while it does.
 */
struct Input {
  std::string prompt;
  std::vector<std::vector<uint8_t>> files;  // each input views its file's own buffer, which growing this list never moves
  std::optional<TestConfig::Clip> clip;
  std::vector<MediaInput> media;

  explicit Input(const nlohmann::json& parts) {
    files.reserve(parts.size());
    for (const auto& part : parts) {
      if (part.contains("text")) {
        prompt += part["text"].get<std::string>();
        continue;
      }
      prompt += mtmd_default_marker();
      media.push_back(media_of(part));
    }
  }
  Input(const Input&) = delete;
  Input& operator=(const Input&) = delete;

 private:
  MediaInput media_of(const nlohmann::json& part) {
    if (part.contains("video")) {
      // The server reads the clip with its video helper; these are the frames it samples, at their times.
      // One clip per input: its frames are views into this one, so a second would move them.
      REQUIRE(part["video"] == "video/red-square-moves.mp4");
      REQUIRE_FALSE(clip.has_value());
      return clip.emplace().input();
    }
    const bool audio = part.contains("audio");
    files.push_back(TestConfig::read_fixture(part[audio ? "audio" : "image"].get<std::string>().c_str()));
    REQUIRE_FALSE(files.back().empty());
    return MediaInput{audio ? MediaInput::Kind::Audio : MediaInput::Kind::Image, files.back()};
  }
};

/** An entry's input as a segment source, framed with the model's own BOS and EOS, or bare when `framed` is off. */
MtmdSource source_of(mtmd_context* mtmd, const llama_model* model, const Input& input, bool framed = true) {
  return MtmdSource(mtmd, input.prompt, input.media, {}, llama_model_n_embd_inp(model), ROOM_FOR_FIXTURES, framed);
}

/** An input as the reference describes it: text, or ordered parts at the entry's image budget. */
std::vector<float> embed(llama_context* ctx, const llama_model* model, const nlohmann::json& entry) {
  if (entry["kind"] == "text") {
    auto tokens = tokenizer::tokenize(llama_model_get_vocab(model), entry["text"].get<std::string>(), true, true);
    embedding::encode(ctx, tokens, static_cast<int32_t>(llama_n_batch(ctx)));
    return embedding::get(ctx, embedding::Normalize::L2);
  }
  auto* mtmd = TestConfig::acquire_mtmd(model, entry.value("image_tokens", 0));
  REQUIRE(mtmd != nullptr);
  const Input input(parts_of(entry));
  auto source = source_of(mtmd, model, input);
  embedding::encode(ctx, source);
  return embedding::get(ctx, embedding::Normalize::L2);
}

/** Each reference input embedded once per process on the full-size context:
 *  an image costs an encode, and on a CPU runner the largest takes minutes. */
const std::vector<float>& embedded(llama_context* ctx, const llama_model* model, const nlohmann::json& entry) {
  static std::map<std::string, std::vector<float>> cache;
  const std::string name = entry["name"];
  auto it = cache.find(name);
  if (it == cache.end()) it = cache.emplace(name, embed(ctx, model, entry)).first;
  return it->second;
}

/** The reference entry of this name. */
const nlohmann::json& entry_named(const nlohmann::json& ref, const std::string& name) {
  const auto& entries = ref["entries"];
  const auto it = std::find_if(entries.begin(), entries.end(), [&](const auto& e) { return e["name"] == name; });
  REQUIRE_MESSAGE(it != entries.end(), "no reference entry named " << name);
  return *it;
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
  auto* ctx = full_context(model.get());
  REQUIRE(ctx);

  const auto ref = reference();
  for (const auto& entry : ref["entries"]) {
    const auto expected = entry["vector"].get<std::vector<float>>();
    const auto& actual = embedded(ctx, model.get(), entry);
    INFO(entry["name"].get<std::string>());
    REQUIRE(actual.size() == expected.size());
    CHECK(std::none_of(actual.begin(), actual.end(), [](float v) { return std::isnan(v); }));
    CHECK(std::sqrt(dot(actual, actual)) == doctest::Approx(1.0f).epsilon(0.001));
    CHECK(dot(actual, expected) >= 0.999f);
  }
}

TEST_CASE("embedding multimodal: each query finds its own image, clip or recording, in any language"
          * doctest::skip(embedding_multimodal_disabled())) {
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  auto* ctx = full_context(model.get());
  REQUIRE(ctx);

  // Every lone image, clip and recording at the projector's own budget, by fixture.
  const std::array searchable{"image", "video", "audio"};
  std::map<std::string, std::vector<float>> media;
  std::vector<std::pair<std::string, std::vector<float>>> queries;  // with the input each must find
  const auto ref = reference();
  for (const auto& entry : ref["entries"]) {
    const std::string kind = entry["kind"];
    if (std::find(searchable.begin(), searchable.end(), kind) != searchable.end() && !entry.contains("image_tokens")) {
      media.emplace(entry[kind].get<std::string>(), embedded(ctx, model.get(), entry));
    }
    if (entry.contains("finds")) queries.emplace_back(entry["finds"], embedded(ctx, model.get(), entry));
  }
  REQUIRE_FALSE(queries.empty());
  for (const auto& [finds, query] : queries) {
    INFO(finds);
    REQUIRE(media.count(finds) == 1);
    const auto best = std::max_element(media.begin(), media.end(), [&](const auto& a, const auto& b) {
      return dot(query, a.second) < dot(query, b.second);
    });
    CHECK(best->first == finds);
  }
}

TEST_CASE("embedding multimodal: a mixed input costs its parts and the model's framing"
          * doctest::skip(embedding_multimodal_disabled())) {
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  auto* mtmd = TestConfig::acquire_mtmd(model.get());
  REQUIRE(mtmd != nullptr);

  const auto ref = reference();
  const auto& mixed = entry_named(ref, "text, image, video and audio");
  const Input whole(mixed["parts"]);
  size_t parts = 0;
  for (const auto& part : mixed["parts"]) {
    const Input alone(nlohmann::json::array({part}));
    parts += source_of(mtmd, model.get(), alone, false).cells();
  }
  // What the model adds around a whole input: its BOS and EOS, as its own file declares them.
  const size_t framing = tokenizer::tokenize(llama_model_get_vocab(model.get()), "", true, true).size();
  CHECK(source_of(mtmd, model.get(), whole).cells() == parts + framing);
}

TEST_CASE("embedding multimodal: the image budget sets what an image costs"
          * doctest::skip(embedding_multimodal_disabled())) {
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);

  // The budgets the reference pins, on its image; the projector's own lies between the smallest and largest.
  const auto ref = reference();
  std::vector<std::pair<int32_t, nlohmann::json>> budgeted;
  for (const auto& entry : ref["entries"]) {
    if (entry.contains("image_tokens")) budgeted.emplace_back(entry["image_tokens"].get<int32_t>(), entry);
  }
  REQUIRE(budgeted.size() >= 2);
  std::sort(budgeted.begin(), budgeted.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  const Input image(parts_of(budgeted.front().second));
  const auto cost = [&](int32_t image_tokens) {
    auto* mtmd = TestConfig::acquire_mtmd(model.get(), image_tokens);
    REQUIRE(mtmd != nullptr);
    return source_of(mtmd, model.get(), image).cells();
  };
  CHECK(cost(budgeted.front().first) < cost(0));
  CHECK(cost(0) < cost(budgeted.back().first));
}

TEST_CASE("embedding multimodal: an input larger than the context's batch is refused untouched"
          * doctest::skip(embedding_multimodal_disabled())) {
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  // The infographic takes over a thousand entries; this batch holds 512.
  auto ctx = embedding_context(model.get(), 512);
  REQUIRE(ctx);

  const auto ref = reference();
  const auto& entries = ref["entries"];
  const auto infographic = std::find_if(entries.begin(), entries.end(), [](const auto& e) {
    return e["kind"] == "image" && e["image"].template get<std::string>().find("Global-X") != std::string::npos;
  });
  REQUIRE(infographic != entries.end());
  CHECK_THROWS_AS(embed(ctx.get(), model.get(), *infographic), std::runtime_error);

  // Nothing ran, so the context still embeds as the reference does.
  const auto cat = std::find_if(entries.begin(), entries.end(), [](const auto& e) {
    return e["kind"] == "image" && e["image"] == "cat.jpg";
  });
  REQUIRE(cat != entries.end());
  CHECK(dot(embed(ctx.get(), model.get(), *cat), (*cat)["vector"].get<std::vector<float>>()) >= 0.999f);
}

TEST_CASE("embedding multimodal: an empty input is refused, never read as the input before it"
          * doctest::skip(embedding_multimodal_disabled())) {
  LlamaBackendGuard backend;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model != nullptr);
  auto* mtmd = TestConfig::acquire_mtmd(model.get());
  REQUIRE(mtmd != nullptr);
  auto ctx = embedding_context(model.get(), 512);
  REQUIRE(ctx);

  // A first input leaves its vector in the context.
  const auto tokens = tokenizer::tokenize(llama_model_get_vocab(model.get()), "a cat", true, true);
  embedding::encode(ctx.get(), tokens, 512);

  // No text, no media and no framing: a stream with nothing in it.
  const Input nothing(nlohmann::json::array({{{"text", ""}}}));
  auto empty = source_of(mtmd, model.get(), nothing, false);
  REQUIRE(empty.cells() == 0);
  CHECK_THROWS_AS(embedding::encode(ctx.get(), empty), std::runtime_error);
}
