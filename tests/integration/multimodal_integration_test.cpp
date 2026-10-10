/**
 * Multimodal Integration Test
 *
 * Exercises the embedding rail end to end against a real VL model:
 * mtmd codec → decode::SegmentSource → BranchStore::decode_segments →
 * decode::embd, then the continuous-tree-batching fan-out on top of it.
 *
 * The rest of this suite asserts `cells_used == <token count>` throughout,
 * because on the token rail cells and position are the same number. An image
 * is the first thing that separates them: an M-RoPE image occupies n_rows KV
 * cells but advances the position by only n_pos = max(nx, ny). These cases
 * cover that split and the slack bookkeeping that keeps the pressure gauge
 * exact across fork, release and retainOnly.
 *
 * What the fan-out case proves is the claim in lloyal-node's README: the image
 * lands in the KV as a shared prefix, so forking after it costs zero cells and
 * zero re-encode, and N children then answer N different questions about the
 * same image through one batched dispatch per token.
 *
 * Image cases require a matched VL decoder/projector. Audio cases require
 * LLOYAL_ASR_TEST=1 and a matched ASR pair in a separate runner process.
 */

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/branch.hpp>
#include <lloyal/chat_in.hpp>
#include <lloyal/mtmd.hpp>
#include <lloyal/tokenizer.hpp>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace lloyal;
using namespace lloyal::branch;

static const char* MODEL_PATH  = std::getenv("LLAMA_TEST_MODEL");
static const char* MMPROJ_PATH = std::getenv("LLAMA_MMPROJ_MODEL");

#define REQUIRE_VL()                                                           \
  if (!MODEL_PATH || !*MODEL_PATH || !MMPROJ_PATH || !*MMPROJ_PATH) {          \
    MESSAGE("[ SKIP ] LLAMA_TEST_MODEL / LLAMA_MMPROJ_MODEL not set");         \
    return;                                                                    \
  }

struct LlamaBackendGuard {
  LlamaBackendGuard() { llama_backend_init(); }
  ~LlamaBackendGuard() { llama_backend_free(); }
};

struct TestParams {
  float temperature = 0.0f;   // greedy — grounded assertions need determinism
  int32_t top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float typical_p = 1.0f;
  float penalty_repeat = 1.0f;
  float penalty_freq = 0.0f;
  float penalty_present = 0.0f;
  int32_t penalty_last_n = 64;
  uint32_t seed = 42;
};

// ============================================================================
// Fixtures
// ============================================================================

static std::vector<uint8_t> read_fixture(const char* name) {
  const std::string path = std::string(LLOYAL_TEST_FIXTURES_DIR) + "/" + name;
  std::vector<uint8_t> bytes;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return bytes;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n > 0) {
    bytes.resize(static_cast<size_t>(n));
    if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
  }
  std::fclose(f);
  return bytes;
}

/**
 * Shared mtmd context — the projector is ~700 MB and init runs a warmup
 * encode, so it is loaded once for the suite. Mirrors acquire_test_model();
 * constructed after the model, so it is destroyed before it (mtmd holds a
 * borrowed llama_model*).
 */
struct MtmdDeleter {
  void operator()(mtmd_context* c) const { if (c) mtmd_free(c); }
};

static mtmd_context* acquire_mtmd(const llama_model* model) {
  static std::unique_ptr<mtmd_context, MtmdDeleter> cached;
  if (!cached) {
    auto p = mtmd_context_params_default();
    p.use_gpu        = TestConfig::n_gpu_layers() != 0;
    p.print_timings  = false;
    p.n_threads      = 4;
    p.warmup         = false;  // each media case executes the encoder
    cached.reset(mtmd_init_from_file(MMPROJ_PATH, model, p));
  }
  return cached.get();
}

/// The user turn carrying the image, as a media_marker content part.
static std::string image_prompt(const llama_model* model, const std::string& ask) {
  chat_in::FormatInputs in;
  in.messages_json =
      std::string(R"([{"role":"system","content":"You are a vision assistant. Answer briefly."},)"
                  R"({"role":"user","content":[{"type":"text","text":")") +
      ask +
      std::string(R"("},{"type":"media_marker","text":")") + mtmd_default_marker() +
      std::string(R"("}]}])");
  in.enable_thinking = false;
  return chat_in::format(model, in).prompt;
}

/// A follow-up user turn, tokenized for the token rail (the child suffix).
static std::vector<llama_token> question_tokens(const llama_model* model,
                                                const std::string& question) {
  chat_in::FormatInputs in;
  in.messages_json = std::string(R"([{"role":"system","content":""},)"
                                 R"({"role":"user","content":")") +
                     question + R"("}])";
  in.enable_thinking = false;
  const std::string prompt = chat_in::format(model, in).prompt;

  const auto* vocab = llama_model_get_vocab(model);
  auto sep  = chat_in::get_turn_separator(model);
  auto toks = tokenizer::tokenize(vocab, prompt, /*add_special*/ false,
                                  /*parse_special*/ true);
  sep.insert(sep.end(), toks.begin(), toks.end());
  return sep;
}

/// Greedy-decode `budget` tokens on one branch, returning the text.
static std::string generate(BranchHandle h, BranchStore& store,
                            const llama_vocab* vocab, int budget) {
  std::string out;
  for (int i = 0; i < budget; ++i) {
    const llama_token t = sample(h, store);
    if (t < 0 || tokenizer::is_eog(vocab, t)) break;
    accept_token(h, t, store);
    out += tokenizer::detokenize(vocab, t, false);
    DecodeEachItem item{h, t};
    store.decode_each(std::span<const DecodeEachItem>(&item, 1));
  }
  return out;
}

/** Grounded content assertions require a capable VL model and are enabled by default. */
static bool vl_strict() {
  const char* e = std::getenv("LLOYAL_VL_STRICT");
  return !(e && std::string(e) == "0");
}

static bool contains_any(const std::string& haystack,
                         const std::vector<std::string>& needles) {
  std::string lower;
  lower.reserve(haystack.size());
  // unsigned char: std::tolower is UB for negative values, and model
  // output is UTF-8, so continuation bytes (>= 0x80) reach here.
  for (unsigned char c : haystack) {
    lower += static_cast<char>(std::tolower(c));
  }
  for (const auto& n : needles) {
    if (lower.find(n) != std::string::npos) return true;
  }
  return false;
}

// ============================================================================
// Mechanics — the cells/position split the token rail never exercises
// ============================================================================

TEST_CASE("multimodal: image prefill decouples position from cells") {
  REQUIRE_VL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  mtmd_context* mtmd = acquire_mtmd(model.get());
  REQUIRE_MESSAGE(mtmd, "mmproj failed to load — is it matched to the model?");

  auto image = read_fixture("cat.jpg");
  REQUIRE_MESSAGE(!image.empty(), "fixtures/cat.jpg missing");

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx     = 4096;
  cparams.n_batch   = 512;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle root = create(ctx, model.get(), store, 0, params, 512);
  REQUIRE(root != INVALID_HANDLE);

  const uint32_t cells_before = store.kv_pressure().cells_used;

  const std::string prompt = image_prompt(model.get(), "Describe this image.");
  CHECK_MESSAGE(prompt.find(mtmd_default_marker()) != std::string::npos,
                "marker survives the template verbatim");

  std::vector<std::vector<uint8_t>> images{image};
  MtmdSource source(mtmd, prompt, images, std::span<const llama_token>(),
                    llama_model_n_embd_inp(model.get()));

  // Cost is known BEFORE anything decodes — counted at construction, after
  // mtmd_tokenize and before any clip encode. This is what lets a caller
  // admit or refuse media against a context budget the way it already can
  // for text, whose length it can measure by tokenizing.
  const size_t predicted = source.cells();
  CHECK(predicted > 0);

  const auto r = store.decode_segments(root, source);

  // The prediction must be exact, or an admission gate built on it is a lie.
  CHECK_MESSAGE(static_cast<int64_t>(predicted) == r.cells,
                "pre-decode cell estimate must match what was decoded");

  CHECK(r.cells > 0);
  CHECK(r.advance > 0);

  // The whole point, and it is a per-model property: an M-RoPE image costs
  // more cells than it advances position (n_pos = max(nx, ny), cells = nx*ny),
  // while a plain-position model advances one per row. Probe, don't assume.
  if (mtmd_decode_use_mrope(mtmd)) {
    CHECK_MESSAGE(r.advance < r.cells,
                  "M-RoPE image: position advance is below the cell count");
  } else {
    CHECK_MESSAGE(r.advance == r.cells,
                  "plain positions: one position per embedding row");
  }

  BranchState* rs = store.get(root);
  REQUIRE(rs);
  CHECK(rs->position == r.advance);
  CHECK(store.kv_pressure().cells_used == cells_before + r.cells);

  // Decoding ended on an image-terminal prefill only if the template put no
  // text after the marker; either way logits must exist to produce from.
  const llama_token first = sample(root, store);
  CHECK(first >= 0);

  // The slack fix: release must recover the cells, not the position delta.
  // Without img_slack_own this leaves (cells - advance) permanently stranded.
  prune(root, store);
  CHECK_MESSAGE(store.kv_pressure().cells_used == 0,
                "release recovers embedding-row slack, not just position delta");

  store.drain();
  llama_free(ctx);
}

TEST_CASE("multimodal: retainOnly promotes inherited image slack") {
  REQUIRE_VL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  mtmd_context* mtmd = acquire_mtmd(model.get());
  REQUIRE(mtmd);

  auto image = read_fixture("cat.jpg");
  REQUIRE(!image.empty());

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx     = 4096;
  cparams.n_batch   = 512;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle root = create(ctx, model.get(), store, 0, params, 512);
  REQUIRE(root != INVALID_HANDLE);

  const std::string prompt = image_prompt(model.get(), "Describe this image.");
  std::vector<std::vector<uint8_t>> images{image};
  MtmdSource source(mtmd, prompt, images, std::span<const llama_token>(),
                    llama_model_n_embd_inp(model.get()));
  store.decode_segments(root, source);

  // Fork inherits the image slack as total but owns none of it.
  BranchHandle winner = fork(root, store);
  REQUIRE(winner != INVALID_HANDLE);

  const auto* vocab = llama_model_get_vocab(model.get());
  auto suffix = question_tokens(model.get(), "What animal is this?");
  DecodeScatterItem item{winner, std::span<const llama_token>(suffix)};
  store.decode_scatter(std::span<const DecodeScatterItem>(&item, 1));
  generate(winner, store, vocab, 4);

  // retainOnly promotes the winner to root: fork_head → 0, and the inherited
  // slack must become its OWN, or the final release under-subtracts by
  // exactly the image's slack and the gauge never returns to zero.
  store.retainOnly(winner);
  CHECK(get_fork_head(winner, store) == 0);

  prune(winner, store);
  CHECK_MESSAGE(store.kv_pressure().cells_used == 0,
                "retainOnly promotes inherited slack to the winner's own");

  store.drain();
  llama_free(ctx);
}

// ============================================================================
// The moat — one encode, N branches, batched fan-out
// ============================================================================

TEST_CASE("multimodal: one encode, N branches fan out over the shared image") {
  REQUIRE_VL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  mtmd_context* mtmd = acquire_mtmd(model.get());
  REQUIRE(mtmd);

  auto image = read_fixture("cat.jpg");
  REQUIRE(!image.empty());

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx     = 8192;
  cparams.n_batch   = 512;
  cparams.n_seq_max = 8;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(16);
  store.init_tenancy(ctx);
  TestParams params;

  const auto* vocab = llama_model_get_vocab(model.get());

  // --- Encode the image ONCE, onto the spine ---
  BranchHandle spine = create(ctx, model.get(), store, 0, params, 512);
  REQUIRE(spine != INVALID_HANDLE);

  const std::string prompt = image_prompt(model.get(), "Study this image.");
  std::vector<std::vector<uint8_t>> images{image};
  MtmdSource source(mtmd, prompt, images, std::span<const llama_token>(),
                    llama_model_n_embd_inp(model.get()));
  const auto spine_r = store.decode_segments(spine, source);
  REQUIRE(spine_r.cells > 0);

  const uint32_t cells_after_image = store.kv_pressure().cells_used;

  // --- Fan out: four agents, four questions about the same cat ---
  struct Ask {
    const char* question;
    std::vector<std::string> expect;  // any-of, lowercased substrings
  };
  const std::vector<Ask> asks = {
    {"What animal is in this picture? Answer in one word.",
     {"cat", "kitten", "feline"}},
    {"What color is the animal's fur? Answer in one word.",
     {"white", "cream", "grey", "gray", "light", "pale", "silver"}},
    {"Is there snow on the ground in this picture? Answer yes or no.",
     {"yes", "snow"}},
    {"What is the animal sitting on? Answer in one word.",
     {"fence", "rail", "wood", "post", "ledge", "beam", "deck"}},
  };

  std::vector<BranchHandle> children;
  for (size_t i = 0; i < asks.size(); ++i) {
    BranchHandle c = fork(spine, store);
    REQUIRE(c != INVALID_HANDLE);
    children.push_back(c);
  }

  // The README's claim, asserted: forking after the image copies no cells.
  CHECK_MESSAGE(store.kv_pressure().cells_used == cells_after_image,
                "fork x4 after the image adds zero cells — KV shared, not copied");

  // Each child's own question, all bin-packed into ONE scatter dispatch.
  std::vector<std::vector<llama_token>> suffixes;
  suffixes.reserve(asks.size());
  for (const auto& a : asks) {
    suffixes.push_back(question_tokens(model.get(), a.question));
  }
  std::vector<DecodeScatterItem> items;
  items.reserve(children.size());
  size_t suffix_total = 0;
  for (size_t i = 0; i < children.size(); ++i) {
    items.push_back({children[i], std::span<const llama_token>(suffixes[i])});
    suffix_total += suffixes[i].size();
  }
  store.decode_scatter(std::span<const DecodeScatterItem>(items));

  CHECK_MESSAGE(store.kv_pressure().cells_used ==
                    cells_after_image + static_cast<uint32_t>(suffix_total),
                "the fan-out costs only the text suffixes, never the image again");

  // --- Continuous tree batching: N branches, one dispatch per token ---
  std::vector<std::string> answers(children.size());
  std::vector<bool> done(children.size(), false);

  for (int step = 0; step < 24; ++step) {
    std::vector<DecodeEachItem> batch;
    batch.reserve(children.size());

    for (size_t i = 0; i < children.size(); ++i) {
      if (done[i]) continue;
      const llama_token t = sample(children[i], store);
      if (t < 0 || tokenizer::is_eog(vocab, t)) {
        done[i] = true;
        continue;
      }
      accept_token(children[i], t, store);
      answers[i] += tokenizer::detokenize(vocab, t, false);
      batch.push_back({children[i], t});
    }
    if (batch.empty()) break;

    // One llama_decode for every live branch this tick — the moat.
    store.decode_each(std::span<const DecodeEachItem>(batch));
  }

  for (size_t i = 0; i < children.size(); ++i) {
    INFO("Q: " << std::string(asks[i].question) << "  |  A: " << answers[i]);
    CHECK_MESSAGE(!answers[i].empty(), "child produced an answer");
    if (vl_strict()) {
      CHECK_MESSAGE(contains_any(answers[i], asks[i].expect),
                    "answer is grounded in the shared image");
    }
  }

  // --- Releasing a child recovers its suffix only, never the shared image ---
  const uint32_t before_release = store.kv_pressure().cells_used;
  BranchState* cs = store.get(children[0]);
  REQUIRE(cs);
  const uint32_t own = static_cast<uint32_t>(cs->position - cs->fork_head);
  prune(children[0], store);
  CHECK_MESSAGE(before_release - store.kv_pressure().cells_used == own,
                "a fork releases its own cells, leaving the image prefix intact");

  for (size_t i = 1; i < children.size(); ++i) prune(children[i], store);

  // The image's cells belong to the spine and survive every child's release.
  CHECK(store.kv_pressure().cells_used == cells_after_image);

  prune(spine, store);
  CHECK_MESSAGE(store.kv_pressure().cells_used == 0,
                "releasing the spine recovers the image, slack included");

  store.drain();
  llama_free(ctx);
}

// ============================================================================
// Error paths — no inference, all cheap
// ============================================================================

TEST_CASE("multimodal: MtmdSource rejects bad input") {
  REQUIRE_VL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  mtmd_context* mtmd = acquire_mtmd(model.get());
  REQUIRE(mtmd);

  auto image = read_fixture("cat.jpg");
  REQUIRE(!image.empty());

  const int32_t n_embd_inp = llama_model_n_embd_inp(model.get());
  const std::string marker = mtmd_default_marker();
  const std::string one_marker = "look: " + marker;

  // --- NULL context ---
  {
    std::vector<std::vector<uint8_t>> images{image};
    CHECK_THROWS_WITH(
        MtmdSource(nullptr, one_marker, images,
                   std::span<const llama_token>(), n_embd_inp),
        "MtmdSource - NULL mtmd context");
  }

  // --- Undecodable bytes (stb_image handles jpg/png/bmp/gif; not this) ---
  {
    std::vector<std::vector<uint8_t>> images{
        std::vector<uint8_t>{0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07}};
    CHECK_THROWS_AS(
        MtmdSource(mtmd, one_marker, images,
                   std::span<const llama_token>(), n_embd_inp),
        std::runtime_error);
  }

  // --- Count mismatch is diagnosed as such, not as a preprocessing failure.
  // mtmd_tokenize's header documents rc==1 here, but the check throws from
  // mtmd_tokenizer's constructor and its catch-all reports rc==2 — so
  // MtmdSource counts markers itself. Both directions:
  {
    std::vector<std::vector<uint8_t>> images{image, image};
    CHECK_THROWS_WITH(
        MtmdSource(mtmd, one_marker, images,
                   std::span<const llama_token>(), n_embd_inp),
        "MtmdSource - media marker count (1) does not match media count (2)");
  }
  {
    const std::string two_markers = "a " + marker + " b " + marker;
    std::vector<std::vector<uint8_t>> images{image};
    CHECK_THROWS_WITH(
        MtmdSource(mtmd, two_markers, images,
                   std::span<const llama_token>(), n_embd_inp),
        "MtmdSource - media marker count (2) does not match media count (1)");
  }
  {
    std::vector<std::vector<uint8_t>> none;
    CHECK_THROWS_WITH(
        MtmdSource(mtmd, one_marker, none,
                   std::span<const llama_token>(), n_embd_inp),
        "MtmdSource - media marker count (1) does not match media count (0)");
  }

  // --- Audio must be rejected by the CONSTRUCTOR, before decode_segments can
  // commit any preceding chunk. mtmd sniffs audio by magic bytes, so a WAV
  // header is enough to route it. ---
  {
    auto u32 = [](std::vector<uint8_t>& v, uint32_t x) {
      v.push_back(x & 0xff); v.push_back((x >> 8) & 0xff);
      v.push_back((x >> 16) & 0xff); v.push_back((x >> 24) & 0xff);
    };
    auto u16 = [](std::vector<uint8_t>& v, uint16_t x) {
      v.push_back(x & 0xff); v.push_back((x >> 8) & 0xff);
    };
    std::vector<uint8_t> wav;
    const uint32_t n_samples = 160, data_bytes = n_samples * 2;
    for (char c : std::string("RIFF")) wav.push_back(static_cast<uint8_t>(c));
    u32(wav, 36 + data_bytes);
    for (char c : std::string("WAVEfmt ")) wav.push_back(static_cast<uint8_t>(c));
    u32(wav, 16); u16(wav, 1); u16(wav, 1);
    u32(wav, 16000); u32(wav, 32000); u16(wav, 2); u16(wav, 16);
    for (char c : std::string("data")) wav.push_back(static_cast<uint8_t>(c));
    u32(wav, data_bytes);
    wav.insert(wav.end(), data_bytes, 0);

    // Image calls cannot bypass explicit audio admission and its limits.
    std::vector<std::vector<uint8_t>> audio{wav};
    CHECK_THROWS_WITH(
        MtmdSource(mtmd, one_marker, audio,
                   std::span<const llama_token>(), n_embd_inp),
        "MtmdSource - audio bytes require a typed audio input");
    if (!mtmd_support_audio(mtmd)) {
      const std::array typed{MediaInput{MediaInput::Kind::Audio, wav}};
      CHECK_THROWS_WITH(MtmdSource(mtmd, one_marker, typed, {}, n_embd_inp,
                                   AudioLimits{wav.size(), n_samples}),
                        "MtmdSource - projector does not support audio input");
    }
  }

  // --- A bare marker: the shape most likely to yield an empty TEXT chunk,
  // since there is no surrounding text. decode_segments REJECTS empty
  // segments (terminality is positional), so mtmd must not produce one. ---
  {
    std::vector<std::vector<uint8_t>> images{image};
    MtmdSource bare(mtmd, marker, images, std::span<const llama_token>(),
                    n_embd_inp);
    REQUIRE(bare.size() > 0);
    for (size_t i = 0; i < bare.size(); ++i) {
      auto seg = bare.at(i);
      if (seg.kind == decode::Segment::Kind::Text) {
        CHECK_MESSAGE(!seg.tokens.empty(),
                      "mtmd must not emit an empty TEXT segment");
      }
    }
  }

  // --- A leading sep run becomes segment 0, ahead of the chunk walk ---
  {
    const auto sep = chat_in::get_turn_separator(model.get());
    REQUIRE(!sep.empty());
    std::vector<std::vector<uint8_t>> images{image};
    MtmdSource with_sep(mtmd, one_marker, images,
                        std::span<const llama_token>(sep), n_embd_inp);
    MtmdSource without_sep(mtmd, one_marker, images,
                           std::span<const llama_token>(), n_embd_inp);
    CHECK(with_sep.size() == without_sep.size() + 1);

    auto seg = with_sep.at(0);
    CHECK(seg.kind == decode::Segment::Kind::Text);
    CHECK(seg.tokens.size() == sep.size());
  }
}

// ============================================================================
// cells() is a SegmentSource contract, not an MtmdSource extra. An admission
// gate holds the base reference — it knows it has segments to place, not that
// a vision projector produced them — so the quote has to be reachable and
// exact through that view.
// ============================================================================

TEST_CASE("multimodal: SegmentSource prices a prefill before it decodes") {
  REQUIRE_VL();
  LlamaBackendGuard guard;

  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  mtmd_context* mtmd = acquire_mtmd(model.get());
  REQUIRE_MESSAGE(mtmd, "mmproj failed to load — is it matched to the model?");

  auto image = read_fixture("cat.jpg");
  REQUIRE_MESSAGE(!image.empty(), "fixtures/cat.jpg missing");

  const int32_t n_embd_inp = llama_model_n_embd_inp(model.get());
  const std::string prompt = image_prompt(model.get(), "Describe this image.");
  const auto sep = chat_in::get_turn_separator(model.get());
  REQUIRE(!sep.empty());

  // --- The quote is the contract's arithmetic: the sum over the segments the
  // source will yield, sep run included. Walked on its OWN instance, because
  // SegmentSource forbids revisiting — a source that has been walked can no
  // longer be handed to decode_segments. ---
  {
    std::vector<std::vector<uint8_t>> images{image};
    MtmdSource walked(mtmd, prompt, images,
                      std::span<const llama_token>(sep), n_embd_inp);
    decode::SegmentSource& src = walked;

    const size_t quoted = src.cells();
    size_t summed = 0;
    for (size_t i = 0; i < src.size(); ++i) {
      const auto seg = src.at(i);
      summed += seg.kind == decode::Segment::Kind::Text
                    ? seg.tokens.size()
                    : static_cast<size_t>(seg.n_rows);
    }
    CHECK_MESSAGE(quoted == summed,
                  "cells() must equal the sum of the segments at() yields");
    CHECK_MESSAGE(quoted > sep.size(),
                  "the quote covers the image rows, not just the sep run");
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx     = 4096;
  cparams.n_batch   = 512;
  cparams.n_seq_max = 4;
  llama_context* ctx = llama_init_from_model(model.get(), cparams);
  REQUIRE(ctx);

  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  BranchHandle h = create(ctx, model.get(), store, 0, params, 512);
  REQUIRE(h != INVALID_HANDLE);
  const uint32_t cells_before = store.kv_pressure().cells_used;

  std::vector<std::vector<uint8_t>> images{image};
  MtmdSource source(mtmd, prompt, images,
                    std::span<const llama_token>(sep), n_embd_inp);
  decode::SegmentSource& gate = source;

  // --- Refusing costs nothing. A caller with less headroom than the quote
  // declines HERE, and the branch is still exactly as it was found: the
  // expensive part (bitmap decode + mtmd_tokenize) happened in the source's
  // constructor, which touches no KV. That is the property the quote buys —
  // decode_segments is not atomic, so discovering the overflow midway would
  // poison the branch instead of merely refusing it. ---
  const size_t quote = gate.cells();
  CHECK(quote > 0);

  BranchState* st = store.get(h);
  REQUIRE(st);
  CHECK_MESSAGE(st->position == 0, "quoting must not touch the branch");
  CHECK_MESSAGE(store.kv_pressure().cells_used == cells_before,
                "quoting must not spend cells");

  // --- Admitted, the charge equals the quote exactly. An admission gate built
  // on an approximation would mis-commit the context on every image. ---
  const auto r = store.decode_segments(h, source);
  CHECK_MESSAGE(static_cast<int64_t>(quote) == r.cells,
                "charge must equal the quote taken through SegmentSource");
  CHECK(store.kv_pressure().cells_used == cells_before + r.cells);

  prune(h, store);
  store.drain();
  llama_free(ctx);
}

// Audio uses the same real-model runner and helpers as images. Select it in a
// separate process so TestConfig's cached model/projector remain a matched pair.
static bool audio_tests_disabled() {
  const char* flag = std::getenv("LLOYAL_ASR_TEST");
  return !flag || std::string(flag) != "1";
}

static std::string audio_prompt(const llama_model* model) {
  chat_in::FormatInputs in;
  in.messages_json = R"([{"role":"user","content":[{"type":"media_marker","text":"<__media__>"}]}])";
  in.enable_thinking = false;
  return chat_in::format(model, in).prompt;
}

static MtmdSource audio_source(mtmd_context* ctx, const llama_model* model,
                               std::span<const uint8_t> bytes,
                               AudioLimits limits = {2 * 1024 * 1024, 320000}) {
  const std::array media{MediaInput{MediaInput::Kind::Audio, bytes}};
  return MtmdSource(ctx, audio_prompt(model), media, {},
                    llama_model_n_embd_inp(model), limits);
}

struct AudioRig {
  std::shared_ptr<llama_model> model = TestConfig::acquire_test_model();
  std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
  BranchStore store{16};
  TestParams params;

  explicit AudioRig(uint32_t cells = 4096) {
    REQUIRE(model);
    auto p = llama_context_default_params();
    p.n_ctx = cells;
    p.n_batch = 128;
    p.n_ubatch = 128;
    p.n_seq_max = 8;
    p.kv_unified = true;
    ctx.reset(llama_init_from_model(model.get(), p));
    REQUIRE(ctx);
    store.init_tenancy(ctx.get());
  }

  ~AudioRig() { store.drain(); }

  BranchHandle root() {
    return create(ctx.get(), model.get(), store, 0, params, 128);
  }
};

// Delegates the actual encode and position geometry; counts only the work
// requested through this source, so forks cannot conceal a second projection.
struct CountedSource : decode::SegmentSource {
  decode::SegmentSource& source;
  size_t encodes = 0;
  explicit CountedSource(decode::SegmentSource& source) : source(source) {}
  size_t size() override { return source.size(); }
  size_t cells() const override { return source.cells(); }
  decode::Segment at(size_t i) override {
    auto segment = source.at(i);
    encodes += segment.kind == decode::Segment::Kind::Embd ? 1 : 0;
    return segment;
  }
  void positions(size_t i, llama_pos base, llama_pos* out) override {
    source.positions(i, base, out);
  }
};

static void check_audio_transcript(const std::string& text) {
  INFO("ASR: " << text);
  CHECK(contains_any(text, {"one, two, three, four, five"}));
  CHECK(contains_any(text, {"the meeting is on thursday"}));
  CHECK(contains_any(text, {"nine thirty", "9:30"}));
}

TEST_CASE("audio: Qwen ASR prompt and tokenization match upstream"
          * doctest::skip(audio_tests_disabled())) {
  REQUIRE(MODEL_PATH);
  REQUIRE(MMPROJ_PATH);
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  auto* mtmd = acquire_mtmd(model.get());
  REQUIRE(mtmd);
  REQUIRE(mtmd_support_audio(mtmd));
  CHECK(mtmd_get_audio_sample_rate(mtmd) == 16000);

  auto templates = common_chat_templates_init(model.get(), "");
  common_chat_msg user;
  user.role = "user";
  user.content = mtmd_default_marker();
  const auto upstream = common_chat_format_single(templates.get(), {}, user,
                                                 true, true);
  const auto actual = audio_prompt(model.get());
  INFO("upstream: " << upstream);
  INFO("liblloyal: " << actual);
  REQUIRE(actual == upstream);

  // Qwen-ASR's template adds no system prefix, so explicit suppression
  // produces the same prompt as omitting the system message.
  chat_in::FormatInputs suppressed;
  suppressed.messages_json = R"([{"role":"system","content":""},{"role":"user","content":"<__media__>"}])";
  suppressed.enable_thinking = false;
  CHECK(chat_in::format(model.get(), suppressed).prompt == upstream);

  const auto bytes = read_fixture("asr-counting.wav");
  REQUIRE(!bytes.empty());
  auto decoded = mtmd_helper_bitmap_init_from_buf(mtmd, bytes.data(), bytes.size(), false);
  REQUIRE(decoded.bitmap);
  REQUIRE(decoded.video_ctx == nullptr);
  ::mtmd::bitmap_ptr bitmap(decoded.bitmap);
  REQUIRE(mtmd_bitmap_is_audio(bitmap.get()));
  const mtmd_bitmap* ptr = bitmap.get();
  ::mtmd::input_chunks_ptr expected(mtmd_input_chunks_init());
  ::mtmd::input_chunks_ptr observed(mtmd_input_chunks_init());
  mtmd_input_text reference{upstream.c_str(), true, true};
  mtmd_input_text input{actual.c_str(), false, true};
  REQUIRE(mtmd_tokenize(mtmd, expected.get(), &reference, &ptr, 1) == 0);
  REQUIRE(mtmd_tokenize(mtmd, observed.get(), &input, &ptr, 1) == 0);
  REQUIRE(mtmd_input_chunks_size(observed.get()) == mtmd_input_chunks_size(expected.get()));
  for (size_t i = 0; i < mtmd_input_chunks_size(expected.get()); ++i) {
    const auto* a = mtmd_input_chunks_get(expected.get(), i);
    const auto* b = mtmd_input_chunks_get(observed.get(), i);
    REQUIRE(mtmd_input_chunk_get_type(a) == mtmd_input_chunk_get_type(b));
    CHECK(mtmd_input_chunk_get_n_tokens(a) == mtmd_input_chunk_get_n_tokens(b));
    CHECK(mtmd_input_chunk_get_n_pos(a) == mtmd_input_chunk_get_n_pos(b));
    if (mtmd_input_chunk_get_type(a) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t na = 0, nb = 0;
      const auto* ta = mtmd_input_chunk_get_tokens_text(a, &na);
      const auto* tb = mtmd_input_chunk_get_tokens_text(b, &nb);
      REQUIRE(na == nb);
      CHECK(std::equal(ta, ta + na, tb));
    }
  }
}

TEST_CASE("audio: projected parent is inherited by batched children"
          * doctest::skip(audio_tests_disabled())) {
  REQUIRE(MODEL_PATH);
  REQUIRE(MMPROJ_PATH);
  LlamaBackendGuard guard;
  AudioRig rig;
  auto* mtmd = acquire_mtmd(rig.model.get());
  REQUIRE(mtmd);
  REQUIRE(mtmd_support_audio(mtmd));
  const auto bytes = read_fixture("asr-counting.wav");
  REQUIRE(!bytes.empty());
  const auto parent = rig.root();
  REQUIRE(parent != INVALID_HANDLE);

  auto source = audio_source(mtmd, rig.model.get(), bytes);
  CountedSource counted(source);
  const auto projected = rig.store.decode_segments(parent, counted);
  REQUIRE(projected.cells == static_cast<int64_t>(source.cells()));
  REQUIRE(projected.advance == projected.cells);
  REQUIRE(counted.encodes == 1);
  const auto parent_cells = rig.store.kv_pressure().cells_used;
  REQUIRE(parent_cells == projected.cells);

  std::vector<BranchHandle> children;
  for (size_t i = 0; i < 4; ++i) {
    const auto child = fork(parent, rig.store);
    REQUIRE(child != INVALID_HANDLE);
    REQUIRE(rig.store.get(child)->position == projected.advance);
    children.push_back(child);
  }
  CHECK(rig.store.kv_pressure().cells_used == parent_cells);
  CHECK(counted.encodes == 1);

  const auto* vocab = llama_model_get_vocab(rig.model.get());
  std::vector<std::string> answers(children.size());
  std::vector<bool> done(children.size(), false);
  size_t generated = 0;
  for (int step = 0; step < 96; ++step) {
    std::vector<DecodeEachItem> batch;
    for (size_t i = 0; i < children.size(); ++i) {
      if (done[i]) continue;
      const auto token = sample(children[i], rig.store);
      REQUIRE(token >= 0);
      done[i] = tokenizer::is_eog(vocab, token);
      if (done[i]) continue;
      accept_token(children[i], token, rig.store);
      answers[i] += tokenizer::detokenize(vocab, token, true);
      batch.push_back({children[i], token});
    }
    if (batch.empty()) break;
    rig.store.decode_each(batch);
    generated += batch.size();
    CHECK(rig.store.kv_pressure().cells_used == parent_cells + generated);
  }

  for (size_t i = 0; i < children.size(); ++i) {
    REQUIRE(done[i]);
    check_audio_transcript(answers[i]);
    CHECK(answers[i].find("language English<asr_text>") == 0);
  }
  CHECK(counted.encodes == 1);
  const auto* first = rig.store.get(children.front());
  const auto own = first->position - first->fork_head;
  const auto before = rig.store.kv_pressure().cells_used;
  prune(children.front(), rig.store);
  CHECK(rig.store.kv_pressure().cells_used == before - own);
  CHECK(rig.store.get(parent)->position == projected.advance);
  for (size_t i = 1; i < children.size(); ++i) {
    CHECK(rig.store.get(children[i])->position > projected.advance);
    prune(children[i], rig.store);
  }
  CHECK(rig.store.kv_pressure().cells_used == parent_cells);
  prune(parent, rig.store);
  CHECK(rig.store.kv_pressure().cells_used == 0);
}

TEST_CASE("audio: multiple encoder chunks survive winner retention"
          * doctest::skip(audio_tests_disabled())) {
  REQUIRE(MODEL_PATH);
  REQUIRE(MMPROJ_PATH);
  LlamaBackendGuard guard;
  AudioRig rig;
  auto* mtmd = acquire_mtmd(rig.model.get());
  REQUIRE(mtmd);
  REQUIRE(mtmd_support_audio(mtmd));
  const auto bytes = read_fixture("asr-repeated.wav");
  REQUIRE(!bytes.empty());
  const auto parent = rig.root();
  REQUIRE(parent != INVALID_HANDLE);
  auto source = audio_source(mtmd, rig.model.get(), bytes);
  CountedSource counted(source);
  const auto projected = rig.store.decode_segments(parent, counted);
  CHECK(projected.cells == static_cast<int64_t>(source.cells()));
  CHECK(projected.advance == projected.cells);
  REQUIRE(counted.encodes >= 2);
  const auto parent_cells = rig.store.kv_pressure().cells_used;
  const auto winner = fork(parent, rig.store);
  REQUIRE(winner != INVALID_HANDLE);
  CHECK(rig.store.kv_pressure().cells_used == parent_cells);
  rig.store.retainOnly(winner);
  CHECK(get_fork_head(winner, rig.store) == 0);
  CHECK(rig.store.kv_pressure().cells_used == parent_cells);
  const auto transcript = generate(winner, rig.store,
      llama_model_get_vocab(rig.model.get()), 128);
  check_audio_transcript(transcript);
  const auto first = transcript.find("Thursday");
  REQUIRE(first != std::string::npos);
  CHECK(transcript.find("Thursday", first + 1) != std::string::npos);
  prune(winner, rig.store);
  CHECK(rig.store.kv_pressure().cells_used == 0);
}

TEST_CASE("audio: partial embedding failure reclaims rows and preserves a sibling"
          * doctest::skip(audio_tests_disabled())) {
  REQUIRE(MODEL_PATH);
  REQUIRE(MMPROJ_PATH);
  LlamaBackendGuard guard;
  AudioRig rig(256);
  auto* mtmd = acquire_mtmd(rig.model.get());
  REQUIRE(mtmd);
  const auto short_audio = read_fixture("asr-counting.wav");
  const auto long_audio = read_fixture("asr-repeated.wav");
  REQUIRE(!short_audio.empty());
  REQUIRE(!long_audio.empty());
  const auto parent = rig.root();
  REQUIRE(parent != INVALID_HANDLE);
  auto prefix = audio_source(mtmd, rig.model.get(), short_audio);
  rig.store.decode_segments(parent, prefix);
  const auto parent_cells = rig.store.kv_pressure().cells_used;
  const auto parent_position = get_position(parent, rig.store);
  const auto sibling = fork(parent, rig.store);
  const auto failing = fork(parent, rig.store);
  REQUIRE(sibling != INVALID_HANDLE);
  REQUIRE(failing != INVALID_HANDLE);
  rig.store.get(failing)->n_batch = 16;
  const auto seq = rig.store.get(failing)->seq_id;

  // Leave room for the leading text and one 16-row audio decode only.
  const auto capacity = llama_n_ctx(rig.ctx.get());
  REQUIRE(capacity > parent_cells + 32);
  const auto* vocab = llama_model_get_vocab(rig.model.get());
  const auto dot = tokenizer::tokenize(vocab, ".", false, false);
  REQUIRE(!dot.empty());
  const std::vector<llama_token> padding(capacity - parent_cells - 32, dot.front());
  prefill(failing, padding.data(), padding.size(), rig.store);

  auto source = audio_source(mtmd, rig.model.get(), long_audio);
  CountedSource counted(source);
  bool failed = false;
  try {
    rig.store.decode_segments(failing, counted);
  } catch (const decode::DecodeError& error) {
    failed = true;
    CHECK(error.rc == 1);
    CHECK(error.partial);
  }
  REQUIRE(failed);
  CHECK(counted.encodes == 1);
  CHECK(kv::pos_max(rig.ctx.get(), seq) >= get_position(failing, rig.store));
  CHECK(get_position(sibling, rig.store) == parent_position);
  CHECK(get_position(parent, rig.store) == parent_position);

  const auto probe = rig.root();
  REQUIRE(probe != INVALID_HANDLE);
  const std::vector<llama_token> claimed_free(
      capacity - rig.store.kv_pressure().cells_used, dot.front());
  REQUIRE(!claimed_free.empty());
  CHECK_THROWS_AS(prefill(probe, claimed_free.data(), claimed_free.size(), rig.store),
                  decode::DecodeError);
  CHECK(get_position(probe, rig.store) == 0);

  prune(failing, rig.store);
  CHECK(kv::pos_max(rig.ctx.get(), seq) == -1);
  CHECK(rig.store.kv_pressure().cells_used == parent_cells);
  CHECK_NOTHROW(prefill(probe, claimed_free.data(), claimed_free.size(), rig.store));
  prune(probe, rig.store);
  check_audio_transcript(generate(sibling, rig.store, vocab, 96));
  prune(sibling, rig.store);
  CHECK(rig.store.kv_pressure().cells_used == parent_cells);
  prune(parent, rig.store);
  CHECK(rig.store.kv_pressure().cells_used == 0);
}

TEST_CASE("audio: invalid inputs are refused before branch mutation"
          * doctest::skip(audio_tests_disabled())) {
  REQUIRE(MODEL_PATH);
  REQUIRE(MMPROJ_PATH);
  LlamaBackendGuard guard;
  AudioRig rig;
  auto* mtmd = acquire_mtmd(rig.model.get());
  REQUIRE(mtmd);
  const auto bytes = read_fixture("asr-counting.wav");
  REQUIRE(bytes.size() == 180646);
  const auto parent = rig.root();
  const AudioLimits exact{bytes.size(), 90284};

  CHECK_NOTHROW(audio_source(mtmd, rig.model.get(), bytes, exact));
  CHECK_THROWS_WITH(audio_source(mtmd, rig.model.get(), bytes, {bytes.size() - 1, 90284}),
                    "MtmdSource - audio byte limit exceeded");
  CHECK_THROWS_WITH(audio_source(mtmd, rig.model.get(), bytes, {bytes.size(), 90283}),
                    "MtmdSource - audio sample limit exceeded");
  CHECK_THROWS_WITH(audio_source(mtmd, rig.model.get(), bytes, {}),
                    "MtmdSource - audio limits must be positive");
  CHECK_THROWS_AS(audio_source(nullptr, rig.model.get(), bytes), std::runtime_error);
  CHECK_THROWS_AS(audio_source(mtmd, rig.model.get(), {}), std::runtime_error);
  CHECK_THROWS_AS(audio_source(mtmd, rig.model.get(), read_fixture("cat.jpg")),
                  std::runtime_error);

  struct InvalidWav { const char* reason; size_t offset; uint8_t value; };
  const InvalidWav invalid[] = {
      {"RIFF size", 4, 0}, {"container", 8, 'X'},
      {"fmt size", 16, 15}, {"compressed format", 20, 6},
      {"channels", 22, 0}, {"rate", 24, 0},
      {"byte rate", 29, 0}, {"block alignment", 32, 1},
      {"sample width", 34, 12}, {"unsupported LIST", 44, 'X'},
      {"data extent", 74, 0},
  };
  for (const auto& invalid_wav : invalid) {
    INFO(invalid_wav.reason);
    auto malformed = bytes;
    malformed[invalid_wav.offset] = invalid_wav.value;
    CHECK_THROWS_AS(audio_source(mtmd, rig.model.get(), malformed), std::runtime_error);
  }
  auto truncated = bytes;
  truncated.pop_back();
  CHECK_THROWS_AS(audio_source(mtmd, rig.model.get(), truncated), std::runtime_error);

  const std::array media{MediaInput{MediaInput::Kind::Audio, bytes}};
  const auto width = llama_model_n_embd_inp(rig.model.get());
  CHECK_THROWS_AS(MtmdSource(mtmd, "no marker", media, {}, width, exact),
                  std::runtime_error);
  const std::array duplicate{media.front(), media.front()};
  CHECK_THROWS_WITH(MtmdSource(mtmd, "<__media__><__media__>", duplicate, {}, width, exact),
                    "MtmdSource - audio byte limit exceeded");
  CHECK_THROWS_WITH(MtmdSource(mtmd, "<__media__><__media__>", duplicate, {}, width,
                              AudioLimits{2 * bytes.size(), 90284}),
                    "MtmdSource - audio sample limit exceeded");

  CHECK(get_position(parent, rig.store) == 0);
  CHECK(rig.store.kv_pressure().cells_used == 0);
  auto valid = audio_source(mtmd, rig.model.get(), bytes);
  CHECK_NOTHROW(rig.store.decode_segments(parent, valid));
  check_audio_transcript(generate(parent, rig.store,
      llama_model_get_vocab(rig.model.get()), 96));
}

static void wav_u32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  for (size_t i = 0; i < 4; ++i) bytes[offset + i] = static_cast<uint8_t>((value >> (8 * i)) & 0xff);
}

TEST_CASE("audio: multiple admitted recordings use ordered encoder outputs"
          * doctest::skip(audio_tests_disabled())) {
  REQUIRE(MODEL_PATH);
  REQUIRE(MMPROJ_PATH);
  LlamaBackendGuard guard;
  AudioRig rig;
  auto* mtmd = acquire_mtmd(rig.model.get());
  REQUIRE(mtmd);
  const auto bytes = read_fixture("asr-counting.wav");
  const std::array inputs{MediaInput{MediaInput::Kind::Audio, bytes},
                          MediaInput{MediaInput::Kind::Audio, bytes}};
  chat_in::FormatInputs in;
  in.messages_json = R"([{"role":"user","content":[{"type":"media_marker","text":"<__media__>"},{"type":"media_marker","text":"<__media__>"}]}])";
  in.enable_thinking = false;
  MtmdSource source(mtmd, chat_in::format(rig.model.get(), in).prompt, inputs, {},
                    llama_model_n_embd_inp(rig.model.get()),
                    AudioLimits{2 * bytes.size(), 2 * 90284});
  CountedSource counted(source);
  const auto parent = rig.root();
  const auto result = rig.store.decode_segments(parent, counted);
  CHECK(counted.encodes == 2);
  CHECK(result.cells == static_cast<int64_t>(source.cells()));
  CHECK(result.advance == result.cells);
  CHECK(rig.store.kv_pressure().cells_used == source.cells());
  check_audio_transcript(generate(parent, rig.store,
      llama_model_get_vocab(rig.model.get()), 128));
}

TEST_CASE("audio: overflowing WAV chunk extents fail before decoding"
          * doctest::skip(audio_tests_disabled())) {
  REQUIRE(MODEL_PATH);
  REQUIRE(MMPROJ_PATH);
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  auto* mtmd = acquire_mtmd(model.get());
  REQUIRE(mtmd);
  auto bytes = read_fixture("asr-counting.wav");
  wav_u32(bytes, 74, std::numeric_limits<uint32_t>::max());
  CHECK_THROWS_WITH(audio_source(mtmd, model.get(), bytes),
                    "MtmdSource - truncated WAV chunk");
}

static std::vector<uint8_t> stereo_48khz_fixture() {
  const auto original = read_fixture("asr-counting.wav");
  REQUIRE(original.size() == 180646);
  constexpr size_t data_offset = 78; // The versioned fixture includes LIST/INFO.
  std::vector<uint8_t> stereo(original.begin(), original.begin() + data_offset);
  for (size_t offset = data_offset; offset < original.size(); offset += 2) {
    for (size_t copy = 0; copy < 6; ++copy) {
      stereo.push_back(original[offset]);
      stereo.push_back(original[offset + 1]);
    }
  }
  stereo[22] = 2;
  stereo[32] = 4;
  wav_u32(stereo, 24, 48000);
  wav_u32(stereo, 28, 48000 * 4);
  wav_u32(stereo, 74, static_cast<uint32_t>(stereo.size() - data_offset));
  wav_u32(stereo, 4, static_cast<uint32_t>(stereo.size() - 8));
  return stereo;
}

TEST_CASE("audio: upstream resamples stereo and bounds the decoded sample count"
          * doctest::skip(audio_tests_disabled())) {
  REQUIRE(MODEL_PATH);
  REQUIRE(MMPROJ_PATH);
  LlamaBackendGuard guard;
  AudioRig rig;
  auto* mtmd = acquire_mtmd(rig.model.get());
  REQUIRE(mtmd);
  auto bytes = stereo_48khz_fixture();
  CHECK_THROWS_WITH(audio_source(mtmd, rig.model.get(), bytes, {bytes.size(), 90284}),
                    "MtmdSource - audio sample limit exceeded");
  auto source = audio_source(mtmd, rig.model.get(), bytes, {bytes.size(), 90285});
  bytes.clear();
  bytes.shrink_to_fit();
  const auto parent = rig.root();
  const auto result = rig.store.decode_segments(parent, source);
  CHECK(result.cells == static_cast<int64_t>(source.cells()));
  CHECK(result.advance == result.cells);
  check_audio_transcript(generate(parent, rig.store,
      llama_model_get_vocab(rig.model.get()), 96));
}

TEST_CASE("audio: empty, single-sample and misaligned WAV data fail before preprocessing"
          * doctest::skip(audio_tests_disabled())) {
  REQUIRE(MODEL_PATH);
  REQUIRE(MMPROJ_PATH);
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  auto* mtmd = acquire_mtmd(model.get());
  REQUIRE(mtmd);
  const auto original = read_fixture("asr-counting.wav");
  REQUIRE(original.size() == 180646);
  for (uint32_t data_bytes : {0u, 1u, 2u, 3u}) {
    INFO("data bytes: " << data_bytes);
    auto bytes = original;
    bytes.resize(78 + data_bytes + (data_bytes & 1));
    wav_u32(bytes, 74, data_bytes);
    wav_u32(bytes, 4, static_cast<uint32_t>(bytes.size() - 8));
    CHECK_THROWS_AS(audio_source(mtmd, model.get(), bytes), std::runtime_error);
  }
}

TEST_CASE("audio: position overflow is refused before filling decoder positions"
          * doctest::skip(audio_tests_disabled())) {
  REQUIRE(MODEL_PATH);
  REQUIRE(MMPROJ_PATH);
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);
  auto* mtmd = acquire_mtmd(model.get());
  REQUIRE(mtmd);
  auto source = audio_source(mtmd, model.get(), read_fixture("asr-counting.wav"));
  size_t audio_chunks = 0;
  for (size_t i = 0; i < source.size(); ++i) {
    const auto segment = source.at(i);
    if (segment.kind != decode::Segment::Kind::Embd) continue;
    ++audio_chunks;
    const auto last_base = std::numeric_limits<llama_pos>::max() - segment.n_rows;
    std::vector<llama_pos> positions(segment.n_rows * segment.n_pos_per_embd, -1);
    source.positions(i, last_base, positions.data());
    CHECK(positions.front() == last_base);
    CHECK(positions.back() == std::numeric_limits<llama_pos>::max() - 1);
    std::fill(positions.begin(), positions.end(), -1);
    CHECK_THROWS_WITH(source.positions(i, last_base + 1, positions.data()),
                      "MtmdSource - media position overflow");
    CHECK(std::all_of(positions.begin(), positions.end(), [](auto p) { return p == -1; }));
  }
  CHECK(audio_chunks == 1);
}
