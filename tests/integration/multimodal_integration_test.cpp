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
 * Gated on LLAMA_TEST_MODEL + LLAMA_MMPROJ_MODEL naming a matched VL pair.
 */

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
    p.warmup         = false;  // every case encodes a real image anyway
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

/**
 * Whether to assert on what the model SAYS, not just what the KV does.
 *
 * Content assertions need a capable VL model (the Qwen3.5-4B tier). The CI
 * tier is SmolVLM-256M, which exercises the plain-position rail and the whole
 * cell/slack mechanism but cannot be held to a grounded answer. Default on, so
 * a local run never quietly weakens itself; CI opts out explicitly.
 */
static bool vl_strict() {
  const char* e = std::getenv("LLOYAL_VL_STRICT");
  return !(e && std::string(e) == "0");
}

static bool contains_any(const std::string& haystack,
                         const std::vector<std::string>& needles) {
  std::string lower;
  lower.reserve(haystack.size());
  for (char c : haystack) lower += static_cast<char>(std::tolower(c));
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
  const auto r = store.decode_segments(root, source);

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
        "MtmdSource - media marker count (1) does not match image count (2)");
  }
  {
    const std::string two_markers = "a " + marker + " b " + marker;
    std::vector<std::vector<uint8_t>> images{image};
    CHECK_THROWS_WITH(
        MtmdSource(mtmd, two_markers, images,
                   std::span<const llama_token>(), n_embd_inp),
        "MtmdSource - media marker count (2) does not match image count (1)");
  }
  {
    std::vector<std::vector<uint8_t>> none;
    CHECK_THROWS_WITH(
        MtmdSource(mtmd, one_marker, none,
                   std::span<const llama_token>(), n_embd_inp),
        "MtmdSource - media marker count (1) does not match image count (0)");
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
