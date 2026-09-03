/**
 * A failed decode says what landed — on real weights.
 *
 * The stub suite pins the rule (DecodeError: the failing call restored its
 * state iff rc is 1 or -1; the branch is intact iff that AND nothing before
 * it landed). These cases make the KV itself the witness. Exhaustion is the
 * one failure a real context produces on demand: a small unified cache and
 * chunks sized so that a LATER chunk cannot find a slot — llama_decode
 * returns 1 for exactly that call and nothing else.
 *
 * Beyond the flag, each case checks what a caller relies on:
 *   - landed branches' books agree with the KV (position and cells),
 *   - branches that did not land did not move, and still work,
 *   - a poisoned branch (KV ahead of its books) is reclaimed whole by prune,
 *   - the store is usable afterwards — a fresh branch prefills.
 *
 * Gated on LLAMA_TEST_MODEL (the embedding-rail case also on
 * LLAMA_MMPROJ_MODEL naming a matched VL pair, like the multimodal suite).
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <doctest/doctest.h>
#include "test_config.hpp"
#include <llama/llama.h>
#include <lloyal/branch.hpp>
#include <lloyal/chat_in.hpp>
#include <lloyal/decode.hpp>
#include <lloyal/kv.hpp>
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

#define REQUIRE_MODEL()                                                        \
  if (!MODEL_PATH || !*MODEL_PATH) {                                           \
    MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set");                              \
    return;                                                                    \
  }
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
  float temperature = 0.0f;
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
// Rig
// ============================================================================

/// A context whose KV is ONE pool of `n_ctx` cells shared by every sequence,
/// so the arithmetic in each case counts cells, not per-stream slots. Batch
/// and micro-batch agree, so a chunk is exactly one llama_decode.
static llama_context* small_ctx(llama_model* model, uint32_t n_ctx,
                                uint32_t n_batch, uint32_t n_seq_max) {
  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx      = n_ctx;
  cparams.n_batch    = n_batch;
  cparams.n_ubatch   = n_batch;
  cparams.n_seq_max  = n_seq_max;
  cparams.kv_unified = true;
  return llama_init_from_model(model, cparams);
}

/// Ordinary vocabulary ids, distinct per run so nothing here is a prefix of
/// anything else. Content is irrelevant: only the cell count matters.
static std::vector<llama_token> filler(int32_t n, int32_t base, int32_t n_vocab) {
  std::vector<llama_token> t(static_cast<size_t>(n));
  for (int32_t j = 0; j < n; ++j) t[static_cast<size_t>(j)] = static_cast<llama_token>((base + j) % n_vocab);
  return t;
}

/// The failure, caught as the data it carries.
struct Caught {
  bool threw = false;
  int32_t rc = 0;
  bool partial = false;
};
template <class F>
static Caught attempt(F&& f) {
  Caught c;
  try {
    f();
  } catch (const decode::DecodeError& e) {
    c.threw = true;
    c.rc = e.rc;
    c.partial = e.partial;
  }
  return c;
}

/// A SegmentSource of ready token runs — the token-rail half of a prefill,
/// enough to exercise decode_segments' composition without a projector.
struct TextSegments : decode::SegmentSource {
  std::vector<std::span<const llama_token>> segs;
  size_t size() override { return segs.size(); }
  size_t cells() const override {
    size_t n = 0;
    for (const auto& s : segs) n += s.size();
    return n;
  }
  decode::Segment at(size_t i) override {
    decode::Segment s;
    s.kind = decode::Segment::Kind::Text;
    s.tokens = segs[i];
    return s;
  }
  void positions(size_t, llama_pos, llama_pos*) override {}
};

// ============================================================================
// Token rail
// ============================================================================

TEST_CASE("decode failure: a later scatter chunk with no KV slot reports partial; landed branches stay consistent") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context* ctx = small_ctx(model.get(), 256, 128, 4);
  REQUIRE(ctx);
  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;

  const auto* vocab = llama_model_get_vocab(model.get());
  const int32_t n_vocab = llama_vocab_n_tokens(vocab);
  auto prompt = tokenizer::tokenize(vocab, "Hello", true, false);
  REQUIRE(!prompt.empty());
  REQUIRE(prompt.size() < 40);  // the arithmetic below assumes a short prefix

  BranchHandle root = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(root != INVALID_HANDLE);
  prefill(root, prompt.data(), prompt.size(), store);
  const llama_pos p = get_position(root, store);
  const uint32_t cells0 = store.kv_pressure().cells_used;

  // Three children, 100 tokens each: under n_batch=128 that bin-packs into
  // THREE chunks. The pool holds 256 cells: p + 200 fit, p + 300 do not, so
  // the third chunk is the one llama_decode refuses.
  BranchHandle c[3];
  for (auto& h : c) {
    h = fork(root, store);
    REQUIRE(h != INVALID_HANDLE);
  }
  const auto t0 = filler(100, 1000, n_vocab);
  const auto t1 = filler(100, 2000, n_vocab);
  const auto t2 = filler(100, 3000, n_vocab);
  DecodeScatterItem items[] = {{c[0], t0}, {c[1], t1}, {c[2], t2}};

  const Caught got = attempt([&] { store.decode_scatter(items); });
  REQUIRE(got.threw);
  CHECK(got.rc == 1);
  CHECK(got.partial == true);

  // Landed: the books moved with the KV. Not landed: nothing moved, and
  // llama_decode left no trace of the refused chunk on that sequence.
  CHECK(get_position(c[0], store) == p + 100);
  CHECK(get_position(c[1], store) == p + 100);
  CHECK(get_position(c[2], store) == p);
  CHECK(store.kv_pressure().cells_used == cells0 + 200);
  CHECK(kv::pos_max(ctx, store.get(c[0])->seq_id) == p + 100 - 1);
  CHECK(kv::pos_max(ctx, store.get(c[2])->seq_id) == p - 1);

  // The intact child is exactly that: a prefill that fits still lands on it.
  const auto tail = filler(10, 4000, n_vocab);
  DecodeScatterItem one{c[2], tail};
  CHECK_NOTHROW(store.decode_scatter(std::span<const DecodeScatterItem>(&one, 1)));
  CHECK(get_position(c[2], store) == p + 10);

  pruneSubtree(root, store);
  CHECK(store.kv_pressure().cells_used == 0);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("decode failure: a chunked prefill that dies mid-way is poisoned, says so, and prune reclaims it") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context* ctx = small_ctx(model.get(), 256, 64, 2);
  REQUIRE(ctx);
  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;
  const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);
  const llama_seq_id seq = store.get(h)->seq_id;

  // 300 tokens in 64-token chunks: four land (256 cells, the pool is full),
  // the fifth cannot.
  const auto toks = filler(300, 1000, n_vocab);
  const Caught got = attempt([&] { prefill(h, toks.data(), toks.size(), store); });
  REQUIRE(got.threw);
  CHECK(got.rc == 1);
  CHECK(got.partial == true);

  // The books never moved; the KV did. That gap is what "poisoned" means.
  CHECK(get_position(h, store) == 0);
  CHECK(store.kv_pressure().cells_used == 0);
  CHECK(get_logits(h, store) == nullptr);
  CHECK(kv::pos_max(ctx, seq) == 255);

  // Prune reclaims every orphan: the sequence is clean, the pool is whole,
  // and a fresh branch prefills where the poisoned one could not.
  prune(h, store);
  CHECK(kv::pos_max(ctx, seq) == -1);
  BranchHandle fresh = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(fresh != INVALID_HANDLE);
  const auto small = filler(10, 5000, n_vocab);
  CHECK_NOTHROW(prefill(fresh, small.data(), small.size(), store));
  CHECK(get_position(fresh, store) == 10);
  CHECK(store.kv_pressure().cells_used == 10);

  prune(fresh, store);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("decode failure: an oversized item that lands some chunks is partial with an UNMOVED position") {
  // The case a caller cannot tell apart from intact by position alone —
  // which is why partial travels as data. decode::many landed four chunks
  // of the item before the fifth was refused; the branch's books say 0.
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context* ctx = small_ctx(model.get(), 256, 64, 2);
  REQUIRE(ctx);
  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;
  const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);
  const llama_seq_id seq = store.get(h)->seq_id;

  const auto big = filler(300, 1000, n_vocab);  // > n_batch: the oversized path
  DecodeScatterItem item{h, big};
  const Caught got = attempt([&] {
    store.decode_scatter(std::span<const DecodeScatterItem>(&item, 1));
  });
  REQUIRE(got.threw);
  CHECK(got.rc == 1);
  CHECK(got.partial == true);
  CHECK(get_position(h, store) == 0);
  CHECK(store.kv_pressure().cells_used == 0);
  CHECK(kv::pos_max(ctx, seq) == 255);

  prune(h, store);
  CHECK(kv::pos_max(ctx, seq) == -1);
  CHECK(store.kv_pressure().cells_used == 0);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("decode failure: a segment after a landed one reports partial even when its own first call failed") {
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context* ctx = small_ctx(model.get(), 256, 128, 2);
  REQUIRE(ctx);
  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;
  const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

  BranchHandle h = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(h != INVALID_HANDLE);
  const llama_seq_id seq = store.get(h)->seq_id;

  // Segment 1 (200 tokens) lands whole; segment 2 (100 tokens) is a single
  // chunk that finds only 56 free cells. Its OWN call restored cleanly, so
  // the flag is the composition's: a prefill is one operation to its caller.
  const auto a = filler(200, 1000, n_vocab);
  const auto b = filler(100, 3000, n_vocab);
  TextSegments src;
  src.segs = {a, b};

  const Caught got = attempt([&] { store.decode_segments(h, src); });
  REQUIRE(got.threw);
  CHECK(got.rc == 1);
  CHECK(got.partial == true);
  CHECK(get_position(h, store) == 200);
  CHECK(store.kv_pressure().cells_used == 200);
  CHECK(kv::pos_max(ctx, seq) == 199);  // segment 2 left nothing behind

  prune(h, store);
  CHECK(kv::pos_max(ctx, seq) == -1);
  CHECK(store.kv_pressure().cells_used == 0);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("decode failure: a repeated handle is refused before anything is dispatched, in both cohorts") {
  // One rule for every path that batches by handle (require_distinct_handles):
  // decode_each used to state none and put two tokens on one cell; the KV is
  // the witness that nothing moves now, and that the branch is intact after.
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context* ctx = small_ctx(model.get(), 256, 64, 4);
  REQUIRE(ctx);
  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;
  const auto* vocab = llama_model_get_vocab(model.get());
  const int32_t n_vocab = llama_vocab_n_tokens(vocab);
  auto prompt = tokenizer::tokenize(vocab, "Hello", true, false);
  REQUIRE(!prompt.empty());

  BranchHandle h = create(ctx, model.get(), store, 0, params, 64);
  REQUIRE(h != INVALID_HANDLE);
  prefill(h, prompt.data(), prompt.size(), store);
  const llama_pos p = get_position(h, store);
  const llama_seq_id seq = store.get(h)->seq_id;
  const uint32_t cells0 = store.kv_pressure().cells_used;

  // decode_each: two tokens for one branch would share a cell.
  DecodeEachItem each[] = {{h, 42}, {h, 43}};
  CHECK_THROWS_WITH(store.decode_each(each),
                    doctest::Contains("decode_each - duplicate handle at indices 0 and 1"));
  CHECK(get_position(h, store) == p);
  CHECK(kv::pos_max(ctx, seq) == p - 1);
  CHECK(store.kv_pressure().cells_used == cells0);

  // decode_scatter: two runs for one branch would overlap.
  const auto a = filler(5, 1000, n_vocab);
  const auto b = filler(6, 2000, n_vocab);
  DecodeScatterItem twice[] = {{h, a}, {h, b}};
  CHECK_THROWS_WITH(store.decode_scatter(twice),
                    doctest::Contains("decode_scatter - duplicate handle at indices 0 and 1"));
  CHECK(get_position(h, store) == p);
  CHECK(kv::pos_max(ctx, seq) == p - 1);

  // An empty span beside a real one is not a repeat: it occupies no cells.
  DecodeScatterItem mixed[] = {{h, std::span<const llama_token>()}, {h, a}};
  CHECK_NOTHROW(store.decode_scatter(mixed));
  CHECK(get_position(h, store) == p + 5);
  CHECK(kv::pos_max(ctx, seq) == p + 5 - 1);

  // Refused, not poisoned: the branch keeps working.
  DecodeEachItem one[] = {{h, 44}};
  CHECK_NOTHROW(store.decode_each(one));
  CHECK(get_position(h, store) == p + 6);
  CHECK(store.kv_pressure().cells_used == cells0 + 6);

  prune(h, store);
  CHECK(store.kv_pressure().cells_used == 0);
  store.drain();
  llama_free(ctx);
}

// ============================================================================
// Embedding rail — a real projector, rows chunked by the branch's n_batch
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

struct MtmdDeleter {
  void operator()(mtmd_context* c) const { if (c) mtmd_free(c); }
};

static std::string image_prompt(const llama_model* model) {
  chat_in::FormatInputs in;
  in.messages_json =
      std::string(R"([{"role":"system","content":"You are a vision assistant."},)"
                  R"({"role":"user","content":[{"type":"text","text":"Describe this image."},)"
                  R"({"type":"media_marker","text":")") + mtmd_default_marker() +
      std::string(R"("}]}])");
  in.enable_thinking = false;
  return chat_in::format(model, in).prompt;
}

TEST_CASE("decode failure: an image whose rows outrun the KV reports partial on the embedding rail") {
  REQUIRE_VL();
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  auto mp = mtmd_context_params_default();
  mp.use_gpu       = TestConfig::n_gpu_layers() != 0;
  mp.print_timings = false;
  mp.n_threads     = 4;
  mp.warmup        = false;
  std::unique_ptr<mtmd_context, MtmdDeleter> mtmd(mtmd_init_from_file(MMPROJ_PATH, model.get(), mp));
  REQUIRE(mtmd);

  const auto image = read_fixture("cat.jpg");
  REQUIRE(!image.empty());
  const std::string prompt = image_prompt(model.get());
  std::vector<std::vector<uint8_t>> images{image};
  const int32_t n_embd_inp = llama_model_n_embd_inp(model.get());

  // Measure the walk once: the text that precedes the image, and the rows
  // the image costs. `at()` on the image encodes it; a second walk below
  // encodes it again — the source's in-order contract allows exactly that.
  int32_t text_before = 0;
  int32_t n_rows = 0;
  {
    MtmdSource probe(mtmd.get(), prompt, images, std::span<const llama_token>(), n_embd_inp);
    for (size_t i = 0; i < probe.size(); ++i) {
      const auto seg = probe.at(i);
      if (seg.kind == decode::Segment::Kind::Embd) { n_rows = seg.n_rows; break; }
      text_before += static_cast<int32_t>(seg.tokens.size());
    }
  }
  REQUIRE(n_rows > 24);  // room for a first 16-row chunk to land and a later one to fail

  llama_context* ctx = small_ctx(model.get(), 256, 16, 2);
  REQUIRE(ctx);
  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;
  const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

  // Leave the image exactly (n_rows - 8) free cells: its first 16-row chunk
  // lands, a later one cannot. The filler rides the branch's own n_batch.
  const int32_t free_for_rows = n_rows - 8;
  const int32_t fill = 256 - text_before - free_for_rows;
  REQUIRE(fill > 0);

  BranchHandle h = create(ctx, model.get(), store, 0, params, 16);
  REQUIRE(h != INVALID_HANDLE);
  const llama_seq_id seq = store.get(h)->seq_id;
  const auto pad = filler(fill, 1000, n_vocab);
  prefill(h, pad.data(), pad.size(), store);
  REQUIRE(get_position(h, store) == fill);

  MtmdSource source(mtmd.get(), prompt, images, std::span<const llama_token>(), n_embd_inp);
  const Caught got = attempt([&] { store.decode_segments(h, source); });
  REQUIRE(got.threw);
  CHECK(got.rc == 1);
  CHECK(got.partial == true);

  // The text before the image landed and moved the books; the image's rows
  // that landed did not (decode_embd's books move only on success), so the
  // KV sits ahead of the branch — poisoned, and prune reclaims it.
  CHECK(get_position(h, store) == fill + text_before);
  CHECK(store.kv_pressure().cells_used == static_cast<uint32_t>(fill + text_before));
  CHECK(kv::pos_max(ctx, seq) >= fill + text_before + 16 - 1);

  prune(h, store);
  CHECK(kv::pos_max(ctx, seq) == -1);
  CHECK(store.kv_pressure().cells_used == 0);
  store.drain();
  llama_free(ctx);
}
