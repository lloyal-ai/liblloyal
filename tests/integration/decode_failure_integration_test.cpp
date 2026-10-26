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
#include "media_fixtures.hpp"
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

/// A stream whose source fails at segment `fail_at` with a decode error of its
/// own, as an encoder that runs a model of its own and finds no room would.
/// The error says nothing of what the prefill landed before it.
struct FailingSegments : TextSegments {
  size_t fail_at = 0;
  decode::Segment at(size_t i) override {
    if (i == fail_at) throw decode::DecodeError(1, false, "the source's own encoder found no KV slot");
    return TextSegments::at(i);
  }
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

TEST_CASE("decode failure: a decode error from the source after a segment landed is partial, and the branch is poisoned") {
  // The rail marks its own failures partial; a source's error cannot know
  // what landed before it. The branch's position does: every segment is
  // material, so a moved position is proof the prefill is part-way through.
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
  const auto a = filler(50, 1000, n_vocab);
  const auto b = filler(50, 2000, n_vocab);

  // After segment 0 landed: partial, poisoned, the source's own words kept.
  BranchHandle h = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(h != INVALID_HANDLE);
  FailingSegments late;
  late.segs = {a, b};
  late.fail_at = 1;
  std::string said;
  const Caught got = attempt([&] {
    try {
      store.decode_segments(h, late);
    } catch (const decode::DecodeError& e) {
      said = e.what();
      throw;
    }
  });
  REQUIRE(got.threw);
  CHECK(got.rc == 1);
  CHECK(got.partial == true);
  CHECK(said.find("the source's own encoder found no KV slot") != std::string::npos);
  CHECK(is_poisoned(h, store));
  CHECK(get_position(h, store) == 50);

  // Before anything landed: the error passes through as it was, and the
  // branch is intact, so the same prefill lands on it.
  BranchHandle fresh = create(ctx, model.get(), store, 0, params, 128);
  REQUIRE(fresh != INVALID_HANDLE);
  FailingSegments early;
  early.segs = {a, b};
  early.fail_at = 0;
  const Caught got0 = attempt([&] { store.decode_segments(fresh, early); });
  REQUIRE(got0.threw);
  CHECK(got0.rc == 1);
  CHECK(got0.partial == false);
  CHECK_FALSE(is_poisoned(fresh, store));
  CHECK(get_position(fresh, store) == 0);
  TextSegments whole;
  whole.segs = {a, b};
  CHECK_NOTHROW(store.decode_segments(fresh, whole));
  CHECK(get_position(fresh, store) == 100);

  prune(h, store);
  prune(fresh, store);
  CHECK(store.kv_pressure().cells_used == 0);
  store.drain();
  llama_free(ctx);
}

TEST_CASE("decode failure: a chunk larger than the context's batch is refused, never dispatched") {
  // The pinned llama.cpp asserts n_tokens <= n_batch inside llama_decode — an
  // abort, not an rc. Chunk sizes come from the caller (a branch's n_batch,
  // a free function's argument), so a caller that overshoots the context's
  // batch must be refused by the kernel before the batch is built.
  REQUIRE_MODEL();
  LlamaBackendGuard guard;
  auto model = TestConfig::acquire_test_model();
  REQUIRE(model);

  llama_context* ctx = small_ctx(model.get(), 256, 32, 2);   // the context's batch is 32
  REQUIRE(ctx);
  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;
  const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

  BranchHandle h = create(ctx, model.get(), store, 0, params, /*n_batch*/ 64);  // the branch asks for 64
  REQUIRE(h != INVALID_HANDLE);
  const auto toks = filler(64, 1000, n_vocab);
  CHECK_THROWS_WITH(prefill(h, toks.data(), toks.size(), store),
                    doctest::Contains("exceeds the context's n_batch"));
  CHECK(get_position(h, store) == 0);
  CHECK(kv::pos_max(ctx, store.get(h)->seq_id) == -1);

  // A chunk that fits is unaffected.
  const auto small = filler(32, 2000, n_vocab);
  CHECK_NOTHROW(prefill(h, small.data(), small.size(), store));
  CHECK(get_position(h, store) == 32);

  prune(h, store);
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

using TestConfig::read_fixture;

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

  auto* mtmd = TestConfig::acquire_mtmd(model.get());
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
    MtmdSource probe(mtmd, prompt, images, std::span<const llama_token>(), n_embd_inp);
    for (size_t i = 0; i < probe.size(); ++i) {
      const auto seg = probe.at(i);
      if (seg.kind == decode::Segment::Kind::Embd) { n_rows = seg.n_rows; break; }
      text_before += static_cast<int32_t>(seg.tokens.size());
    }
  }
  REQUIRE(n_rows > 24);  // room for a first 16-row chunk to land and a later one to fail

  // The context takes whole batches (256); the IMAGE branch chunks its rows
  // by its own n_batch of 16, so a later chunk can fail while earlier ones
  // stand. Batch and micro-batch agree, so a chunk is one llama_decode.
  llama_context* ctx = small_ctx(model.get(), 256, 256, 2);
  REQUIRE(ctx);
  BranchStore store(8);
  store.init_tenancy(ctx);
  TestParams params;
  const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

  // Leave the image fewer free cells than it has rows — (n_rows - 8) when
  // the pool could otherwise hold it, and no padding at all when the image
  // alone outruns the pool (a dynamic-resolution projector on this fixture).
  // Either way its first 16-row chunk lands and a later one cannot.
  const int32_t fill = std::max(0, 256 - text_before - (n_rows - 8));
  REQUIRE(256 - text_before - fill >= 16);

  BranchHandle h = create(ctx, model.get(), store, 0, params, 16);
  REQUIRE(h != INVALID_HANDLE);
  const llama_seq_id seq = store.get(h)->seq_id;
  if (fill > 0) {
    const auto pad = filler(fill, 1000, n_vocab);
    prefill(h, pad.data(), pad.size(), store);
  }
  REQUIRE(get_position(h, store) == fill);

  MtmdSource source(mtmd, prompt, images, std::span<const llama_token>(), n_embd_inp);
  const Caught got = attempt([&] { store.decode_segments(h, source); });
  REQUIRE(got.threw);
  CHECK(got.rc == 1);
  CHECK(got.partial == true);

  // The text before the image landed and moved the books; the image's rows
  // that landed did not (decode_embd's books move only on success), so the
  // KV holds cells the books do not admit to — poisoned. Position is not the
  // witness here: an M-RoPE projector holds every row at ONE temporal
  // position, so pos_max cannot count rows. The pool can: a prefill sized
  // to the books' idea of free space must fail while the orphans stand.
  CHECK(get_position(h, store) == fill + text_before);
  CHECK(store.kv_pressure().cells_used == static_cast<uint32_t>(fill + text_before));
  const int32_t claimed_free = 256 - (fill + text_before);
  BranchHandle probe = create(ctx, model.get(), store, 0, params, 256);  // one chunk: all or nothing
  REQUIRE(probe != INVALID_HANDLE);
  const auto probe_toks = filler(claimed_free - 15, 7000, n_vocab);       // fits iff < 16 rows landed
  const Caught blocked = attempt([&] { prefill(probe, probe_toks.data(), probe_toks.size(), store); });
  CHECK(blocked.threw);
  CHECK(blocked.rc == 1);
  CHECK(blocked.partial == false);
  CHECK(get_position(probe, store) == 0);

  // Prune reclaims the poisoned branch whole — text and orphaned rows — and
  // the same prefill now lands.
  prune(h, store);
  CHECK(kv::pos_max(ctx, seq) == -1);
  CHECK_NOTHROW(prefill(probe, probe_toks.data(), probe_toks.size(), store));
  CHECK(get_position(probe, store) == static_cast<llama_pos>(probe_toks.size()));
  prune(probe, store);
  CHECK(store.kv_pressure().cells_used == 0);
  store.drain();
  llama_free(ctx);
}
