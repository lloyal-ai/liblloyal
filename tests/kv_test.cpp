#include "llama_stubs.h"
#include <doctest/doctest.h>
#include <lloyal/kv.hpp>
#include <memory>
#include <vector>

using namespace lloyal::kv;

TEST_CASE("KV: empty cache guard - pos_max returns -1") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = -1; // Empty KV cache

  llama_pos max_pos = pos_max(&ctx, 0);
  CHECK(max_pos == -1);
}

TEST_CASE("KV: empty cache guard - state_size returns 0") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = -1; // Empty KV cache

  size_t size = state_size(&ctx, 0);
  CHECK(size == 0);
}

TEST_CASE("KV: empty cache guard - state_save returns 0") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = -1; // Empty KV cache

  std::vector<uint8_t> buffer(1024);
  size_t written = state_save(&ctx, 0, buffer.data(), buffer.size());

  CHECK(written == 0);
}

TEST_CASE("KV: state_size ignores per-seq size when KV empty") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = -1;        // Empty KV cache
  llamaStubConfig().per_seq_size = 4096; // Would be non-zero without guard

  size_t size = state_size(&ctx, 0);
  CHECK(size == 0); // Guard overrides per-seq size
}

TEST_CASE("KV: null/invalid parameter guards") {
  resetStubConfig();

  std::vector<uint8_t> buf(16, 0);

  SUBCASE("null context returns safe defaults") {
    CHECK(state_size(nullptr, 0) == 0);
    CHECK(state_save(nullptr, 0, buf.data(), buf.size()) == 0);
    CHECK(state_load(nullptr, 0, buf.data(), buf.size()) == 0);
    CHECK(pos_max(nullptr, 0) == -1);
    CHECK_FALSE(remove_range(nullptr, 0, 0, 1));
  }

  SUBCASE("null buffer returns 0") {
    llama_context ctx{};
    CHECK(state_save(&ctx, 0, nullptr, 16) == 0);
    CHECK(state_load(&ctx, 0, nullptr, 16) == 0);
  }

  SUBCASE("zero size returns 0") {
    llama_context ctx{};
    CHECK(state_save(&ctx, 0, buf.data(), 0) == 0);
    CHECK(state_load(&ctx, 0, buf.data(), 0) == 0);
  }
}

TEST_CASE("KV: remove_range success") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().rm_ok = true;

  bool success = remove_range(&ctx, 0, 10, 20);
  CHECK(success);
}

TEST_CASE("KV: remove_range failure") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().rm_ok = false;

  bool success = remove_range(&ctx, 0, 10, 20);
  CHECK_FALSE(success);
}

TEST_CASE("KV: pos_max tracking") {
  resetStubConfig();

  llama_context ctx{};

  SUBCASE("Empty cache") {
    llamaStubConfig().pos_max = -1;
    CHECK(pos_max(&ctx, 0) == -1);
  }

  SUBCASE("Cache with 10 tokens") {
    llamaStubConfig().pos_max = 9; // 10 tokens (0-9)
    CHECK(pos_max(&ctx, 0) == 9);
  }

  SUBCASE("Cache with 100 tokens") {
    llamaStubConfig().pos_max = 99;
    CHECK(pos_max(&ctx, 0) == 99);
  }
}

TEST_CASE("KV: state_size per-sequence success") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;        // Non-empty cache
  llamaStubConfig().per_seq_size = 2048; // Per-seq succeeds

  size_t size = state_size(&ctx, 0);

  // Should return per-seq size (no fallback)
  CHECK(size == 2048);
}

TEST_CASE("KV: state_size fallback to global") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;       // Non-empty cache
  llamaStubConfig().per_seq_size = 0;   // Per-seq fails
  llamaStubConfig().global_size = 4096; // Global succeeds

  size_t size = state_size(&ctx, 0);

  // Should fallback to global size
  CHECK(size == 4096);
}

TEST_CASE("KV: state_size both operations fail") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;     // Non-empty cache
  llamaStubConfig().per_seq_size = 0; // Per-seq fails
  llamaStubConfig().global_size = 0;  // Global fails

  size_t size = state_size(&ctx, 0);

  // Should return 0
  CHECK(size == 0);
}

TEST_CASE("KV: state_save per-sequence success") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;      // Non-empty cache
  llamaStubConfig().per_seq_rw = 2048; // Per-seq succeeds

  std::vector<uint8_t> buffer(4096);
  size_t written = state_save(&ctx, 0, buffer.data(), buffer.size());

  // Should write per-seq bytes (no fallback)
  CHECK(written == 2048);
}

TEST_CASE("KV: state_save fallback to global") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;     // Non-empty cache
  llamaStubConfig().per_seq_rw = 0;   // Per-seq fails (fragmentation)
  llamaStubConfig().global_rw = 4096; // Global succeeds

  std::vector<uint8_t> buffer(8192);
  size_t written = state_save(&ctx, 0, buffer.data(), buffer.size());

  // Should fallback to global save
  CHECK(written == 4096);
}

TEST_CASE("KV: state_save both operations fail") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;   // Non-empty cache
  llamaStubConfig().per_seq_rw = 0; // Per-seq fails
  llamaStubConfig().global_rw = 0;  // Global fails

  std::vector<uint8_t> buffer(4096);
  size_t written = state_save(&ctx, 0, buffer.data(), buffer.size());

  // Should return 0
  CHECK(written == 0);
}

TEST_CASE("KV: state_load per-sequence success") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;      // Non-empty cache
  llamaStubConfig().per_seq_rw = 2048; // Per-seq succeeds

  std::vector<uint8_t> buffer(4096, 0xAA); // Fake saved state
  size_t read = state_load(&ctx, 0, buffer.data(), buffer.size());

  // Should read per-seq bytes (no fallback)
  CHECK(read == 2048);
}

TEST_CASE("KV: state_load fallback to global") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;     // Non-empty cache
  llamaStubConfig().per_seq_rw = 0;   // Per-seq fails
  llamaStubConfig().global_rw = 4096; // Global succeeds

  std::vector<uint8_t> buffer(8192, 0xBB);
  size_t read = state_load(&ctx, 0, buffer.data(), buffer.size());

  // Should fallback to global restore
  CHECK(read == 4096);
}

TEST_CASE("KV: state_load both operations fail") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;   // Non-empty cache
  llamaStubConfig().per_seq_rw = 0; // Per-seq fails
  llamaStubConfig().global_rw = 0;  // Global fails

  std::vector<uint8_t> buffer(4096, 0xCC);
  size_t read = state_load(&ctx, 0, buffer.data(), buffer.size());

  // Should return 0 (caller must replay from checkpoint)
  CHECK(read == 0);
}

TEST_CASE("KV: state_load proceeds even if pos_max < 0") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = -1;     // Empty cache → should warn only
  llamaStubConfig().per_seq_rw = 0;   // Force fallback
  llamaStubConfig().global_rw = 1024; // Fallback succeeds

  std::vector<uint8_t> buffer(2048, 0xEE);
  size_t read = state_load(&ctx, 0, buffer.data(), buffer.size());

  // Should still succeed (logs warning but doesn't bail)
  CHECK(read == 1024);
}

TEST_CASE("KV: global_state_size explicit call") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().global_size = 8192;

  size_t size = global_state_size(&ctx);
  CHECK(size == 8192);
}

TEST_CASE("KV: global_state_save explicit call") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().global_rw = 4096;

  std::vector<uint8_t> buffer(8192);
  size_t written = global_state_save(&ctx, buffer.data(), buffer.size());

  CHECK(written == 4096);
}

TEST_CASE("KV: global_state_load explicit call") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().global_rw = 4096;

  std::vector<uint8_t> buffer(8192, 0xDD);
  size_t read = global_state_load(&ctx, buffer.data(), buffer.size());

  CHECK(read == 4096);
}

// ===== PHASE 3: CACHE CLEARING TESTS =====

TEST_CASE("KV: clear_all - null context guard") {
  resetStubConfig();

  CHECK_THROWS_AS(clear_all(nullptr), std::runtime_error);
}

TEST_CASE("KV: clear_all - successful operation") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;  // Simulates non-empty cache

  // Should not throw
  CHECK_NOTHROW(clear_all(&ctx));

  // Note: Stub doesn't actually clear, but validates call sequence
  // Real behavior validated in integration tests
}

TEST_CASE("KV: clear_metadata - successful operation") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;  // Simulates non-empty cache

  // Should not throw
  CHECK_NOTHROW(clear_metadata(&ctx));

  // Note: Stub doesn't distinguish metadata-only vs full clear
  // Real behavior validated in integration tests
}

// ===== FILE PERSISTENCE TESTS =====

TEST_CASE("KV: write_file - null context guard") {
  resetStubConfig();

  std::vector<llama_token> tokens = {1, 2, 3};
  size_t bytes = write_file(nullptr, 0, "test.llama", tokens);

  CHECK(bytes == 0);
}

TEST_CASE("KV: write_file - empty filepath guard") {
  resetStubConfig();

  llama_context ctx{};
  std::vector<llama_token> tokens = {1, 2, 3};
  size_t bytes = write_file(&ctx, 0, "", tokens);

  CHECK(bytes == 0);
}

TEST_CASE("KV: write_file - empty KV cache guard") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = -1;  // Empty cache
  std::vector<llama_token> tokens = {1, 2, 3};

  size_t bytes = write_file(&ctx, 0, "test.llama", tokens);

  CHECK(bytes == 0);
}

TEST_CASE("KV: write_file - success") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;  // Non-empty cache
  llamaStubConfig().file_operation_succeeds = true;
  llamaStubConfig().file_write_bytes = 8192;

  std::vector<llama_token> tokens = {1, 2, 3, 4, 5};
  size_t bytes = write_file(&ctx, 0, "test.llama", tokens);

  CHECK(bytes == 8192);
}

TEST_CASE("KV: write_file - operation fails") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().pos_max = 50;
  llamaStubConfig().file_operation_succeeds = false;

  std::vector<llama_token> tokens = {1, 2, 3};
  size_t bytes = write_file(&ctx, 0, "test.llama", tokens);

  CHECK(bytes == 0);
}

TEST_CASE("KV: read_file - null context throws") {
  resetStubConfig();

  CHECK_THROWS_AS(read_file(nullptr, 0, "test.llama"), std::runtime_error);
}

TEST_CASE("KV: read_file - empty filepath throws") {
  resetStubConfig();

  llama_context ctx{};
  CHECK_THROWS_AS(read_file(&ctx, 0, ""), std::runtime_error);
}

TEST_CASE("KV: read_file - success") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().file_operation_succeeds = true;
  llamaStubConfig().file_read_bytes = 8192;
  llamaStubConfig().file_token_count = 5;

  auto data = read_file(&ctx, 0, "test.llama");

  CHECK(data.bytes_read == 8192);
  CHECK(data.tokens.size() == 5);
  // Verify tokens are sequential (100, 101, 102, 103, 104)
  for (size_t i = 0; i < data.tokens.size(); ++i) {
    CHECK(data.tokens[i] == 100 + static_cast<llama_token>(i));
  }
}

TEST_CASE("KV: read_file - operation fails throws") {
  resetStubConfig();

  llama_context ctx{};
  llamaStubConfig().file_operation_succeeds = false;

  CHECK_THROWS_AS(read_file(&ctx, 0, "test.llama"), std::runtime_error);
}

// ===== SEQUENCE COPY TESTS (System 2) =====

TEST_CASE("KV: seq_cp null context guard") {
  resetStubConfig();
  seq_cp(nullptr, 0, 1, 0, 100);
  CHECK_FALSE(llamaStubConfig().seq_cp_called);
}

TEST_CASE("KV: seq_cp parameter propagation") {
  resetStubConfig();
  llama_context ctx{};
  seq_cp(&ctx, 0, 1, 10, 50);
  CHECK(llamaStubConfig().seq_cp_called);
  CHECK(llamaStubConfig().seq_cp_src == 0);
  CHECK(llamaStubConfig().seq_cp_dst == 1);
  CHECK(llamaStubConfig().seq_cp_p0 == 10);
  CHECK(llamaStubConfig().seq_cp_p1 == 50);
}

TEST_CASE("KV: seq_cp default parameters") {
  resetStubConfig();
  llama_context ctx{};
  seq_cp(&ctx, 0, 1);  // p0=0, p1=-1 defaults
  CHECK(llamaStubConfig().seq_cp_called);
  CHECK(llamaStubConfig().seq_cp_p0 == 0);
  CHECK(llamaStubConfig().seq_cp_p1 == -1);
}

// ===== SEQUENCE KEEP TESTS (System 2) =====

TEST_CASE("KV: seq_keep null context guard") {
  resetStubConfig();
  seq_keep(nullptr, 0);
  CHECK_FALSE(llamaStubConfig().seq_keep_called);
}

TEST_CASE("KV: seq_keep single sequence") {
  resetStubConfig();
  llama_context ctx{};
  seq_keep(&ctx, 7);
  CHECK(llamaStubConfig().seq_keep_called);
  CHECK(llamaStubConfig().seq_keep_seq == 7);
}
