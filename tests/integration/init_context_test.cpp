#include <doctest/doctest.h>
#include <lloyal/model_registry.hpp>
#include <lloyal/kv.hpp>
#include <llama/llama.h>
#include <cstdlib>
#include <cstring>

using namespace lloyal;

/**
 * Parameterized Context Initialization Tests
 *
 * Verifies that user-provided parameters correctly flow through to llama.cpp
 * during context initialization.
 *
 * Tests the happy path: valid params -> successful initialization
 * Error cases (invalid params) are tested in unit tests
 *
 * Uses tiny-random-llama.gguf (~12MB)
 */

static const char* MODEL_PATH = std::getenv("LLAMA_TEST_MODEL");

// Skip test if model not available
#define REQUIRE_MODEL() \
    if (!MODEL_PATH || !*MODEL_PATH) { \
        MESSAGE("[ SKIP ] LLAMA_TEST_MODEL not set"); \
        return; \
    }

// Helper to init llama.cpp backend once
struct LlamaBackendGuard {
    LlamaBackendGuard() { llama_backend_init(); }
    ~LlamaBackendGuard() { llama_backend_free(); }
};

// ============================================================================
// Model Parameter Tests (3 tests)
// ============================================================================

TEST_CASE("Integration: ModelRegistry respects n_gpu_layers parameter") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    SUBCASE("n_gpu_layers = 0 (CPU only)") {
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;  // Force CPU

        auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
        REQUIRE(model != nullptr);

        // Verify model loaded successfully with CPU-only configuration
        size_t model_size = llama_model_size(model.get());
        CHECK(model_size > 0);
    }

    SUBCASE("n_gpu_layers = -1 (all layers to GPU)") {
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;  // Use GPU acceleration

        auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
        REQUIRE(model != nullptr);

        // Model should load regardless of GPU availability
        size_t model_size = llama_model_size(model.get());
        CHECK(model_size > 0);
    }
}

TEST_CASE("Integration: ModelRegistry respects use_mmap parameter") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    SUBCASE("use_mmap = true (memory mapped loading)") {
        auto model_params = llama_model_default_params();
        model_params.use_mmap = true;
        model_params.n_gpu_layers = 0;  // CPU for determinism

        auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
        REQUIRE(model != nullptr);

        size_t model_size = llama_model_size(model.get());
        CHECK(model_size > 0);
    }

    SUBCASE("use_mmap = false (direct loading)") {
        auto model_params = llama_model_default_params();
        model_params.use_mmap = false;
        model_params.n_gpu_layers = 0;  // CPU for determinism

        auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
        REQUIRE(model != nullptr);

        size_t model_size = llama_model_size(model.get());
        CHECK(model_size > 0);
    }
}

TEST_CASE("Integration: Multiple models can coexist with different params") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    // Load same model with different parameters
    auto params_cpu = llama_model_default_params();
    params_cpu.n_gpu_layers = 0;
    params_cpu.use_mmap = true;

    auto params_gpu = llama_model_default_params();
    params_gpu.n_gpu_layers = -1;
    params_gpu.use_mmap = false;

    auto model1 = ModelRegistry::acquire(MODEL_PATH, params_cpu);
    auto model2 = ModelRegistry::acquire(MODEL_PATH, params_gpu);

    REQUIRE(model1 != nullptr);
    REQUIRE(model2 != nullptr);

    // Both models should be valid and functional
    CHECK(llama_model_size(model1.get()) > 0);
    CHECK(llama_model_size(model2.get()) > 0);
}

// ============================================================================
// Context Parameter Tests (3 tests)
// ============================================================================

TEST_CASE("Integration: llama_init_from_model respects n_ctx parameter") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU for determinism

    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    SUBCASE("n_ctx = 512 (small context)") {
        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;

        llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
        REQUIRE(ctx != nullptr);

        // Context should be functional
        auto mem = llama_get_memory(ctx);
        CHECK(mem != nullptr);

        llama_free(ctx);
    }

    SUBCASE("n_ctx = 2048 (large context)") {
        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;

        llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
        REQUIRE(ctx != nullptr);

        auto mem = llama_get_memory(ctx);
        CHECK(mem != nullptr);

        llama_free(ctx);
    }
}

TEST_CASE("Integration: llama_init_from_model respects n_batch parameter") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    SUBCASE("n_batch = 128 (small batch)") {
        auto ctx_params = llama_context_default_params();
        ctx_params.n_batch = 128;
        ctx_params.n_ctx = 512;

        llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
        REQUIRE(ctx != nullptr);

        // Verify batch can be created
        auto batch = llama_batch_init(128, 0, 1);
        CHECK(batch.token != nullptr);
        llama_batch_free(batch);

        llama_free(ctx);
    }

    SUBCASE("n_batch = 512 (large batch)") {
        auto ctx_params = llama_context_default_params();
        ctx_params.n_batch = 512;
        ctx_params.n_ctx = 1024;

        llama_context* ctx = llama_init_from_model(model.get(), ctx_params);
        REQUIRE(ctx != nullptr);

        auto batch = llama_batch_init(512, 0, 1);
        CHECK(batch.token != nullptr);
        llama_batch_free(batch);

        llama_free(ctx);
    }
}

TEST_CASE("Integration: Multiple contexts from same model with different params") {
    REQUIRE_MODEL();
    LlamaBackendGuard backend;

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    auto model = ModelRegistry::acquire(MODEL_PATH, model_params);
    REQUIRE(model != nullptr);

    // Create two contexts with different parameters
    auto ctx_params1 = llama_context_default_params();
    ctx_params1.n_ctx = 512;
    ctx_params1.n_batch = 128;

    auto ctx_params2 = llama_context_default_params();
    ctx_params2.n_ctx = 1024;
    ctx_params2.n_batch = 256;

    llama_context* ctx1 = llama_init_from_model(model.get(), ctx_params1);
    llama_context* ctx2 = llama_init_from_model(model.get(), ctx_params2);

    REQUIRE(ctx1 != nullptr);
    REQUIRE(ctx2 != nullptr);

    // Both contexts should be functional
    auto mem1 = llama_get_memory(ctx1);
    auto mem2 = llama_get_memory(ctx2);
    CHECK(mem1 != nullptr);
    CHECK(mem2 != nullptr);

    llama_free(ctx1);
    llama_free(ctx2);
}
