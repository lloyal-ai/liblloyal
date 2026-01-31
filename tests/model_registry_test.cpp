#include "llama_stubs.h"
#include <doctest/doctest.h>
#include <lloyal/model_registry.hpp>
#include <memory>

using namespace lloyal;

TEST_CASE("ModelKey: equality and comparison") {
  resetStubConfig();

  SUBCASE("Same values produce equal keys") {
    ModelKey key1{"/path/to/model.gguf", 32, true};
    ModelKey key2{"/path/to/model.gguf", 32, true};

    CHECK(key1 == key2);
  }

  SUBCASE("Different paths produce unequal keys") {
    ModelKey key1{"/path/to/model1.gguf", 32, true};
    ModelKey key2{"/path/to/model2.gguf", 32, true};

    CHECK_FALSE(key1 == key2);
  }

  SUBCASE("Different GPU layers produce unequal keys") {
    ModelKey key1{"/path/to/model.gguf", 32, true};
    ModelKey key2{"/path/to/model.gguf", 16, true};

    CHECK_FALSE(key1 == key2);
  }

  SUBCASE("Different mmap settings produce unequal keys") {
    ModelKey key1{"/path/to/model.gguf", 32, true};
    ModelKey key2{"/path/to/model.gguf", 32, false};

    CHECK_FALSE(key1 == key2);
  }
}

TEST_CASE("ModelKeyHash: deterministic hashing") {
  resetStubConfig();

  ModelKey key1{"/path/to/model.gguf", 32, true};
  ModelKey key2{"/path/to/model.gguf", 32, true};

  ModelKeyHash hasher;

  // Same key should produce same hash
  CHECK(hasher(key1) == hasher(key2));
}

TEST_CASE("ModelRegistry: path normalization") {
  resetStubConfig();

  auto params = llama_model_params{.n_gpu_layers = 32, .use_mmap = true};

  // file:// URI and filesystem path should normalize to same key
  std::string path1 = "file:///tmp/test/model.gguf";
  std::string path2 = "/tmp/test/model.gguf";

  auto model1 = ModelRegistry::acquire(path1, params);
  auto model2 = ModelRegistry::acquire(path2, params);

  REQUIRE(model1 != nullptr);
  REQUIRE(model2 != nullptr);

  // Both should return the same model instance (cache hit)
  CHECK(model1.get() == model2.get());
}

TEST_CASE("ModelRegistry: cache hit behavior") {
  resetStubConfig();

  auto params = llama_model_params{.n_gpu_layers = 32, .use_mmap = true};

  std::string path = "/path/to/model.gguf";

  SUBCASE("Multiple acquires return same model pointer") {
    auto model1 = ModelRegistry::acquire(path, params);
    auto model2 = ModelRegistry::acquire(path, params);

    REQUIRE(model1 != nullptr);
    REQUIRE(model2 != nullptr);

    // Cache hit: same underlying pointer
    CHECK(model1.get() == model2.get());
  }

  SUBCASE("Refcount increases with multiple acquires") {
    auto model1 = ModelRegistry::acquire(path, params);
    REQUIRE(model1 != nullptr);
    CHECK(model1.use_count() == 1);

    auto model2 = ModelRegistry::acquire(path, params);
    REQUIRE(model2 != nullptr);

    // Both references held
    CHECK(model1.use_count() == 2);
    CHECK(model2.use_count() == 2);
  }
}

TEST_CASE("ModelRegistry: cache miss behavior") {
  resetStubConfig();

  std::string path = "/path/to/model.gguf";

  SUBCASE("Different GPU layers produce different models") {
    auto params1 = llama_model_params{.n_gpu_layers = 32, .use_mmap = true};

    auto params2 = llama_model_params{.n_gpu_layers = 16, .use_mmap = true};

    auto model1 = ModelRegistry::acquire(path, params1);
    auto model2 = ModelRegistry::acquire(path, params2);

    REQUIRE(model1 != nullptr);
    REQUIRE(model2 != nullptr);

    // Cache miss: different pointers
    CHECK(model1.get() != model2.get());
  }

  SUBCASE("Different mmap settings produce different models") {
    auto params1 = llama_model_params{.n_gpu_layers = 32, .use_mmap = true};

    auto params2 = llama_model_params{.n_gpu_layers = 32, .use_mmap = false};

    auto model1 = ModelRegistry::acquire(path, params1);
    auto model2 = ModelRegistry::acquire(path, params2);

    REQUIRE(model1 != nullptr);
    REQUIRE(model2 != nullptr);

    // Cache miss: different pointers
    CHECK(model1.get() != model2.get());
  }
}

TEST_CASE("ModelRegistry: reference counting") {
  resetStubConfig();

  auto params = llama_model_params{.n_gpu_layers = 32, .use_mmap = true};

  std::string path = "/path/to/model.gguf";

  SUBCASE("Initial acquire has refcount 1") {
    auto model = ModelRegistry::acquire(path, params);
    REQUIRE(model != nullptr);
    CHECK(model.use_count() == 1);
  }

  SUBCASE("Releasing reference decrements refcount") {
    auto model1 = ModelRegistry::acquire(path, params);
    REQUIRE(model1 != nullptr);

    {
      auto model2 = ModelRegistry::acquire(path, params);
      CHECK(model1.use_count() == 2);
    } // model2 destroyed here

    // Refcount should drop back to 1
    CHECK(model1.use_count() == 1);
  }
}

TEST_CASE("ModelRegistry: automatic cleanup") {
  resetStubConfig();

  auto params = llama_model_params{.n_gpu_layers = 32, .use_mmap = true};

  std::string path = "/path/to/model.gguf";

  std::weak_ptr<llama_model> weak;

  {
    auto model = ModelRegistry::acquire(path, params);
    REQUIRE(model != nullptr);

    weak = model;
    CHECK_FALSE(weak.expired());
  } // model destroyed here, last reference released

  // Model should be freed when last shared_ptr released
  CHECK(weak.expired());
}

TEST_CASE("ModelRegistry: after last release, next acquire makes new model") {
  resetStubConfig();

  auto params = llama_model_params{.n_gpu_layers = 32, .use_mmap = true};

  std::string path = "/path/to/model.gguf";

  std::weak_ptr<llama_model> weak;
  uintptr_t first_addr = 0;

  {
    auto model1 = ModelRegistry::acquire(path, params);
    REQUIRE(model1 != nullptr);
    weak = model1;
    first_addr = reinterpret_cast<uintptr_t>(model1.get());
  } // model1 destroyed, registry should evict entry

  // Verify cleanup
  CHECK(weak.expired());

  // Next acquire should create NEW model (not reuse old pointer)
  auto model2 = ModelRegistry::acquire(path, params);
  REQUIRE(model2 != nullptr);

  uintptr_t second_addr = reinterpret_cast<uintptr_t>(model2.get());
  CHECK(second_addr != first_addr); // New load happened
}

TEST_CASE("ModelRegistry: error handling") {
  resetStubConfig();

  auto params = llama_model_params{.n_gpu_layers = 32, .use_mmap = true};

  std::string path = "/path/to/model.gguf";

  SUBCASE("Returns nullptr on load failure") {
    // Configure stub to fail model loading
    llamaStubConfig().model_load_succeeds = false;

    auto model = ModelRegistry::acquire(path, params);
    CHECK(model == nullptr);
  }
}

TEST_CASE("ModelRegistry: model size reporting") {
  resetStubConfig();

  auto params = llama_model_params{.n_gpu_layers = 32, .use_mmap = true};

  std::string path = "/path/to/model.gguf";

  // Configure stub to return specific model size
  size_t expected_size = 3ULL * 1024 * 1024 * 1024; // 3GB
  llamaStubConfig().model_size = expected_size;

  auto model = ModelRegistry::acquire(path, params);
  REQUIRE(model != nullptr);

  size_t reported_size = llama_model_size(model.get());
  CHECK(reported_size == expected_size);
}
