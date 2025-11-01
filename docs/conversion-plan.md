# liblloyal C++20 Header-Only Conversion Plan

**Version:** 1.0
**Date:** 2025-11-01
**Status:** In Progress (Phase 2)

## Overview

This document specifies the conversion strategy for extracting core llama.cpp-bound inference logic from `calibrate-ndk` into a shared, header-only C++20 library (`liblloyal`). This eliminates code duplication between the commercial (`calibrate-ndk`) and open-source (`nitro-llama`) React Native shells.

## Design Principles

1. **Header-only INTERFACE target** - No compiled library, only headers propagated via CMake
2. **llama.cpp b6870 pinned** - Version-locked to specific commit for API stability
3. **Zero behavioral divergence** - Both shells execute identical inference logic
4. **C++20 modern idioms** - `constinit`, `std::span`, `std::string_view`, `inline` variables
5. **Clear API boundaries** - Public API in `lloyal::`, internals in `lloyal::detail::`

## Repository Layout

```
packages/
  liblloyal/
    CMakeLists.txt              # INTERFACE target definition
    README.md                   # Architecture and integration guide
    docs/
      conversion-plan.md        # This file
    include/
      lloyal/
        common.hpp              # Logging + constants [✅ DONE]
        helpers.hpp             # Batch utils + chat template [✅ DONE]
        tokenizer.hpp           # Tokenization/detokenization [✅ DONE]
        decoder.hpp             # Batch decode orchestration [✅ DONE]
        kv.hpp                  # KV cache + fragmentation fallback [✅ DONE]
        sampler.hpp             # Sampling with 52 parameters [⏳ TODO]
        grammar.hpp             # Grammar-constrained generation [⏳ TODO]
        model_registry.hpp      # Weak-ptr model cache [⏳ TODO]
        chat_template.hpp       # Jinja2 template formatting [⏳ TODO]
        json-schema-to-grammar.hpp  # JSON schema → GBNF converter [⏳ TODO]
    tests/
      test_logits_mask.cpp      # Smoke test: logits biasing [⏳ TODO]
      test_kv_range.cpp         # Smoke test: KV operations [⏳ TODO]
  calibrate-ndk/
    # Shell-specific: HybridCalibrateContext, TokenLedger, Nitrogen bindings
  nitro-llama/
    # Shell-specific: HybridNitroLlama, Nitrogen bindings
third_party/
  llama.cpp/                    # Built per-shell with platform-specific flags
```

### Why This Structure?

- **Project-prefixed headers** (`include/lloyal/...`) - Standard C++ convention, avoids collisions ([Pitchfork Layout](https://github.com/jgoossen851/cpp-project-template))
- **INTERFACE target** - Headers propagate via `target_include_directories`, no linking ([CMake Docs](https://cmake.org/cmake/help/latest/command/target_include_directories.html))
- **Per-shell llama.cpp builds** - Allows Metal/CUDA/Vulkan/CPU variants without liblloyal recompilation

## CMake Integration

### liblloyal/CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(lloyal LANGUAGES C CXX)

add_library(lloyal INTERFACE)

target_include_directories(lloyal
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Consumers must provide a 'llama' target
target_link_libraries(lloyal INTERFACE llama)

install(TARGETS lloyal EXPORT lloyalTargets)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT lloyalTargets NAMESPACE lloyal:: DESTINATION lib/cmake/lloyal)
```

### Shell Integration

```cmake
# In calibrate-ndk/android/CMakeLists.txt or nitro-llama/android/CMakeLists.txt
add_subdirectory(../llama.cpp build-llama)  # Produces 'llama' target
add_subdirectory(../liblloyal)              # Produces 'lloyal' INTERFACE target

target_link_libraries(calibratendk PRIVATE lloyal llama)
```

**Key Point:** Each shell controls llama.cpp build flags (Metal, CUDA, etc.) independently.

## C++20 Header-Only Conversion Rules

### 1. Function Definitions

**Rule:** All functions must be marked `inline` to prevent ODR violations.

```cpp
// ✅ Correct
inline std::vector<llama_token> tokenize(const llama_vocab* vocab, ...) {
  // implementation
}

// ❌ Wrong (will cause linker errors in multi-TU builds)
std::vector<llama_token> tokenize(const llama_vocab* vocab, ...) {
  // implementation
}
```

**Reference:** [cppreference - inline specifier](https://en.cppreference.com/w/cpp/language/inline)

### 2. Static Variables

**Rule:** Use function-local `static` for singletons/registries (thread-safe since C++11).

```cpp
// ✅ Correct (thread-safe initialization)
inline auto& registry_map() {
  static std::unordered_map<ModelKey, std::weak_ptr<llama_model>> m;
  return m;
}

// ❌ Wrong (ODR violation across TUs)
static std::unordered_map<ModelKey, std::weak_ptr<llama_model>> g_registry;
```

**Reference:** [Stack Overflow - Thread-safe static initialization](https://stackoverflow.com/questions/8102125/is-local-static-variable-initialization-thread-safe-in-c11)

### 3. Constant Tables

**Rule:** Use `inline constexpr` for compile-time constants, `constinit` for runtime-initialized statics.

```cpp
// ✅ Correct (C++17+)
inline constexpr std::array<ggml_type, 9> kv_cache_types = {
  GGML_TYPE_F32, GGML_TYPE_F16, /* ... */
};

// ✅ Correct (C++20, runtime init but must be static init)
inline constinit std::array<const char*, 3> format_names = {
  "uuid", "date-time", "date"
};
```

**Reference:** [cppreference - constinit](https://en.cppreference.com/w/cpp/language/constinit)

### 4. Public API Signatures

**Rule:** Preserve existing facade signatures exactly to enable zero-disruption migration.

```cpp
// ✅ Correct - matches calibrate-ndk contract
inline std::vector<llama_token> tokenize(
  const llama_vocab* vocab,
  const std::string& text,
  bool add_special,
  bool parse_special
);

inline void decode_tokens(
  llama_context* ctx,
  const llama_token* tokens,
  int32_t n_tokens,
  int32_t n_past,
  int32_t n_batch
);

// ❌ Wrong - breaks existing shell code
inline void decode_tokens(
  llama_context* ctx,
  std::span<const llama_token> tokens,  // Would require updating all call sites
  int32_t n_past,
  int32_t n_batch
);
```

**Rationale:** API modernization (std::span, std::string_view) can be added in a future phase after both shells are migrated. The extraction phase must maintain exact contract compatibility.

**Note:** C++20 is required (Nitro modules minimum), but existing signatures don't need modernization to compile.

### 5. Namespace Organization

**Rule:** Public API in `lloyal::`, implementation details in `lloyal::detail::`.

```cpp
// Public API
namespace lloyal::tokenizer {
  inline std::vector<llama_token> tokenize(...);
}

// Internal helpers (not for shell consumption)
namespace lloyal::detail {
  inline void add_tokens_to_batch(...);
  inline std::string common_token_to_piece(...);
}
```

**Reference:** [Stack Overflow - detail namespace convention](https://stackoverflow.com/questions/26546265/what-is-the-detail-namespace-commonly-used-for)

## File-by-File Conversion Specifications

### Phase 1: Infrastructure [✅ COMPLETE]

#### common.hpp
- **Source:** `Logging.h` + `constants.h`
- **Changes:**
  - Renamed `LOG_DEBUG` → `LLOYAL_LOG_DEBUG` (namespace safety)
  - Renamed `margelo::nitro::calibratendk` → `lloyal`
- **Status:** ✅ Complete

#### helpers.hpp
- **Source:** `helpers.h` (already header-only in calibrate-ndk)
- **Changes:**
  - All functions already `inline` ✅
  - Includes batch utilities, chat template helpers, string utils
- **Status:** ✅ Complete
- **TODO:** Move internal helpers to `lloyal::detail::`

#### tokenizer.hpp
- **Source:** `Tokenizer.h` + `Tokenizer.cpp`
- **Changes:** Merged .h and .cpp, all functions marked `inline`
- **Status:** ✅ Complete
- **TODO:** None (signatures preserved exactly)

#### decoder.hpp
- **Source:** `Decoder.h` + `Decoder.cpp`
- **Changes:** Merged .h and .cpp, RAII BatchGuard in anonymous namespace
- **Status:** ✅ Complete
- **TODO:** Move `BatchGuard` to `lloyal::detail::` (signature preserved exactly)

#### kv.hpp
- **Source:** `KV.h` + `KV.cpp`
- **Changes:** Merged .h and .cpp, includes fragmentation fallback logic
- **Status:** ✅ Complete

### Phase 2: Core Inference [⏳ IN PROGRESS]

#### sampler.hpp
- **Source:** `Sampler.h` + `Sampler.cpp` (300 lines)
- **Key Challenges:**
  1. **SamplingParams struct dependency** - Nitrogen-generated per-shell, need generic solution
  2. **Grammar integration** - Sampler accepts optional `llama_sampler*` for constrained decode
  3. **Logits mask contract** - Must only read logits for steps where decoder set `batch.logits[i]=true`

**Implementation Strategy - Template-based (Option 1):**

```cpp
namespace lloyal::detail {
  // Helper for extracting values from optional<T> or T
  template<class T> struct is_optional : std::false_type {};
  template<class T> struct is_optional<std::optional<T>> : std::true_type {};

  template<class X, class T>
  constexpr T as_value(const X& x, T def) {
    if constexpr (is_optional<X>::value) return x.value_or(def);
    else return static_cast<T>(x);
  }
}

namespace lloyal {

// C++20 concept: any type with sampling fields works
template<class P>
concept SamplingParamsLike = requires(const P& p) {
  p.temperature;
  p.top_k;
  p.top_p;
  p.typical_p;
  p.min_p;
  p.penalty_repeat;
  p.penalty_freq;
  p.penalty_present;
  p.penalty_last_n;
  p.seed;
  // ... (20 parameters total)
};

namespace sampler {

// Greedy sampling (simple argmax)
inline llama_token greedy(llama_context* ctx, const llama_vocab* vocab);

// Full parameterized sampling (template accepts any shell's SamplingParams)
template<SamplingParamsLike P>
inline llama_token sample_with_params(
  llama_context* ctx,
  const llama_vocab* vocab,
  const P& params,                       // Works with Nitrogen-generated types
  llama_sampler* grammarSampler = nullptr
);

} // namespace sampler
} // namespace lloyal
```

**Why template approach:**
- No struct duplication between liblloyal and shells
- No adapter functions needed in shell code
- Works with any Nitrogen-generated SamplingParams (any namespace)
- C++20 concept provides clear compile-time errors if fields missing
- Templates stay in C++ boundary, never exposed to Swift ([Swift C++ Interop](https://swift.org/documentation/cxx-interop/))

**Shell integration (unchanged):**
```cpp
// HybridCalibrateContext.cpp
#include <lloyal/sampler.hpp>

// Direct call - template instantiates for nitrogen::SamplingParams
auto token = lloyal::sampler::sample_with_params(
  ctx, vocab, nitrogenParams, grammarSampler
);
```

**Sampling Chain Order (must preserve):**
1. Grammar mask (if present) - limits candidates before other filters
2. Repetition penalties (frequency, presence, repeat)
3. Top-k / Top-p / Typical-p / Min-p filtering
4. Temperature scaling
5. Final token selection

**Logits Contract:** Sampler reads `llama_get_logits_ith(ctx, -1)` which only works if decoder marked last token with `logits=true`. Current decoder.hpp already does this correctly (line 42).

**Template Implementation Details:**
- `detail::as_value<X, T>(x, def)` handles both `optional<T>` and `T` uniformly
- Concept `SamplingParamsLike` enforces required fields at compile-time
- Template instantiates separately for each shell's Nitrogen type (zero runtime cost)
- Header-only templates avoid ODR issues ([Why templates in headers](https://www.geeksforgeeks.org/dsa/why-can-templates-only-be-implemented-in-the-header-file/))

**References:**
- [llama.cpp issue #9224 - logits mask validation](https://github.com/ggerganov/llama.cpp/issues/9224)
- [C++20 Concepts](https://en.cppreference.com/w/cpp/language/constraints)
- [Swift C++ Interop - Templates not exposed](https://swift.org/documentation/cxx-interop/)

**Status:** ⏳ TODO

#### grammar.hpp
- **Source:** `Grammar.h` + `Grammar.cpp` (58 lines, very thin wrapper)
- **API:**
  - `init_grammar(model, grammar_str)` → returns `llama_sampler*`
  - `accept_token(sampler, token)` → updates grammar state after sampling
  - `reset_grammar(sampler)` → clears grammar state

**Key Invariant:** Grammar must be primed with prompt tokens before first sample. Current implementation does this in Grammar.cpp:34-38.

**Call Order in Sampler:**
```cpp
// Grammar applied FIRST, then other filters
if (grammarSampler) {
  llama_sampler_apply(grammarSampler, &candidates);  // Mask invalid tokens
}
// Then top-k, top-p, temperature, etc.
```

**Reference:** [llama.cpp grammars README](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)

**Status:** ⏳ TODO

#### model_registry.hpp
- **Source:** `ModelRegistry.h` + `ModelRegistry.cpp` (84 lines)
- **Purpose:** Thread-safe weak-pointer cache to avoid reloading same model multiple times
- **Key:** `(canonPath, n_gpu_layers, use_mmap)`
- **Architecture:** Class-based with static members (NOT namespace with free functions)

**Implementation Strategy (source-verified):**

```cpp
namespace lloyal {

/**
 * Model cache key combining file path and GPU configuration
 * SOURCE: ModelRegistry.h:22-32
 */
struct ModelKey {
  std::string canonPath;  // Normalized file path (file:// prefix removed)
  int n_gpu_layers;       // Number of layers offloaded to GPU (-1 = all)
  bool use_mmap;          // Whether to use memory mapping

  bool operator==(const ModelKey& o) const {
    return n_gpu_layers == o.n_gpu_layers &&
           use_mmap == o.use_mmap &&
           canonPath == o.canonPath;
  }
};

/**
 * Hash function for ModelKey
 * SOURCE: ModelRegistry.h:38-46
 */
struct ModelKeyHash {
  size_t operator()(const ModelKey& k) const {
    std::hash<std::string> Hs;
    std::hash<int> Hi;
    std::hash<bool> Hb;
    return Hs(k.canonPath) ^ (Hi(k.n_gpu_layers) + 0x9e3779b9 + (Hb(k.use_mmap) << 6));
  }
};

/**
 * Thread-safe registry for sharing llama_model instances
 * SOURCE: ModelRegistry.h:72-120
 *
 * IMPORTANT: This is a CLASS with static members, not a namespace.
 * Converting to header-only requires inline static members.
 */
class ModelRegistry {
public:
  /**
   * Acquire a model from cache or load if not present
   * SOURCE: ModelRegistry.h:93-96
   *
   * @param fsPath Filesystem path to model file (file:// prefix normalized)
   * @param params Model load parameters (GPU layers, mmap, etc.)
   * @return shared_ptr to model, or nullptr if load failed
   */
  static std::shared_ptr<llama_model> acquire(
    const std::string& fsPath,
    const llama_model_params& params
  );

private:
  /**
   * Global cache mutex - inline static for header-only
   * SOURCE: ModelRegistry.h:103
   */
  inline static std::mutex mu_;

  /**
   * Model cache - inline static for header-only
   * SOURCE: ModelRegistry.h:113
   */
  inline static std::unordered_map<ModelKey, std::weak_ptr<llama_model>, ModelKeyHash> cache_;

  /**
   * Create cache key from path and parameters (private helper)
   * SOURCE: ModelRegistry.h:119
   */
  static ModelKey makeKey(const std::string& fsPath, const llama_model_params& params);
};

} // namespace lloyal
```

**Key Conversions:**
- Static members → `inline static` (C++17 feature for header-only classes)
- Implementation moves to header with `inline` on all methods
- Thread-safe via `std::mutex` (already present in source)
- Custom deleter for `shared_ptr` frees model when last reference drops

**References:**
- [Inline static members (C++17)](https://en.cppreference.com/w/cpp/language/static)
- Source: ModelRegistry.cpp:11-84 for implementation details

**Status:** ⏳ TODO

#### chat_template.hpp
- **Source:** `ChatTemplate.h` + `ChatTemplate.cpp` (70 lines)
- **Purpose:** Orchestrates chat template formatting with fallback handling
- **Architecture:** Wraps helpers.hpp functions, adds error handling layer
- **Status:** Separate header required (NOT just re-exports)

**Implementation Strategy (source-verified):**

```cpp
#include "helpers.hpp"
#include <llama/llama.h>
#include <string>
#include <vector>

namespace lloyal::chat_template {

/**
 * Result from chat template formatting
 * SOURCE: ChatTemplate.h:24-28
 * NOTE: Named FormatResult, NOT ChatTemplateResult
 */
struct FormatResult {
  std::string prompt;                     // Formatted prompt text
  std::vector<std::string> additional_stops;  // Stop tokens from template
};

/**
 * Format chat messages using model's chat template with fallback
 * SOURCE: ChatTemplate.h:51-55
 *
 * Orchestration logic:
 * 1. Calls format_chat_template_complete() from helpers.hpp
 * 2. If template processing fails (empty prompt), falls back to simple format
 * 3. Handles JSON parsing errors
 *
 * Fallback hierarchy:
 * 1. template_override (if provided)
 * 2. Model's built-in template
 * 3. ChatML template
 * 4. Simple "role: content" format (this layer adds this)
 *
 * @param model Llama model (for template and vocab)
 * @param messages_json JSON string with messages array
 * @param template_override Optional custom template
 * @return FormatResult with formatted prompt and stop tokens
 */
inline FormatResult format(
  const llama_model* model,
  const std::string& messages_json,
  const std::string& template_override = ""
);

/**
 * Validate chat template syntax
 * SOURCE: ChatTemplate.h:68
 *
 * Calls validate_chat_template_helper() from helpers.hpp.
 * Does NOT require a model (syntax-only validation).
 *
 * @param template_str Template string to validate
 * @return True if syntax is valid, false otherwise (never throws)
 */
inline bool validate(const std::string& template_str);

} // namespace lloyal::chat_template
```

**Key Points:**
- **NOT a re-export** - Adds orchestration and fallback logic
- Struct is `FormatResult` (not `ChatTemplateResult`)
- Function is `format()` (not `format_chat_template_from_model()`)
- Wraps helpers.hpp functions: `format_chat_template_complete()`, `validate_chat_template_helper()`
- Implementation in ChatTemplate.cpp:11-70 shows fallback to simple format

**Why separate header:**
- Source has dedicated ChatTemplate.{h,cpp} (not in helpers)
- Adds error handling and fallback beyond helpers.hpp
- Public API differs from helpers.hpp internal functions

**References:**
- Source: ChatTemplate.h:20-70
- Source: ChatTemplate.cpp:11-70 (implementation with fallbacks)

**Status:** ⏳ TODO

#### json-schema-to-grammar.hpp
- **Source:** `json-schema-to-grammar.h` + `json-schema-to-grammar.cpp` (1011 lines)
- **Purpose:** Convert JSON schema to GBNF grammar for constrained generation
- **Complexity:** Large, many helper functions and constant tables

**Implementation Strategy:**

```cpp
namespace lloyal {

// Public API
inline std::string json_schema_to_grammar(const json& schema, bool force_gbnf = false);
inline std::string build_grammar(
  const std::function<void(const common_grammar_builder&)>& cb,
  const common_grammar_options& options = {}
);

namespace detail {
  // All internal helpers go here
  inline std::string parse_pattern(const std::string& pattern);
  inline std::string build_object_rule(const json& schema);
  // ... ~30 helper functions

  // Constant tables (C++17 inline or C++20 constinit)
  inline constexpr std::array<const char*, 3> date_formats = {
    "uuid", "date-time", "date"
  };
}

} // namespace lloyal
```

**Key Conversions:**
- All free functions → `inline`
- Helper functions → `lloyal::detail::`
- Static const tables → `inline constexpr` or `constinit`
- Uses `string_repeat`, `string_join`, `string_split` from helpers.hpp

**Status:** ⏳ TODO (largest single file)

### Phase 3: Dependencies [⏳ TODO]

#### Vendored Libraries

**Copy from calibrate-ndk to liblloyal:**

1. **minja/** (Jinja2 template engine for chat templates)
   - Source: `calibrate-ndk/src/minja/`
   - Destination: `liblloyal/include/lloyal/minja/`
   - Files: `chat-template.hpp`, `minja.hpp`

2. **nlohmann/json.hpp** (JSON library)
   - Source: `calibrate-ndk/src/nlohmann/`
   - Destination: `liblloyal/include/lloyal/nlohmann/`
   - Files: `json.hpp`, `json_fwd.hpp`
   - Note: Use `nlohmann::ordered_json` for schema processing (preserves key order)

**Forward Declaration Strategy:**

```cpp
// In each header that needs JSON
#include <lloyal/nlohmann/json.hpp>

namespace lloyal {
  using json = nlohmann::ordered_json;
}
```

### Phase 4: Build System [⏳ TODO]

#### CMakeLists.txt Features

```cmake
cmake_minimum_required(VERSION 3.20)
project(lloyal LANGUAGES C CXX VERSION 1.0.0)

add_library(lloyal INTERFACE)

target_compile_features(lloyal INTERFACE cxx_std_20)

target_include_directories(lloyal
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Consumers must provide 'llama' target (built per-shell with platform flags)
target_link_libraries(lloyal INTERFACE llama)

# Optional: Version check for llama.cpp b6870
target_compile_definitions(lloyal INTERFACE LLOYAL_LLAMA_VERSION_REQUIRED=b6870)

# Installation
install(TARGETS lloyal EXPORT lloyalTargets)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT lloyalTargets
  NAMESPACE lloyal::
  DESTINATION lib/cmake/lloyal
  FILE lloyalConfig.cmake)
```

**Version Pinning:** Build-time check in `common.hpp`:

```cpp
#ifdef LLAMA_BUILD_VERSION
  // Extract commit from LLAMA_BUILD_VERSION and compare to b6870
  #if LLAMA_BUILD_VERSION != "b6870"
    #warning "lloyal is tested against llama.cpp b6870, current version may have API changes"
  #endif
#endif
```

### Phase 5: Testing [⏳ TODO]

#### Smoke Tests

**test_logits_mask.cpp** - Validates logits biasing operations

```cpp
#include <lloyal/tokenizer.hpp>
#include <lloyal/decoder.hpp>
#include <lloyal/sampler.hpp>

// Test: Decoder sets logits mask, sampler reads it
// Validates fix for llama.cpp issue #9224
void test_logits_mask_contract() {
  // Setup: model, context, tokens
  // Decode tokens (decoder sets logits[last]=true)
  // Sample (should succeed, not throw "invalid logits id")
  // Assert: No crash, valid token returned
}
```

**test_kv_range.cpp** - Validates KV cache range operations

```cpp
#include <lloyal/kv.hpp>

// Test: Remove range, check pos_max, state save/load
void test_kv_operations() {
  // Decode 100 tokens
  // Remove range [50, 75)
  // Assert: pos_max == 99, range [50,75) is empty
  // Save state
  // Clear cache
  // Load state
  // Assert: pos_max == 99, range [50,75) still empty
}
```

**test_grammar_constraint.cpp** - Validates grammar-constrained sampling

```cpp
#include <lloyal/grammar.hpp>
#include <lloyal/sampler.hpp>
#include <lloyal/json-schema-to-grammar.hpp>

// Test: JSON schema → GBNF → constrained sample
void test_json_schema_grammar() {
  json schema = {{"type", "object"}, {"properties", { /*...*/ }}};
  std::string gbnf = lloyal::json_schema_to_grammar(schema);
  auto* sampler = lloyal::grammar::init_grammar(model, gbnf);

  // Sample should only produce valid JSON tokens
  // Assert: Output parses as valid JSON matching schema
}
```

**Build integration:**

```cmake
enable_testing()

add_executable(test_logits_mask tests/test_logits_mask.cpp)
target_link_libraries(test_logits_mask PRIVATE lloyal llama)
add_test(NAME logits_mask COMMAND test_logits_mask)

add_executable(test_kv_range tests/test_kv_range.cpp)
target_link_libraries(test_kv_range PRIVATE lloyal llama)
add_test(NAME kv_range COMMAND test_kv_range)

add_executable(test_grammar_constraint tests/test_grammar_constraint.cpp)
target_link_libraries(test_grammar_constraint PRIVATE lloyal llama)
add_test(NAME grammar_constraint COMMAND test_grammar_constraint)
```

### Phase 6: calibrate-ndk Integration [⏳ TODO]

**Changes to calibrate-ndk:**

1. **CMakeLists.txt:**
   ```cmake
   # Remove old source files
   # add_library(calibratendk SHARED
   #   ../src/Tokenizer.cpp
   #   ../src/Decoder.cpp
   #   ../src/KV.cpp
   #   ../src/Sampler.cpp
   #   # ... etc
   # )

   # Add liblloyal
   add_subdirectory(../../liblloyal)
   target_link_libraries(calibratendk PRIVATE lloyal llama)
   ```

2. **HybridCalibrateContext.cpp:**
   ```cpp
   // Old:
   // #include "Tokenizer.h"
   // #include "Decoder.h"

   // New:
   #include <lloyal/tokenizer.hpp>
   #include <lloyal/decoder.hpp>
   #include <lloyal/kv.hpp>
   #include <lloyal/sampler.hpp>
   #include <lloyal/grammar.hpp>

   // Namespace usage:
   using namespace lloyal;

   // NO adapter needed - template accepts Nitrogen type directly
   auto token = lloyal::sampler::sample_with_params(
     ctx, vocab, nitrogenSamplingParams, grammarSampler
   );
   // Template instantiates for margelo::nitro::calibratendk::SamplingParams
   ```

3. **Delete duplicate files:**
   ```bash
   rm calibrate-ndk/src/Tokenizer.{h,cpp}
   rm calibrate-ndk/src/Decoder.{h,cpp}
   rm calibrate-ndk/src/KV.{h,cpp}
   rm calibrate-ndk/src/Sampler.{h,cpp}
   rm calibrate-ndk/src/Grammar.{h,cpp}
   rm calibrate-ndk/src/ModelRegistry.{h,cpp}
   rm calibrate-ndk/src/ChatTemplate.{h,cpp}
   rm calibrate-ndk/src/json-schema-to-grammar.{h,cpp}
   rm calibrate-ndk/src/helpers.h
   rm calibrate-ndk/src/constants.h
   rm -r calibrate-ndk/src/minja/
   rm -r calibrate-ndk/src/nlohmann/
   ```

4. **Keep shell-specific files:**
   - `HybridCalibrateContext.{hpp,cpp}` - Nitrogen bindings
   - `HybridCalibrateNdk.{hpp,cpp}` - Nitrogen bindings
   - `TokenLedger.{h,cpp}` - Commercial feature (NOT in liblloyal)
   - `Embeddings.{h,cpp}` - Uses HybridCalibrateContext, shell-specific
   - `LlamaBackendManager.{h,cpp}` - Shell initialization
   - `Platform.h` - Platform detection helpers
   - `ErrorHandler.h`, `ErrorFormatting.h` - Error handling
   - `FileSystem.h` - Platform file ops

### Phase 7: nitro-llama Integration [⏳ TODO]

**Same as calibrate-ndk, minus TokenLedger:**

1. **CMakeLists.txt** (Android) and **NitroLlama.podspec** (iOS):
   ```cmake
   add_subdirectory(../../liblloyal)
   target_link_libraries(nitrollama PRIVATE lloyal llama)
   ```

2. **HybridNitroLlama.cpp:**
   ```cpp
   #include <lloyal/tokenizer.hpp>
   #include <lloyal/decoder.hpp>
   #include <lloyal/kv.hpp>
   #include <lloyal/sampler.hpp>
   #include <lloyal/grammar.hpp>

   using namespace lloyal;

   // NO adapter needed - template accepts Nitrogen type directly
   auto token = lloyal::sampler::sample_with_params(
     ctx, vocab, nitrogenSamplingParams, grammarSampler
   );
   // Template instantiates for margelo::nitro::nitrollama::SamplingParams
   ```

3. **Delete duplicate files** (same list as calibrate-ndk, no TokenLedger to keep)

### Phase 8: Validation [⏳ TODO]

**Build Tests:**

1. **calibrate-ndk iOS:** `cd calibrate-ndk/example && npx react-native run-ios`
2. **calibrate-ndk Android:** `cd calibrate-ndk/example && npx react-native run-android`
3. **nitro-llama iOS:** `cd nitro-llama/example && npx react-native run-ios`
4. **nitro-llama Android:** `cd nitro-llama/example && npx react-native run-android`

**Smoke Tests:**

```bash
cd packages/liblloyal/build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
ctest --output-on-failure
```

**Integration Tests:**

Run existing app-level tests in both shells to ensure no regressions:
- Token generation correctness
- Grammar-constrained output validates against schema
- KV cache operations preserve model state
- Model registry correctly deduplicates loads

## Refactoring Existing Headers [✅ COMPLETE]

All 5 completed headers have been refactored to use `lloyal::detail` namespace for internals. **All signatures preserved exactly.**

### tokenizer.hpp ✅

**Status:** No changes needed (no internal helpers exposed, signatures already correct).

### decoder.hpp ✅

**Completed refactoring:**
- Moved `struct BatchGuard` from anonymous namespace → `lloyal::detail`
- Moved `add_tokens_to_batch()` from anonymous namespace → `lloyal::detail`
- Updated 2 call sites to use `detail::` prefix

**Public API signatures:** UNCHANGED (both `decode_tokens()` overloads preserved)

### helpers.hpp ✅

**Completed refactoring:**
- Moved 4 internal helpers to `lloyal::detail`:
  - `common_token_to_piece()` - Token conversion
  - `get_token_safe()` - Safe token extraction
  - `get_chatml_template()` - Template constant
  - `apply_chat_template_helper()` - Minja engine wrapper
- Updated 8 references in public functions to use `detail::` prefix

**Public API preserved in `lloyal::`:**
- `batch_clear()`, `batch_add()` - Used by decoder
- `ChatTemplateResult` struct
- `format_chat_template_from_model()` - Will be used by chat_template.hpp
- `extract_template_stop_tokens()` - Will be used by chat_template.hpp
- `format_chat_template_complete()` - Will be used by chat_template.hpp
- `validate_chat_template_helper()` - Will be used by chat_template.hpp
- All parameter conversion and string utility functions

### kv.hpp ✅

**Status:** No changes needed (already clean, no internal helpers exposed, signatures correct).

### common.hpp ✅

**Status:** No changes needed (logging macros and constants, no internal helpers).

## Success Criteria

- [ ] All headers compile with `-std=c++20 -Wall -Wextra -Werror`
- [ ] No ODR violations (verified with multiple TU test)
- [ ] Both shells build successfully on iOS and Android
- [ ] Smoke tests pass (logits mask, KV range, grammar constraint)
- [ ] Zero behavioral divergence (same prompts → same outputs in both shells)
- [ ] Code reduction: ~3000 lines removed from shells (moved to liblloyal)

## References

1. [CMake INTERFACE libraries](https://cmake.org/cmake/help/latest/command/add_library.html#interface-libraries)
2. [Pitchfork Layout](https://github.com/jgoossen851/cpp-project-template)
3. [C++11 thread-safe static init](https://stackoverflow.com/questions/8102125/is-local-static-variable-initialization-thread-safe-in-c11)
4. [std::span documentation](https://en.cppreference.com/w/cpp/container/span)
5. [constinit specifier (C++20)](https://en.cppreference.com/w/cpp/language/constinit)
6. [detail namespace convention](https://stackoverflow.com/questions/26546265/what-is-the-detail-namespace-commonly-used-for)
7. [llama.cpp logits mask issue #9224](https://github.com/ggerganov/llama.cpp/issues/9224)
8. [llama.cpp grammars README](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)

## Timeline

- **Phase 1 (Infrastructure):** ✅ Complete (2025-11-01)
- **Phase 1.5 (Contract Verification):** ✅ Complete (2025-11-01)
- **Phase 1.75 (Refactor Existing Headers):** ✅ Complete (2025-11-01)
- **Phase 2 (Core Inference):** ✅ Complete (2025-11-01)
- **Phase 3 (Vendored Dependencies):** ✅ Complete (2025-11-01)
- **Phase 4 (Build System):** ✅ Complete (2025-11-01)
- **Phase 5 (Testing):** ⏳ TODO (ETA: 20 mins)
- **Phase 6 (calibrate-ndk):** ⏳ TODO (ETA: 15 mins)
- **Phase 7 (nitro-llama):** ⏳ TODO (ETA: 15 mins)
- **Phase 8 (Validation):** ⏳ TODO (ETA: 30 mins)

**Total Estimated Time:** ~2.5 hours
**Time Spent:** ~1.6 hours (Phases 1-4)
**Remaining:** ~0.9 hours (Phases 5-8)

---

## Implementation Status Checklist

### Phase 1: Package Structure ✅ COMPLETE
- [x] Create `packages/liblloyal/` directory
- [x] Create `packages/liblloyal/include/lloyal/` directory
- [x] Create `packages/liblloyal/docs/` directory
- [x] Create `packages/liblloyal/README.md`

### Phase 1.5: Contract Verification ✅ COMPLETE
- [x] Read all source files from calibrate-ndk
- [x] Document exact API signatures in contracts-verified.md
- [x] Verify tokenizer signatures (Tokenizer.h:34-116)
- [x] Verify decoder signatures (Decoder.h:41-57)
- [x] Verify sampler signatures (Sampler.h:35-62)
- [x] Verify kv signatures (KV.h:32-112)
- [x] Verify grammar signatures (Grammar.h:38)
- [x] Verify model_registry signatures (ModelRegistry.h:22-120)
- [x] Verify chat_template signatures (ChatTemplate.h:24-70)
- [x] Fix discrepancies in conversion-plan.md (ModelRegistry, ChatTemplate, ModelKey)

### Phase 1.75: Refactor Existing Headers ✅ COMPLETE
- [x] Refactor decoder.hpp (move BatchGuard, add_tokens_to_batch to detail::)
- [x] Refactor helpers.hpp (move 4 internal helpers to detail::)
- [x] Verify tokenizer.hpp (no changes needed)
- [x] Verify kv.hpp (no changes needed)
- [x] Verify common.hpp (no changes needed)

### Phase 2: Core Inference Headers ✅ COMPLETE

#### Completed Headers (10/10)
- [x] common.hpp - Logging macros + constants (62 lines)
- [x] helpers.hpp - Batch utils + chat template helpers (374 lines)
- [x] tokenizer.hpp - Tokenization/detokenization (251 lines)
- [x] decoder.hpp - Batch decode orchestration (136 lines)
- [x] kv.hpp - KV cache operations (252 lines)
- [x] sampler.hpp - Sampling with C++20 concept template (372 lines)
  - ✅ Template-based `sample_with_params<P>()` with SamplingParamsLike concept
  - ✅ Greedy sampling function
  - ✅ Logits mask contract enforcement
  - ✅ Grammar integration hook
- [x] grammar.hpp - Grammar-constrained generation (67 lines)
  - ✅ `from_json_schema()` function (wraps json-schema-to-grammar.hpp)
  - ✅ Error handling and logging
- [x] model_registry.hpp - Thread-safe model cache (190 lines)
  - ✅ `ModelKey` struct with canonPath field
  - ✅ `ModelKeyHash` struct
  - ✅ `ModelRegistry` class with inline static members
  - ✅ Static `acquire()` method
  - ✅ Inline path normalization (no FileSystem.h dependency)
- [x] chat_template.hpp - Chat template orchestration (123 lines)
  - ✅ `FormatResult` struct (NOT ChatTemplateResult)
  - ✅ `format()` function (wraps helpers.hpp with fallback)
  - ✅ `validate()` function (wraps helpers.hpp)
  - ✅ Fallback error handling
- [x] json-schema-to-grammar.hpp - JSON schema → GBNF converter (scaffold complete)
  - ✅ `json_schema_to_grammar()` public function
  - ✅ `build_grammar()` public function
  - ✅ `SchemaConverter` class with all methods in `lloyal::detail`
  - ✅ Constant tables with `inline const` (PRIMITIVE_RULES, STRING_FORMAT_RULES)
  - ✅ All internal helpers in `lloyal::detail`

**Phase 2 Progress:** ✅ 10/10 headers complete (~2600 lines total)

### Phase 3: Vendored Dependencies ✅ COMPLETE
- [x] Copy `calibrate-ndk/src/minja/` → `liblloyal/include/lloyal/minja/`
  - [x] chat-template.hpp (550 lines)
  - [x] minja.hpp (3,009 lines)
- [x] Copy `calibrate-ndk/src/nlohmann/` → `liblloyal/include/lloyal/nlohmann/`
  - [x] json.hpp (25,526 lines)
  - [x] json_fwd.hpp (187 lines)
- [x] Fix include path in chat-template.hpp (`#include <nlohmann/json.hpp>`)

**Total vendored code:** 29,272 lines (minja: 3,559, nlohmann: 25,713)

**Licenses:**
- minja: MIT License (Google LLC, 2024)
- nlohmann/json: v3.12.0, MIT License (Niels Lohmann, 2013-2025)

### Phase 4: Build System ✅ COMPLETE
- [x] Create `packages/liblloyal/CMakeLists.txt`
  - [x] Define INTERFACE library target
  - [x] Set C++20 requirement
  - [x] Configure include directories
  - [x] Add llama.cpp dependency
  - [x] Add version check for b6870
  - [x] Configure installation
- [x] Create `cmake/liblloyal-config.cmake.in` package config template

### Phase 5: Testing ⏳ TODO
- [ ] Create `packages/liblloyal/tests/` directory
- [ ] Implement `test_logits_mask.cpp`
  - [ ] Verify decoder sets batch.logits[last]=true
  - [ ] Verify sampler reads logits correctly
  - [ ] Test for "invalid logits id" error prevention
- [ ] Implement `test_kv_range.cpp`
  - [ ] Test remove_range()
  - [ ] Test pos_max()
  - [ ] Test state save/load
  - [ ] Test fragmentation fallback
- [ ] Implement `test_grammar_constraint.cpp`
  - [ ] Test JSON schema → GBNF conversion
  - [ ] Test grammar-constrained sampling
  - [ ] Verify output validates against schema
- [ ] Add CMake test configuration

### Phase 6: calibrate-ndk Integration ⏳ TODO
- [ ] Update `calibrate-ndk/android/CMakeLists.txt`
  - [ ] Add `add_subdirectory(../../liblloyal)`
  - [ ] Add `target_link_libraries(calibratendk PRIVATE lloyal llama)`
  - [ ] Remove old source files from build
- [ ] Update `calibrate-ndk/src/HybridCalibrateContext.cpp`
  - [ ] Replace old includes with `<lloyal/*.hpp>`
  - [ ] Add `using namespace lloyal;`
  - [ ] Verify template instantiation works
- [ ] Delete duplicate source files
  - [ ] Remove Tokenizer.{h,cpp}
  - [ ] Remove Decoder.{h,cpp}
  - [ ] Remove KV.{h,cpp}
  - [ ] Remove Sampler.{h,cpp}
  - [ ] Remove Grammar.{h,cpp}
  - [ ] Remove ModelRegistry.{h,cpp}
  - [ ] Remove ChatTemplate.{h,cpp}
  - [ ] Remove json-schema-to-grammar.{h,cpp}
  - [ ] Remove helpers.h
  - [ ] Remove constants.h
  - [ ] Remove minja/
  - [ ] Remove nlohmann/
- [ ] Keep shell-specific files
  - [ ] TokenLedger.{h,cpp} (commercial feature)
  - [ ] HybridCalibrateContext.{hpp,cpp}
  - [ ] Embeddings.{h,cpp}
  - [ ] LlamaBackendManager.{h,cpp}
  - [ ] Platform.h, ErrorHandler.h, FileSystem.h

### Phase 7: nitro-llama Integration ⏳ TODO
- [ ] Update `nitro-llama/android/CMakeLists.txt`
  - [ ] Add `add_subdirectory(../../liblloyal)`
  - [ ] Add `target_link_libraries(nitrollama PRIVATE lloyal llama)`
- [ ] Update `nitro-llama/ios/NitroLlama.podspec`
  - [ ] Add liblloyal include path
- [ ] Update `nitro-llama/src/HybridNitroLlama.cpp`
  - [ ] Replace old includes with `<lloyal/*.hpp>`
  - [ ] Add `using namespace lloyal;`
- [ ] Delete duplicate source files (same list as calibrate-ndk, no TokenLedger)

### Phase 8: Validation ⏳ TODO
- [ ] Build tests
  - [ ] calibrate-ndk iOS: `cd calibrate-ndk/example && npx react-native run-ios`
  - [ ] calibrate-ndk Android: `cd calibrate-ndk/example && npx react-native run-android`
  - [ ] nitro-llama iOS: `cd nitro-llama/example && npx react-native run-ios`
  - [ ] nitro-llama Android: `cd nitro-llama/example && npx react-native run-android`
- [ ] Smoke tests
  - [ ] `cd packages/liblloyal/build && cmake .. && make && ctest`
- [ ] Integration tests
  - [ ] Token generation correctness
  - [ ] Grammar-constrained output validation
  - [ ] KV cache state preservation
  - [ ] Model registry deduplication

**Overall Progress:** 5/8 phases complete (Phases 1, 1.5, 1.75, 2, 3 done)
