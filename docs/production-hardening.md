# Production Hardening Checklist

This document tracks hardening tasks required before production release of liblloyal.

## Status Legend

- **Done**: Implemented and tested
- **Partial**: Some coverage, needs expansion
- **TODO**: Not yet implemented

---

## 1. Memory Safety

| Task | Status | Notes |
|------|--------|-------|
| AddressSanitizer in CI | Done | `sanitizer-tests` job in tests.yml |
| LeakSanitizer in CI | Done | `leak-tests` job (Linux only, macOS ARM lacks LSan) |
| Use-after-free tests | Done | Handle invalidation tests in branch_test.cpp |
| Double-free tests | Done | RAII destructor tests |
| Buffer overflow tests | Done | Covered by ASan |
| Stack buffer overflow | Done | Covered by ASan |

### Gaps

- **MemorySanitizer (MSan)**: Detects uninitialized memory reads. Not enabled because it requires entire toolchain (libc++) to be MSan-instrumented. Consider for dedicated CI image.

---

## 2. Numeric Safety

| Task | Status | Notes |
|------|--------|-------|
| Signed integer overflow | Done | UBSan `-fsanitize=undefined` |
| Unsigned integer overflow | Done | `-fsanitize=integer` catches implicit truncation |
| size_t underflow patterns | Done | Fixed `for (size_t i = n; i-- > 0;)` pattern |
| uint16_t generation wrap | Done | Test verifies wrap-around is safe |
| Float division by zero | Done | `-fsanitize=float-divide-by-zero` |
| Float overflow/underflow | Partial | Numerically stable softmax implemented |
| NaN/Inf propagation | Done | Tests verify `isinf()` checks before `log()` |

### Implementation Details

**size_t underflow fix** (was infinite loop):
```cpp
// WRONG: infinite loop when i wraps to SIZE_MAX
for (size_t i = n - 1; i >= 0; --i) { ... }

// CORRECT: post-decrement idiom
for (size_t i = n; i-- > 0; ) { ... }
```

**Integer sanitizer note**: Standard UBSan doesn't catch unsigned overflow (it's defined behavior in C++). Use `-fsanitize=integer` to catch implicit truncation bugs like:
```cpp
uint16_t gen = 0xFFFF;
gen = gen + 1;  // Promoted to int (65536), truncates to 0 - caught by integer sanitizer
```

---

## 3. Input Validation

| Task | Status | Notes |
|------|--------|-------|
| Invalid handle rejection | Done | `get()` returns nullptr for invalid handles |
| Stale handle rejection | Done | Generation mismatch check |
| Reserved slot 0 protection | Done | Explicit `if (index == 0) return nullptr` |
| Empty input handling | Done | `decode()` with 0 tokens throws, tested |
| Null pointer checks | Partial | Critical paths covered |
| Capacity bounds | Done | BranchStore grow() tested |

---

## 4. Exception Safety

| Task | Status | Notes |
|------|--------|-------|
| RAII resource cleanup | Done | Branch destructor calls `prune()` |
| No-throw destructors | Done | Destructors don't throw |
| Self-move-assign safety | Done | `Branch` handles self-assignment |
| Partial construction cleanup | Partial | Needs review for multi-resource ctors |

### RAII Pattern

```cpp
~Branch() {
    if (handle_ != INVALID_HANDLE && store_) {
        branch::prune(handle_, store_);  // Clean up KV cache
        handle_ = INVALID_HANDLE;
    }
}
```

**Critical**: Must call `prune()` not `destroy()` - prune cleans up KV cache, destroy only frees the slot.

---

## 5. Concurrency Safety

| Task | Status | Notes |
|------|--------|-------|
| ThreadSanitizer in CI | TODO | Detects data races |
| Lock ordering documentation | TODO | Document mutex acquisition order |
| Atomic operations audit | TODO | Verify memory ordering |
| Thread-safe handle validation | Partial | Single-threaded design assumed |

### Gaps

- **ThreadSanitizer (TSan)**: Not yet in CI. Add job:
  ```yaml
  thread-tests:
    runs-on: ubuntu-latest
    steps:
      - name: Configure with TSan
        run: cmake -DCMAKE_CXX_FLAGS="-fsanitize=thread -g" ...
  ```

- **Concurrent access patterns**: BranchStore is not thread-safe by design. Document this clearly and ensure callers synchronize.

---

## 6. FFI Stability (Dynamic Linking)

liblloyal-node dynamically links `libllama.dylib`/`libllama.so`. ABI stability across this boundary is critical.

| Task | Status | Notes |
|------|--------|-------|
| Version pinning | Partial | Documented as "b6870" but not enforced |
| Struct size assertions | TODO | Compile-time checks for ABI stability |
| Symbol validation | TODO | Runtime check for expected exports |
| Calling convention audit | TODO | Verify all cross-boundary calls |

### Risk Analysis

**Dynamic link boundary**:
```
lloyal.node (.node addon)
    │
    ├── libllama.dylib (macOS)
    ├── libllama.so (Linux)
    └── llama.lib (Windows)
```

**Types crossing FFI boundary**:

| Type | Risk | Mitigation |
|------|------|------------|
| `llama_context*` | Opaque pointer, low risk | None needed |
| `llama_model*` | Opaque pointer, low risk | None needed |
| `llama_batch` | Struct with specific layout | Size assertion |
| `llama_token_data` | Struct used in arrays | Size assertion |
| `llama_sampler*` | Opaque pointer, low risk | None needed |
| `llama_pos`, `llama_seq_id` | Typedefs (int32_t) | Size assertion |

### Recommended ABI Tests

Create `tests/abi_test.cpp`:

```cpp
#include <llama.h>
#include <doctest/doctest.h>

TEST_CASE("ABI: struct sizes match expected") {
    // These sizes are for llama.cpp b6870
    // Update when upgrading llama.cpp version
    CHECK(sizeof(llama_token_data) == 12);  // int32 + float + float
    CHECK(sizeof(llama_batch) == 72);       // Platform-specific, verify
    CHECK(sizeof(llama_pos) == 4);
    CHECK(sizeof(llama_seq_id) == 4);
    CHECK(sizeof(llama_token) == 4);
}

TEST_CASE("ABI: critical symbols exist") {
    // Verify symbols we depend on are exported
    CHECK(llama_decode != nullptr);
    CHECK(llama_get_logits != nullptr);
    CHECK(llama_sampler_sample != nullptr);
    CHECK(llama_kv_cache_seq_cp != nullptr);
}
```

### Build-time Version Enforcement

Add to CMakeLists.txt:

```cmake
# Enforce llama.cpp version at build time
file(READ "${LLAMA_CPP_DIR}/CMakeLists.txt" LLAMA_CMAKE)
string(REGEX MATCH "LLAMA_BUILD_NUMBER ([0-9]+)" _ ${LLAMA_CMAKE})
if(NOT CMAKE_MATCH_1 EQUAL 6870)
    message(FATAL_ERROR "Expected llama.cpp b6870, found b${CMAKE_MATCH_1}")
endif()
```

---

## 7. Branch API: Purpose and Constraints

### What branch.hpp Is

`branch.hpp` is a **state management primitive** for branching inference. It provides:

- **Handle-based lifecycle** - Create, fork, prune branches via opaque handles
- **Atomic state cloning** - Fork copies KV cache, grammar, sampler, logits together
- **Resource cleanup** - Prune cleans up KV cache and frees resources

**branch.hpp is NOT an MCTS implementation.** It provides the building blocks. Policy decisions (when to fork, how many branches, token vs chunk granularity) belong to the caller.

### Actual API Constraints

| Constraint | What It Means | Impact |
|------------|---------------|--------|
| Grammar per-branch | Grammar state cloned on fork, can't share | Each branch tracks its own grammar position |
| Sampler chain per-branch | Sampler params fixed at creation | Use TypeScript tsampler to bypass if dynamic params needed |
| Logits copied on capture | `capture_logits()` copies ~512KB (128k vocab) | Memory scales O(active_branches × vocab_size) |
| Linear position per-branch | Position increments after decode | No backtracking - fork before uncertain points |
| Single seq_id per branch | Each branch owns one KV sequence | Limited by context's `n_seq_max` |

### NOT Constraints (Common Misconceptions)

| Misconception | Reality |
|---------------|---------|
| "Must expand one token at a time" | **No.** `decode_tokens()` exists for batch decode. `decode_one()` is a convenience. |
| "Must fork at every token" | **No.** Fork when your policy decides. Chunk-level forking is fine. |
| "Can't do adaptive temperature" | **Workaround:** TypeScript Stepper uses tsampler, bypassing branch.hpp's sampler chain entirely. |
| "Policy is baked in" | **No.** branch.hpp is policy-agnostic. The integration test implements PUCT, but that's one choice among many. |

### Integration Test vs. API Capability

The `mcts_integration_test.cpp` implements per-token PUCT expansion:

```cpp
// Test's policy choice - NOT a branch.hpp requirement
int expand(int parent_idx) {
  BranchHandle child = fork(parent.branch, child_seq, &store_);
  llama_token token = sample(child, &store_);  // One token
  // ...
}
```

This could equally be chunk-level:

```cpp
// Also valid - branch.hpp supports this
int expand_chunk(int parent_idx) {
  BranchHandle child = fork(parent.branch, child_seq, &store_);
  for (int i = 0; i < 10; i++) {  // Generate chunk
    llama_token token = sample(child, &store_);
    accept_token(child, token, &store_);
    decode_and_capture_one(child, token, &store_);
  }
  // Evaluate chunk, not single token
}
```

### Memory Planning

For tree search with B active branches and V vocabulary size:

```
Memory per branch ≈ V × 4 bytes (logits) + KV cache share
Example: 128k vocab, 100 active branches
         = 128,000 × 4 × 100 = ~50 MB logits alone
         + KV overhead for divergent tokens only (copy-on-write)
```

**Mitigation strategies:**
- Prune aggressively - call `prune()` on low-value branches
- Use `kv::seq_keep()` for bulk cleanup (keep winner, remove all others)
- Limit active branches - evaluate and prune before expanding more
- Chunk-level forking - fewer branches needed than token-level

### TypeScript Stepper Integration

Production MCTS typically uses TypeScript Stepper, which:
- **Uses** branch.hpp concepts: KV forking via `kvSeqCopy()`, grammar cloning via `cloneSampler()`
- **Bypasses** branch.hpp's sampler chain: Uses tsampler for full control over sampling params
- **Adds** Tier 1 filtering: `produce()` rejects bad tokens before KV commit (free)
- **Decides** fork granularity: Boundary-driven (sentences, clauses) not token-driven

See `@lloyal/client/docs/future/mcts-ideas-and-patterns.md` for production MCTS patterns.

### ABI Safety

The `decode_one()` function uses stack-allocated `llama_batch` by default for zero-allocation performance. This is ABI-fragile.

**LLOYAL_STACK_BATCH switch:**
- `LLOYAL_STACK_BATCH=1` (default): Fast path, breaks if llama_batch changes
- `LLOYAL_STACK_BATCH=0`: Safe path via `llama_batch_init()`, survives ABI changes

**After llama.cpp update:**
1. Build with `LLOYAL_STACK_BATCH=0` to unblock immediately
2. Run ABI stability test to detect struct changes
3. Update `decode_one()` stack layout if needed
4. Re-enable `LLOYAL_STACK_BATCH=1`

---

## 8. CI Pipeline Status (Updated)

| Job | Platform | Status | Notes |
|-----|----------|--------|-------|
| unit-tests | Ubuntu | Done | Stub-based, fast |
| sanitizer-tests | Ubuntu | Done | ASan + UBSan + LeakSan combined |
| abi-safe-path-tests | Ubuntu | Done | LLOYAL_STACK_BATCH=0 fallback |
| integration-tests-macos | macOS | Done | Real model tests with caching |
| lint | Ubuntu | Done | clang-format check |

### Missing CI Jobs

| Job | Priority | Notes |
|-----|----------|-------|
| thread-tests (TSan) | Medium | Requires dedicated config |
| windows-tests | Low | Windows CI not yet configured |
| msan-tests | Low | Requires instrumented libc++ |

---

## 9. Pre-Release Checklist

Before production release:

- [ ] All CI jobs passing on main branch
- [ ] No sanitizer warnings (treat warnings as errors)
- [ ] ABI tests passing for target llama.cpp version
- [ ] Version pinning enforced in build system
- [ ] RAII cleanup verified (no KV cache leaks)
- [ ] Handle invalidation tested for all code paths
- [ ] Documentation updated with thread-safety guarantees

---

## References

- [llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases)
- [AddressSanitizer docs](https://clang.llvm.org/docs/AddressSanitizer.html)
- [UndefinedBehaviorSanitizer docs](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
- [ThreadSanitizer docs](https://clang.llvm.org/docs/ThreadSanitizer.html)
