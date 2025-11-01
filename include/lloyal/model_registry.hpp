#pragma once

#include "common.hpp"
#include <llama/llama.h>
#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <functional>

/**
 * Model Registry (Header-Only)
 *
 * Purpose: Thread-safe weak-pointer cache to avoid reloading same model multiple times.
 * Uses inline static members (C++17) to enable header-only class with static state.
 *
 * Key: (canonPath, n_gpu_layers, use_mmap)
 * Value: weak_ptr to llama_model (auto-cleanup when last context releases)
 *
 * Thread-safe via std::mutex for all cache operations.
 */

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
 * Converting to header-only requires inline static members (C++17).
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

namespace lloyal::detail {

/**
 * Custom deleter for llama_model shared_ptr
 * Logs model free for debugging
 */
inline void freeModel(llama_model* model) {
  LLOYAL_LOG_DEBUG("[ModelRegistry] Freeing model: ptr=%p (last reference released)", (void*)model);
  llama_model_free(model);
  LLOYAL_LOG_DEBUG("[ModelRegistry] Model freed: ptr=%p", (void*)model);
}

} // namespace lloyal::detail

namespace lloyal {

// ===== IMPLEMENTATION =====

// Normalize path to ensure "file:///path" and "/path" map to same key
inline ModelKey ModelRegistry::makeKey(const std::string& fsPath, const llama_model_params& params) {
  // Inline path normalization (removes file:// prefix if present)
  std::string canonPath = fsPath;
  const std::string filePrefix = "file://";
  if (canonPath.substr(0, filePrefix.length()) == filePrefix) {
    canonPath = canonPath.substr(filePrefix.length());
  }

  return {
    canonPath,
    params.n_gpu_layers,
    params.use_mmap
  };
}

// Acquire model from cache or load new
// 1. Check cache (thread-safe)
// 2. Return existing if found (cache hit)
// 3. Load new if expired (cache miss)
// 4. Store as weak_ptr, return shared_ptr
inline std::shared_ptr<llama_model> ModelRegistry::acquire(
  const std::string& fsPath,
  const llama_model_params& params
) {
  ModelKey key = makeKey(fsPath, params);

  LLOYAL_LOG_DEBUG("[ModelRegistry] Acquiring model: path='%s', n_gpu_layers=%d, use_mmap=%s",
            key.canonPath.c_str(),
            key.n_gpu_layers,
            key.use_mmap ? "true" : "false");

  std::lock_guard<std::mutex> lock(mu_);

  auto cacheEntry = cache_.find(key);
  if (cacheEntry != cache_.end()) {
    // Try to upgrade weak_ptr to shared_ptr
    if (auto existingModel = cacheEntry->second.lock()) {
      long refCount = existingModel.use_count();
      LLOYAL_LOG_DEBUG("[ModelRegistry] Cache HIT - Reusing model: ptr=%p, refcount=%ld",
                (void*)existingModel.get(),
                refCount);
      return existingModel;
    } else {
      LLOYAL_LOG_DEBUG("[ModelRegistry] Cache entry expired (model was freed), removing stale entry");
      cache_.erase(cacheEntry);
    }
  }

  LLOYAL_LOG_DEBUG("[ModelRegistry] Cache MISS - Loading NEW model from disk");
  LLOYAL_LOG_DEBUG("[ModelRegistry]   Path: %s", key.canonPath.c_str());
  LLOYAL_LOG_DEBUG("[ModelRegistry]   GPU layers: %d", key.n_gpu_layers);
  LLOYAL_LOG_DEBUG("[ModelRegistry]   Memory mapping: %s", key.use_mmap ? "enabled" : "disabled");

  llama_model* rawModel = llama_model_load_from_file(key.canonPath.c_str(), params);

  if (!rawModel) {
    // Let caller handle error (will throw structured error)
    LLOYAL_LOG_DEBUG("[ModelRegistry] ERROR: llama_model_load_from_file returned NULL");
    return nullptr;
  }

  size_t modelSize = llama_model_size(rawModel);
  LLOYAL_LOG_DEBUG("[ModelRegistry] Model loaded:");
  LLOYAL_LOG_DEBUG("[ModelRegistry]   Pointer: %p", (void*)rawModel);
  LLOYAL_LOG_DEBUG("[ModelRegistry]   Size: %zu bytes (%.2f MB)",
            modelSize,
            modelSize / (1024.0 * 1024.0));

  auto sharedModel = std::shared_ptr<llama_model>(rawModel, detail::freeModel);

  // Store as weak_ptr (allows automatic cleanup when all contexts release the model)
  cache_[key] = sharedModel;
  LLOYAL_LOG_DEBUG("[ModelRegistry] Model cached as weak_ptr, returning shared_ptr (refcount=1)");

  return sharedModel;
}

} // namespace lloyal
