#pragma once

// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Lloyal Labs

#include "common.hpp"
#include <functional>
#include <llama/llama.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

/**
 * @file model_registry.hpp
 * @brief Thread-Safe Model Cache
 *
 * Provides weak-pointer cache to avoid reloading same model multiple times.
 * Uses inline static members (C++17) for header-only implementation with static state.
 *
 * Cache key: (canonPath, n_gpu_layers, use_mmap)
 * Cache value: weak_ptr to llama_model (auto-cleanup when last context releases)
 *
 * Thread-safe via std::mutex for all cache operations.
 */

namespace lloyal {

/**
 * @brief Model cache key combining file path and GPU configuration
 *
 * Used as key in ModelRegistry cache to uniquely identify model instances.
 * Different GPU configurations of the same model are cached separately.
 */
struct ModelKey {
  std::string canonPath; ///< Normalized file path (file:// prefix removed)
  int n_gpu_layers;      ///< Number of layers offloaded to GPU (-1 = all)
  bool use_mmap;         ///< Whether to use memory mapping

  bool operator==(const ModelKey &o) const {
    return n_gpu_layers == o.n_gpu_layers && use_mmap == o.use_mmap &&
           canonPath == o.canonPath;
  }
};

/**
 * @brief Hash functor for ModelKey
 *
 * Computes combined hash of path, GPU layers, and mmap flag for use in
 * std::unordered_map. Uses XOR with golden ratio constant for good distribution.
 */
struct ModelKeyHash {
  /**
   * @brief Compute hash for ModelKey
   *
   * Combines path hash with GPU/mmap configuration using XOR and golden ratio.
   *
   * @param k Key to hash
   * @return Combined hash value
   */
  size_t operator()(const ModelKey &k) const {
    std::hash<std::string> Hs;
    std::hash<int> Hi;
    std::hash<bool> Hb;
    return Hs(k.canonPath) ^
           (Hi(k.n_gpu_layers) + 0x9e3779b9 + (Hb(k.use_mmap) << 6));
  }
};

/**
 * Thread-safe registry for sharing llama_model instances
 *
 * IMPORTANT: This is a CLASS with static members, not a namespace.
 * Converting to header-only requires inline static members (C++17).
 */
class ModelRegistry {
public:
  /**
   * Acquire a model from cache or load if not present
   *
   * @param fsPath Filesystem path to model file (file:// prefix normalized)
   * @param params Model load parameters (GPU layers, mmap, etc.)
   * @return shared_ptr to model, or nullptr if load failed
   */
  static std::shared_ptr<llama_model> acquire(const std::string &fsPath,
                                              const llama_model_params &params);

private:
  /**
   * Global cache mutex - inline static for header-only
   */
  inline static std::mutex mu_;

  /**
   * Model cache - inline static for header-only
   */
  inline static std::unordered_map<ModelKey, std::weak_ptr<llama_model>,
                                   ModelKeyHash>
      cache_;

  /**
   * @brief Create normalized cache key from path and parameters
   *
   * Normalizes filesystem path by removing file:// prefix to ensure
   * "file:///path" and "/path" map to the same cache entry.
   *
   * @param fsPath Filesystem path (may include file:// prefix)
   * @param params Model parameters for GPU/mmap configuration
   * @return Normalized ModelKey
   */
  static ModelKey makeKey(const std::string &fsPath,
                          const llama_model_params &params);
};

} // namespace lloyal

namespace lloyal::detail {

/**
 * Custom deleter for llama_model shared_ptr
 * Logs model free for debugging
 */
inline void freeModel(llama_model *model) {
  LLOYAL_LOG_DEBUG(
      "[ModelRegistry] Freeing model: ptr=%p (last reference released)",
      (void *)model);
  llama_model_free(model);
  LLOYAL_LOG_DEBUG("[ModelRegistry] Model freed: ptr=%p", (void *)model);
}

} // namespace lloyal::detail

namespace lloyal {

// ===== IMPLEMENTATION =====

// Normalize path to ensure "file:///path" and "/path" map to same key
inline ModelKey ModelRegistry::makeKey(const std::string &fsPath,
                                       const llama_model_params &params) {
  // Inline path normalization (removes file:// prefix if present)
  std::string canonPath = fsPath;
  const std::string filePrefix = "file://";
  if (canonPath.substr(0, filePrefix.length()) == filePrefix) {
    canonPath = canonPath.substr(filePrefix.length());
  }

  return {canonPath, params.n_gpu_layers, params.use_mmap};
}

// Acquire model from cache or load new
// 1. Check cache (thread-safe)
// 2. Return existing if found (cache hit)
// 3. Load new if expired (cache miss)
// 4. Store as weak_ptr, return shared_ptr
inline std::shared_ptr<llama_model>
ModelRegistry::acquire(const std::string &fsPath,
                       const llama_model_params &params) {
  ModelKey key = makeKey(fsPath, params);

  LLOYAL_LOG_DEBUG("[ModelRegistry] Acquiring model: path='%s', "
                   "n_gpu_layers=%d, use_mmap=%s",
                   key.canonPath.c_str(), key.n_gpu_layers,
                   key.use_mmap ? "true" : "false");

  std::lock_guard<std::mutex> lock(mu_);

  auto cacheEntry = cache_.find(key);
  if (cacheEntry != cache_.end()) {
    // Try to upgrade weak_ptr to shared_ptr
    if (auto existingModel = cacheEntry->second.lock()) {
      long refCount = existingModel.use_count();
      LLOYAL_LOG_DEBUG(
          "[ModelRegistry] Cache HIT - Reusing model: ptr=%p, refcount=%ld",
          (void *)existingModel.get(), refCount);
      return existingModel;
    } else {
      LLOYAL_LOG_DEBUG("[ModelRegistry] Cache entry expired (model was freed), "
                       "removing stale entry");
      cache_.erase(cacheEntry);
    }
  }

  LLOYAL_LOG_DEBUG("[ModelRegistry] Cache MISS - Loading NEW model from disk");
  LLOYAL_LOG_DEBUG("[ModelRegistry]   Path: %s", key.canonPath.c_str());
  LLOYAL_LOG_DEBUG("[ModelRegistry]   GPU layers: %d", key.n_gpu_layers);
  LLOYAL_LOG_DEBUG("[ModelRegistry]   Memory mapping: %s",
                   key.use_mmap ? "enabled" : "disabled");

  llama_model *rawModel =
      llama_model_load_from_file(key.canonPath.c_str(), params);

  if (!rawModel) {
    // Let caller handle error (will throw structured error)
    LLOYAL_LOG_DEBUG(
        "[ModelRegistry] ERROR: llama_model_load_from_file returned NULL");
    return nullptr;
  }

  size_t modelSize = llama_model_size(rawModel);
  LLOYAL_LOG_DEBUG("[ModelRegistry] Model loaded:");
  LLOYAL_LOG_DEBUG("[ModelRegistry]   Pointer: %p", (void *)rawModel);
  LLOYAL_LOG_DEBUG("[ModelRegistry]   Size: %zu bytes (%.2f MB)", modelSize,
                   modelSize / (1024.0 * 1024.0));

  auto sharedModel = std::shared_ptr<llama_model>(rawModel, detail::freeModel);

  // Store as weak_ptr (allows automatic cleanup when all contexts release the
  // model)
  cache_[key] = sharedModel;
  LLOYAL_LOG_DEBUG("[ModelRegistry] Model cached as weak_ptr, returning "
                   "shared_ptr (refcount=1)");

  return sharedModel;
}

} // namespace lloyal
