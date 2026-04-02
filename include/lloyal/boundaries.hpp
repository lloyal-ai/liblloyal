// Copyright (c) 2024-2026 Lloyal Labs
// Licensed under the Business Source License 1.1 (the "License");
// you may not use this file except in compliance with the License.
// See LICENSE in the root of this repository.

#pragma once

/**
 * Boundary Tracker Stub for OSS liblloyal
 *
 * This stub provides the minimal interface needed by branch.hpp
 *
 * When using branch.hpp without boundaries:
 * - Pass nullptr for boundary_tracker parameter
 * - fork() will skip boundary_tracker cloning
 * - Boundary-aware features (per-boundary expansion in tree search) unavailable
 */

#include <memory>

namespace lloyal {
namespace boundaries {

/**
 * Stub BoundaryTracker - does nothing
 */
class BoundaryTracker {
public:
  virtual ~BoundaryTracker() = default;

  // Clone interface required by branch::fork()
  virtual std::unique_ptr<BoundaryTracker> clone() const {
    return nullptr;  // Stub returns nullptr
  }
};

}  // namespace boundaries
}  // namespace lloyal
