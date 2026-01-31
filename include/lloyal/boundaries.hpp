#pragma once

/**
 * Boundary Tracker Stub for OSS liblloyal
 *
 * This stub provides the minimal interface needed by branch.hpp
 *
 * When using branch.hpp without boundaries:
 * - Pass nullptr for boundary_tracker parameter
 * - fork() will skip boundary_tracker cloning
 * - Boundary-aware features (MCTS per-boundary expansion) unavailable
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
