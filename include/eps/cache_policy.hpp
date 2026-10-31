#pragma once

#include <cstdint>
#include <vector>
#include "eps/types.hpp"

namespace eps {
enum class CacheAdmissionMode { manual, frequency, reuse_cost };

struct CachePolicyOptions {
  CacheAdmissionMode mode = CacheAdmissionMode::manual;
  std::vector<AgentId> agents;
  std::uint32_t window_requests = 128;
  std::uint32_t max_tracked_rows = 65536;
  double window_decay = 0.5;
  double minimum_accesses = 2;
  std::uint32_t minimum_residence_windows = 2;
  double replacement_margin = 0.25;
  std::uint32_t max_rows_per_batch = 64;
  // Priors in microseconds; measurements include acknowledged migration work.
  double saved_microseconds_per_access = 1;
  double load_microseconds_per_byte = 0.001;
  double writeback_microseconds_per_byte = 0.001;
  double cost_average_alpha = 0.2;
  bool learn_migration_costs = true;
  // Shared across a decision window. Migration batches execute sequentially;
  // max_inflight_batches also caps the total batches issued in that window.
  MigrationBudget budget{1ULL << 20, 1ULL << 20, 4, 16};
};
}  // namespace eps
