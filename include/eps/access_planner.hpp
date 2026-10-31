#pragma once

#include <cstdint>
#include <vector>

#include "eps/types.hpp"

namespace eps {

struct AccessObservation {
  TableId table;
  std::uint32_t feature;
  Pooling pooling;
  std::uint64_t occurrences;
  std::uint64_t unique_rows;
  std::uint64_t bags;
  double cpu_route_microseconds;
  double gpu_route_microseconds;
  double queue_microseconds;
};

struct AdmissionCandidate {
  RowKey row;
  double expected_accesses;
  double saved_microseconds_per_access;
  double loading_microseconds;
  double writeback_microseconds;
  std::uint64_t state_bytes;
};

class AccessPlanner {
 public:
  virtual ~AccessPlanner() = default;
  virtual void Observe(AccessObservation observation) = 0;
  // Applies to CPU-backed contributions; resident rows retain their owner.
  virtual AccessRoute SelectRoute(const FeatureInput& feature) const = 0;
  // Produces proposals; the coordinator acquires row references/slots separately.
  virtual std::vector<MigrationPlan> PlanMigration(
      std::vector<AdmissionCandidate> candidates,
      MigrationBudget budget) const = 0;
};

}  // namespace eps
