#pragma once

#include <memory>
#include "eps/access_planner.hpp"

namespace eps {
struct PlannerOptions {
  double moving_average_alpha = 0.2;
  double switch_margin = 0.15;
  std::uint32_t minimum_observations = 4;
  ShardId shard = 0;
  std::uint32_t shard_count = 1;
  AgentId agent = 0;
};

class CostBasedPlanner final : public AccessPlanner {
 public:
  explicit CostBasedPlanner(PlannerOptions options = {});
  ~CostBasedPlanner() override;
  void Observe(AccessObservation observation) override;
  AccessRoute SelectRoute(const FeatureInput& feature) const override;
  // Capacity observations are per table/layout and include only reusable slots.
  void SetFreeSlots(TableId table, std::uint64_t slots);
  std::vector<MigrationPlan> PlanMigration(std::vector<AdmissionCandidate> candidates,
                                         MigrationBudget budget) const override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace eps
