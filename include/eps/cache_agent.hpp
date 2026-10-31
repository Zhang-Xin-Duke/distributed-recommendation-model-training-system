#pragma once

#include <future>
#include <vector>

#include "eps/types.hpp"

namespace eps {

// Runs inside the existing GPU worker; device allocations and streams stay there.
class GpuCacheAgent {
 public:
  virtual ~GpuCacheAgent() = default;

  virtual std::future<Result<Ack>> Load(
      MigrationPlan plan, std::vector<RowState> state) = 0;
  virtual std::future<Result<std::vector<DeviceTensorBlock>>> Gather(
      std::string ticket, std::vector<AgentWork> work,
      std::vector<FeatureInput> layouts) = 0;
  virtual std::future<Result<Ack>> Apply(
      std::string ticket, UpdateContext context, DeviceRowGradients gradients) = 0;
  // Returns current weights + optimizer state after referenced GPU work finishes.
  virtual std::future<Result<std::vector<RowState>>> WriteBack(
      MigrationPlan plan) = 0;
};

}  // namespace eps
