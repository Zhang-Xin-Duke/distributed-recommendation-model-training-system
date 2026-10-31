#pragma once

#include <future>
#include <memory>
#include <vector>
#include "eps/types.hpp"

namespace eps {
// Control/data interface hosted by the existing training worker. The CUDA
// implementation keeps weights and moments on device; the host implementation
// is an explicit protocol/reference backend, not a GPU-performance substitute.
class CacheBackend {
 public:
  virtual ~CacheBackend() = default;
  virtual std::future<Result<Ack>> Define(TableDefinition table) = 0;
  virtual std::future<Result<std::uint64_t>> FreeSlots(TableId table) = 0;
  virtual std::future<Result<std::vector<LocatedRow>>> Load(
      MigrationPlan plan, std::vector<RowState> states) = 0;
  virtual std::future<Result<std::vector<TensorBlock>>> GatherHost(
      std::vector<AgentWork> work, std::vector<FeatureInput> layouts) = 0;
  virtual std::future<Result<Ack>> ApplyHost(
      UpdateContext context, RowGradients gradients) = 0;
  // The default reference implementation expands on the host. CUDA overrides
  // this operation to reduce the references directly into device gradients.
  virtual std::future<Result<Ack>> ApplyCompact(UpdateContext,CompactGradients);
  // Readback retains the rows. Slot reclamation is a separate acknowledged step.
  virtual std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan plan) = 0;
  virtual std::future<Result<Ack>> CommitEviction(MigrationPlan plan) = 0;
};

class HostCacheBackend final : public CacheBackend {
 public:
  explicit HostCacheBackend(AgentId agent, std::uint64_t state_bytes = 64ULL << 20);
  ~HostCacheBackend() override;
  std::future<Result<Ack>> Define(TableDefinition) override;
  std::future<Result<std::uint64_t>> FreeSlots(TableId) override;
  std::future<Result<std::vector<LocatedRow>>> Load(MigrationPlan,std::vector<RowState>) override;
  std::future<Result<std::vector<TensorBlock>>> GatherHost(std::vector<AgentWork>,std::vector<FeatureInput>) override;
  std::future<Result<Ack>> ApplyHost(UpdateContext,RowGradients) override;
  std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan) override;
  std::future<Result<Ack>> CommitEviction(MigrationPlan) override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace eps
