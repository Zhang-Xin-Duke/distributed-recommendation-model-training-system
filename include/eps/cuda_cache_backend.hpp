#pragma once
#include "eps/cache_agent.hpp"
#include "eps/device_features.hpp"
namespace eps {
struct CudaCacheOptions {
  AgentId agent=0;
  int device=0;
  std::uint64_t state_bytes=64ULL<<20;
  std::uint64_t rows_per_table=4096;
  std::uint64_t transfer_bytes=16ULL<<20;
  std::uint64_t outstanding_output_bytes=64ULL<<20;
};
struct CudaMemoryStats {
  std::uint64_t resident_state_bytes;
  std::uint64_t staging_device_bytes;
  std::uint64_t outstanding_output_bytes;
  std::uint64_t output_allocations;
  bool operator==(const CudaMemoryStats&) const = default;
};
class CudaCacheBackend final : public GpuCacheAgent {
 public:
  explicit CudaCacheBackend(CudaCacheOptions options={});
  ~CudaCacheBackend() override;
  std::future<Result<Ack>> Define(TableDefinition) override;
  std::future<Result<std::uint64_t>> FreeSlots(TableId) override;
  std::future<Result<std::vector<LocatedRow>>> Load(MigrationPlan,std::vector<RowState>) override;
  std::future<Result<std::vector<TensorBlock>>> GatherHost(std::vector<AgentWork>,std::vector<FeatureInput>) override;
  std::future<Result<std::vector<DeviceTensorBlock>>> GatherDevice(std::vector<AgentWork>,std::vector<FeatureInput>) override;
  std::future<Result<Ack>> ApplyHost(UpdateContext,RowGradients) override;
  std::future<Result<Ack>> ApplyCompact(UpdateContext,CompactGradients) override;
  std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan) override;
  std::future<Result<Ack>> CommitEviction(MigrationPlan) override;
  std::future<Result<CudaMemoryStats>> MemoryStats();
  std::future<Result<std::vector<DeviceTensorBlock>>> AssembleFeatures(DeviceAssemblyRequest);
 private:struct Impl;std::unique_ptr<Impl> impl_;
};
}  // namespace eps
