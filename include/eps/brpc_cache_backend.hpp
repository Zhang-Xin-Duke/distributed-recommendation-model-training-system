#pragma once

#include "eps/cache_backend.hpp"
#include "eps/agent_admission.hpp"

namespace eps {
class BrpcCacheBackend final : public CacheBackend {
 public:
  BrpcCacheBackend(std::string endpoint,std::string run,ShardId home,AgentId agent,int timeout_ms=5000);
  ~BrpcCacheBackend() override;
  std::future<Result<AgentAdmissionStats>> AdmissionStats();
  std::future<Result<Ack>> Define(TableDefinition) override;
  std::future<Result<std::uint64_t>> FreeSlots(TableId) override;
  std::future<Result<std::vector<LocatedRow>>> Load(MigrationPlan,std::vector<RowState>) override;
  std::future<Result<std::vector<TensorBlock>>> GatherHost(std::vector<AgentWork>,std::vector<FeatureInput>) override;
  std::future<Result<Ack>> ApplyHost(UpdateContext,RowGradients) override;
  std::future<Result<Ack>> ApplyCompact(UpdateContext,CompactGradients) override;
  std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan) override;
  std::future<Result<Ack>> CommitEviction(MigrationPlan) override;
 private:
  struct Impl;std::shared_ptr<Impl> impl_;
};
}  // namespace eps
