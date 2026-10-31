#pragma once

#include <memory>
#include "eps/cache_backend.hpp"

namespace eps {
// Serializes monotonically numbered control commands and retains the previous result
// per home shard. A readback remains resident until CommitEviction is acknowledged.
class AgentRuntime {
 public:
  AgentRuntime(std::string run,std::shared_ptr<CacheBackend> backend,
               std::uint64_t replay_bytes=64ULL<<20);
  ~AgentRuntime();
  std::future<Result<Ack>> Define(AgentCommand,TableDefinition);
  std::future<Result<std::uint64_t>> FreeSlots(std::string run,TableId);
  std::future<Result<std::vector<LocatedRow>>> Load(AgentCommand,MigrationPlan,std::vector<RowState>);
  std::future<Result<std::vector<TensorBlock>>> Gather(std::string run,std::vector<AgentWork>,std::vector<FeatureInput>);
  std::future<Result<Ack>> Apply(AgentCommand,UpdateContext,RowGradients);
  std::future<Result<Ack>> ApplyCompact(AgentCommand,UpdateContext,CompactGradients);
  std::future<Result<std::vector<RowState>>> ReadBack(AgentCommand,MigrationPlan);
  std::future<Result<Ack>> CommitEviction(AgentCommand,MigrationPlan);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace eps
