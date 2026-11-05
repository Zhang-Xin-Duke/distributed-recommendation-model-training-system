#pragma once
#include "eps/brpc_replica.hpp"
#include "eps/runtime.hpp"
namespace eps::transport {
struct JournalParentGuard;
class ReplicatedRoundWriter {
 public:
  ReplicatedRoundWriter(std::string endpoint,ReplicaLogOptions options,ReplicaLease lease);
  ReplicaHead Start(CpuParameterService&,bool recover,bool leave_paused);
  void BeforeRound(std::uint64_t);
  void Commit(CpuParameterService&,std::uint64_t,const std::vector<RowKey>&);
 private:
  ReplicaLogOptions options_;
  ReplicaLease lease_,local_lease_;
  std::shared_ptr<JournalParentGuard> parent_guard_;
  BrpcReplicaLog replica_;
  ReplicaLog local_;
};
}  // namespace eps::transport
