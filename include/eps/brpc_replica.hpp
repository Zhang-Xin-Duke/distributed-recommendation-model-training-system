#pragma once
#include <future>
#include "eps/replica_log.hpp"
namespace eps {
class BrpcReplicaLog {
 public:
  BrpcReplicaLog(std::string endpoint, ReplicaLogOptions scope, int timeout_ms=10000);
  ~BrpcReplicaLog();
  std::future<Result<ReplicaHead>> Head();
  std::future<Result<ReplicaHead>> Fence(std::uint64_t expected,std::string primary);
  std::future<Result<Ack>> Write(ReplicaLease,bool baseline,std::uint64_t round,std::uint64_t total,
                               std::uint64_t offset,std::string data);
  std::future<Result<SnapshotChunk>> Read(ReplicaLease,bool baseline,std::uint64_t round,
                                        std::uint64_t offset,std::uint32_t max_bytes=1<<20);
 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace eps
