#pragma once
#include <filesystem>
#include <memory>
#include <functional>
#include "eps/types.hpp"

namespace eps {
struct ReplicatedUpdate {
  std::string run;
  ShardId shard;
  std::uint32_t shards;
  std::uint64_t epoch, round;
  std::vector<UpdateContext> contexts;
  std::vector<RowState> rows;
  std::vector<RequestFrontier> requests;
};
std::string EncodeReplicatedUpdate(const ReplicatedUpdate&, std::uint64_t max_bytes = 64ULL << 20);
ReplicatedUpdate DecodeReplicatedUpdate(const std::string&, std::uint64_t max_bytes = 64ULL << 20);
struct ReplicaLease {
  std::uint64_t term = 0;
  std::string primary;
  bool operator==(const ReplicaLease&) const = default;
};
struct ReplicaLogOptions {
  std::filesystem::path directory;
  std::string run, binding;
  ShardId shard = 0;
  std::uint32_t shards = 1;
  std::uint64_t epoch = 1, base_round = 0, record_bytes = 64ULL << 20, disk_bytes = 64ULL << 30;
  std::function<void(std::uint64_t)> before_publish;
};
struct ReplicaHead {
  ReplicaLease lease;
  bool baseline = false;
  std::uint64_t base_round = 0, next_round = 0, bytes = 0, baseline_bytes = 0;
};
// Disk-backed immutable records. Chunk acknowledgement becomes durable commit
// only when the last chunk verifies, fsyncs and publishes the complete blob.
class ReplicaLog {
 public:
  explicit ReplicaLog(ReplicaLogOptions);
  ~ReplicaLog();
  ReplicaLog(const ReplicaLog&) = delete;
  ReplicaLog& operator=(const ReplicaLog&) = delete;
  ReplicaHead Head() const;
  ReplicaLease Fence(std::uint64_t expected, std::string primary);
  void Check(const ReplicaLease&) const;
  void Write(const ReplicaLease&, bool baseline, std::uint64_t round, std::uint64_t total,
             std::uint64_t offset, const std::string& data);
  SnapshotChunk Read(const ReplicaLease&, bool baseline, std::uint64_t round,
                     std::uint64_t offset, std::uint32_t max_bytes) const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace eps
