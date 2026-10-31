#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "eps/parameter_service.hpp"

namespace eps {
class DistributedBatch {
 public:
  const std::vector<TensorBlock>& features() const { return features_; }
 private:
  friend class DistributedClient;
  LookupRequest original_;
  std::vector<TensorBlock> features_;
  struct Part {
    std::string ticket;
    std::vector<std::vector<std::uint64_t>> positions;
  };
  std::vector<Part> shards_;
};

// Fan-out/fan-in client. It preserves global bag denominators and sequence
// positions; each server still owns an independent asynchronous update stream.
class DistributedClient {
 public:
  explicit DistributedClient(std::vector<std::shared_ptr<ParameterService>> shards);
  Result<Ack> CreateTable(TableDefinition table);
  Result<DistributedBatch> Lookup(LookupRequest request);
  Result<Ack> Push(const DistributedBatch& batch, std::vector<TensorBlock> gradients);
  Result<Ack> Release(const DistributedBatch& batch);
 private:
  std::vector<std::shared_ptr<ParameterService>> shards_;
  std::unordered_map<TableId, TableDefinition> tables_;
  // One client may be used by multiple graph threads. The transport calls are
  // asynchronous across shards; this lock protects its metadata and failure state.
  std::mutex mutex_;
  bool failed_ = false;
};
}  // namespace eps
