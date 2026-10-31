#pragma once

#include <memory>
#include <functional>
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
  // Receives coalesced work from all shards with ORIGINAL feature layouts and
  // occurrence positions. Returns unnormalized partial sums or full-position
  // sequence contributions. It must finish all submitted reads before returning
  // (including errors), so lookup-failure cleanup may safely release row pins.
  using CacheResolver=std::function<Result<std::vector<TensorBlock>>(
      std::vector<AgentWork>,std::vector<FeatureInput>)>;
  void SetCacheResolver(CacheResolver resolver);
 private:
  std::vector<std::shared_ptr<ParameterService>> shards_;
  std::unordered_map<TableId, TableDefinition> tables_;
  // One client may be used by multiple graph threads. The transport calls are
  // asynchronous across shards; this lock protects its metadata and failure state.
  std::mutex mutex_;
  bool failed_ = false;
  CacheResolver resolver_;
};
}  // namespace eps
