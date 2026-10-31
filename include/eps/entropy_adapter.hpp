#pragma once

#include <optional>
#include "eps/distributed_client.hpp"

namespace eps::entropy {
struct Feature {
  TableId table;
  std::uint32_t feature;
  Pooling pooling;
  std::vector<Key> keys;
  std::vector<std::uint64_t> offsets;
  std::optional<Key> padding_key;
};
struct PullResult {
  std::uint64_t handle;
  std::vector<TensorBlock> features;
};
struct AdapterOptions {
  std::string run;
  std::uint32_t learner = 0;
  std::uint64_t partition_epoch = 1;
  std::size_t max_live_batches = 128;
  std::size_t completion_history = 4096;
};

// Embedding Pull/Push/Join boundary. Dense computation, loss scaling and
// collectives remain in the caller's existing training graph.
class TableAdapter {
 public:
  TableAdapter(AdapterOptions options, std::vector<std::shared_ptr<ParameterService>> shards);
  ~TableAdapter();
  Result<Ack> CreateTable(TableDefinition table);
  void SetCacheResolver(DistributedClient::CacheResolver resolver,bool coalesce=false);
  Result<PullResult> Pull(std::vector<Feature> features, bool read_only = false);
  // Gradients are dL/d(completed feature), before mean-pooling unnormalization.
  Result<Ack> Push(std::uint64_t handle, std::vector<TensorBlock> gradients);
  Result<Ack> Release(std::uint64_t handle);
  Result<Ack> Join();
  // Called after the framework closes input and completes every Pull lifecycle.
  // A successful save leaves the services drained until restart or explicit resume.
  Result<std::vector<SnapshotMetadata>> Save(std::string snapshot_id);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace eps::entropy
