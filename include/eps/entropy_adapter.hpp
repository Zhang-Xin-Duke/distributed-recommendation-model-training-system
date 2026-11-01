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
struct PreparationOptions {
  std::size_t max_pending_batches = 8;
  std::uint64_t payload_bytes = 64ULL << 20;
  PreparationLimits lookup_limits;
};
struct PreparationStats {
  std::uint64_t accepted=0, completed=0, failed=0, consumed=0;
  std::uint64_t cancelled=0, rejected=0, pending=0, claimed=0, reserved_jobs=0;
  std::uint64_t reserved_bytes=0, peak_reserved_bytes=0, prepare_nanoseconds=0;
};
struct AdapterOptions {
  std::string run;
  std::uint32_t learner = 0;
  std::uint64_t partition_epoch = 1;
  std::size_t max_live_batches = 128;
  std::size_t completion_history = 4096;
  PreparationOptions preparation;
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
  // Prepares only IDs/layouts; parameter reads begin at PullPrepared. Tokens
  // must be consumed in FIFO order or cancelled before checkpointing.
  Result<std::uint64_t> PrepareAsync(std::vector<Feature> features);
  Result<PullResult> PullPrepared(std::uint64_t token, bool read_only = false);
  Result<Ack> CancelPrepared(std::uint64_t token);
  PreparationStats PreparationStatus() const;
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
