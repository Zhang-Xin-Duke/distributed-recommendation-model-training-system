#pragma once

#include <chrono>
#include <functional>
#include "eps/training_progress.hpp"

namespace eps {
enum class GradientReduction { sum, worker_mean, sample_mean };
struct GradientTable {
  TableId table;
  std::uint32_t dimension;
  // Sorted unique keys and row-major dL/d(parameter). Empty tables participate.
  std::vector<Key> keys;
  std::vector<float> values;
  bool operator==(const GradientTable&) const = default;
};
struct GradientContribution {
  TrainingTask task;
  // Gradients are the local batch mean when using sample_mean.
  std::uint64_t sample_count;
  std::vector<GradientTable> tables;
  bool operator==(const GradientContribution&) const = default;
};
struct SynchronousGradientOptions {
  std::string run;
  std::uint64_t epoch = 1;
  std::uint32_t workers = 1;
  ShardId shard = 0;
  std::uint32_t shards = 1;
  // The fixed table schema; keys/values must be empty here.
  std::vector<GradientTable> tables;
  GradientReduction reduction = GradientReduction::sample_mean;
  std::uint64_t gradient_bytes = 64ULL << 20;
  std::chrono::milliseconds round_timeout{30000};
};
struct SynchronousGradientStats {
  std::uint64_t next_round = 0;
  std::uint32_t received_workers = 0;
  std::uint64_t retained_bytes = 0;
  std::uint64_t duplicate_submissions = 0;
  bool applying = false;
  bool failed = false;
};

// One fixed-group, shard-local synchronous update stream. The application
// callback runs once per complete round, outside the admission lock, and must
// return after the parameter/optimizer update acknowledgement. A failed apply
// is terminal; partial external changes require job-level recovery.
class SynchronousGradientCollector {
 public:
  using Apply = std::function<Result<Ack>(std::uint64_t, std::vector<GradientTable>)>;
  SynchronousGradientCollector(SynchronousGradientOptions options, Apply apply);
  ~SynchronousGradientCollector();
  SynchronousGradientCollector(const SynchronousGradientCollector&) = delete;
  SynchronousGradientCollector& operator=(const SynchronousGradientCollector&) = delete;
  std::shared_future<Result<Ack>> Submit(GradientContribution contribution);
  SynchronousGradientStats Stats() const;
  // Wakes callers immediately; already-dispatched external operations drain
  // before destruction. This does not undo an update already being applied.
  void Abort(std::string reason);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace eps
