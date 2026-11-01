#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include "eps/entropy_adapter.hpp"

namespace eps::detail {
struct PreparedAdapterInput {
  PreparedLookup lookup;
  std::vector<std::vector<std::uint64_t>> positions;
  std::vector<std::uint64_t> original_sizes;
  std::vector<Pooling> pooling;
};
Result<std::uint64_t> EstimateAdapterPreparation(const std::vector<entropy::Feature>&,
    std::uint32_t shards, PreparationLimits limits);
Result<PreparedAdapterInput> PrepareAdapterInput(std::vector<entropy::Feature>,
    std::uint32_t shards, PreparationLimits limits);

class PrepareQueue {
 public:
  using Processor=std::function<Result<PreparedAdapterInput>(std::vector<entropy::Feature>)>;
  PrepareQueue(std::uint32_t shards, entropy::PreparationOptions options, Processor processor={});
  ~PrepareQueue();
  Result<std::uint64_t> Submit(std::vector<entropy::Feature> features);
  Result<PreparedAdapterInput> Take(std::uint64_t token);
  Result<Ack> Cancel(std::uint64_t token);
  bool Empty() const;
  Result<Ack> CloseIfEmpty();
  Result<Ack> Reopen();
  void Close();
  entropy::PreparationStats Stats() const;
 private:
  struct Job;
  void Run();
  void Reclaim(Job& job);
  void CancelAll();
  std::uint32_t shards_;
  entropy::PreparationOptions options_;
  Processor processor_;
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<std::shared_ptr<Job>> jobs_;
  std::map<std::uint64_t,std::shared_ptr<Job>> pending_;
  entropy::PreparationStats stats_;
  std::uint64_t next_=0;
  bool closed_=false, stopping_=false;
  std::thread worker_;
};
}  // namespace eps::detail
