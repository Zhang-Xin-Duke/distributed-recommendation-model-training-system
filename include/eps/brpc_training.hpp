#pragma once

#include "eps/training_progress.hpp"

namespace eps {
// Async RPCs retain their channel through completion, including after this
// wrapper is destroyed. A timed-out Begin may be retried with the same task.
class BrpcTrainingProgress {
 public:
  BrpcTrainingProgress(std::string endpoint, std::string run,
                       std::uint64_t epoch = 1, int timeout_ms = 30000);
  ~BrpcTrainingProgress();
  std::future<Result<TrainingPermit>> Begin(std::uint32_t worker, std::uint64_t iteration);
  std::future<Result<Ack>> Complete(std::uint32_t worker, std::uint64_t iteration);
  std::future<Result<TrainingProgressStats>> Stats();
  std::future<Result<Ack>> Pause();
  std::future<Result<TrainingProgressCheckpoint>> Checkpoint();
  std::future<Result<Ack>> Resume();
  std::future<Result<Ack>> Abort(std::string reason);
 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace eps
