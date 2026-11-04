#pragma once

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <vector>

#include "eps/types.hpp"

namespace eps {

enum class TrainingConsistency { sequential, eventual, bounded_delay };

struct TrainingProgressOptions {
  std::string run;
  std::uint64_t epoch = 1;
  std::uint32_t worker_count = 1;
  TrainingConsistency consistency = TrainingConsistency::sequential;
  std::uint64_t max_delay = 0;
};

struct TrainingTask {
  std::string run;
  std::uint64_t epoch;
  std::uint32_t worker;
  std::uint64_t iteration;
  bool operator==(const TrainingTask&) const = default;
};

struct TrainingPermit {
  TrainingTask task;
  // Number of globally completed iterations when this permit was granted.
  // This is not a parameter version or a consistent cross-shard model view.
  std::uint64_t observed_completion_floor;
};

struct TrainingProgressCheckpoint {
  std::string source_run;
  std::uint64_t epoch;
  TrainingConsistency consistency;
  std::uint64_t max_delay;
  // c[w] means worker w has completed every iteration in [0,c[w]).
  std::vector<std::uint64_t> completed;
};

struct TrainingProgressStats {
  std::vector<std::uint64_t> completed;
  std::uint64_t completion_floor = 0;
  std::uint64_t granted_tasks = 0;
  std::uint64_t completed_tasks = 0;
  std::uint64_t duplicate_completions = 0;
  std::uint32_t active = 0;
  std::uint32_t waiting = 0;
  bool paused = false;
  bool failed = false;
};

// Fixed-group training-task dependencies. The training executor owns the
// parameter operations and must call Complete only after their acknowledged
// completion. A permit does not aggregate gradients or fence parameter owners.
class TrainingProgressController {
 public:
  explicit TrainingProgressController(TrainingProgressOptions options);
  ~TrainingProgressController();
  TrainingProgressController(const TrainingProgressController&) = delete;
  TrainingProgressController& operator=(const TrainingProgressController&) = delete;

  // With allow_wait=false, return capacity_exceeded rather than registering a
  // new dependency waiter. Immediately eligible work can still make progress.
  std::shared_future<Result<TrainingPermit>> Begin(TrainingTask task, bool allow_wait = true);
  Result<Ack> Complete(TrainingTask task);
  Result<Ack> Pause();
  Result<TrainingProgressCheckpoint> Checkpoint() const;
  Result<Ack> Restore(TrainingProgressCheckpoint checkpoint);
  // Trusted startup after the manager reconciles replica-backed shard rounds.
  Result<Ack> RecoverSynchronousRound(std::uint64_t next);
  Result<Ack> Resume();
  void Abort(std::string reason);
  TrainingProgressStats Stats() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace eps
