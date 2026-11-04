#include "eps/training_progress.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace eps {
namespace {
template <class T>
std::shared_future<Result<T>> Ready(Error error) {
  std::promise<Result<T>> result;
  result.set_value(std::move(error));
  return result.get_future().share();
}

bool ValidPolicy(TrainingConsistency policy) {
  return policy == TrainingConsistency::sequential ||
         policy == TrainingConsistency::eventual ||
         policy == TrainingConsistency::bounded_delay;
}
}  // namespace

class TrainingProgressController::Impl {
 public:
  struct Request {
    TrainingTask task;
    std::promise<Result<TrainingPermit>> promise;
    std::shared_future<Result<TrainingPermit>> future = promise.get_future().share();
    bool active = false;
    explicit Request(TrainingTask value) : task(std::move(value)) {}
  };

  explicit Impl(TrainingProgressOptions value) : options(std::move(value)) {
    if (options.run.empty() || !options.epoch || !options.worker_count ||
        options.worker_count > 1024 || !ValidPolicy(options.consistency) ||
        (options.consistency != TrainingConsistency::bounded_delay && options.max_delay)) {
      throw std::invalid_argument("invalid training progress configuration");
    }
    completed.resize(options.worker_count);
    requests.resize(options.worker_count);
  }

  std::optional<Error> Validate(const TrainingTask& task) const {
    if (task.run != options.run || task.epoch != options.epoch) {
      return Error{Error::Code::stale_generation, "training run or epoch differs"};
    }
    if (task.worker >= options.worker_count ||
        task.iteration == std::numeric_limits<std::uint64_t>::max()) {
      return Error{Error::Code::invalid_argument, "invalid worker or iteration"};
    }
    return std::nullopt;
  }

  std::uint64_t Floor() const { return *std::min_element(completed.begin(), completed.end()); }
  bool Eligible(std::uint64_t iteration, std::uint64_t floor) const {
    if (options.consistency == TrainingConsistency::eventual) return true;
    const auto delay = options.consistency == TrainingConsistency::bounded_delay ? options.max_delay : 0;
    return iteration <= delay || floor >= iteration - delay;
  }

  void GrantReady() {
    const auto floor = Floor();
    for (auto& request : requests) {
      if (!request || request->active || !Eligible(request->task.iteration, floor)) continue;
      request->active = true;
      ever_started = true;
      ++granted;
      request->promise.set_value(TrainingPermit{request->task, floor});
    }
  }

  Error Failed() const { return {Error::Code::run_failed, failure}; }

  TrainingProgressOptions options;
  mutable std::mutex mutex;
  std::vector<std::uint64_t> completed;
  std::vector<std::unique_ptr<Request>> requests;
  std::uint64_t granted = 0, finished = 0, duplicates = 0;
  bool paused = false, ever_started = false, restored = false;
  std::string failure;
};

TrainingProgressController::TrainingProgressController(TrainingProgressOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}

TrainingProgressController::~TrainingProgressController() {
  try { Abort("training progress controller destroyed"); } catch (...) {}
}

std::shared_future<Result<TrainingPermit>> TrainingProgressController::Begin(TrainingTask task, bool allow_wait) {
  std::lock_guard lock(impl_->mutex);
  if (auto error = impl_->Validate(task)) return Ready<TrainingPermit>(*error);
  if (!impl_->failure.empty()) return Ready<TrainingPermit>(impl_->Failed());
  const auto done = impl_->completed[task.worker];
  if (task.iteration < done) {
    return Ready<TrainingPermit>({Error::Code::retired_request, "training iteration already completed"});
  }
  if (task.iteration != done) {
    return Ready<TrainingPermit>({Error::Code::invalid_argument, "worker iterations must be contiguous"});
  }
  auto& request = impl_->requests[task.worker];
  if (request) {
    if (!allow_wait && !request->active)
      return Ready<TrainingPermit>({Error::Code::capacity_exceeded, "no capacity for a dependency waiter"});
    return request->future;
  }
  if (impl_->paused) return Ready<TrainingPermit>({Error::Code::unavailable, "training admission paused"});
  if (!allow_wait && !impl_->Eligible(task.iteration, impl_->Floor()))
    return Ready<TrainingPermit>({Error::Code::capacity_exceeded, "no capacity for a dependency waiter"});
  request = std::make_unique<Impl::Request>(std::move(task));
  auto future = request->future;
  impl_->GrantReady();
  return future;
}

Result<Ack> TrainingProgressController::Complete(TrainingTask task) {
  std::lock_guard lock(impl_->mutex);
  if (auto error = impl_->Validate(task)) return *error;
  if (!impl_->failure.empty()) return impl_->Failed();
  auto& count = impl_->completed[task.worker];
  if (task.iteration < count) {
    ++impl_->duplicates;
    return Ack{};
  }
  auto& request = impl_->requests[task.worker];
  if (task.iteration != count || !request || !request->active) {
    return Error{Error::Code::invalid_argument, "iteration has no active training permit"};
  }
  ++count;
  ++impl_->finished;
  request.reset();
  if (!impl_->paused) impl_->GrantReady();
  return Ack{};
}

Result<Ack> TrainingProgressController::Pause() {
  std::lock_guard lock(impl_->mutex);
  if (!impl_->failure.empty()) return impl_->Failed();
  impl_->paused = true;
  for (auto& request : impl_->requests) {
    if (!request || request->active) continue;
    request->promise.set_value(Error{Error::Code::unavailable, "training admission paused"});
    request.reset();
  }
  return Ack{};
}

Result<TrainingProgressCheckpoint> TrainingProgressController::Checkpoint() const {
  std::lock_guard lock(impl_->mutex);
  if (!impl_->failure.empty()) return impl_->Failed();
  if (!impl_->paused || std::any_of(impl_->requests.begin(), impl_->requests.end(),
                                  [](const auto& request) { return bool(request); })) {
    return Error{Error::Code::unavailable, "pause and finish admitted training tasks before checkpoint"};
  }
  return TrainingProgressCheckpoint{impl_->options.run, impl_->options.epoch,
      impl_->options.consistency, impl_->options.max_delay, impl_->completed};
}

Result<Ack> TrainingProgressController::Restore(TrainingProgressCheckpoint checkpoint) {
  std::lock_guard lock(impl_->mutex);
  if (!impl_->failure.empty()) return impl_->Failed();
  if (impl_->ever_started || impl_->restored ||
      std::any_of(impl_->requests.begin(), impl_->requests.end(), [](const auto& r) { return bool(r); })) {
    return Error{Error::Code::invalid_argument, "restore requires a fresh training controller"};
  }
  if (checkpoint.source_run.empty() || checkpoint.source_run == impl_->options.run ||
      checkpoint.epoch != impl_->options.epoch ||
      checkpoint.consistency != impl_->options.consistency ||
      checkpoint.max_delay != impl_->options.max_delay ||
      checkpoint.completed.size() != impl_->options.worker_count) {
    return Error{Error::Code::invalid_argument, "training checkpoint identity or group configuration differs"};
  }
  if (checkpoint.consistency != TrainingConsistency::eventual) {
    const auto [low, high] = std::minmax_element(checkpoint.completed.begin(), checkpoint.completed.end());
    const auto difference = *high - *low;
    // A permitted t may complete while another worker is still before t-tau.
    // Completed counts can therefore differ by tau+1, but no further.
    if (difference && difference - 1 > checkpoint.max_delay) {
      return Error{Error::Code::invalid_argument, "training checkpoint clocks violate the dependency policy"};
    }
  }
  impl_->completed = std::move(checkpoint.completed);
  impl_->restored = true;
  impl_->paused = true;
  return Ack{};
}

Result<Ack> TrainingProgressController::Resume() {
  std::lock_guard lock(impl_->mutex);
  if (!impl_->failure.empty()) return impl_->Failed();
  if (impl_->paused && std::any_of(impl_->requests.begin(), impl_->requests.end(),
                                  [](const auto& r) { return bool(r); })) {
    return Error{Error::Code::unavailable, "admitted training tasks have not drained"};
  }
  impl_->paused = false;
  return Ack{};
}

void TrainingProgressController::Abort(std::string reason) {
  std::lock_guard lock(impl_->mutex);
  if (!impl_->failure.empty()) return;
  impl_->failure = reason.empty() ? "training run aborted" : std::move(reason);
  for (auto& request : impl_->requests) {
    if (!request || request->active) continue;
    request->promise.set_value(impl_->Failed());
    request.reset();
  }
}

TrainingProgressStats TrainingProgressController::Stats() const {
  std::lock_guard lock(impl_->mutex);
  TrainingProgressStats result;
  result.completed = impl_->completed;
  result.completion_floor = impl_->Floor();
  result.granted_tasks = impl_->granted;
  result.completed_tasks = impl_->finished;
  result.duplicate_completions = impl_->duplicates;
  result.paused = impl_->paused;
  result.failed = !impl_->failure.empty();
  for (const auto& request : impl_->requests) {
    if (!request) continue;
    if (request->active) ++result.active;
    else ++result.waiting;
  }
  return result;
}

}  // namespace eps
