#include "eps/training_progress.hpp"
#include "eps/entropy_adapter.hpp"
#include "eps/runtime.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace eps;
using namespace std::chrono_literals;

namespace {
void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
template <class T>
T Take(Result<T> result) {
  if (auto* error = std::get_if<Error>(&result)) throw std::runtime_error(error->message);
  return std::get<T>(std::move(result));
}
template <class T>
T Take(std::future<Result<T>> result) { return Take(result.get()); }
template <class T>
void Reject(Result<T> result, Error::Code code) {
  auto* error = std::get_if<Error>(&result);
  Check(error && error->code == code, "unexpected rejection code");
}
TrainingTask Task(std::uint32_t worker, std::uint64_t iteration,
                  std::string run = "training") {
  return {std::move(run), 1, worker, iteration};
}
bool Ready(const std::shared_future<Result<TrainingPermit>>& future) {
  return future.wait_for(0s) == std::future_status::ready;
}
TrainingPermit Permit(std::shared_future<Result<TrainingPermit>> future) {
  Check(future.wait_for(2s) == std::future_status::ready, "training admission did not complete");
  return Take(future.get());
}

void Sequential(TrainingConsistency policy = TrainingConsistency::sequential) {
  TrainingProgressController controller({"training", 1, 2, policy});
  auto first = Permit(controller.Begin(Task(0, 0)));
  auto repeated = Permit(controller.Begin(Task(0, 0)));
  Check(first.task == repeated.task && controller.Stats().granted_tasks == 1,
        "duplicate admission allocated another task");
  Take(controller.Complete(first.task));
  auto waiting = controller.Begin(Task(0, 1));
  auto repeated_wait = controller.Begin(Task(0, 1));
  Check(!Ready(waiting) && !Ready(repeated_wait), "sequential iteration passed a missing contribution");
  Reject(controller.Begin(Task(0, 2)).get(), Error::Code::invalid_argument);
  Reject(controller.Complete(Task(0, 1)), Error::Code::invalid_argument);
  Take(controller.Complete(Task(0, 0)));
  Check(!Ready(waiting), "duplicate completion advanced progress twice");
  auto slow = Permit(controller.Begin(Task(1, 0)));
  Take(controller.Complete(slow.task));
  auto next = Permit(waiting);
  Check(next.observed_completion_floor == 1, "wrong sequential admission floor");
  Check(Permit(repeated_wait).observed_completion_floor == 1, "waiting retries disagree");
  Take(controller.Complete(next.task));
  Reject(controller.Begin(Task(0, 0)).get(), Error::Code::retired_request);
  auto stats = controller.Stats();
  Check(stats.completed == std::vector<std::uint64_t>({2, 1}) &&
        stats.completion_floor == 1 && stats.completed_tasks == 3 &&
        stats.duplicate_completions == 1, "incorrect sequential clocks");
}

void Backpressure() {
  TrainingProgressController c({"training", 1, 2});
  Take(c.Complete(Permit(c.Begin(Task(0, 0), false)).task));
  Reject(c.Begin(Task(0, 1), false).get(), Error::Code::capacity_exceeded);
  Check(!c.Stats().waiting && !c.Stats().active, "rejected admission changed task state");
  auto waiting = c.Begin(Task(0, 1));
  Reject(c.Begin(Task(0, 1), false).get(), Error::Code::capacity_exceeded);
  Check(c.Stats().waiting == 1, "backpressure duplicated an existing waiter");
  Take(c.Complete(Permit(c.Begin(Task(1, 0), false)).task));
  const auto retried = Permit(c.Begin(Task(0, 1), false));
  Check(retried.task == Permit(waiting).task && c.Stats().granted_tasks == 3,
        "backpressure blocked a ready slow-worker permit or duplicated a task");
  Take(c.Complete(retried.task));
}

void BoundedDelay() {
  TrainingProgressController controller({"training", 1, 2, TrainingConsistency::bounded_delay, 1});
  for (std::uint64_t iteration = 0; iteration < 2; ++iteration) {
    auto permit = Permit(controller.Begin(Task(0, iteration)));
    Check(permit.observed_completion_floor == 0, "slow worker should still be at zero");
    Take(controller.Complete(permit.task));
  }
  // Slide 18: t=2 must wait for iteration 0, while t=1 did not.
  auto second = controller.Begin(Task(0, 2));
  Check(!Ready(second), "bounded delay admitted t=2 before iteration zero finished");
  auto slow = Permit(controller.Begin(Task(1, 0)));
  Take(controller.Complete(slow.task));
  auto permit = Permit(second);
  Check(permit.observed_completion_floor == 1, "bounded delay released at wrong floor");
  Take(controller.Complete(permit.task));
  auto third = controller.Begin(Task(0, 3));
  Check(!Ready(third), "bounded delay admitted t=3 before iteration one finished");
  Take(controller.Complete(Permit(controller.Begin(Task(1, 1))).task));
  Check(Permit(third).observed_completion_floor == 2, "bounded delay did not release t=3");
}

void Eventual() {
  TrainingProgressController controller({"training", 1, 2, TrainingConsistency::eventual});
  for (std::uint64_t iteration = 0; iteration < 100; ++iteration) {
    auto permit = Permit(controller.Begin(Task(0, iteration)));
    Check(permit.observed_completion_floor == 0, "eventual policy moved the idle worker clock");
    Take(controller.Complete(permit.task));
  }
  Check(controller.Stats().completed == std::vector<std::uint64_t>({100, 0}),
        "eventual policy imposed a finite delay bound");
}

void PauseRestore() {
  TrainingProgressController controller({"training", 1, 2});
  auto active = Permit(controller.Begin(Task(0, 0)));
  Reject(controller.Checkpoint(), Error::Code::unavailable);
  Take(controller.Pause());
  Reject(controller.Resume(), Error::Code::unavailable);
  Reject(controller.Checkpoint(), Error::Code::unavailable);
  Take(controller.Complete(active.task));
  Take(controller.Resume());
  auto waiting = controller.Begin(Task(0, 1));
  Check(!Ready(waiting), "expected waiting iteration before checkpoint pause");
  Take(controller.Pause());
  Reject(waiting.get(), Error::Code::unavailable);
  auto checkpoint = Take(controller.Checkpoint());
  Check(checkpoint.completed == std::vector<std::uint64_t>({1, 0}), "checkpoint lost per-worker clocks");
  Reject(controller.Begin(Task(1, 0)).get(), Error::Code::unavailable);

  TrainingProgressController resumed({"resumed", 1, 2});
  auto malformed = checkpoint;
  malformed.completed.pop_back();
  Reject(resumed.Restore(malformed), Error::Code::invalid_argument);
  malformed = checkpoint;
  malformed.completed[0] = 3;
  Reject(resumed.Restore(malformed), Error::Code::invalid_argument);
  Check(!resumed.Stats().paused && resumed.Stats().completed[0] == 0,
        "failed restore changed controller state");
  Take(resumed.Restore(checkpoint));
  Check(resumed.Stats().paused && resumed.Stats().completed_tasks == 0,
        "restore reopened admission or persisted live-run RPC counters");
  Reject(resumed.Begin(Task(1, 0, "resumed")).get(), Error::Code::unavailable);
  Take(resumed.Resume());
  auto fast = resumed.Begin(Task(0, 1, "resumed"));
  Check(!Ready(fast), "restore forgot the unfinished slow iteration");
  Take(resumed.Complete(Permit(resumed.Begin(Task(1, 0, "resumed"))).task));
  Check(Permit(fast).observed_completion_floor == 1, "restored progress did not release the next task");
  Reject(resumed.Complete(Task(0, 1)), Error::Code::stale_generation);
  Reject(resumed.Restore(checkpoint), Error::Code::invalid_argument);
  TrainingProgressController same_run({"training", 1, 2});
  Reject(same_run.Restore(checkpoint), Error::Code::invalid_argument);
}

void FailureAndValidation() {
  TrainingProgressController controller({"training", 1, 2});
  Reject(controller.Begin({"other", 1, 0, 0}).get(), Error::Code::stale_generation);
  Reject(controller.Begin({"training", 2, 0, 0}).get(), Error::Code::stale_generation);
  Reject(controller.Begin(Task(2, 0)).get(), Error::Code::invalid_argument);
  Reject(controller.Begin(Task(0, std::numeric_limits<std::uint64_t>::max())).get(),
         Error::Code::invalid_argument);
  Take(controller.Complete(Permit(controller.Begin(Task(0, 0))).task));
  auto waiting = controller.Begin(Task(0, 1));
  controller.Abort("worker failed");
  Reject(waiting.get(), Error::Code::run_failed);
  Reject(controller.Begin(Task(1, 0)).get(), Error::Code::run_failed);
  Reject(controller.Complete(Task(0, 0)), Error::Code::run_failed);
  Reject(controller.Resume(), Error::Code::run_failed);
  Reject(controller.Checkpoint(), Error::Code::run_failed);
  Check(controller.Stats().failed && !controller.Stats().waiting,
        "abort did not release blocked callers");
  std::shared_future<Result<TrainingPermit>> after_destroy;
  {
    TrainingProgressController owned({"training", 1, 2});
    Take(owned.Complete(Permit(owned.Begin(Task(0, 0))).task));
    after_destroy = owned.Begin(Task(0, 1));
  }
  Reject(after_destroy.get(), Error::Code::run_failed);
  for (auto options : std::vector<TrainingProgressOptions>{
      {"", 1, 2}, {"bad", 0, 2}, {"bad", 1, 0}, {"bad", 1, 1025},
      {"bad", 1, 2, TrainingConsistency::sequential, 1},
      {"bad", 1, 2, static_cast<TrainingConsistency>(99), 0}}) {
    bool rejected = false;
    try { TrainingProgressController invalid(options); }
    catch (const std::invalid_argument&) { rejected = true; }
    Check(rejected, "invalid training group accepted");
  }
}

void ConcurrentWorkers(TrainingConsistency policy, std::uint64_t delay) {
  constexpr unsigned workers = 8, iterations = 80;
  TrainingProgressController controller({"training", 1, workers, policy, delay});
  std::atomic<unsigned> failures = 0;
  std::vector<std::jthread> threads;
  for (unsigned worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        for (unsigned iteration = 0; iteration < iterations; ++iteration) {
          auto permit = Permit(controller.Begin(Task(worker, iteration)));
          if (policy != TrainingConsistency::eventual && iteration > delay) {
            Check(permit.observed_completion_floor >= iteration - delay,
                  "concurrent worker exceeded its admission dependency bound");
          }
          if (worker == workers - 1 && iteration % 11 == 0) std::this_thread::sleep_for(1ms);
          Take(controller.Complete(permit.task));
        }
      } catch (...) {
        ++failures;
        controller.Abort("test worker failed");
      }
    });
  }
  threads.clear();
  auto stats = controller.Stats();
  Check(!failures && stats.completed == std::vector<std::uint64_t>(workers, iterations) &&
        stats.completion_floor == iterations && stats.granted_tasks == workers * iterations &&
        stats.completed_tasks == workers * iterations && !stats.active && !stats.waiting,
        "concurrent training progress did not drain correctly");
}

// A controlled training integration: real sparse logistic-loss gradients,
// sharded parameters and acknowledged LazyAdam updates. The four workers and
// two parameter-service actors run in one process; this is not an RPC/GPU or
// common-quality performance measurement.
void SparseTrainingIntegration(TrainingConsistency policy, std::uint64_t delay) {
  constexpr unsigned workers = 4, iterations = 40;
  std::vector<std::shared_ptr<CpuParameterService>> homes;
  std::vector<std::shared_ptr<ParameterService>> endpoints;
  for (unsigned shard = 0; shard < 2; ++shard) {
    RuntimeOptions options;
    options.run_id = "training";
    options.shard_id = shard;
    options.shard_count = 2;
    homes.push_back(std::make_shared<CpuParameterService>(options));
    endpoints.push_back(homes.back());
  }
  entropy::TableAdapter setup({"training", 1000}, endpoints);
  TableDefinition definition{1, 1, 2, 1, {.02f, .9f, .999f, 1e-8f, .9f, .999f}, 0, 0, 42};
  Take(setup.CreateTable(definition));
  auto features = [](unsigned worker) {
    return std::vector<entropy::Feature>{{1, 0, Pooling::sum, {1, Key(worker + 2)}, {0, 2}, {}}};
  };
  auto loss = [](double score, double label) {
    return std::max(score, 0.) - label * score + std::log1p(std::exp(-std::abs(score)));
  };
  TrainingProgressController controller({"training", 1, workers, policy, delay});
  std::atomic<unsigned> failures = 0;
  std::vector<std::jthread> threads;
  for (unsigned worker = 0; worker < workers; ++worker) {
    threads.emplace_back([&, worker] {
      try {
        entropy::TableAdapter adapter({"training", worker}, endpoints);
        Take(adapter.CreateTable(definition));
        for (unsigned iteration = 0; iteration < iterations; ++iteration) {
          auto permit = Permit(controller.Begin(Task(worker, iteration)));
          auto batch = Take(adapter.Pull(features(worker)));
          const double score = batch.features.at(0).values.at(0);
          const float gradient = float(1. / (1. + std::exp(-score)) - (worker % 2));
          auto gradients = batch.features;
          gradients.at(0).values.at(0) = gradient;
          Take(adapter.Push(batch.handle, std::move(gradients)));
          // Training progress advances after the parameter update's ack.
          Take(controller.Complete(permit.task));
        }
      } catch (...) {
        ++failures;
        controller.Abort("sparse training integration failed");
      }
    });
  }
  threads.clear();
  Check(!failures && controller.Stats().completed_tasks == workers * iterations,
        "sparse training workers failed or lost a completed update");
  double final_loss = 0;
  for (unsigned worker = 0; worker < workers; ++worker) {
    auto batch = Take(setup.Pull(features(worker), true));
    final_loss += loss(batch.features.at(0).values.at(0), worker % 2) / workers;
    Take(setup.Release(batch.handle));
  }
  Check(std::isfinite(final_loss) && final_loss < std::log(2.), "controlled sparse training did not reduce loss");
  for (Key key = 1; key < workers + 2; ++key) {
    auto shard = key % 2;
    auto rows = Take(homes[shard]->Locations({{1, key}}));
    auto state = Take(homes[shard]->ReadState(rows)).at(0);
    Check(state.row_updates == (key == 1 ? workers * iterations : iterations),
          "worker progress and actual sparse row updates disagree");
    Check(std::isfinite(state.weights[0]) && std::isfinite(state.first_moment[0]) &&
          std::isfinite(state.second_moment[0]), "training produced non-finite optimizer state");
  }
  for (const auto& home : homes) Check(Take(home->Stats()).live_tickets == 0, "training leaked a parameter ticket");
  std::cout << "controlled_training policy=" << int(policy) << " workers=4 shards=2 updates=160"
            << " initial_loss=" << std::log(2.) << " final_loss=" << final_loss << '\n';
}
}  // namespace

int main() {
  try {
    TrainingProgressController recovered({"recover",1,2,TrainingConsistency::sequential,0});
    Reject(recovered.RecoverSynchronousRound(UINT64_MAX),Error::Code::invalid_argument);
    Take(recovered.RecoverSynchronousRound(50000));
    Check(recovered.Stats().paused && recovered.Stats().completed==std::vector<std::uint64_t>({50000,50000}),"recovered round/cursor differs");
    Reject(recovered.Begin({"recover",1,0,50000}).get(),Error::Code::unavailable);
    Reject(recovered.RecoverSynchronousRound(50000),Error::Code::invalid_argument);
    Take(recovered.Resume());Take(recovered.Begin({"recover",1,0,50000}).get());Take(recovered.Complete({"recover",1,0,50000}));
    TrainingProgressController eventual_recovery({"eventual-recovery",1,2,TrainingConsistency::eventual,0});
    Reject(eventual_recovery.RecoverSynchronousRound(1),Error::Code::invalid_argument);
    Sequential();
    Sequential(TrainingConsistency::bounded_delay);
    Backpressure();
    BoundedDelay();
    Eventual();
    PauseRestore();
    FailureAndValidation();
    ConcurrentWorkers(TrainingConsistency::sequential, 0);
    ConcurrentWorkers(TrainingConsistency::bounded_delay, 3);
    ConcurrentWorkers(TrainingConsistency::eventual, 0);
    SparseTrainingIntegration(TrainingConsistency::sequential, 0);
    SparseTrainingIntegration(TrainingConsistency::bounded_delay, 2);
    SparseTrainingIntegration(TrainingConsistency::eventual, 0);
    std::cout << "PASS training task consistency: sequential/eventual/bounded-delay, "
                 "slide-18 boundaries, duplicate completion, pause/restore, abort, "
                 "and 8-thread progress across all policies\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL training progress: " << error.what() << '\n';
    return 1;
  }
}
