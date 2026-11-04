#include <brpc/server.h>
#include <butil/endpoint.h>
#include <chrono>
#include <iostream>
#include <thread>
#include "eps/brpc_training.hpp"
#include "transport/brpc_training_server.hpp"

using namespace eps;
using namespace std::chrono_literals;
namespace {
void Check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class T> Result<T> ResultOf(std::future<Result<T>> f) {
  Check(f.wait_for(5s) == std::future_status::ready, "training RPC failed to finish"); return f.get();
}
template<class T> T Take(std::future<Result<T>> f) {
  auto r = ResultOf(std::move(f));
  if (auto* e = std::get_if<Error>(&r)) throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
template<class T> void ErrorIs(std::future<Result<T>> f, Error::Code code) {
  const auto r = ResultOf(std::move(f)); const auto* e = std::get_if<Error>(&r);
  Check(e && e->code == code, "wrong training RPC error");
}
struct Harness {
  transport::BrpcTrainingServer adapter;
  brpc::Server server;
  Harness(std::string run = "rpc", std::size_t capacity = 16)
      : adapter({std::move(run), 1, 2, TrainingConsistency::sequential, 0}, capacity) {
    Check(server.AddService(&adapter, brpc::SERVER_DOESNT_OWN_SERVICE) == 0, "register training RPC");
    brpc::ServerOptions options; options.max_concurrency = 0;
    Check(server.Start("127.0.0.1:0", &options) == 0, "start training RPC");
  }
  std::string Address() const { return butil::endpoint2str(server.listen_address()).c_str(); }
  ~Harness() { adapter.Shutdown(); server.Stop(0); server.Join(); }
};
void SaturationAndLifecycle() {
  Harness h;
  BrpcTrainingProgress c(h.Address(), "rpc");
  Take(c.Begin(0, 0)); Take(c.Complete(0, 0));
  std::vector<std::future<Result<TrainingPermit>>> pending;
  for (unsigned i = 0; i < 80; ++i) pending.push_back(c.Begin(0, 1));
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  std::size_t ready = 0;
  do {
    ready = 0;
    for (auto& f : pending) if (f.wait_for(0s) == std::future_status::ready) ++ready;
    if (ready == 64) break;
    std::this_thread::sleep_for(2ms);
  } while (std::chrono::steady_clock::now() < deadline);
  Check(ready == 64, "training RPC capacity not enforced");
  const auto start = std::chrono::steady_clock::now();
  auto stats = Take(c.Stats());
  Check(stats.completed == std::vector<std::uint64_t>({1, 0}) && stats.waiting == 1,
        "wrong progress while fast worker waits");
  const auto latency = std::chrono::steady_clock::now() - start;
  Check(latency < 1s, "waiting admissions starved stats");
  BrpcTrainingProgress stale(h.Address(), "old-run");
  ErrorIs(stale.Pause(), Error::Code::stale_generation);
  ErrorIs(stale.Abort("stale abort"), Error::Code::stale_generation);
  Take(c.Begin(1, 0)); Take(c.Complete(1, 0));
  unsigned permits = 0, rejected = 0;
  for (auto& f : pending) {
    auto result = ResultOf(std::move(f));
    if (auto* error = std::get_if<Error>(&result)) {
      Check(error->code == Error::Code::capacity_exceeded, "wrong saturation error"); ++rejected;
    } else {
      const auto& p = std::get<TrainingPermit>(result);
      Check(p.task.iteration == 1 && p.observed_completion_floor == 1,
            "waiting permit had wrong observed floor"); ++permits;
    }
  }
  Check(permits == 16 && rejected == 64, "duplicate waiting RPC count changed");
  Take(c.Complete(0, 1)); Take(c.Complete(0, 1));
  auto waits = c.Begin(0, 2);
  while (Take(c.Stats()).waiting != 1) std::this_thread::sleep_for(1ms);
  Take(c.Pause());
  ErrorIs(std::move(waits), Error::Code::unavailable);
  auto snapshot = Take(c.Checkpoint());
  Check(snapshot.completed == std::vector<std::uint64_t>({2, 1}), "checkpoint lost task clocks");
  Take(c.Resume());
  Take(c.Begin(1, 1)); Take(c.Complete(1, 1));
  auto blocked = c.Begin(1, 2);
  Take(c.Begin(0, 2));
  // Both may run round 2. Put worker 0 ahead and then abort its round 3 waiter.
  Take(c.Complete(0, 2));
  auto abort_wait = c.Begin(0, 3);
  Take(c.Abort("worker process failed"));
  ErrorIs(std::move(abort_wait), Error::Code::run_failed);
  Take(std::move(blocked));
  ErrorIs(c.Complete(1, 2), Error::Code::run_failed);
  Check(Take(c.Stats()).failed, "abort did not fail training job");
  std::cout << "training_rpc saturated_waiters=16 rejected=64 stats_latency_us="
            << std::chrono::duration_cast<std::chrono::microseconds>(latency).count()
            << " duplicate_permits=16 lifecycle=pause/checkpoint/resume/abort\n";
}
void ShutdownAndTimeout() {
  Harness h;
  BrpcTrainingProgress c(h.Address(), "rpc");
  Take(c.Begin(0, 0)); Take(c.Complete(0, 0));
  BrpcTrainingProgress timeout(h.Address(), "rpc", 1, 30);
  ErrorIs(timeout.Begin(0, 1), Error::Code::unavailable);
  auto retry = c.Begin(0, 1);
  Take(c.Begin(1, 0)); Take(c.Complete(1, 0));
  Take(std::move(retry)); Take(c.Complete(0, 1));
  auto waiting = c.Begin(0, 2);
  while (Take(c.Stats()).waiting != 1) std::this_thread::sleep_for(1ms);
  h.adapter.Shutdown();
  ErrorIs(std::move(waiting), Error::Code::run_failed);
  Check(Take(c.Stats()).completed == std::vector<std::uint64_t>({2, 1}),
        "timeout retry changed task clocks");
}
}  // namespace
int main() {
  try { SaturationAndLifecycle(); ShutdownAndTimeout(); }
  catch (const std::exception& e) { std::cerr << "training RPC: " << e.what() << '\n'; return 1; }
}
