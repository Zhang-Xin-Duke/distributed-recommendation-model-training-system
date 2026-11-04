#include "eps/synchronous_gradients.hpp"
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>

using namespace eps;
using namespace std::chrono_literals;
namespace {
void Check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class T> T Take(std::shared_future<Result<T>> f) {
  Check(f.wait_for(3s) == std::future_status::ready, "collective completion timed out");
  auto r = f.get(); if (auto* e = std::get_if<Error>(&r)) throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
void Reject(std::shared_future<Result<Ack>> f, Error::Code code) {
  Check(f.wait_for(3s) == std::future_status::ready, "rejection timed out");
  auto r = f.get(); auto* e = std::get_if<Error>(&r);
  Check(e && e->code == code, "unexpected collective error");
}
GradientContribution Packet(unsigned worker, std::uint64_t round, std::vector<Key> keys,
                             std::vector<float> values, std::uint64_t samples = 1) {
  return {{"sync", 1, worker, round}, samples, {{1, 2, std::move(keys), std::move(values)}}};
}
SynchronousGradientOptions Options(GradientReduction reduction) {
  SynchronousGradientOptions o; o.run = "sync"; o.workers = 3;
  o.tables = {{1, 2, {}, {}}}; o.reduction = reduction; return o;
}
void Averaging(GradientReduction reduction, bool reverse) {
  auto options = Options(reduction);
  std::vector<GradientTable> actual;
  std::atomic<unsigned> calls = 0;
  SynchronousGradientCollector c(options, [&](auto round, auto tables) -> Result<Ack> {
    Check(round == 0, "wrong update round"); actual = std::move(tables); ++calls; return Ack{};
  });
  std::vector<GradientContribution> packets{
    Packet(0, 0, {1, 4}, {3, 6, 9, 12}, 1),
    Packet(1, 0, {1, 2}, {6, 9, 12, 15}, 3), Packet(2, 0, {}, {}, 2)};
  std::vector<std::shared_future<Result<Ack>>> pending;
  for (unsigned k = 0; k < 3; ++k) {
    pending.push_back(c.Submit(packets[reverse ? 2-k : k]));
    if (k < 2) Check(pending.back().wait_for(0s) != std::future_status::ready,
                     "round updated before every worker contributed");
  }
  for (auto& p : pending) Take(p);
  Check(calls == 1 && actual[0].keys == std::vector<Key>({1, 2, 4}), "wrong reduction rows");
  const std::vector<float> expected = reduction == GradientReduction::sum ?
    std::vector<float>{9, 15, 12, 15, 9, 12} : reduction == GradientReduction::worker_mean ?
    std::vector<float>{3, 5, 4, 5, 3, 4} : std::vector<float>{3.5, 5.5, 6, 7.5, 1.5, 2};
  Check(actual[0].values == expected, "wrong missing-key or weighted average");
  Take(c.Submit(packets[0]));
  Check(calls == 1 && c.Stats().next_round == 1 && !c.Stats().retained_bytes,
        "completed retry repeated update or retained gradients");
}
void ValidationAndFailure() {
  auto options = Options(GradientReduction::worker_mean);
  options.gradient_bytes = 4096;
  std::atomic<unsigned> calls = 0;
  SynchronousGradientCollector c(options, [&](auto, auto) -> Result<Ack> { ++calls; return Ack{}; });
  auto p = Packet(0, 0, {1}, {1, 2});
  auto first = c.Submit(p), retry = c.Submit(p);
  auto changed = p; changed.tables[0].values[0] = 2;
  Reject(c.Submit(changed), Error::Code::invalid_argument);
  changed = p; changed.task.run = "old"; Reject(c.Submit(changed), Error::Code::stale_generation);
  changed = p; changed.task.iteration = 1; Reject(c.Submit(changed), Error::Code::invalid_argument);
  changed = Packet(1, 0, {1, 1}, {1, 2, 3, 4}); Reject(c.Submit(changed), Error::Code::invalid_argument);
  changed = Packet(1, 0, {1}, {1, std::numeric_limits<float>::infinity()});
  Reject(c.Submit(changed), Error::Code::invalid_argument);
  changed = Packet(1, 0, {1}, {1}); Reject(c.Submit(changed), Error::Code::invalid_argument);
  changed = Packet(1, 0, {}, {}); changed.sample_count = 0;
  Reject(c.Submit(changed), Error::Code::invalid_argument);
  std::vector<Key> keys(100); for (unsigned i = 0; i < keys.size(); ++i) keys[i] = i;
  Reject(c.Submit(Packet(1, 0, keys, std::vector<float>(200))), Error::Code::capacity_exceeded);
  Check(c.Stats().received_workers == 1 && c.Stats().duplicate_submissions == 1 && !calls,
        "invalid packets changed accepted contribution state");
  c.Abort("worker died");
  Reject(first, Error::Code::run_failed); Reject(retry, Error::Code::run_failed);
  Reject(c.Submit(Packet(1, 0, {}, {})), Error::Code::run_failed);
}
void AckAndTimeout() {
  auto options = Options(GradientReduction::sum); options.workers = 1;
  std::promise<void> entered, release;
  auto gate = release.get_future();
  SynchronousGradientCollector c(options, [&](auto, auto) -> Result<Ack> {
    entered.set_value(); gate.wait(); return Ack{};
  });
  auto result = c.Submit(Packet(0, 0, {1}, {1, 2}));
  Check(entered.get_future().wait_for(3s) == std::future_status::ready, "apply never entered");
  Check(c.Stats().applying && c.Stats().next_round == 0 &&
        result.wait_for(0s) != std::future_status::ready, "ack preceded parameter application");
  Check(std::holds_alternative<Error>(c.Pause()) && !c.Stats().paused,
        "an applying optimizer round was checkpointed");
  c.Abort("aborted during apply"); Reject(result, Error::Code::run_failed);
  release.set_value();
  options.workers = 2; options.round_timeout = 20ms;
  SynchronousGradientCollector timed(options, [](auto, auto) -> Result<Ack> {
    throw std::runtime_error("incomplete round must not apply");
  });
  Reject(timed.Submit(Packet(0, 0, {}, {})), Error::Code::run_failed);
  Check(timed.Stats().failed && !timed.Stats().next_round, "missing worker did not fail round");
}
void ConcurrentRounds() {
  constexpr unsigned workers = 8, rounds = 40;
  auto options = Options(GradientReduction::worker_mean); options.workers = workers;
  std::atomic<unsigned> calls = 0, failures = 0;
  SynchronousGradientCollector c(options, [&](auto round, auto tables) -> Result<Ack> {
    Check(round == calls++ && tables[0].values == std::vector<float>({3.5f, 1.f}),
          "concurrent round lost/duplicated gradients"); return Ack{};
  });
  std::vector<std::jthread> threads;
  for (unsigned w = 0; w < workers; ++w) threads.emplace_back([&, w] {
    try {
      for (unsigned r = 0; r < rounds; ++r) {
        auto p = Packet(w, r, {1}, {float(w), 1});
        Take(c.Submit(p)); Take(c.Submit(std::move(p)));
      }
    } catch (...) { ++failures; c.Abort("test worker failed"); }
  });
  threads.clear();
  Check(!failures && calls == rounds && c.Stats().next_round == rounds,
        "concurrent collective did not finish exactly once per round");
}
void CheckpointBoundary() {
  auto options = Options(GradientReduction::sum); options.workers = 2;
  SynchronousGradientCollector original(options, [](auto, auto) -> Result<Ack> { return Ack{}; });
  Check(std::holds_alternative<Error>(original.Checkpoint()), "unpaused checkpoint accepted");
  auto first = original.Submit(Packet(0, 0, {}, {}));
  Check(std::holds_alternative<Error>(original.Pause()) && !original.Stats().paused,
        "partial round was frozen");
  Take(original.Submit(Packet(1, 0, {}, {}))); Take(first);
  Check(std::holds_alternative<Ack>(original.Pause()), "boundary pause failed");
  Reject(original.Submit(Packet(0, 1, {}, {})), Error::Code::unavailable);
  const auto saved = std::get<SynchronousGradientCheckpoint>(original.Checkpoint());
  Check(saved.next_round == 1, "wrong saved next round");
  options.run = "restored";
  std::atomic<unsigned> calls = 0;
  SynchronousGradientCollector restored(options, [&](auto round, auto) -> Result<Ack> {
    Check(round == 1, "restore repeated retired round"); ++calls; return Ack{};
  });
  auto wrong = saved; wrong.workers = 3;
  Check(std::holds_alternative<Error>(restored.Restore(wrong)), "mismatched membership restored");
  Check(std::holds_alternative<Ack>(restored.Restore(saved)) && restored.Stats().paused,
        "restore did not preserve paused state");
  auto packet = Packet(0, 0, {}, {}); packet.task.run = options.run;
  Reject(restored.Submit(packet), Error::Code::retired_request);
  packet.task.iteration = 1;
  Reject(restored.Submit(packet), Error::Code::unavailable);
  Check(std::holds_alternative<Ack>(restored.Resume()), "resume failed");
  auto pending = restored.Submit(packet); packet.task.worker = 1;
  Take(restored.Submit(packet)); Take(pending); Take(restored.Submit(packet));
  Check(calls == 1 && restored.Stats().next_round == 2, "resumed retry repeated application");
  Check(std::holds_alternative<Error>(restored.Restore(saved)), "live collector restored twice");
}
}
int main() {
  try {
    for (auto reduction : {GradientReduction::sum, GradientReduction::worker_mean, GradientReduction::sample_mean})
      for (bool reverse : {false, true}) Averaging(reduction, reverse);
    ValidationAndFailure(); AckAndTimeout(); ConcurrentRounds(); CheckpointBoundary();
    std::cout << "PASS synchronous gradients: sum/worker/sample mean, sparse union/empty contribution, "
                 "duplicates, invalid payloads, budget, timeout/abort, ack ordering, 8-worker/40-round execution\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
