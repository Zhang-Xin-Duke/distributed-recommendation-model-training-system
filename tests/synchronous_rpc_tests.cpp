#include "eps/brpc_client.hpp"
#include "eps/brpc_synchronous_gradients.hpp"
#include "transport/brpc_server.hpp"
#include "transport/brpc_synchronous_server.hpp"
#include <brpc/server.h>
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <butil/endpoint.h>
#include <chrono>
#include <iostream>
#include <thread>

using namespace eps;
using namespace std::chrono_literals;
namespace {
void Check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class T> T Take(std::future<Result<T>> f) {
  Check(f.wait_for(5s) == std::future_status::ready, "synchronous RPC completion timed out");
  auto r = f.get(); if (auto* e = std::get_if<Error>(&r)) throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
template<class T> void Reject(std::future<Result<T>> f, Error::Code code) {
  Check(f.wait_for(5s) == std::future_status::ready, "synchronous RPC rejection timed out");
  const auto r = f.get(); const auto* e = std::get_if<Error>(&r);
  if (!e || e->code != code) throw std::runtime_error(e ?
    "unexpected synchronous RPC error: " + e->message : "expected an RPC error");
}
TableDefinition Table() { return {1, 2, 1, 1, {.02f, .9f, .999f, 1e-8f, .9f, .999f}, 0, 0, 42}; }
struct Harness {
  std::filesystem::path directory = std::filesystem::temp_directory_path() /
    ("eps-sync-rpc-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  CpuParameterService parameters;
  transport::BrpcServer access;
  transport::BrpcSynchronousServer rounds;
  brpc::Server server;
  Harness(unsigned workers = 3, unsigned timeout = 30000)
      : parameters([this] { RuntimeOptions o; o.run_id = "sync"; o.exclusive_update_learner = 1023;
          o.snapshot_directory = directory; return o; }()),
        access(parameters, 0), rounds(parameters, [&] {
          SynchronousGradientOptions o; o.run = "sync"; o.workers = workers;
          o.reduction = GradientReduction::worker_mean; o.tables = {{1, 2, {}, {}}};
          o.round_timeout = std::chrono::milliseconds(timeout); return o;
        }()) {
    Take(parameters.CreateTable(Table()));
    Check(server.AddService(&access, brpc::SERVER_DOESNT_OWN_SERVICE) == 0 &&
      server.AddService(&rounds, brpc::SERVER_DOESNT_OWN_SERVICE) == 0, "register synchronous services");
    brpc::ServerOptions o; o.max_concurrency = 0;
    Check(server.Start("127.0.0.1:0", &o) == 0, "start synchronous server");
  }
  std::string Address() const { return butil::endpoint2str(server.listen_address()).c_str(); }
  ~Harness() { rounds.Shutdown(); server.Stop(0); server.Join(); std::filesystem::remove_all(directory); }
};
GradientContribution Packet(unsigned worker, unsigned round, bool empty = false) {
  return {{"sync", 1, worker, round}, 1,
    {{1, 2, empty ? std::vector<Key>{} : std::vector<Key>{1},
      empty ? std::vector<float>{} : std::vector<float>{float(worker * 3 + 3), float(worker * 3 + 6)}}}};
}
void RoundApplicationAndAdmission() {
  Harness h;
  BrpcSynchronousGradients c(h.Address(), "sync");
  BrpcParameterService access(h.Address());
  // Worker reads remain available; ordinary writes cannot bypass aggregation.
  LookupRequest lookup{{"sync", 0, 0, Phase::lookup}, 1, false,
    {{1, 0, Pooling::sum, {1}, {0, 1}, {1}}}};
  Reject(access.Lookup(lookup), Error::Code::invalid_argument);
  lookup.read_only = true;
  auto ticket = Take(access.Lookup(lookup));
  Reject(access.Push({{"sync", 0, 0, Phase::push}, ticket.ticket, {{0, {1, 2}, {1, 1}}}}),
    Error::Code::invalid_argument);
  Take(access.Release({{"sync", 0, 0, Phase::release}, ticket.ticket}));
  auto first = c.Submit(Packet(0, 0));
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (Take(c.Stats()).received_workers != 1) {
    Check(std::chrono::steady_clock::now() < deadline, "first contribution not admitted");
    std::this_thread::sleep_for(1ms);
  }
  for (unsigned i = 0; i < 64; ++i) Reject(c.Submit(Packet(0, 0)), Error::Code::capacity_exceeded);
  BrpcSynchronousGradients stale(h.Address(), "old");
  Reject(stale.Abort("old abort"), Error::Code::stale_generation);
  Check(Take(c.Stats()).received_workers == 1 && !Take(c.Stats()).failed,
    "duplicate RPCs or stale abort changed membership");
  auto second = c.Submit(Packet(1, 0)), third = c.Submit(Packet(2, 0, true));
  Take(std::move(first)); Take(std::move(second)); Take(std::move(third));
  Take(c.Submit(Packet(0, 0)));
  auto location = Take(access.Locations({{1, 1}}));
  auto state = Take(access.ReadState(location)).at(0);
  Check(state.row_updates == 1, "round performed more than one optimizer update");
  RuntimeOptions options; options.run_id = "reference";
  CpuParameterService reference(options); Take(reference.CreateTable(Table()));
  auto apply = [&](std::uint64_t round, bool empty) {
    auto q = LookupRequest{{"reference", 0, round, Phase::lookup}, 1, false,
      {{1, 0, Pooling::sequence, empty ? std::vector<Key>{} : std::vector<Key>{1},
        {0, empty ? 0UL : 1UL}, {empty ? 0UL : 1UL}}}};
    auto t = Take(reference.Lookup(q));
    Take(reference.Push({{"reference", 0, round, Phase::push}, t.ticket,
      {{0, {empty ? 0UL : 1UL, 2}, empty ? std::vector<float>{} : std::vector<float>{3, 5}}}}));
  };
  apply(0, false);
  auto expected = Take(reference.ReadState({{{1, 1}, 0, Placement::cpu, 0, 0, 0}})).at(0);
  Check(state.weights == expected.weights && state.first_moment == expected.first_moment &&
    state.second_moment == expected.second_moment, "round optimizer differs from one averaged update");
  std::vector<std::future<Result<Ack>>> empty;
  for (unsigned w = 0; w < 3; ++w) empty.push_back(c.Submit(Packet(w, 1, true)));
  for (auto& f : empty) Take(std::move(f));
  apply(1, true);
  auto a = c.Submit(Packet(0, 2)), b = c.Submit(Packet(1, 2)), d = c.Submit(Packet(2, 2, true));
  Take(std::move(a)); Take(std::move(b)); Take(std::move(d)); apply(2, false);
  state = Take(access.ReadState(location)).at(0);
  expected = Take(reference.ReadState({{{1, 1}, 0, Placement::cpu, 0, 0, 0}})).at(0);
  Check(state.weights == expected.weights && state.first_moment == expected.first_moment &&
    state.second_moment == expected.second_moment && state.row_updates == 2,
    "empty round failed to advance optimizer context or modified absent rows");
  auto saved = Take(access.DrainAndFlush("sync-check"));
  Check(saved.optimizer_contexts.at(0).optimizer_sequence == 3,
    "optimizer context did not count each complete round exactly once");
  auto pending = c.Submit(Packet(0, 3, true));
  const auto shutdown_deadline = std::chrono::steady_clock::now() + 3s;
  while (Take(c.Stats()).received_workers != 1) {
    Check(std::chrono::steady_clock::now() < shutdown_deadline, "shutdown waiter never admitted");
    std::this_thread::sleep_for(1ms);
  }
  h.rounds.Shutdown(); Reject(std::move(pending), Error::Code::run_failed);
  std::cout << "PASS synchronous RPC: per-worker response slots, 64 duplicate rejections, "
               "averaged optimizer state exact, empty-round clock, write guard, stale control, shutdown\n";
}
void MissingWorker() {
  Harness h(2, 30); BrpcSynchronousGradients c(h.Address(), "sync");
  Reject(c.Submit(Packet(0, 0)), Error::Code::run_failed);
  Check(Take(c.Stats()).failed && Take(h.parameters.Stats()).updates == 0,
    "incomplete round changed parameters or remained live");
}
void CheckpointAndRestore() {
  Harness h(2); BrpcSynchronousGradients c(h.Address(), "sync");
  Reject(c.Checkpoint(), Error::Code::unavailable);
  auto a = c.Submit(Packet(0, 0)); Take(c.Submit(Packet(1, 0))); Take(std::move(a));
  Take(c.Pause()); const auto saved = Take(c.Checkpoint());
  Check(saved.next_round == 1 && Take(c.Stats()).paused, "RPC checkpoint boundary mismatch");
  Reject(c.Submit(Packet(0, 1)), Error::Code::unavailable);
  BrpcParameterService access(h.Address());
  const auto snapshot = Take(access.DrainAndFlush("round-one"));
  const auto retry = Take(access.DrainAndFlush("round-one"));
  Check(retry.state_files == snapshot.state_files, "lost save acknowledgement retry changed snapshot");
  const auto original = Take(access.ReadState(Take(access.Locations({{1, 1}})))).at(0);
  RuntimeOptions o; o.run_id = "restored"; o.start_paused = true; o.exclusive_update_learner = 1023;
  CpuParameterService parameters(o); Take(parameters.Restore(snapshot.state_files.at(0)));
  Take(parameters.CreateTable(Table()));
  auto changed = Table(); changed.optimizer.learning_rate *= 2;
  Check(std::holds_alternative<Error>(parameters.CreateTable(changed).get()), "different optimizer restored");
  SynchronousGradientOptions options; options.run = o.run_id; options.workers = 2;
  options.tables = {{1, 2, {}, {}}}; options.reduction = GradientReduction::worker_mean;
  transport::BrpcSynchronousServer rounds(parameters, options);
  const auto provenance = Take(parameters.RestoredFrom());
  auto wrong = provenance; ++wrong.checksum;
  Check(std::holds_alternative<Error>(rounds.Restore(saved, wrong)), "wrong snapshot identity restored");
  auto clock = saved; ++clock.next_round;
  Check(std::holds_alternative<Error>(rounds.Restore(clock, provenance)), "wrong optimizer clock restored");
  Check(std::holds_alternative<Ack>(rounds.Restore(saved, provenance)), "valid group restore rejected");
  brpc::Server server;
  Check(server.AddService(&rounds, brpc::SERVER_DOESNT_OWN_SERVICE) == 0, "register restored RPC");
  Check(server.Start("127.0.0.1:0", nullptr) == 0, "start restored RPC");
  BrpcSynchronousGradients resumed(butil::endpoint2str(server.listen_address()).c_str(), o.run_id);
  auto packet = Packet(0, 0); packet.task.run = o.run_id;
  Reject(resumed.Submit(packet), Error::Code::retired_request);
  packet.task.iteration = 1; Reject(resumed.Submit(packet), Error::Code::unavailable);
  Take(resumed.Resume());
  a = resumed.Submit(packet); packet = Packet(1, 1); packet.task.run = o.run_id;
  Take(resumed.Submit(packet)); Take(std::move(a)); Take(resumed.Submit(packet));
  Take(c.Resume()); a = c.Submit(Packet(0, 1)); Take(c.Submit(Packet(1, 1))); Take(std::move(a));
  const auto expected = Take(access.ReadState(Take(access.Locations({{1, 1}})))).at(0);
  const auto actual = Take(parameters.ReadState({{{1, 1}, 0, Placement::cpu, 0, 0, 0}})).at(0);
  Check(actual.weights == expected.weights && actual.first_moment == expected.first_moment &&
    actual.second_moment == expected.second_moment && actual.row_updates == 2 && original.row_updates == 1,
    "restored round differs from uninterrupted optimizer update");
  rounds.Shutdown(); server.Stop(0); server.Join();
}
}
int main() {
  try { RoundApplicationAndAdmission(); MissingWorker(); CheckpointAndRestore(); }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
