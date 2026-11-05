#include "eps/brpc_client.hpp"
#include "eps/brpc_synchronous_gradients.hpp"
#include "transport/brpc_server.hpp"
#include "transport/brpc_synchronous_server.hpp"
#include "transport/codec.hpp"
#include <brpc/channel.h>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <iostream>
#include <thread>
using namespace eps;
using namespace std::chrono_literals;
namespace {
void Check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
template <class T> T Take(std::future<Result<T>> future) {
  Check(future.wait_for(8s) == std::future_status::ready,
        "gradient layout RPC did not complete");
  auto value = future.get();
  if (auto *error = std::get_if<Error>(&value))
    throw std::runtime_error(error->message);
  return std::get<T>(std::move(value));
}
struct Harness {
  CpuParameterService parameters{[] {
    RuntimeOptions o;
    o.run_id = "gradient-layouts";
    o.exclusive_update_learner = 1023;
    return o;
  }()};
  transport::BrpcServer access{parameters, 0};
  transport::BrpcSynchronousServer rounds;
  brpc::Server server;
  Harness(GradientLayoutCacheOptions cache = {}, unsigned workers = 1)
      : rounds(
            parameters,
            [&] {
              SynchronousGradientOptions o;
              o.run = "gradient-layouts";
              o.workers = workers;
              o.tables = {{1, 2, {}, {}}};
              return o;
            }(),
            {}, cache) {
    Take(parameters.CreateTable(
        {1, 2, 1, 1, {.02f, .9f, .999f, 1e-8f, .9f, .999f}, 0, 0, 42}));
    Check(!server.AddService(&access, brpc::SERVER_DOESNT_OWN_SERVICE) &&
              !server.AddService(&rounds, brpc::SERVER_DOESNT_OWN_SERVICE),
          "register gradient services");
    Check(!server.Start("127.0.0.1:0", nullptr), "start gradient server");
  }
  std::string Address() const {
    return butil::endpoint2str(server.listen_address()).c_str();
  }
  ~Harness() {
    rounds.Shutdown();
    server.Stop(0);
    server.Join();
  }
};
GradientContribution Packet(unsigned round, Key first = 0, unsigned count = 512,
                            unsigned worker = 0) {
  GradientContribution c{
      {"gradient-layouts", 1, worker, round}, 3, {{1, 2, {}, {}}}};
  for (unsigned i = 0; i < count; ++i) {
    c.tables[0].keys.push_back(first + i);
    c.tables[0].values.push_back(float(round + 1) * .01f);
    c.tables[0].values.push_back(float(round + 2) * .02f);
  }
  return c;
}
struct RunResult {
  std::vector<RowState> rows;
  GradientTransportStats transport;
};
RunResult Run(bool cache, GradientLayoutCacheOptions server_cache = {}) {
  Harness h(server_cache);
  BrpcSynchronousGradients c(h.Address(), "gradient-layouts");
  c.SetGradientLayoutCache(cache);
  BrpcParameterService access(h.Address());
  for (unsigned round = 0; round < 6; ++round)
    Take(c.Submit(Packet(round)));
  auto stats = c.TransportStats();
  Check(Take(c.Stats()).next_round == 6, "gradient round clock differs");
  std::vector<RowKey> keys;
  for (Key key = 0; key < 512; ++key)
    keys.push_back({1, key});
  auto rows = Take(access.ReadState(Take(access.Locations(keys))));
  for (const auto &row : rows)
    Check(row.row_updates == 6, "cached gradient was skipped or repeated");
  Take(c.Submit(Packet(5)));
  Check(Take(c.Stats()).next_round == 6,
        "completed cached retry repeated optimizer update");
  if (cache && server_cache.entries) {
    Check(stats.full_requests == 1 && stats.token_requests == 5 &&
              stats.avoided_request_bytes > 0,
          "repeated gradients did not reuse Key metadata");
    Check(Take(c.LayoutStats()).hits == 6,
          "server did not resolve gradient tokens");
  } else
    Check(stats.full_requests == 6 && !stats.token_requests,
          "disabled registry sent token requests");
  return {std::move(rows), stats};
}
void ExactStateAndBytes() {
  const auto full = Run(false), cached = Run(true),
             disabled = Run(true, {0, 1 << 20, 1000ms});
  Check(cached.rows.size() == full.rows.size(),
        "cached gradient row count differs");
  for (std::size_t i = 0; i < full.rows.size(); ++i) {
    for (const auto *current : {&cached.rows[i], &disabled.rows[i]})
      Check(current->weights == full.rows[i].weights &&
                current->first_moment == full.rows[i].first_moment &&
                current->second_moment == full.rows[i].second_moment &&
                current->row_updates == full.rows[i].row_updates,
            "fresh gradient/Adam state changed under metadata reuse");
  }
  Check(cached.transport.request_bytes < full.transport.request_bytes,
        "gradient metadata did not reduce request bytes");
  std::cout << "GRADIENT_LAYOUT_BYTES {\"full\":"
            << full.transport.request_bytes
            << ",\"cached\":" << cached.transport.request_bytes
            << ",\"rows\":512,\"rounds\":6,\"dimension\":2}\n";
}
void ExpiryAndEviction() {
  Harness h({1, 1 << 20, 100ms});
  BrpcSynchronousGradients c(h.Address(), "gradient-layouts");
  c.SetGradientLayoutCache(true);
  Take(c.Submit(Packet(0, 0)));
  Take(c.Submit(Packet(1, 1024)));
  Take(c.Submit(Packet(2, 0)));
  Check(c.TransportStats().fallback_requests == 1,
        "evicted gradient token did not fall back");
  std::this_thread::sleep_for(150ms);
  Take(c.Submit(Packet(3, 0)));
  Check(c.TransportStats().fallback_requests == 2 &&
            Take(c.Stats()).next_round == 4,
        "expired gradient token fallback changed update count");
  auto empty = Packet(4, 0, 0);
  Take(c.Submit(empty));
  empty.task.iteration = 5;
  Take(c.Submit(empty));
  Check(Take(c.Stats()).next_round == 6 && !Take(c.Stats()).received_workers,
        "empty cached contribution failed to advance exactly one round");
  const auto server = c.LayoutStats().get();
  Check(std::get<GradientLayoutCacheStats>(server).entries <= 1,
        "gradient registry exceeded bounds");
}
void ScopeAndMalformedTokens() {
  Harness h({}, 2);
  BrpcSynchronousGradients c(h.Address(), "gradient-layouts");
  auto other = c.Submit(Packet(0, 0, 4, 1));
  brpc::Channel channel;
  Check(!channel.Init(h.Address().c_str(), nullptr), "raw gradient channel");
  v1::SynchronousGradientService_Stub stub(&channel);
  brpc::Controller ctl;
  v1::GradientContribution full;
  v1::GradientReply reply;
  auto p = Packet(0, 0, 4);
  auto *task = full.mutable_task();
  task->set_run(p.task.run);
  task->set_epoch(1);
  task->set_worker(0);
  task->set_iteration(0);
  full.set_sample_count(3);
  full.set_register_layout(true);
  auto *table = full.add_tables();
  table->set_table_id(1);
  table->set_dimension(2);
  transport::Pack(p.tables[0].keys, table->mutable_keys(),
                  ctl.request_attachment());
  transport::Pack(p.tables[0].values, table->mutable_values(),
                  ctl.request_attachment());
  stub.Submit(&ctl, &full, &reply, nullptr);
  Take(std::move(other));
  Check(!ctl.Failed() && !transport::GetError(reply.status()) &&
            !reply.layout_token().empty(),
        "raw gradient registration failed");
  const auto token = reply.layout_token();
  for (unsigned change = 0; change < 6; ++change) {
    brpc::Controller call;
    v1::GradientContribution q;
    v1::GradientReply out;
    *q.mutable_task() = full.task();
    q.mutable_task()->set_iteration(1);
    q.set_sample_count(3);
    q.set_layout_token(token);
    auto *t = q.add_tables();
    t->set_table_id(1);
    t->set_dimension(2);
    transport::Pack(p.tables[0].values, t->mutable_values(),
                    call.request_attachment());
    if (change == 0)
      q.mutable_task()->set_worker(1);
    if (change == 1)
      q.mutable_task()->set_run("other");
    if (change == 2)
      q.mutable_task()->set_epoch(2);
    if (change == 3)
      t->set_dimension(3);
    if (change == 4)
      transport::Pack(p.tables[0].keys, t->mutable_keys(),
                      call.request_attachment());
    if (change == 5)
      q.set_layout_token("unknown-process-token");
    stub.Submit(&call, &q, &out, nullptr);
    Check(!call.Failed(), "raw gradient transport failed");
    if (change == 5)
      Check(out.layout_miss() && !transport::GetError(out.status()),
            "unknown token was acknowledged as an update");
    else {
      const auto e = transport::GetError(out.status());
      Check(e && e->code == (change < 3 ? Error::Code::stale_generation
                                        : Error::Code::invalid_argument),
            "gradient token crossed scope/schema");
    }
    Check(Take(c.Stats()).next_round == 1 && !Take(c.Stats()).received_workers,
          "gradient miss/rejection reached collector admission");
  }
}
} // namespace
int main() {
  try {
    ExactStateAndBytes();
    ExpiryAndEviction();
    ScopeAndMalformedTokens();
    std::cout << "PASS gradient Key reuse: exact fresh-value Adam updates, "
                 "scoped tokens, bytes, LRU/TTL fallback, empty rounds and "
                 "no-admission misses\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
