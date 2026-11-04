#include <chrono>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "eps/brpc_client.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/brpc_synchronous_gradients.hpp"

using namespace eps;
namespace {
template<class T> T Take(std::future<Result<T>> future) {
  auto r = future.get(); if (auto* e = std::get_if<Error>(&r)) throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
}
int main(int argc, char** argv) {
  try {
    if (argc != 3 || std::string(argv[1]) != "--config")
      throw std::runtime_error("usage: eps_training_prepare --config FILE");
    std::ifstream f(argv[2]); if (!f) throw std::runtime_error("cannot open preparation config");
    nlohmann::json j; f >> j;
    const auto addresses = j.at("shards").get<std::vector<std::string>>();
    const auto rows = j.at("rows").get<std::uint64_t>();
    const auto dim = j.at("dimension").get<std::uint64_t>();
    const auto table = j.value("table_id", TableId(1));
    const std::string run = j.at("run");
    const auto epoch = j.value("epoch", std::uint64_t(1));
    if (addresses.empty() || addresses.size() > 1024 || !rows || rows > (1ULL<<34) || !dim || dim > 65536)
      throw std::runtime_error("invalid training preparation layout");
    const auto batch = std::min<std::uint64_t>(j.value("batch_rows", std::uint64_t(256)),
      (8ULL<<20)/(dim*12+256));
    if (!batch) throw std::runtime_error("invalid preparation batch size");
    const bool resident = j.value("resident", true);
    const bool restoring = j.value("restore_existing", false);
    if (!j.value("workers", 2u) || j.value("workers", 2u) > 1022)
      throw std::runtime_error("preparation reserves learner 1022; use at most 1022 training workers");
    if (j.contains("only_shard") && j.at("only_shard").get<std::size_t>() >= addresses.size())
      throw std::runtime_error("preparation shard index is outside the topology");
    nlohmann::json result = {{"rows", rows}, {"dimension", dim}, {"resident", resident}, {"restore_existing", restoring},
      {"logical_parameter_optimizer_bytes", rows*dim*12}, {"shards", nlohmann::json::array()}};
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t home = 0; home < addresses.size(); ++home) {
      if (j.contains("only_shard") && j.at("only_shard").get<std::size_t>() != home) continue;
      BrpcParameterService shard(addresses[home], 30000);
      if (restoring) {
        const auto source = Take(shard.RestoredFrom());
        if (!source.loaded || !source.pristine || source.run != run || source.shard != home ||
            source.shard_count != addresses.size() || source.partition_epoch != epoch)
          throw std::runtime_error("rehydration requires a pristine restored parameter shard");
        BrpcSynchronousGradients group(addresses[home], run, epoch);
        const auto checkpoint = Take(group.Checkpoint()); // Requires paused, empty and matching optimizer clocks.
        if (!std::any_of(checkpoint.tables.begin(), checkpoint.tables.end(), [&](const auto& t) {
              return t.table == table && t.dimension == dim;
            })) throw std::runtime_error("rehydration table dimension differs from restored schema");
      }
      std::uint64_t sequence = 0, initialized = 0;
      for (std::uint64_t begin = home; begin < rows;) {
        std::vector<Key> keys; std::vector<RowKey> moving;
        for (std::uint64_t count = 0; count < batch && begin < rows; ++count, begin += addresses.size()) {
          keys.push_back(begin); moving.push_back({table, begin});
        }
        const auto count = keys.size();
        if (!restoring) {
          const auto id = RequestId{run, 1022, sequence++, Phase::lookup};
          auto lookup = Take(shard.Lookup({id, epoch, true,
            {{table, 0, Pooling::sequence, std::move(keys), {0, count}, {count}}}}));
          Take(shard.Release({{run, 1022, id.sequence, Phase::release}, lookup.ticket}));
        } else {
          // Existing rows only: never initialize a missing row or consume a training read ticket.
          auto locations = Take(shard.Locations(moving));
          if (locations.size() != count) throw std::runtime_error("rehydration omitted restored rows");
          moving.clear();
          for (const auto& location : locations) {
            if (location.placement == Placement::cpu) moving.push_back(location.row);
            else if (!resident || location.placement != Placement::gpu ||
                     location.agent != j.at("agents").at(home).at("id").get<AgentId>())
              throw std::runtime_error("restored row has unexpected placement");
          }
        }
        if (resident && !moving.empty()) {
          const auto agent = j.at("agents").at(home).at("id").get<AgentId>();
          const auto migrating = moving.size();
          auto placed = Take(shard.MoveRows({run, std::move(moving), agent, Placement::gpu,
            {16ULL<<20, 16ULL<<20, 1, 0}}));
          if (placed.size() != migrating) throw std::runtime_error("migration omitted parameter rows");
          for (const auto& row : placed)
            if (row.placement != Placement::gpu || row.agent != agent || row.home != home)
              throw std::runtime_error("migration returned incorrect parameter ownership");
        }
        initialized += count;
      }
      const auto stats = Take(shard.Stats());
      if (stats.rows != initialized || stats.live_tickets || (resident && stats.cached_rows != initialized))
        throw std::runtime_error("prepared row/residency counts disagree");
      nlohmann::json record = {{"shard", home}, {"initialized_rows", initialized},
        {"resident_rows", stats.cached_rows}, {"cpu_state_bytes", stats.state_bytes},
        {"logical_resident_state_bytes", resident ? initialized*dim*12 : 0}};
      if (resident) {
        const auto& a = j.at("agents").at(home);
        BrpcCacheBackend agent(a.at("endpoint"), run, home, a.at("id"));
        record["remaining_slots"] = Take(agent.FreeSlots(table));
      }
      result["shards"].push_back(std::move(record));
    }
    result["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    std::cout << result.dump() << std::endl;
  } catch (const std::exception& e) { std::cerr << "eps_training_prepare: " << e.what() << '\n'; return 1; }
}
