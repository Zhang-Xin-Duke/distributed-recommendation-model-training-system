#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "eps/brpc_client.hpp"
#include "eps/brpc_training.hpp"
#include "eps/brpc_synchronous_gradients.hpp"

using namespace eps;
namespace {
template<class T> T Take(std::future<Result<T>> f) {
  auto result = f.get();
  if (auto* e = std::get_if<Error>(&result)) throw std::runtime_error(e->message);
  return std::get<T>(std::move(result));
}
}
int main(int argc, char** argv) {
  try {
    if (argc != 5 || std::string(argv[1]) != "--config" || std::string(argv[3]) != "--action")
      throw std::runtime_error("usage: eps_training_control --config FILE --action stats|inspect|pause|checkpoint|resume|abort");
    std::ifstream file(argv[2]);
    if (!file) throw std::runtime_error("cannot open training control config");
    nlohmann::json j; file >> j;
    BrpcTrainingProgress c(j.at("coordinator"), j.at("run"), j.value("epoch", std::uint64_t(1)), 5000);
    const std::string action = argv[4];
    nlohmann::json result;
    if (action == "stats" || action == "inspect") {
      const auto s = Take(c.Stats());
      result = {{"completed", s.completed}, {"completion_floor", s.completion_floor},
        {"granted_tasks", s.granted_tasks}, {"completed_tasks", s.completed_tasks},
        {"duplicate_completions", s.duplicate_completions}, {"active", s.active},
        {"waiting", s.waiting}, {"paused", s.paused}, {"failed", s.failed}};
      if (action == "inspect") {
        const auto addresses = j.at("shards").get<std::vector<std::string>>();
        if (addresses.empty() || addresses.size() > 1024) throw std::runtime_error("invalid shard group");
        const auto keys = j.at("keys").get<std::vector<Key>>();
        if (keys.size() > 100000) throw std::runtime_error("inspection key limit exceeded");
        const auto table = j.value("table", TableId(1));
        result["rows"] = nlohmann::json::array();
        result["shard_stats"] = nlohmann::json::array();
        for (std::size_t home = 0; home < addresses.size(); ++home) {
          BrpcParameterService shard(addresses[home]);
          std::vector<RowKey> local;
          for (auto key : keys) if (key % addresses.size() == home) local.push_back({table, key});
          auto locations = Take(shard.Locations(std::move(local)));
          for (const auto& row : Take(shard.ReadState(std::move(locations))))
            result["rows"].push_back({{"key", row.row.key}, {"home", home},
              {"weights", row.weights}, {"first_moment", row.first_moment},
              {"second_moment", row.second_moment}, {"row_updates", row.row_updates}});
          const auto stats = Take(shard.Stats());
          result["shard_stats"].push_back({{"live_tickets", stats.live_tickets},
            {"updates", stats.updates}, {"rows", stats.rows}});
          if (j.value("synchronous", false)) {
            BrpcSynchronousGradients group(addresses[home], j.at("run"), j.value("epoch", std::uint64_t(1)));
            const auto rounds = Take(group.Stats());
            result["shard_stats"].back()["synchronous"] = {{"next_round", rounds.next_round},
              {"received_workers", rounds.received_workers}, {"retained_bytes", rounds.retained_bytes},
              {"duplicate_submissions", rounds.duplicate_submissions}, {"failed", rounds.failed}};
          }
        }
      }
    } else if (action == "checkpoint") {
      const auto s = Take(c.Checkpoint());
      result = {{"source_run", s.source_run}, {"epoch", s.epoch},
        {"consistency", static_cast<unsigned>(s.consistency)},
        {"max_delay", s.max_delay}, {"completed", s.completed}};
    } else {
      if (action == "pause") Take(c.Pause());
      else if (action == "resume") Take(c.Resume());
      else if (action == "abort") Take(c.Abort(j.value("reason", std::string("operator requested abort"))));
      else throw std::runtime_error("unknown training action");
      result = {{"ok", true}};
    }
    std::cout << result.dump() << '\n';
  } catch (const std::exception& e) {
    std::cerr << "eps_training_control: " << e.what() << '\n'; return 1;
  }
}
