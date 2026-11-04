#pragma once
#include <nlohmann/json.hpp>
#include "eps/synchronous_gradients.hpp"

namespace eps::training {
inline nlohmann::json Encode(const SynchronousGradientCheckpoint& c) {
  auto tables=nlohmann::json::array();
  for (const auto& t:c.tables) tables.push_back({{"id",t.table},{"dimension",t.dimension}});
  return {{"source_run",c.source_run},{"epoch",c.epoch},{"workers",c.workers},{"shard",c.shard},
    {"shards",c.shards},{"tables",tables},{"reduction",static_cast<unsigned>(c.reduction)},{"next_round",c.next_round}};
}
inline SynchronousGradientCheckpoint DecodeGroup(const nlohmann::json& j) {
  const auto reduction=j.at("reduction").get<unsigned>();
  if (reduction>2 || j.at("tables").size()>256) throw std::runtime_error("invalid gradient checkpoint metadata");
  SynchronousGradientCheckpoint c{j.at("source_run"),j.at("epoch"),j.at("workers"),j.at("shard"),j.at("shards"),{},
    static_cast<GradientReduction>(reduction),j.at("next_round")};
  for (const auto& t:j.at("tables")) c.tables.push_back({t.at("id"),t.at("dimension"),{}, {}});
  return c;
}
inline TableDefinition DecodeTable(const nlohmann::json& t,std::uint32_t count,std::uint64_t epoch) {
  if (t.value("dtype",std::string("float32"))!="float32" || t.at("optimizer").at("kind")!="sparse_lazy_adam")
    throw std::runtime_error("training checkpoints require FP32 sparse_lazy_adam tables");
  const auto& o=t.at("optimizer");const auto& init=t.at("initialization");
  TableDefinition result{t.at("id"),t.at("dimension"),count,epoch,
    {o.at("learning_rate"),o.at("beta1"),o.at("beta2"),o.at("epsilon"),o.at("initial_beta1_power"),o.at("initial_beta2_power")},
    init.at("min"),init.at("max"),init.at("seed")};
  if (o.contains("weight_bounds")) {
    const auto& b=o.at("weight_bounds");
    if (!b.is_array() || b.size()!=2) throw std::runtime_error("invalid checkpoint weight bounds");
    result.optimizer.weight_bounds=WeightBounds{b.at(0),b.at(1)};
  }
  return result;
}
}  // namespace eps::training
