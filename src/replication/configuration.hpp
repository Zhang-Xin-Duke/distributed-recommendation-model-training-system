#pragma once
#include <algorithm>
#include <nlohmann/json.hpp>
#include "eps/replica_log.hpp"
namespace eps::replication {
inline ReplicaLogOptions Configuration(const nlohmann::json& j) {
  auto tables=j.at("tables");std::sort(tables.begin(),tables.end(),[](const auto& a,const auto& b){return a.at("id")<b.at("id");});
  const auto& sync=j.at("synchronous_training");
  const auto binding=nlohmann::json{{"tables",tables},{"workers",sync.at("workers")},{"reduction",sync.value("reduction",std::string("sample_mean"))},
    {"initial_rows",j.value("replica_table_rows",nlohmann::json::object())}}.dump();
  return {j.value("replica_directory",std::string()),j.at("run_id"),binding,j.at("partition").at("shard_id"),
    j.at("partition").at("shard_count"),j.at("partition").at("epoch"),j.value("replica_base_round",std::uint64_t(0)),
    j.value("replica_record_bytes",64ULL<<20),j.value("replica_disk_bytes",64ULL<<30)};
}
}  // namespace eps::replication
