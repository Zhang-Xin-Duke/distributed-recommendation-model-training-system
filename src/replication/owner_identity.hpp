#pragma once
#include <iomanip>
#include <sstream>
#include <openssl/sha.h>
#include <nlohmann/json.hpp>
#include "eps/replica_log.hpp"

namespace eps::replication {
inline std::string OwnerRun(const std::string& run,std::uint64_t epoch,ShardId shard,const ReplicaLease& lease) {
  const auto data=nlohmann::json::array({run,epoch,shard,lease.term,lease.primary}).dump();
  unsigned char digest[SHA256_DIGEST_LENGTH];
  if(!SHA256(reinterpret_cast<const unsigned char*>(data.data()),data.size(),digest))
    throw std::runtime_error("owner identity digest failed");
  std::ostringstream result;result<<"eps-owner-"<<std::hex<<std::setfill('0');
  for(auto byte:digest)result<<std::setw(2)<<static_cast<unsigned>(byte);
  return result.str();
}
}  // namespace eps::replication
