#pragma once

#include <map>
#include "eps/cache_backend.hpp"
#include "eps/distributed_client.hpp"

namespace eps {
// One asynchronous gather per destination. All submitted reads finish before
// an error is returned, so the caller can then release its parameter-row pins.
DistributedClient::CacheResolver MakeCacheResolver(
    std::map<AgentId,std::shared_ptr<CacheBackend>> agents);
}  // namespace eps
