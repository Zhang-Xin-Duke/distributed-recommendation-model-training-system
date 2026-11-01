#pragma once

#include <map>
#include "eps/cache_backend.hpp"
#include "eps/distributed_client.hpp"

namespace eps {
// One asynchronous gather per destination. Send only referenced input positions;
// restore sequence contributions to the caller's layout before returning.
// All submitted read completions are collected before returning an error, so
// the caller can then release its parameter-row pins.
DistributedClient::CacheResolver MakeCacheResolver(
    std::map<AgentId,std::shared_ptr<CacheBackend>> agents);
using NativeDeviceGather=std::function<std::future<Result<std::vector<DeviceTensorBlock>>>(
    std::vector<AgentWork>,std::vector<FeatureInput>)>;
DistributedClient::DeviceResolver MakeDeviceCacheResolver(
    std::map<AgentId,NativeDeviceGather> native,
    std::map<AgentId,std::shared_ptr<CacheBackend>> remote={});
}  // namespace eps
