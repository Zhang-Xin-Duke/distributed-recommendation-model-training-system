#pragma once

#include "eps/cache_backend.hpp"

namespace eps {

struct DeviceTensor {
  virtual ~DeviceTensor()=default;
  virtual void* data() const=0;
  virtual std::uint64_t elements() const=0;
  virtual int device() const=0;
  virtual void WaitOnStream(void* cuda_stream) const=0;
  virtual std::vector<float> CopyToHost() const=0;
};

// Native model code consumes device results directly; the control interface
// is also available over BRPC through CacheBackend.
class GpuCacheAgent : public CacheBackend {
 public:
  virtual std::future<Result<std::vector<DeviceTensorBlock>>> GatherDevice(
      std::vector<AgentWork> work,
      std::vector<FeatureInput> layouts) = 0;
};

}  // namespace eps
