#pragma once
#include "eps/cache_agent.hpp"

namespace eps {
struct DeviceContribution {
  DeviceTensorBlock block;
  // Projected sequence row -> compact global occurrence position.
  std::vector<std::uint64_t> positions;
  bool projected=false;
};
struct ResolvedDeviceFeatures {
  std::vector<TensorBlock> host;
  std::vector<DeviceContribution> device;
};
struct SequenceProjection {
  std::uint32_t feature;
  std::uint64_t rows;
  // Compact occurrence -> original sequence position, including padding gaps.
  std::vector<std::uint64_t> positions;
};
struct DeviceAssemblyRequest {
  std::vector<TensorBlock> host;
  std::vector<DeviceContribution> device;
  std::vector<FeatureInput> layouts;
  std::vector<SequenceProjection> projections;
};
}  // namespace eps
