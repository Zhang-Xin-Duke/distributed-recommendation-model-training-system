#pragma once
#include <algorithm>
#include <cmath>
#include "eps/types.hpp"
#include "common/error.hpp"
namespace eps::detail {
inline void ValidateWeightBounds(const OptimizerConfig& optimizer) {
  if (optimizer.weight_bounds) {
    const auto b = *optimizer.weight_bounds;
    Require(std::isfinite(b.minimum) && std::isfinite(b.maximum) &&
            b.minimum <= b.maximum, "invalid optimizer weight bounds");
  }
}
inline float BoundWeight(float value, const OptimizerConfig& optimizer) {
  return optimizer.weight_bounds
      ? std::clamp(value, optimizer.weight_bounds->minimum, optimizer.weight_bounds->maximum)
      : value;
}
}  // namespace eps::detail
