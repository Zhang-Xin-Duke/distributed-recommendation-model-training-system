#pragma once
#include "eps/types.hpp"
namespace eps {
// Validates the complete packet before any model state is modified.
void ValidateCompactGradients(const CompactGradients& packet);
RowGradients ExpandCompactGradients(const CompactGradients& packet);
std::uint64_t CompactGradientBytes(const CompactGradients& packet);
}  // namespace eps
