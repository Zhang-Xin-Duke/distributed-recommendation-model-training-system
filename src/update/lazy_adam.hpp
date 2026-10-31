#pragma once
#include "eps/types.hpp"
#include "store/shard_store.hpp"

namespace eps::detail {
void ValidateLazyAdam(const TableState& table, const RowGradients& gradients);
void ApplyLazyAdam(TableState& table, const RowGradients& gradients);
}
