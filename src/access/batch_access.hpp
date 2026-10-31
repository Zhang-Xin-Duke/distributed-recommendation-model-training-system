#pragma once

#include "eps/types.hpp"
#include "store/shard_store.hpp"

namespace eps::detail {

struct PreparedFeature {
  FeatureInput input;
  std::vector<std::size_t> unique_slots;
  std::vector<std::size_t> occurrence_to_unique;
  TensorBlock contribution;
};

void ValidateFeature(const FeatureInput& feature, const ShardStore& store);
PreparedFeature PrepareFeature(FeatureInput feature, ShardStore& store);
// Returned vectors are unique per key across all features of one Push.
std::unordered_map<TableId, RowGradients> ReduceGradients(
    const std::vector<PreparedFeature>& features,
    const std::vector<TensorBlock>& gradients, const ShardStore& store);

}  // namespace eps::detail
