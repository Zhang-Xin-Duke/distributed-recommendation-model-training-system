#include "access/batch_access.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

#include "common/error.hpp"

namespace eps::detail {
void ValidateFeature(const FeatureInput& f, const ShardStore& store) {
  store.Table(f.table);
  Require(f.pooling == Pooling::sum || f.pooling == Pooling::mean ||
          f.pooling == Pooling::sequence, "unsupported pooling mode");
  Require(!f.offsets.empty() && f.offsets.front() == 0 &&
          f.offsets.back() == f.keys.size() &&
          std::is_sorted(f.offsets.begin(), f.offsets.end()), "invalid bag offsets");
  Require(f.valid_lengths.size() + 1 == f.offsets.size(), "missing original bag lengths");
  for (std::size_t b = 0; b < f.valid_lengths.size(); ++b)
    Require(f.valid_lengths[b] >= f.offsets[b + 1] - f.offsets[b],
            "original bag length is smaller than shard contribution");
  for (auto key : f.keys)
    Require(key % store.shard_count() == store.shard(), "key routed to wrong shard");
}

PreparedFeature PrepareFeature(FeatureInput f, ShardStore& store,
    const std::function<bool(RowKey)>& cpu_owned) {
  PreparedFeature out;
  out.input = std::move(f);
  std::unordered_map<Key, std::size_t> unique;
  unique.reserve(out.input.keys.size());
  out.occurrence_to_unique.reserve(out.input.keys.size());
  for (auto key : out.input.keys) {
    auto [it, inserted] = unique.try_emplace(key, unique.size());
    if (inserted) out.unique_slots.push_back(store.Ensure(out.input.table, key));
    out.occurrence_to_unique.push_back(it->second);
  }
  auto& t = store.Table(out.input.table);
  const auto dim = t.definition.dimension;
  const auto rows = out.input.pooling == Pooling::sequence
      ? out.input.keys.size() : out.input.valid_lengths.size();
  out.contribution = {out.input.feature, {rows, dim}, std::vector<float>(rows * dim, 0)};
  for (std::size_t b = 0; b < out.input.valid_lengths.size(); ++b) {
    for (auto i = out.input.offsets[b]; i < out.input.offsets[b + 1]; ++i) {
      if(cpu_owned && !cpu_owned({out.input.table,out.input.keys[i]}))continue;
      const auto row = t.At(out.unique_slots[out.occurrence_to_unique[i]]);
      auto dest = out.contribution.values.data() +
          (out.input.pooling == Pooling::sequence ? i : b) * dim;
      for (std::uint32_t d = 0; d < dim; ++d) dest[d] += row.weights[d];
    }
  }
  return out;
}

std::unordered_map<TableId, RowGradients> ReduceGradients(
    const std::vector<PreparedFeature>& features,
    const std::vector<TensorBlock>& gradients, const ShardStore& store) {
  Require(features.size() == gradients.size(), "gradient feature count mismatch");
  std::unordered_map<std::uint32_t, const TensorBlock*> by_feature;
  for (const auto& g : gradients)
    Require(by_feature.emplace(g.feature, &g).second, "duplicate gradient feature");
  // Validate the complete update before touching any parameter state.
  for (const auto& f : features) {
    auto it = by_feature.find(f.input.feature);
    Require(it != by_feature.end(), "missing feature gradient");
    const auto& g = *it->second;
    Require(g.shape == f.contribution.shape &&
            g.values.size() == f.contribution.shape[0] * f.contribution.shape[1],
            "gradient tensor layout mismatch");
    for (auto x : g.values) Require(std::isfinite(x), "non-finite gradient");
  }
  std::unordered_map<TableId, RowGradients> result;
  std::unordered_map<TableId, std::unordered_map<Key, std::size_t>> row_indices;
  for (const auto& f : features) {
    auto dim = store.Table(f.input.table).definition.dimension;
    const auto& g = *by_feature.at(f.input.feature);
    auto& output = result[f.input.table];
    auto& index = row_indices[f.input.table];
    for (std::size_t b = 0; b < f.input.valid_lengths.size(); ++b) {
      const auto scale = f.input.pooling == Pooling::mean && f.input.valid_lengths[b]
          ? 1.0f / f.input.valid_lengths[b] : 1.0f;
      for (auto i = f.input.offsets[b]; i < f.input.offsets[b + 1]; ++i) {
        auto key = f.input.keys[i];
        auto [it, inserted] = index.try_emplace(key, output.rows.size());
        if (inserted) {
          output.rows.push_back({f.input.table, key});
          output.values.resize(output.values.size() + dim, 0);
        }
        auto src = g.values.data() + (f.input.pooling == Pooling::sequence ? i : b) * dim;
        auto dest = output.values.data() + it->second * dim;
        for (std::uint32_t d = 0; d < dim; ++d) dest[d] += src[d] * scale;
      }
    }
  }
  for (const auto& [id, g] : result)
    for (auto value : g.values) Require(std::isfinite(value), "gradient reduction overflow");
  return result;
}
}  // namespace eps::detail
