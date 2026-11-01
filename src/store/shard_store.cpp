#include "store/shard_store.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/error.hpp"
#include "common/optimizer_bounds.hpp"

namespace eps::detail {
namespace {
constexpr std::size_t kChunkRows = 1024;
std::uint64_t Mix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
}

TableState::Row TableState::At(std::size_t slot) const {
  // All chunks except the final budget-limited chunk have kChunkRows slots.
  const auto& chunk = chunks.at(slot / kChunkRows);
  const auto local = slot % kChunkRows;
  Require(local < chunk.used, "row slot is outside allocated state");
  const auto dim = definition.dimension;
  auto* base = chunk.values.get() + local * dim * 3;
  return {base, base + dim, base + dim * 2, chunk.updates.get() + local};
}

ShardStore::ShardStore(ShardId shard, std::uint32_t count, std::uint64_t epoch,
                       std::uint64_t byte_limit)
    : shard_(shard), count_(count), epoch_(epoch), byte_limit_(byte_limit) {
  Require(count > 0 && shard < count, "invalid shard topology");
}

void ShardStore::Create(TableDefinition d) {
  Require(d.dimension > 0 && d.dimension <= 65536, "invalid embedding dimension");
  Require(d.shard_count == count_ && d.partition_epoch == epoch_,
          "table partition does not match server");
  const auto& o = d.optimizer;
  ValidateWeightBounds(o);
  Require(std::isfinite(o.learning_rate) && o.learning_rate > 0 &&
          o.beta1 >= 0 && o.beta1 < 1 && o.beta2 >= 0 && o.beta2 < 1 &&
          std::isfinite(o.epsilon) && o.epsilon > 0 &&
          o.initial_beta1_power >= 0 && o.initial_beta1_power < 1 &&
          o.initial_beta2_power >= 0 && o.initial_beta2_power < 1,
          "invalid sparse LazyAdam configuration");
  Require(std::isfinite(d.initialization_min) &&
          std::isfinite(d.initialization_max) &&
          d.initialization_min <= d.initialization_max,
          "invalid initializer bounds");
  if (auto it = tables_.find(d.id); it != tables_.end()) {
    Require(it->second.definition == d, "table already exists with different configuration");
    return;
  }
  Require(tables_.size() < 1024, "table count exceeds limit", Error::Code::capacity_exceeded);
  TableState t;
  t.definition = d;
  t.beta1_power = o.initial_beta1_power;
  t.beta2_power = o.initial_beta2_power;
  tables_.emplace(d.id, std::move(t));
}

TableState& ShardStore::Table(TableId id) {
  auto it = tables_.find(id);
  Require(it != tables_.end(), "unknown table");
  return it->second;
}
const TableState& ShardStore::Table(TableId id) const {
  auto it = tables_.find(id);
  Require(it != tables_.end(), "unknown table");
  return it->second;
}

std::size_t ShardStore::Ensure(TableId id, Key key) {
  Require(key % count_ == shard_, "key belongs to another shard");
  auto& t = Table(id);
  if (auto it = t.index.find(key); it != t.index.end()) return it->second;
  const auto dim = t.definition.dimension;
  if (t.chunks.empty() || t.chunks.back().used == t.chunks.back().capacity) {
    // A shortened final chunk cannot be followed by another chunk.
    Require(t.chunks.empty() || t.chunks.back().capacity == kChunkRows,
            "parameter-state budget exhausted", Error::Code::capacity_exceeded);
    const auto row_bytes = std::uint64_t(dim) * 3 * sizeof(float) + sizeof(std::uint64_t);
    const auto capacity = std::min<std::uint64_t>(kChunkRows,
        (byte_limit_ - allocated_bytes_) / row_bytes);
    Require(capacity > 0, "parameter-state budget exhausted", Error::Code::capacity_exceeded);
    TableState::Chunk chunk;
    chunk.capacity = capacity;
    chunk.values = std::make_unique<float[]>(capacity * dim * 3);
    chunk.updates = std::make_unique<std::uint64_t[]>(capacity);
    t.chunks.push_back(std::move(chunk));
    allocated_bytes_ += capacity * row_bytes;
  }
  auto& chunk = t.chunks.back();
  const auto slot = t.keys.size();
  // Allocate metadata before publishing the row. Parameters stay in stable chunks.
  t.keys.push_back(key);
  try { t.index.emplace(key, slot); }
  catch (...) { t.keys.pop_back(); throw; }
  ++chunk.used;
  auto row = t.At(slot);
  for (std::uint32_t i = 0; i < dim; ++i) {
    const auto bits = Mix(key ^ Mix(t.definition.initialization_seed) ^
                          Mix(id) ^ Mix(i));
    const double u = static_cast<double>(bits >> 11) * 0x1.0p-53;
    row.weights[i] = static_cast<float>(
        (1 - u) * t.definition.initialization_min + u * t.definition.initialization_max);
    row.weights[i] = BoundWeight(row.weights[i], t.definition.optimizer);
    row.first[i] = row.second[i] = 0;
  }
  *row.updates = 0;
  return slot;
}

RowState ShardStore::Read(TableId id, Key key) const {
  const auto& t = Table(id);
  auto it = t.index.find(key);
  Require(it != t.index.end(), "unknown parameter row");
  const auto row = t.At(it->second);
  const auto dim = t.definition.dimension;
  return {{id, key}, {row.weights, row.weights + dim},
          {row.first, row.first + dim}, {row.second, row.second + dim}, *row.updates};
}

void ShardStore::Import(const RowState& row) {
  auto& t = Table(row.row.table);
  auto dim = t.definition.dimension;
  Require(row.weights.size() == dim && row.first_moment.size() == dim &&
          row.second_moment.size() == dim, "invalid imported state dimensions");
  Require(!t.index.contains(row.row.key), "duplicate imported row");
  auto target = t.At(Ensure(row.row.table, row.row.key));
  std::copy(row.weights.begin(), row.weights.end(), target.weights);
  std::copy(row.first_moment.begin(), row.first_moment.end(), target.first);
  std::copy(row.second_moment.begin(), row.second_moment.end(), target.second);
  *target.updates = row.row_updates;
}

void ShardStore::Overwrite(const RowState& row) {
  auto& t=Table(row.row.table);const auto dim=t.definition.dimension;
  Require(t.index.contains(row.row.key) && row.weights.size()==dim &&
    row.first_moment.size()==dim && row.second_moment.size()==dim,"invalid returned row state");
  auto target=t.At(t.index.at(row.row.key));
  std::copy(row.weights.begin(),row.weights.end(),target.weights);
  std::copy(row.first_moment.begin(),row.first_moment.end(),target.first);
  std::copy(row.second_moment.begin(),row.second_moment.end(),target.second);
  *target.updates=row.row_updates;
}

std::uint64_t ShardStore::rows() const {
  std::uint64_t n = 0;
  for (const auto& [id, t] : tables_) n += t.keys.size();
  return n;
}
}  // namespace eps::detail
