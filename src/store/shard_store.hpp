#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "eps/types.hpp"

namespace eps::detail {

struct TableState {
  TableDefinition definition;
  std::uint64_t optimizer_sequence = 0;
  float beta1_power;
  float beta2_power;
  std::unordered_map<Key, std::size_t> index;
  std::vector<Key> keys;
  struct Chunk {
    std::unique_ptr<float[]> values;
    std::unique_ptr<std::uint64_t[]> updates;
    std::size_t capacity;
    std::size_t used = 0;
  };
  std::vector<Chunk> chunks;
  struct Row {
    float* weights;
    float* first;
    float* second;
    std::uint64_t* updates;
  };
  Row At(std::size_t slot) const;
};

class ShardStore {
 public:
  ShardStore(ShardId shard, std::uint32_t count, std::uint64_t epoch,
             std::uint64_t byte_limit);
  void Create(TableDefinition definition);
  TableState& Table(TableId table);
  const TableState& Table(TableId table) const;
  std::size_t Ensure(TableId table, Key key);
  RowState Read(TableId table, Key key) const;
  void Import(const RowState& row);
  void Overwrite(const RowState& row);
  std::uint64_t bytes() const { return allocated_bytes_; }
  std::uint64_t rows() const;
  bool empty() const { return tables_.empty(); }
  auto& tables() { return tables_; }
  const auto& tables() const { return tables_; }
  ShardId shard() const { return shard_; }
  std::uint32_t shard_count() const { return count_; }
  std::uint64_t epoch() const { return epoch_; }
 private:
  ShardId shard_;
  std::uint32_t count_;
  std::uint64_t epoch_;
  std::uint64_t byte_limit_;
  std::uint64_t allocated_bytes_ = 0;
  std::unordered_map<TableId, TableState> tables_;
};

}  // namespace eps::detail
