#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "store/shard_store.hpp"

namespace eps::detail {
struct SnapshotInspection {
  std::string run;
  ShardId shard;
  std::uint32_t shards;
  std::uint64_t epoch, bytes, checksum;
  std::vector<UpdateContext> optimizer_contexts;
};
// Stream format/identity, optimizer clocks and checksum without allocating rows.
SnapshotInspection InspectSnapshot(const std::filesystem::path& file);
SnapshotChunk ReadSnapshotChunk(const std::filesystem::path& file,
    std::uint64_t offset, std::uint32_t max_bytes);
void WriteSnapshot(const std::filesystem::path& file, const std::string& run,
                   const ShardStore& store);
std::unique_ptr<ShardStore> ReadSnapshot(const std::filesystem::path& file,
    const std::string& new_run, ShardId shard, std::uint32_t count,
    std::uint64_t epoch, std::uint64_t byte_limit,
    RestoredSnapshot* provenance=nullptr, bool replica_same_run=false);
}  // namespace eps::detail
