#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "store/shard_store.hpp"

namespace eps::detail {
SnapshotChunk ReadSnapshotChunk(const std::filesystem::path& file,
    std::uint64_t offset, std::uint32_t max_bytes);
void WriteSnapshot(const std::filesystem::path& file, const std::string& run,
                   const ShardStore& store);
std::unique_ptr<ShardStore> ReadSnapshot(const std::filesystem::path& file,
    const std::string& new_run, ShardId shard, std::uint32_t count,
    std::uint64_t epoch, std::uint64_t byte_limit);
}  // namespace eps::detail
