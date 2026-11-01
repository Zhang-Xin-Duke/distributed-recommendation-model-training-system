#pragma once

#include "eps/types.hpp"
#include <utility>

namespace eps {
struct PreparationLimits {
  std::uint64_t max_input_bytes = 16ULL << 20;
  std::uint64_t max_prepared_bytes = 64ULL << 20;
  std::uint32_t max_features = 1024;
};

class PreparedLookup {
 public:
  PreparedLookup() = default;
  PreparedLookup(PreparedLookup&& other) noexcept { *this=std::move(other); }
  PreparedLookup& operator=(PreparedLookup&& other) noexcept {
    if(this!=&other) {
      shard_count_=std::exchange(other.shard_count_,0);
      bytes_=std::exchange(other.bytes_,0);
      original_=std::move(other.original_);parts_=std::move(other.parts_);
    }
    return *this;
  }
  PreparedLookup(const PreparedLookup&) = delete;
  PreparedLookup& operator=(const PreparedLookup&) = delete;
  const std::vector<FeatureInput>& features() const { return original_; }
  std::uint64_t payload_bytes() const { return bytes_; }
 private:
  friend class DistributedClient;
  friend Result<PreparedLookup> PrepareLookupInputs(std::vector<FeatureInput>,std::uint32_t,PreparationLimits);
  struct Part {
    std::vector<FeatureInput> features;
    std::vector<std::vector<std::uint64_t>> positions;
  };
  std::uint32_t shard_count_ = 0;
  std::uint64_t bytes_ = 0;
  std::vector<FeatureInput> original_;
  std::vector<Part> parts_;
};

// Validates generic layouts and estimates retained metadata before allocating
// shard vectors. Table registration/dimensions are checked at actual Lookup.
Result<std::uint64_t> EstimatePreparedLookupBytes(const std::vector<FeatureInput>&,
    std::uint32_t shard_count,PreparationLimits limits={});
Result<PreparedLookup> PrepareLookupInputs(std::vector<FeatureInput>,
    std::uint32_t shard_count,PreparationLimits limits={});
}  // namespace eps
