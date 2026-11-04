#pragma once
#include <chrono>
#include <memory>
#include <string>
#include "eps/types.hpp"

namespace eps {
struct LookupLayoutScope {
  std::string run;
  std::uint64_t epoch;
  std::uint32_t learner;
  ShardId shard;
  bool operator==(const LookupLayoutScope&) const = default;
};
struct LookupLayoutCacheOptions {
  std::size_t entries = 4096;
  std::uint64_t bytes = 64ULL << 20;
  std::chrono::milliseconds idle_ttl{60000};
};
struct LookupLayoutCacheStats {
  std::uint64_t entries = 0, bytes = 0, hits = 0, misses = 0,
    registrations = 0, evictions = 0, expirations = 0, rejected = 0;
};
// Immutable metadata only. A shared result remains valid through eviction.
class LookupLayoutCache {
 public:
  using Layout = std::vector<FeatureInput>;
  using Clock = std::chrono::steady_clock;
  explicit LookupLayoutCache(LookupLayoutCacheOptions = {});
  ~LookupLayoutCache();
  LookupLayoutCache(const LookupLayoutCache&) = delete;
  LookupLayoutCache& operator=(const LookupLayoutCache&) = delete;
  std::string Find(const LookupLayoutScope&, const Layout&, Clock::time_point now = Clock::now());
  std::shared_ptr<const Layout> Resolve(const LookupLayoutScope&, const std::string&,
                                      Clock::time_point now = Clock::now());
  // Empty token requests a fresh process-specific token. Empty return declines registration.
  std::string Put(LookupLayoutScope, std::shared_ptr<const Layout>, std::string token = {},
                  Clock::time_point now = Clock::now());
  void Erase(const std::string&);
  void Clear();
  LookupLayoutCacheStats Stats(Clock::time_point now = Clock::now());
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace eps
