#pragma once
#include "eps/lookup_layout_cache.hpp"

namespace eps {
// The learner field scopes a synchronous training worker. Values are never
// cached.
using GradientLayoutScope = LookupLayoutScope;
using GradientLayoutCacheOptions = LookupLayoutCacheOptions;
using GradientLayoutCacheStats = LookupLayoutCacheStats;
struct GradientKeyLayout {
  TableId table;
  std::uint32_t dimension;
  std::vector<Key> keys;
  bool operator==(const GradientKeyLayout &) const = default;
};
class GradientLayoutCache {
public:
  using Layout = std::vector<GradientKeyLayout>;
  using Clock = std::chrono::steady_clock;
  explicit GradientLayoutCache(GradientLayoutCacheOptions = {});
  ~GradientLayoutCache();
  std::string Find(const GradientLayoutScope &, const Layout &,
                   Clock::time_point = Clock::now());
  std::shared_ptr<const Layout> Resolve(const GradientLayoutScope &,
                                        const std::string &,
                                        Clock::time_point = Clock::now());
  std::string Put(GradientLayoutScope, std::shared_ptr<const Layout>,
                  std::string token = {}, Clock::time_point = Clock::now());
  void Erase(const std::string &);
  void Clear();
  GradientLayoutCacheStats Stats(Clock::time_point = Clock::now());

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace eps
