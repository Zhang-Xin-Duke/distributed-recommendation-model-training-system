#include "eps/gradient_layout_cache.hpp"
#include "common/immutable_metadata_cache.hpp"

namespace eps {
namespace {
void Number(std::string &out, std::uint64_t n) {
  for (unsigned i = 0; i < 8; ++i)
    out.push_back(static_cast<char>(n >> (8 * i)));
}
std::string Content(const GradientLayoutScope &scope,
                    const GradientLayoutCache::Layout &layout) {
  std::string out;
  Number(out, scope.run.size());
  out += scope.run;
  Number(out, scope.epoch);
  Number(out, scope.learner);
  Number(out, scope.shard);
  Number(out, layout.size());
  for (const auto &table : layout) {
    Number(out, table.table);
    Number(out, table.dimension);
    Number(out, table.keys.size());
    for (auto key : table.keys)
      Number(out, key);
  }
  return out;
}
std::uint64_t Bytes(const GradientLayoutCache::Layout &layout) {
  std::uint64_t bytes = layout.capacity() * sizeof(GradientKeyLayout);
  for (const auto &table : layout)
    bytes += table.keys.capacity() * sizeof(Key);
  return bytes;
}
} // namespace
struct GradientLayoutCache::Impl
    : detail::ImmutableMetadataCache<GradientLayoutScope, Layout> {
  explicit Impl(GradientLayoutCacheOptions o)
      : ImmutableMetadataCache(o, Content, Bytes) {}
};
GradientLayoutCache::GradientLayoutCache(GradientLayoutCacheOptions o)
    : impl_(std::make_unique<Impl>(o)) {}
GradientLayoutCache::~GradientLayoutCache() = default;
std::string GradientLayoutCache::Find(const GradientLayoutScope &s,
                                      const Layout &l, Clock::time_point n) {
  return impl_->Find(s, l, n);
}
std::shared_ptr<const GradientLayoutCache::Layout>
GradientLayoutCache::Resolve(const GradientLayoutScope &s, const std::string &t,
                             Clock::time_point n) {
  return impl_->Resolve(s, t, n);
}
std::string GradientLayoutCache::Put(GradientLayoutScope s,
                                     std::shared_ptr<const Layout> l,
                                     std::string t, Clock::time_point n) {
  return impl_->Put(std::move(s), std::move(l), std::move(t), n);
}
void GradientLayoutCache::Erase(const std::string &t) { impl_->Erase(t); }
void GradientLayoutCache::Clear() { impl_->Clear(); }
GradientLayoutCacheStats GradientLayoutCache::Stats(Clock::time_point n) {
  return impl_->Stats(n);
}
} // namespace eps
