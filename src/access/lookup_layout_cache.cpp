#include "eps/lookup_layout_cache.hpp"
#include "common/immutable_metadata_cache.hpp"
#include <algorithm>
#include <list>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>

namespace eps {
namespace {
void Number(std::string& out, std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) out.push_back(static_cast<char>(value >> (8 * i)));
}
void Vector(std::string& out, const std::vector<std::uint64_t>& v) {
  Number(out, v.size()); for (auto n : v) Number(out, n);
}
std::string Content(const LookupLayoutScope& s, const LookupLayoutCache::Layout& layout) {
  std::string out; Number(out, s.run.size()); out += s.run;
  Number(out, s.epoch); Number(out, s.learner); Number(out, s.shard); Number(out, layout.size());
  for (const auto& f : layout) {
    Number(out, f.table); Number(out, f.feature); Number(out, static_cast<unsigned>(f.pooling));
    Vector(out, f.keys); Vector(out, f.offsets); Vector(out, f.valid_lengths);
  }
  return out;
}
std::uint64_t LayoutBytes(const LookupLayoutCache::Layout& layout) {
  std::uint64_t result = layout.capacity() * sizeof(FeatureInput);
  for (const auto& f : layout) result += 8 * (f.keys.capacity() + f.offsets.capacity() + f.valid_lengths.capacity());
  return result;
}
}  // namespace
struct LookupLayoutCache::Impl : detail::ImmutableMetadataCache<LookupLayoutScope, Layout> {
  explicit Impl(LookupLayoutCacheOptions options)
      : ImmutableMetadataCache(options, Content, LayoutBytes) {}
};
LookupLayoutCache::LookupLayoutCache(LookupLayoutCacheOptions o) : impl_(std::make_unique<Impl>(o)) {}
LookupLayoutCache::~LookupLayoutCache() = default;
std::string LookupLayoutCache::Find(const LookupLayoutScope& s,const Layout& l,Clock::time_point n) { return impl_->Find(s,l,n); }
std::shared_ptr<const LookupLayoutCache::Layout> LookupLayoutCache::Resolve(const LookupLayoutScope& s,const std::string& t,Clock::time_point n) { return impl_->Resolve(s,t,n); }
std::string LookupLayoutCache::Put(LookupLayoutScope s,std::shared_ptr<const Layout> l,std::string t,Clock::time_point n) { return impl_->Put(std::move(s),std::move(l),std::move(t),n); }
void LookupLayoutCache::Erase(const std::string& t) { impl_->Erase(t); }
void LookupLayoutCache::Clear() { impl_->Clear(); }
LookupLayoutCacheStats LookupLayoutCache::Stats(Clock::time_point n) { return impl_->Stats(n); }
}  // namespace eps
