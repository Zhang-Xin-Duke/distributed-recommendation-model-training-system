#include "eps/lookup_layout_cache.hpp"
#include "common/error.hpp"
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
struct LookupLayoutCache::Impl {
  struct Entry {
    LookupLayoutScope scope;
    std::shared_ptr<const Layout> layout;
    std::string content;
    std::uint64_t charge;
    Clock::time_point used;
    std::list<std::string>::iterator lru;
  };
  LookupLayoutCacheOptions options;
  LookupLayoutCacheStats stats;
  std::mutex mutex;
  std::list<std::string> lru;
  std::unordered_map<std::string, Entry> entries;
  std::unordered_map<std::string, std::string> content;
  std::string instance;
  std::uint64_t sequence = 0;
  Clock::time_point latest{};
  explicit Impl(LookupLayoutCacheOptions o) : options(o) {
    detail::Require(o.idle_ttl.count() > 0 && o.idle_ttl <= std::chrono::hours(24), "invalid layout-cache TTL");
    std::random_device random; std::ostringstream out;
    for (int i = 0; i < 4; ++i) out << std::hex << random() << '-';
    instance = out.str();
  }
  void Remove(const std::string& token) {
    auto it = entries.find(token); if (it == entries.end()) return;
    stats.bytes -= it->second.charge; content.erase(it->second.content);
    lru.erase(it->second.lru); entries.erase(it); stats.entries = entries.size();
  }
  Clock::time_point Expire(Clock::time_point now) {
    now = latest = std::max(latest, now);
    while (!lru.empty() && now - entries.at(lru.back()).used >= options.idle_ttl) {
      const auto token = lru.back(); Remove(token); ++stats.expirations;
    }
    return now;
  }
  void Touch(Entry& e, Clock::time_point now) { e.used = now; lru.splice(lru.begin(), lru, e.lru); }
};
LookupLayoutCache::LookupLayoutCache(LookupLayoutCacheOptions o) : impl_(std::make_unique<Impl>(o)) {}
LookupLayoutCache::~LookupLayoutCache() = default;
std::string LookupLayoutCache::Find(const LookupLayoutScope& s, const Layout& layout, Clock::time_point now) {
  const auto key = Content(s, layout);
  std::lock_guard lock(impl_->mutex); now = impl_->Expire(now);
  auto it = impl_->content.find(key);
  if (it == impl_->content.end()) { ++impl_->stats.misses; return {}; }
  auto& entry = impl_->entries.at(it->second); impl_->Touch(entry, now); ++impl_->stats.hits;
  return it->second;
}
std::shared_ptr<const LookupLayoutCache::Layout> LookupLayoutCache::Resolve(
    const LookupLayoutScope& s, const std::string& token, Clock::time_point now) {
  std::lock_guard lock(impl_->mutex); now = impl_->Expire(now);
  const auto it = impl_->entries.find(token);
  if (it == impl_->entries.end()) { ++impl_->stats.misses; return {}; }
  detail::Require(it->second.scope == s, "lookup layout belongs to another run, learner, shard or epoch",
                  Error::Code::stale_generation);
  impl_->Touch(it->second, now); ++impl_->stats.hits; return it->second.layout;
}
std::string LookupLayoutCache::Put(LookupLayoutScope s, std::shared_ptr<const Layout> layout,
                                 std::string token, Clock::time_point now) {
  detail::Require(bool(layout) && !s.run.empty() && s.run.size() <= 256 && token.size() <= 256,
                  "invalid layout-cache identity");
  std::lock_guard lock(impl_->mutex); now = impl_->Expire(now);
  if (!impl_->options.entries || !impl_->options.bytes || LayoutBytes(*layout) > impl_->options.bytes) {
    ++impl_->stats.rejected; return {};
  }
  auto key = Content(s, *layout);
  // Include both exact-key copies, immutable vectors, scope/token and conservative container overhead.
  const auto charge = 1024 + 2 * key.capacity() + LayoutBytes(*layout) + 2 * (s.run.capacity() + 256);
  if (charge > impl_->options.bytes) { ++impl_->stats.rejected; return {}; }
  if (!token.empty() && impl_->entries.contains(token)) {
    auto& entry = impl_->entries.at(token);
    detail::Require(entry.scope == s && entry.content == key, "lookup layout token rebound to different content");
    impl_->Touch(entry, now); return token;
  }
  if (const auto it = impl_->content.find(key); it != impl_->content.end()) {
    if (token.empty() || token == it->second) {
      impl_->Touch(impl_->entries.at(it->second), now); return it->second;
    }
    const auto old = it->second; impl_->Remove(old);
  }
  if (token.empty()) {
    do {
      detail::Require(impl_->sequence != UINT64_MAX, "lookup layout token sequence exhausted");
      token = impl_->instance + std::to_string(++impl_->sequence);
    } while (impl_->entries.contains(token));
  }
  while (impl_->entries.size() >= impl_->options.entries || charge > impl_->options.bytes - impl_->stats.bytes) {
    const auto old = impl_->lru.back(); impl_->Remove(old); ++impl_->stats.evictions;
  }
  // Roll back partial allocations so optional cache admission cannot corrupt future lookups.
  bool indexed = false, listed = false;
  try {
    impl_->entries.emplace(token, Impl::Entry{std::move(s), std::move(layout), key, charge, now, {}});
    impl_->content.emplace(key, token); indexed = true;
    impl_->lru.push_front(token); listed = true;
    impl_->entries.at(token).lru = impl_->lru.begin();
  } catch (...) {
    if (listed) impl_->lru.pop_front();
    if (indexed) impl_->content.erase(key);
    impl_->entries.erase(token); throw;
  }
  impl_->stats.bytes += charge;
  impl_->stats.entries = impl_->entries.size(); ++impl_->stats.registrations;
  return token;
}
void LookupLayoutCache::Erase(const std::string& token) { std::lock_guard lock(impl_->mutex); impl_->Remove(token); }
void LookupLayoutCache::Clear() {
  std::lock_guard lock(impl_->mutex); impl_->entries.clear(); impl_->content.clear(); impl_->lru.clear();
  impl_->stats.bytes = impl_->stats.entries = 0;
}
LookupLayoutCacheStats LookupLayoutCache::Stats(Clock::time_point now) {
  std::lock_guard lock(impl_->mutex); impl_->Expire(now); return impl_->stats;
}
}  // namespace eps
