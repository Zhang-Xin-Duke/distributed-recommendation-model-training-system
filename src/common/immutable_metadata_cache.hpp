#pragma once
#include "common/error.hpp"
#include "eps/lookup_layout_cache.hpp"
#include <algorithm>
#include <list>
#include <mutex>
#include <random>
#include <sstream>
#include <unordered_map>

namespace eps::detail {
// Shared bounded registry; typed callers define exact fingerprints and charges.
template <class Scope, class Layout> class ImmutableMetadataCache {
public:
  using Clock = std::chrono::steady_clock;
  using Fingerprint = std::string (*)(const Scope &, const Layout &);
  using Size = std::uint64_t (*)(const Layout &);
  struct Entry {
    Scope scope;
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
  Fingerprint fingerprint;
  Size layout_bytes;
  explicit ImmutableMetadataCache(LookupLayoutCacheOptions o,
                                  Fingerprint content_fn, Size size_fn)
      : options(o), fingerprint(content_fn), layout_bytes(size_fn) {
    detail::Require(o.idle_ttl.count() > 0 &&
                        o.idle_ttl <= std::chrono::hours(24),
                    "invalid layout-cache TTL");
    std::random_device random;
    std::ostringstream out;
    for (int i = 0; i < 4; ++i)
      out << std::hex << random() << '-';
    instance = out.str();
  }
  void Remove(const std::string &token) {
    auto it = entries.find(token);
    if (it == entries.end())
      return;
    stats.bytes -= it->second.charge;
    content.erase(it->second.content);
    lru.erase(it->second.lru);
    entries.erase(it);
    stats.entries = entries.size();
  }
  Clock::time_point Expire(Clock::time_point now) {
    now = latest = std::max(latest, now);
    while (!lru.empty() &&
           now - entries.at(lru.back()).used >= options.idle_ttl) {
      const auto token = lru.back();
      Remove(token);
      ++stats.expirations;
    }
    return now;
  }
  void Touch(Entry &e, Clock::time_point now) {
    e.used = now;
    lru.splice(lru.begin(), lru, e.lru);
  }
  std::string Find(const Scope &s, const Layout &layout,
                   Clock::time_point now) {
    const auto key = fingerprint(s, layout);
    std::lock_guard lock(mutex);
    now = Expire(now);
    auto it = content.find(key);
    if (it == content.end()) {
      ++stats.misses;
      return {};
    }
    auto &entry = entries.at(it->second);
    Touch(entry, now);
    ++stats.hits;
    return it->second;
  }
  std::shared_ptr<const Layout>
  Resolve(const Scope &s, const std::string &token, Clock::time_point now) {
    std::lock_guard lock(mutex);
    now = Expire(now);
    const auto it = entries.find(token);
    if (it == entries.end()) {
      ++stats.misses;
      return {};
    }
    detail::Require(
        it->second.scope == s,
        "metadata layout belongs to another run, learner, shard or epoch",
        Error::Code::stale_generation);
    Touch(it->second, now);
    ++stats.hits;
    return it->second.layout;
  }
  std::string Put(Scope s, std::shared_ptr<const Layout> layout,
                  std::string token, Clock::time_point now) {
    detail::Require(bool(layout) && !s.run.empty() && s.run.size() <= 256 &&
                        token.size() <= 256,
                    "invalid layout-cache identity");
    std::lock_guard lock(mutex);
    now = Expire(now);
    if (!options.entries || !options.bytes ||
        layout_bytes(*layout) > options.bytes) {
      ++stats.rejected;
      return {};
    }
    auto key = fingerprint(s, *layout);
    // Include both exact-key copies, immutable vectors, scope/token and
    // conservative container overhead.
    const auto charge = 1024 + 2 * key.capacity() + layout_bytes(*layout) +
                        2 * (s.run.capacity() + 256);
    if (charge > options.bytes) {
      ++stats.rejected;
      return {};
    }
    if (!token.empty() && entries.contains(token)) {
      auto &entry = entries.at(token);
      detail::Require(entry.scope == s && entry.content == key,
                      "metadata layout token rebound to different content");
      Touch(entry, now);
      return token;
    }
    if (const auto it = content.find(key); it != content.end()) {
      if (token.empty() || token == it->second) {
        Touch(entries.at(it->second), now);
        return it->second;
      }
      const auto old = it->second;
      Remove(old);
    }
    if (token.empty()) {
      do {
        detail::Require(sequence != UINT64_MAX,
                        "metadata layout token sequence exhausted");
        token = instance + std::to_string(++sequence);
      } while (entries.contains(token));
    }
    while (entries.size() >= options.entries ||
           charge > options.bytes - stats.bytes) {
      const auto old = lru.back();
      Remove(old);
      ++stats.evictions;
    }
    // Roll back partial allocations so optional cache admission cannot corrupt
    // future lookups.
    bool indexed = false, listed = false;
    try {
      entries.emplace(
          token, Entry{std::move(s), std::move(layout), key, charge, now, {}});
      content.emplace(key, token);
      indexed = true;
      lru.push_front(token);
      listed = true;
      entries.at(token).lru = lru.begin();
    } catch (...) {
      if (listed)
        lru.pop_front();
      if (indexed)
        content.erase(key);
      entries.erase(token);
      throw;
    }
    stats.bytes += charge;
    stats.entries = entries.size();
    ++stats.registrations;
    return token;
  }
  void Erase(const std::string &token) {
    std::lock_guard lock(mutex);
    Remove(token);
  }
  void Clear() {
    std::lock_guard lock(mutex);
    entries.clear();
    content.clear();
    lru.clear();
    stats.bytes = stats.entries = 0;
  }
  LookupLayoutCacheStats Stats(Clock::time_point now) {
    std::lock_guard lock(mutex);
    Expire(now);
    return stats;
  }
};
} // namespace eps::detail
