#include "eps/lookup_layout_cache.hpp"
#include <atomic>
#include <iostream>
#include <thread>
using namespace eps;
using namespace std::chrono_literals;
namespace {
void Check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
auto Layout(Key key = 7) {
  return std::make_shared<const LookupLayoutCache::Layout>(LookupLayoutCache::Layout{
    {1, 0, Pooling::mean, {key, key, 9}, {0, 2, 3}, {2, 1}}});
}
void ExactMetadataAndScope() {
  LookupLayoutCache c; LookupLayoutScope s{"run", 1, 3, 2}; auto layout = Layout();
  const auto token = c.Put(s, layout);
  Check(!token.empty() && c.Put(s, layout) == token && c.Stats().registrations == 1, "layout not deduplicated");
  Check(c.Find(s, *layout) == token && c.Resolve(s, token)->at(0).keys == std::vector<Key>({7,7,9}),
        "duplicate IDs or layout changed");
  auto changed = *layout;
  for (int difference = 0; difference < 7; ++difference) {
    changed = *layout;
    if (difference == 0) changed[0].pooling = Pooling::sum;
    if (difference == 1) changed[0].keys[1] = 8;
    if (difference == 2) changed[0].offsets = {0, 1, 3};
    if (difference == 3) changed[0].valid_lengths[0] = 3;
    if (difference == 4) changed[0].table = 2;
    if (difference == 5) changed[0].feature = 1;
    if (difference == 6) changed.push_back(changed[0]);
    Check(c.Find(s, changed).empty(), "different complete layout incorrectly aliased");
  }
  for (int field = 0; field < 4; ++field) {
    auto wrong = s;
    if (field == 0) wrong.run = "other";
    if (field == 1) ++wrong.epoch;
    if (field == 2) ++wrong.learner;
    if (field == 3) ++wrong.shard;
    bool rejected = false;
    try { c.Resolve(wrong, token); } catch (const std::exception&) { rejected = true; }
    Check(rejected && c.Find(wrong, *layout).empty(), "token crossed its scope");
  }
  c.Put(s, layout, "replacement");
  Check(!c.Resolve(s,token) && c.Find(s,*layout)=="replacement", "remote token was not replaced");
  bool rejected = false;
  try { c.Put(s,Layout(11),"replacement"); } catch (const std::exception&) { rejected = true; }
  Check(rejected && c.Find(s,*layout)=="replacement", "token rebound changed old entry");
  const auto future = token.substr(0,token.rfind('-')+1)+"2";
  c.Put(s,Layout(11),future);
  const auto automatic = c.Put(s,Layout(12));
  Check(automatic!=future && c.Resolve(s,future)->at(0).keys[0]==11 &&
        c.Resolve(s,automatic)->at(0).keys[0]==12,"generated token collided with an existing opaque token");
}
void EvictionAndExpiry() {
  LookupLayoutScope s{"run",1,0,0}; LookupLayoutCache c({2, 16384, 100ms});
  const auto now = LookupLayoutCache::Clock::now();
  auto first = c.Put(s,Layout(1),{},now); auto pinned = c.Resolve(s,first,now);
  auto second = c.Put(s,Layout(2),{},now);
  c.Find(s,*Layout(1),now+10ms); c.Put(s,Layout(3),{},now+20ms);
  Check(!c.Resolve(s,second,now+20ms) && c.Resolve(s,first,now+30ms), "LRU evicted the wrong layout");
  Check(c.Stats(now+30ms).entries == 2 && c.Stats(now+30ms).evictions == 1 && pinned->at(0).keys[0]==1,
        "eviction lost pinned metadata or exceeded count");
  Check(c.Stats(now+131ms).entries == 0 && c.Stats(now+131ms).expirations == 2,
        "idle TTL failed to release metadata");
  LookupLayoutCache tiny({8,64,100ms}), disabled({0,16384,100ms});
  Check(tiny.Put(s,Layout()).empty() && tiny.Stats().rejected==1 && disabled.Put(s,Layout()).empty(),
        "cache budget did not reject registration");
  LookupLayoutCache restart;
  const auto fresh = restart.Put(s,Layout(1));
  Check(fresh != first && !restart.Resolve(s,first), "token survived an unrelated process namespace");
}
void ConcurrentReaders() {
  LookupLayoutCache c({32,1 << 20,1000ms}); std::atomic<unsigned> failed=0;
  std::vector<std::jthread> threads;
  for (unsigned w=0; w<8; ++w) threads.emplace_back([&,w] {
    try {
      LookupLayoutScope scope{"concurrent",1,w,0};
      for (unsigned i=0;i<200;++i) {
        auto layout=Layout(i%8); const auto token=c.Put(scope,layout);
        const auto found=c.Resolve(scope,token);
        if (found) Check(found->at(0).keys == layout->at(0).keys,"concurrent registry changed IDs");
        (void)c.Find(scope,*layout);
      }
    } catch (...) { ++failed; }
  });
  threads.clear(); const auto stats=c.Stats();
  Check(!failed && stats.entries <= 32 && stats.bytes <= (1 << 20), "concurrent registry exceeded bounds");
}
}
int main() {
  try { ExactMetadataAndScope(); EvictionAndExpiry(); ConcurrentReaders();
    std::cout << "PASS reusable lookup layouts: exact CSR metadata, scoped tokens, LRU/TTL/budgets, immutable held views, restart namespace and concurrent registry\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
