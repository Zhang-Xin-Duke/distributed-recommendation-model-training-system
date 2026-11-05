#include "eps/gradient_layout_cache.hpp"
#include <atomic>
#include <iostream>
#include <thread>
using namespace eps;
using namespace std::chrono_literals;
namespace {
void Check(bool value, const char *why) {
  if (!value)
    throw std::runtime_error(why);
}
auto Layout(Key key = 1) {
  return std::make_shared<const GradientLayoutCache::Layout>(
      GradientLayoutCache::Layout{{1, 8, {key, key + 1}}, {2, 4, {}}});
}
void IdentityAndBounds() {
  GradientLayoutCache cache({2, 1 << 20, 100ms});
  GradientLayoutScope scope{"run", 1, 0, 2};
  const auto now = GradientLayoutCache::Clock::now();
  const auto layout = Layout();
  const auto token = cache.Put(scope, layout, {}, now);
  Check(!token.empty(), "gradient registration declined");
  for (unsigned change = 0; change < 4; ++change) {
    auto other = *layout;
    if (change == 0)
      ++other[0].dimension;
    if (change == 1)
      ++other[0].table;
    if (change == 2)
      other[0].keys[1] += 1;
    if (change == 3)
      std::swap(other[0], other[1]);
    Check(cache.Find(scope, other, now).empty(),
          "gradient metadata fingerprint aliased schema/order/Keys");
  }
  GradientLayoutCache restart;
  Check(!restart.Resolve(scope, token, now),
        "gradient token survived registry replacement");
  auto held = cache.Resolve(scope, token, now);
  cache.Put(scope, Layout(10), {}, now);
  cache.Put(scope, Layout(20), {}, now);
  Check(!cache.Resolve(scope, token, now) && held->at(0).keys[0] == 1,
        "eviction invalidated held gradient metadata");
  Check(cache.Stats(now + 101ms).entries == 0 &&
            cache.Stats(now + 101ms).expirations == 2,
        "gradient TTL did not release metadata");
  GradientLayoutCache tiny({1, 32, 100ms});
  Check(tiny.Put(scope, layout).empty(), "gradient cache exceeded byte budget");
}
void ConcurrentScopes() {
  GradientLayoutCache cache({32, 1 << 20, 1000ms});
  std::atomic<unsigned> failures = 0;
  std::vector<std::jthread> threads;
  for (unsigned worker = 0; worker < 8; ++worker)
    threads.emplace_back([&, worker] {
      try {
        GradientLayoutScope scope{"workers", 1, worker, 0};
        for (unsigned step = 0; step < 200; ++step) {
          auto layout = Layout(step % 8);
          auto token = cache.Put(scope, layout);
          auto found = cache.Resolve(scope, token);
          if (found)
            Check(*found == *layout,
                  "concurrent registry changed gradient Keys");
          cache.Find(scope, *layout);
        }
      } catch (...) {
        ++failures;
      }
    });
  threads.clear();
  auto stats = cache.Stats();
  Check(!failures && stats.entries <= 32 && stats.bytes <= (1 << 20),
        "concurrent gradient registry exceeded bounds");
}
} // namespace
int main() {
  try {
    IdentityAndBounds();
    ConcurrentScopes();
    std::cout << "PASS gradient metadata identity, process namespace, pinned "
                 "views, budgets, TTL and concurrent worker scopes\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
