#include "eps/synchronous_gradients.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include "common/error.hpp"

namespace eps {
namespace {
std::shared_future<Result<Ack>> Ready(Result<Ack> result) {
  std::promise<Result<Ack>> p; p.set_value(std::move(result)); return p.get_future().share();
}
using detail::Require;
}
struct SynchronousGradientCollector::Impl {
  SynchronousGradientOptions options;
  Apply apply;
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::shared_ptr<const GradientContribution>> inputs;
  std::promise<Result<Ack>> promise;
  std::shared_future<Result<Ack>> future = promise.get_future().share();
  SynchronousGradientStats stats;
  std::string failure;
  bool stopping = false, settled = false;
  std::optional<std::chrono::steady_clock::time_point> deadline;
  std::thread executor;

  Impl(SynchronousGradientOptions o, Apply fn) : options(std::move(o)), apply(std::move(fn)) {
    Require(!options.run.empty() && options.run.size() <= 256 && options.epoch &&
      options.workers && options.workers <= 1023 && options.shards && options.shards <= 1024 &&
      options.shard < options.shards && !options.tables.empty() && options.tables.size() <= 256 &&
      options.gradient_bytes >= 4096 && options.gradient_bytes <= (1ULL << 34) &&
      options.round_timeout >= std::chrono::milliseconds(1) &&
      options.round_timeout <= std::chrono::minutes(10) && bool(apply), "invalid synchronous gradient options");
    Require(options.reduction == GradientReduction::sum ||
      options.reduction == GradientReduction::worker_mean ||
      options.reduction == GradientReduction::sample_mean, "invalid gradient reduction");
    std::sort(options.tables.begin(), options.tables.end(), [](const auto& a, const auto& b) { return a.table < b.table; });
    for (std::size_t i = 0; i < options.tables.size(); ++i) {
      const auto& t = options.tables[i];
      Require(t.dimension && t.dimension <= 65536 && t.keys.empty() && t.values.empty() &&
        (!i || options.tables[i-1].table != t.table), "invalid synchronous table schema");
    }
    inputs.resize(options.workers);
    executor = std::thread([this] { Execute(); });
  }
  void Fail(std::string reason) {
    if (!failure.empty()) return;
    failure = reason.empty() ? "synchronous training aborted" : std::move(reason);
    stats.failed = true;
    if (!settled) { promise.set_value(Error{Error::Code::run_failed, failure}); settled = true; }
  }
  std::uint64_t Validate(const GradientContribution& c) const {
    Require(c.sample_count && c.sample_count <= (1ULL << 40), "invalid sample count");
    Require(c.tables.size() == options.tables.size(), "gradient table set differs from group schema");
    Require(c.tables.capacity() <= options.gradient_bytes / sizeof(GradientTable) &&
      c.task.run.capacity() <= options.gradient_bytes, "gradient metadata exceeds budget", Error::Code::capacity_exceeded);
    std::uint64_t bytes = sizeof(GradientContribution) + c.tables.capacity() * sizeof(GradientTable) + c.task.run.capacity();
    for (std::size_t i = 0; i < c.tables.size(); ++i) {
      const auto& t = c.tables[i]; const auto& schema = options.tables[i];
      Require(t.table == schema.table && t.dimension == schema.dimension, "gradient table schema mismatch");
      Require(t.keys.size() <= options.gradient_bytes / (sizeof(Key) + sizeof(float) * t.dimension),
        "gradient rows exceed budget", Error::Code::capacity_exceeded);
      Require(t.values.size() == t.keys.size() * t.dimension, "gradient shape mismatch");
      Require(t.keys.capacity() <= options.gradient_bytes / sizeof(Key) &&
        t.values.capacity() <= options.gradient_bytes / sizeof(float),
        "gradient retained capacity exceeds budget", Error::Code::capacity_exceeded);
      for (std::size_t k = 0; k < t.keys.size(); ++k)
        Require(t.keys[k] % options.shards == options.shard && (!k || t.keys[k-1] < t.keys[k]),
          "gradient keys must be unique, sorted and owned by this shard");
      for (float value : t.values) Require(std::isfinite(value), "non-finite gradient");
      // Reserve conservative space for retained input, deterministic reduction
      // map/vector storage and output. Duplicate rows can only reduce this bound.
      const auto cost = t.keys.size() * (128ULL + 2 * sizeof(Key) + 4ULL * sizeof(float) * t.dimension) +
        (t.keys.capacity() - t.keys.size()) * sizeof(Key) + (t.values.capacity() - t.values.size()) * sizeof(float);
      Require(cost <= options.gradient_bytes && bytes <= options.gradient_bytes - cost,
        "gradient reduction scratch exceeds budget", Error::Code::capacity_exceeded);
      bytes += cost;
    }
    return bytes;
  }
  std::vector<GradientTable> Reduce(const std::vector<std::shared_ptr<const GradientContribution>>& round) const {
    double denominator = 1;
    if (options.reduction == GradientReduction::worker_mean) denominator = options.workers;
    else if (options.reduction == GradientReduction::sample_mean) {
      denominator = 0;
      for (const auto& c : round) denominator += double(c->sample_count);
    }
    std::vector<GradientTable> output;
    for (std::size_t table = 0; table < options.tables.size(); ++table) {
      const auto& schema = options.tables[table];
      std::map<Key, std::vector<double>> rows;
      // Fixed worker order makes reduction independent of RPC arrival order.
      for (const auto& c : round) {
        const auto& t = c->tables[table];
        const double scale = (options.reduction == GradientReduction::sample_mean ? double(c->sample_count) : 1.) / denominator;
        for (std::size_t row = 0; row < t.keys.size(); ++row) {
          auto [it, inserted] = rows.try_emplace(t.keys[row]);
          if (inserted) it->second.resize(schema.dimension);
          for (std::size_t d = 0; d < schema.dimension; ++d)
            it->second[d] += double(t.values[row * schema.dimension + d]) * scale;
        }
      }
      GradientTable result{schema.table, schema.dimension, {}, {}};
      result.keys.reserve(rows.size()); result.values.reserve(rows.size() * schema.dimension);
      for (const auto& [key, values] : rows) {
        result.keys.push_back(key);
        for (double value : values) {
          const auto f = static_cast<float>(value);
          Require(std::isfinite(f), "gradient reduction overflow", Error::Code::run_failed);
          result.values.push_back(f);
        }
      }
      output.push_back(std::move(result));
    }
    return output;
  }
  void Execute() {
    std::unique_lock lock(mutex);
    for (;;) {
      changed.wait(lock, [&] { return stopping || !failure.empty() || stats.received_workers; });
      if (stopping || !failure.empty()) return;
      if (stats.received_workers != options.workers) {
        changed.wait_until(lock, *deadline, [&] {
          return stopping || !failure.empty() || stats.received_workers == options.workers;
        });
        if (stopping || !failure.empty()) return;
        if (stats.received_workers != options.workers) {
          Fail("synchronous round timed out waiting for worker gradients"); return;
        }
      }
      stats.applying = true;
      const auto round = stats.next_round;
      const auto packets = inputs;
      lock.unlock();
      Result<Ack> result;
      try { result = apply(round, Reduce(packets)); }
      catch (const detail::Failure& e) { result = e.error(); }
      catch (const std::exception& e) { result = Error{Error::Code::run_failed, e.what()}; }
      lock.lock(); stats.applying = false;
      if (!failure.empty() || stopping) return;
      if (const auto* error = std::get_if<Error>(&result)) { Fail(error->message); return; }
      promise.set_value(Ack{}); settled = true;
      ++stats.next_round; stats.received_workers = 0; stats.retained_bytes = 0;
      std::fill(inputs.begin(), inputs.end(), nullptr); deadline.reset();
      promise = std::promise<Result<Ack>>(); future = promise.get_future().share(); settled = false;
    }
  }
};

SynchronousGradientCollector::SynchronousGradientCollector(SynchronousGradientOptions options, Apply apply)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(apply))) {}
SynchronousGradientCollector::~SynchronousGradientCollector() {
  { std::lock_guard lock(impl_->mutex); impl_->stopping = true; impl_->Fail("gradient collector shutting down"); }
  impl_->changed.notify_all();
  if (impl_->executor.joinable()) impl_->executor.join();
}
std::shared_future<Result<Ack>> SynchronousGradientCollector::Submit(GradientContribution c) {
  std::lock_guard lock(impl_->mutex);
  try {
    Require(c.task.run == impl_->options.run && c.task.epoch == impl_->options.epoch,
      "gradient run/epoch mismatch", Error::Code::stale_generation);
    Require(c.task.worker < impl_->options.workers && c.task.iteration != UINT64_MAX,
      "invalid gradient worker or round");
    Require(impl_->failure.empty(), impl_->failure.c_str(), Error::Code::run_failed);
    if (c.task.iteration < impl_->stats.next_round) {
      ++impl_->stats.duplicate_submissions; return Ready(Ack{});
    }
    Require(c.task.iteration == impl_->stats.next_round, "gradient round is not current");
    auto& slot = impl_->inputs[c.task.worker];
    if (slot) {
      Require(*slot == c, "in-flight gradient retry changed its payload");
      ++impl_->stats.duplicate_submissions; return impl_->future;
    }
    const auto bytes = impl_->Validate(c);
    Require(bytes <= impl_->options.gradient_bytes - impl_->stats.retained_bytes,
      "synchronous gradient memory budget exhausted", Error::Code::capacity_exceeded);
    slot = std::make_shared<const GradientContribution>(std::move(c));
    impl_->stats.retained_bytes += bytes; ++impl_->stats.received_workers;
    if (!impl_->deadline) impl_->deadline = std::chrono::steady_clock::now() + impl_->options.round_timeout;
    impl_->changed.notify_one(); return impl_->future;
  } catch (const detail::Failure& e) { return Ready(e.error()); }
}
SynchronousGradientStats SynchronousGradientCollector::Stats() const {
  std::lock_guard lock(impl_->mutex); return impl_->stats;
}
void SynchronousGradientCollector::Abort(std::string reason) {
  { std::lock_guard lock(impl_->mutex); impl_->Fail(std::move(reason)); }
  impl_->changed.notify_all();
}
}  // namespace eps
