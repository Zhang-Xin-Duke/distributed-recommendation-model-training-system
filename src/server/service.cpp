#include "eps/runtime.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "access/batch_access.hpp"
#include "common/error.hpp"
#include "server/request_tracker.hpp"
#include "store/snapshot.hpp"
#include "update/lazy_adam.hpp"

namespace eps {
using detail::Require;

class CpuParameterService::Impl {
 public:
  explicit Impl(RuntimeOptions options)
      : options_(std::move(options)), tracker_(options_.request_history),
        store_(std::make_unique<detail::ShardStore>(options_.shard_id,
          options_.shard_count, options_.partition_epoch, options_.state_bytes)) {
    Require(!options_.run_id.empty() && options_.run_id.size() <= 1024, "invalid run ID");
    Require(options_.request_history >= 4 && options_.max_live_tickets > 0 &&
            options_.max_queued_operations > 0, "invalid request limits");
    worker_ = std::thread([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock lock(mutex_);
          changed_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
          if (queue_.empty() && stopping_) return;
          queued_payload_bytes_ -= queue_.front().payload_bytes;
          task = std::move(queue_.front().function); queue_.pop_front();
        }
        task();
      }
    });
  }
  ~Impl() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    changed_.notify_all(); worker_.join();
  }

  template <typename T, typename Fn>
  std::future<Result<T>> Submit(Fn fn, std::uint64_t payload_bytes = 0) {
    auto promise = std::make_shared<std::promise<Result<T>>>();
    auto future = promise->get_future();
    {
      std::lock_guard lock(mutex_);
      if (stopping_ || queue_.size() >= options_.max_queued_operations ||
          payload_bytes > options_.queue_payload_bytes - queued_payload_bytes_) {
        promise->set_value(Error{Error::Code::capacity_exceeded, "shard operation queue is full"});
        return future;
      }
      queue_.push_back({[promise, fn = std::move(fn)]() mutable {
        try { promise->set_value(fn()); }
        catch (const detail::Failure& e) { promise->set_value(e.error()); }
        catch (const std::bad_alloc&) {
          promise->set_value(Error{Error::Code::capacity_exceeded, "allocation failed"});
        } catch (const std::exception& e) {
          promise->set_value(Error{Error::Code::run_failed, e.what()});
        }
      },payload_bytes});
      queued_payload_bytes_ += payload_bytes;
    }
    changed_.notify_one(); return future;
  }

  void Identity(const RequestId& id, Phase phase) const {
    Require(id.run == options_.run_id && id.phase == phase, "invalid run or request phase");
  }
  static std::string TicketName(const RequestId& id) {
    return std::to_string(id.learner) + "/" + std::to_string(id.sequence);
  }
  void Size(std::uint64_t& bytes, std::size_t count, std::size_t element_size) const {
    Require(count <= options_.max_request_bytes / element_size,
            "request array exceeds limit", Error::Code::capacity_exceeded);
    const auto add = count * element_size;
    Require(add <= options_.max_request_bytes - bytes, "request exceeds byte limit",
            Error::Code::capacity_exceeded);
    bytes += add;
  }

  LookupReply Lookup(LookupRequest request) {
    Identity(request.id, Phase::lookup);
    Require(request.partition_epoch == options_.partition_epoch, "partition epoch mismatch",
            Error::Code::stale_generation);
    const auto name = TicketName(request.id);
    const auto state = tracker_.Reserve(request.id);
    if (state != detail::RequestTracker::State::fresh) {
      ++stats_.duplicate_requests;
      auto it = tickets_.find(name);
      Require(it != tickets_.end(), "lookup is already released", Error::Code::retired_request);
      return it->second.reply;
    }
    try {
      Require(!draining_, "shard is draining", Error::Code::unavailable);
      Require(tickets_.size() < options_.max_live_tickets, "live ticket limit reached",
              Error::Code::capacity_exceeded);
      Require(!request.features.empty(), "lookup has no features");
      std::uint64_t bytes = 0;
      std::unordered_set<std::uint32_t> ids;
      for (const auto& f : request.features) {
        Require(ids.insert(f.feature).second, "duplicate feature ID");
        detail::ValidateFeature(f, *store_);
        Size(bytes, f.keys.size(), sizeof(Key));
        Size(bytes, f.offsets.size(), sizeof(std::uint64_t));
        Size(bytes, f.valid_lengths.size(), sizeof(std::uint64_t));
        const auto rows = f.pooling == Pooling::sequence ? f.keys.size() : f.valid_lengths.size();
        Size(bytes, rows, store_->Table(f.table).definition.dimension * sizeof(float));
      }
      Ticket ticket; ticket.id = request.id; ticket.read_only = request.read_only;
      ticket.reply.ticket = name;
      for (auto& f : request.features) {
        ticket.features.push_back(detail::PrepareFeature(std::move(f), *store_));
        ticket.reply.cpu_contributions.push_back(ticket.features.back().contribution);
        // Only layout is needed for backward; retain one cached feature payload.
        std::vector<float>().swap(ticket.features.back().contribution.values);
      }
      for (const auto& f : ticket.features)
        ticket.payload_bytes += (f.input.keys.capacity() + f.input.offsets.capacity() +
          f.input.valid_lengths.capacity() + f.unique_slots.capacity() +
          f.occurrence_to_unique.capacity()) * sizeof(std::uint64_t);
      for (const auto& f : ticket.reply.cpu_contributions)
        ticket.payload_bytes += f.values.capacity() * sizeof(float);
      Require(ticket.payload_bytes <= options_.ticket_payload_bytes - ticket_payload_bytes_,
              "live ticket payload budget exhausted", Error::Code::capacity_exceeded);
      auto reply = ticket.reply;
      const auto ticket_bytes = ticket.payload_bytes;
      tickets_.emplace(name, std::move(ticket));
      ticket_payload_bytes_ += ticket_bytes;
      ++stats_.lookups;
      return reply;
    } catch (...) { tracker_.Abort(request.id); throw; }
  }

  Ack Push(PushRequest request) {
    Identity(request.id, Phase::push);
    const auto completed = tracker_.CompletedAs(request.id);
    if (completed) {
      Require(*completed == Phase::push, "lookup was released without an update");
      ++stats_.duplicate_requests; return {};
    }
      auto it = tickets_.find(request.ticket);
      Require(it != tickets_.end(), "unknown lookup ticket");
      auto& ticket = it->second;
      Require(!ticket.read_only && TicketName(request.id) == request.ticket &&
              request.id.learner == ticket.id.learner, "ticket does not belong to update");
      std::uint64_t bytes = 0;
      for (const auto& g : request.feature_gradients) Size(bytes, g.values.size(), sizeof(float));
      auto gradients = detail::ReduceGradients(ticket.features, request.feature_gradients, *store_);
      for (const auto& [table, g] : gradients) detail::ValidateLazyAdam(store_->Table(table), g);
      for (const auto& [table, g] : gradients) detail::ApplyLazyAdam(store_->Table(table), g);
      tracker_.Complete(request.id);
      ticket_payload_bytes_ -= ticket.payload_bytes;
      tickets_.erase(it); ++stats_.updates;
      return {};
  }

  Ack Release(ReleaseRequest request) {
    Identity(request.id, Phase::release);
    if (tracker_.CompletedAs(request.id)) {
      ++stats_.duplicate_requests; return {};
    }
      auto it = tickets_.find(request.ticket);
      Require(it != tickets_.end() && TicketName(request.id) == request.ticket,
              "release does not match a live ticket");
      tracker_.Complete(request.id);
      ticket_payload_bytes_ -= it->second.payload_bytes;
      tickets_.erase(it); return {};
  }

  std::vector<RowState> ReadState(const std::vector<LocatedRow>& rows) {
    std::uint64_t bytes = 0;
    for (const auto& located : rows) {
      Require(located.home == options_.shard_id && located.placement == Placement::cpu &&
              located.generation == 0, "row placement changed", Error::Code::stale_generation);
      Size(bytes, store_->Table(located.row.table).definition.dimension, sizeof(float) * 3);
      for (const auto& [name, ticket] : tickets_)
        for (const auto& f : ticket.features)
          if (f.input.table == located.row.table)
            Require(std::find(f.input.keys.begin(), f.input.keys.end(), located.row.key) == f.input.keys.end(),
                    "row is held by an active lookup", Error::Code::unavailable);
    }
    std::vector<RowState> result; result.reserve(rows.size());
    for (const auto& row : rows) result.push_back(store_->Read(row.row.table, row.row.key));
    return result;
  }

  SnapshotMetadata Save(const std::string& id) {
    Require(!id.empty() && id.size() <= 128 && id != "." && id != ".." &&
      std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
      }), "invalid snapshot ID");
    draining_ = true;
    Require(tickets_.empty(), "drain in progress: complete or release outstanding tickets",
            Error::Code::unavailable);
    const auto file = options_.snapshot_directory / id /
        ("shard-" + std::to_string(options_.shard_id) + ".bin");
    detail::WriteSnapshot(file, options_.run_id, *store_);
    SnapshotMetadata result{id, options_.shard_id, options_.partition_epoch, {file.string()}, {}};
    for (const auto& [table, t] : store_->tables())
      result.optimizer_contexts.push_back({{options_.run_id, 0, 0, Phase::maintenance},
        options_.shard_id, table, t.optimizer_sequence,
        t.definition.optimizer.learning_rate, t.beta1_power, t.beta2_power});
    return result;
  }

  struct Ticket {
    RequestId id;
    bool read_only;
    std::vector<detail::PreparedFeature> features;
    LookupReply reply;
    std::uint64_t payload_bytes = 0;
  };
  RuntimeOptions options_;
  detail::RequestTracker tracker_;
  std::unique_ptr<detail::ShardStore> store_;
  std::unordered_map<std::string, Ticket> tickets_;
  RuntimeStats stats_;
  bool draining_ = false;
  std::uint64_t ticket_payload_bytes_ = 0;
  std::mutex mutex_;
  std::condition_variable changed_;
  struct Queued { std::function<void()> function; std::uint64_t payload_bytes; };
  std::deque<Queued> queue_;
  std::uint64_t queued_payload_bytes_ = 0;
  bool stopping_ = false;
  std::thread worker_;
};

CpuParameterService::CpuParameterService(RuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
CpuParameterService::~CpuParameterService() = default;
std::future<Result<Ack>> CpuParameterService::CreateTable(TableDefinition table) {
  return impl_->Submit<Ack>([this, table] { impl_->store_->Create(table); return Ack{}; });
}
std::future<Result<LookupReply>> CpuParameterService::Lookup(LookupRequest request) {
  std::uint64_t bytes = request.features.capacity() * sizeof(FeatureInput);
  for (const auto& f : request.features)
    bytes += (f.keys.capacity() + f.offsets.capacity() + f.valid_lengths.capacity()) * sizeof(std::uint64_t);
  return impl_->Submit<LookupReply>([this, request = std::move(request)]() mutable {
    return impl_->Lookup(std::move(request));
  }, bytes);
}
std::future<Result<Ack>> CpuParameterService::Push(PushRequest request) {
  std::uint64_t bytes = request.feature_gradients.capacity() * sizeof(TensorBlock);
  for (const auto& g : request.feature_gradients)
    bytes += g.values.capacity() * sizeof(float) + g.shape.capacity() * sizeof(std::uint64_t);
  return impl_->Submit<Ack>([this, request = std::move(request)]() mutable {
    return impl_->Push(std::move(request));
  }, bytes);
}
std::future<Result<Ack>> CpuParameterService::Release(ReleaseRequest request) {
  return impl_->Submit<Ack>([this, request = std::move(request)]() mutable {
    return impl_->Release(std::move(request));
  });
}
std::future<Result<std::vector<RowState>>> CpuParameterService::ReadState(std::vector<LocatedRow> rows) {
  const auto bytes = rows.size() * sizeof(LocatedRow);
  return impl_->Submit<std::vector<RowState>>([this, rows = std::move(rows)] {
    return impl_->ReadState(rows);
  }, bytes);
}
std::future<Result<Ack>> CpuParameterService::CompleteMigration(MigrationCompletion) {
  return impl_->Submit<Ack>([]() -> Result<Ack> {
    return Error{Error::Code::unavailable, "GPU cache agent is not attached"};
  });
}
std::future<Result<SnapshotMetadata>> CpuParameterService::DrainAndFlush(std::string id) {
  return impl_->Submit<SnapshotMetadata>([this, id = std::move(id)] { return impl_->Save(id); });
}
std::future<Result<Ack>> CpuParameterService::Restore(std::filesystem::path file) {
  return impl_->Submit<Ack>([this, file = std::move(file)] {
    Require(impl_->store_->empty() && impl_->tickets_.empty(), "restore requires a fresh service");
    auto store = detail::ReadSnapshot(file, impl_->options_.run_id,
      impl_->options_.shard_id, impl_->options_.shard_count,
      impl_->options_.partition_epoch, impl_->options_.state_bytes);
    impl_->store_ = std::move(store); return Ack{};
  });
}
std::future<Result<Ack>> CpuParameterService::Resume() {
  return impl_->Submit<Ack>([this] { impl_->draining_ = false; return Ack{}; });
}
std::future<Result<RuntimeStats>> CpuParameterService::Stats() {
  return impl_->Submit<RuntimeStats>([this] {
    auto s = impl_->stats_;
    s.rows = impl_->store_->rows(); s.state_bytes = impl_->store_->bytes();
    s.live_tickets = impl_->tickets_.size();
    s.ticket_payload_bytes = impl_->ticket_payload_bytes_;
    { std::lock_guard lock(impl_->mutex_); s.queued_payload_bytes = impl_->queued_payload_bytes_; }
    return s;
  });
}
}  // namespace eps
