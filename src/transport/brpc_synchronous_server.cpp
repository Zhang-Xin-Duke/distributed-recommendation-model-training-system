#include "transport/brpc_synchronous_server.hpp"

#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include "transport/codec.hpp"

namespace eps::transport {
namespace {
template<class T> T Take(std::future<Result<T>> f) {
  auto r = f.get(); if (auto* e = std::get_if<Error>(&r)) throw detail::Failure(*e);
  return std::get<T>(std::move(r));
}
template<class R, class F> void Reply(R* r, google::protobuf::Closure* done, F operation) {
  brpc::ClosureGuard guard(done);
  try { operation(); }
  catch (const detail::Failure& e) { SetError(r->mutable_status(), e.error()); }
  catch (const std::exception& e) { SetError(r->mutable_status(), {Error::Code::run_failed, e.what()}); }
}
}
struct BrpcSynchronousServer::Impl {
  struct Pending {
    std::shared_future<Result<Ack>> future;
    brpc::Controller* controller;
    v1::Ack* response;
    google::protobuf::Closure* done;
  };
  CpuParameterService& parameters;
  SynchronousGradientOptions options;
  SynchronousGradientCollector collector;
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::optional<Pending>> pending;
  bool stopping = false;
  std::thread pump;

  Impl(CpuParameterService& ps, SynchronousGradientOptions o)
      : parameters(ps), options(std::move(o)), collector(options,
        [this](auto round, auto tables) { return Apply(round, std::move(tables)); }),
        pending(options.workers), pump([this] { Poll(); }) {}
  Result<Ack> Apply(std::uint64_t round, std::vector<GradientTable> tables) {
    LookupRequest request{{options.run, 1023, round, Phase::lookup}, options.epoch, false, {}};
    std::vector<TensorBlock> gradients;
    for (std::size_t i = 0; i < tables.size(); ++i) {
      auto& table = tables[i];
      const auto count = table.keys.size();
      request.features.push_back({table.table, static_cast<std::uint32_t>(i), Pooling::sequence,
        std::move(table.keys), {0, count}, {count}});
      gradients.push_back({static_cast<std::uint32_t>(i), {count, table.dimension}, std::move(table.values)});
    }
    auto lookup = Take(parameters.Lookup(std::move(request)));
    return parameters.Push({{options.run, 1023, round, Phase::push},
      std::move(lookup.ticket), std::move(gradients)}).get();
  }
  void Validate(google::protobuf::RpcController* c, const google::protobuf::Message& q) const {
    detail::Require(q.ByteSizeLong() <= (1ULL << 20) &&
      static_cast<brpc::Controller*>(c)->request_attachment().size() <= kMaxAttachment,
      "synchronous RPC payload exceeds limit");
  }
  void Control(google::protobuf::RpcController* c, const v1::TrainingControl& q) const {
    Validate(c, q);
    detail::Require(q.run() == options.run && q.epoch() == options.epoch,
      "synchronous group run/epoch mismatch", Error::Code::stale_generation);
  }
  static void Finish(Pending& p) {
    Reply(p.response, p.done, [&] {
      if (p.controller->IsCanceled()) SetError(p.response->mutable_status(),
        {Error::Code::unavailable, "gradient RPC disconnected"});
      else if (const auto* e = std::get_if<Error>(&p.future.get())) SetError(p.response->mutable_status(), *e);
    });
  }
  void Poll() {
    std::unique_lock lock(mutex);
    for (;;) {
      std::vector<Pending> ready;
      bool waiting = false;
      for (auto& p : pending) if (p) {
        if (p->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready || p->controller->IsCanceled()) {
          ready.push_back(std::move(*p)); p.reset();
        } else waiting = true;
      }
      const bool finished = stopping && !waiting;
      lock.unlock(); for (auto& p : ready) Finish(p); lock.lock();
      if (finished) return;
      changed.wait_for(lock, std::chrono::milliseconds(10));
    }
  }
  void Shutdown() {
    { std::lock_guard lock(mutex); if (stopping) return; stopping = true; collector.Abort("gradient service shutting down"); }
    changed.notify_all(); if (pump.joinable()) pump.join();
  }
};
BrpcSynchronousServer::BrpcSynchronousServer(CpuParameterService& service, SynchronousGradientOptions options)
    : impl_(std::make_unique<Impl>(service, std::move(options))) {}
BrpcSynchronousServer::~BrpcSynchronousServer() { Shutdown(); }
void BrpcSynchronousServer::Shutdown() { impl_->Shutdown(); }
void BrpcSynchronousServer::Submit(google::protobuf::RpcController* c, const v1::GradientContribution* q,
                                  v1::Ack* r, google::protobuf::Closure* done) {
  brpc::ClosureGuard guard(done);
  try {
    impl_->Validate(c, *q);
    detail::Require(q->tables_size() <= 256 && q->task().worker() < impl_->options.workers,
      "invalid synchronous worker/table count");
    GradientContribution input{{q->task().run(), q->task().epoch(), q->task().worker(), q->task().iteration()},
      q->sample_count(), {}};
    std::uint64_t decoded = 0;
    const auto& attachment = static_cast<brpc::Controller*>(c)->request_attachment();
    for (const auto& table : q->tables()) {
      // Bound cumulative decoded bytes even when malicious slices overlap.
      for (const auto* slice : {&table.keys(), &table.values()}) {
        detail::Require(slice->byte_length() <= kMaxAttachment - decoded, "gradient decoded payload exceeds limit");
        decoded += slice->byte_length();
      }
      input.tables.push_back({table.table_id(), table.dimension(), Unpack<Key>(table.keys(), attachment),
        Unpack<float>(table.values(), attachment)});
    }
    std::unique_lock lock(impl_->mutex);
    detail::Require(!impl_->stopping, "gradient service shutting down", Error::Code::unavailable);
    auto future = impl_->collector.Submit(std::move(input));
    if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      if (const auto* e = std::get_if<Error>(&future.get())) SetError(r->mutable_status(), *e);
    } else {
      auto& slot = impl_->pending[q->task().worker()];
      if (slot && (slot->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready ||
                   slot->controller->IsCanceled())) {
        auto old = std::move(*slot); slot.reset();
        lock.unlock(); Impl::Finish(old); lock.lock();
      }
      detail::Require(!impl_->stopping, "gradient service shutting down", Error::Code::unavailable);
      detail::Require(!slot, "this worker already has a pending gradient response; retry the same task",
        Error::Code::capacity_exceeded);
      slot = Impl::Pending{std::move(future), static_cast<brpc::Controller*>(c), r, done};
      guard.release(); impl_->changed.notify_one();
    }
  } catch (const detail::Failure& e) { SetError(r->mutable_status(), e.error()); }
    catch (const std::exception& e) { SetError(r->mutable_status(), {Error::Code::run_failed, e.what()}); }
}
void BrpcSynchronousServer::Stats(google::protobuf::RpcController* c, const v1::TrainingControl* q,
                                 v1::SynchronousGradientStatsReply* r, google::protobuf::Closure* done) {
  Reply(r, done, [&] {
    impl_->Control(c, *q); auto s = impl_->collector.Stats();
    r->set_next_round(s.next_round); r->set_received_workers(s.received_workers);
    r->set_retained_bytes(s.retained_bytes); r->set_duplicate_submissions(s.duplicate_submissions);
    r->set_applying(s.applying); r->set_failed(s.failed);
  });
}
void BrpcSynchronousServer::Abort(google::protobuf::RpcController* c, const v1::TrainingControl* q,
                                 v1::Ack* r, google::protobuf::Closure* done) {
  Reply(r, done, [&] { impl_->Control(c, *q); impl_->collector.Abort(q->reason()); impl_->changed.notify_all(); });
}
}  // namespace eps::transport
