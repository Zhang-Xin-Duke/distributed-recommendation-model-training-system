#include "transport/brpc_training_server.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include "transport/codec.hpp"

namespace eps::transport {
namespace {
TrainingTask Task(const v1::TrainingTask& q) {
  return {q.run(), q.epoch(), q.worker(), q.iteration()};
}
void Encode(const TrainingPermit& permit, v1::TrainingPermitReply* r) {
  auto* q = r->mutable_task();
  q->set_run(permit.task.run); q->set_epoch(permit.task.epoch);
  q->set_worker(permit.task.worker); q->set_iteration(permit.task.iteration);
  r->set_observed_completion_floor(permit.observed_completion_floor);
}
template<class Rsp, class Operation>
void Reply(Rsp* response, google::protobuf::Closure* done, Operation operation) {
  brpc::ClosureGuard guard(done);
  try { operation(); }
  catch (const detail::Failure& e) { SetError(response->mutable_status(), e.error()); }
  catch (const std::exception& e) {
    SetError(response->mutable_status(), {Error::Code::run_failed, e.what()});
  }
}
template<class T, class Rsp>
void ErrorOnly(const Result<T>& result, Rsp* response) {
  if (const auto* error = std::get_if<Error>(&result)) SetError(response->mutable_status(), *error);
}
}  // namespace

struct BrpcTrainingServer::Impl {
  struct Pending {
    std::shared_future<Result<TrainingPermit>> future;
    brpc::Controller* controller;
    v1::TrainingPermitReply* response;
    google::protobuf::Closure* done;
  };
  TrainingProgressOptions options;
  TrainingProgressController progress;
  const std::size_t capacity;
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<Pending> pending;
  bool stopping = false;
  std::thread completions;

  Impl(TrainingProgressOptions o, std::size_t limit)
      : options(std::move(o)), progress(options), capacity(limit) {
    detail::Require(options.run.size() <= 256 && limit > 0 && limit <= 65536,
                    "invalid training RPC capacity or run ID");
    completions = std::thread([this] { Poll(); });
  }
  void Validate(google::protobuf::RpcController* c, const google::protobuf::Message& q) const {
    detail::Require(q.ByteSizeLong() <= 4096 &&
                    static_cast<brpc::Controller*>(c)->request_attachment().empty(),
                    "training control payload exceeds limit");
  }
  void Validate(google::protobuf::RpcController* c, const v1::TrainingControl& q) const {
    Validate(c, static_cast<const google::protobuf::Message&>(q));
    if (q.run() != options.run || q.epoch() != options.epoch)
      throw detail::Failure({Error::Code::stale_generation, "training control run/epoch mismatch"});
  }
  static void Finish(Pending& p) {
    Reply(p.response, p.done, [&] {
      if (p.controller->IsCanceled()) {
        SetError(p.response->mutable_status(), {Error::Code::unavailable, "training RPC canceled"});
      } else {
        const auto& r = p.future.get();
        if (const auto* error = std::get_if<Error>(&r)) SetError(p.response->mutable_status(), *error);
        else Encode(std::get<TrainingPermit>(r), p.response);
      }
    });
  }
  void Poll() {
    std::unique_lock lock(mutex);
    for (;;) {
      std::vector<Pending> ready;
      for (std::size_t i = 0; i < pending.size();) {
        if (pending[i].future.wait_for(std::chrono::seconds(0)) == std::future_status::ready ||
            pending[i].controller->IsCanceled()) {
          ready.push_back(std::move(pending[i]));
          if (i + 1 != pending.size()) pending[i] = std::move(pending.back());
          pending.pop_back();
        } else ++i;
      }
      const bool finished = stopping && pending.empty();
      lock.unlock();
      // BRPC may release request/controller/response inside done->Run(). Never
      // inspect a Pending's borrowed pointers after its callback has run.
      for (auto& p : ready) Finish(p);
      lock.lock();
      if (finished) return;
      if (pending.empty()) changed.wait(lock, [&] { return stopping || !pending.empty(); });
      else changed.wait_for(lock, std::chrono::milliseconds(10));
    }
  }
  void Shutdown() {
    {
      std::lock_guard lock(mutex);
      if (stopping) return;
      stopping = true;
      progress.Abort("training coordinator shutting down");
    }
    changed.notify_all();
    if (completions.joinable()) completions.join();
  }
};

BrpcTrainingServer::BrpcTrainingServer(TrainingProgressOptions o, std::size_t limit)
    : impl_(std::make_unique<Impl>(std::move(o), limit)) {}
BrpcTrainingServer::~BrpcTrainingServer() { Shutdown(); }
void BrpcTrainingServer::Shutdown() { impl_->Shutdown(); }
Result<Ack> BrpcTrainingServer::Restore(TrainingProgressCheckpoint checkpoint) {
  return impl_->progress.Restore(std::move(checkpoint));
}

void BrpcTrainingServer::Begin(google::protobuf::RpcController* c, const v1::TrainingTask* q,
                               v1::TrainingPermitReply* r, google::protobuf::Closure* done) {
  brpc::ClosureGuard guard(done);
  try {
    impl_->Validate(c, *q);
    std::lock_guard lock(impl_->mutex);
    if (impl_->stopping) {
      SetError(r->mutable_status(), {Error::Code::unavailable, "training coordinator shutting down"});
      return;
    }
    // A slow worker must still be able to obtain an immediately ready permit
    // when fast workers fill the waiting-RPC capacity. Reject only work that
    // actually needs another pending slot, without mutating its task state.
    auto future = impl_->progress.Begin(Task(*q), impl_->pending.size() < impl_->capacity);
    if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      const auto& result = future.get();
      if (const auto* error = std::get_if<Error>(&result)) SetError(r->mutable_status(), *error);
      else Encode(std::get<TrainingPermit>(result), r);
    } else {
      impl_->pending.push_back({std::move(future), static_cast<brpc::Controller*>(c), r, done});
      guard.release();
      impl_->changed.notify_one();
    }
  } catch (const detail::Failure& e) { SetError(r->mutable_status(), e.error()); }
    catch (const std::exception& e) { SetError(r->mutable_status(), {Error::Code::run_failed, e.what()}); }
}
void BrpcTrainingServer::Complete(google::protobuf::RpcController* c, const v1::TrainingTask* q,
                                  v1::Ack* r, google::protobuf::Closure* done) {
  Reply(r, done, [&] {
    impl_->Validate(c, *q);
    ErrorOnly(impl_->progress.Complete(Task(*q)), r);
    impl_->changed.notify_one();
  });
}
void BrpcTrainingServer::Stats(google::protobuf::RpcController* c, const v1::TrainingControl* q,
                               v1::TrainingStatsReply* r, google::protobuf::Closure* done) {
  Reply(r, done, [&] {
    impl_->Validate(c, *q);
    auto s = impl_->progress.Stats();
    for (auto count : s.completed) r->add_completed(count);
    r->set_completion_floor(s.completion_floor); r->set_granted_tasks(s.granted_tasks);
    r->set_completed_tasks(s.completed_tasks); r->set_duplicate_completions(s.duplicate_completions);
    r->set_active(s.active); r->set_waiting(s.waiting);
    r->set_paused(s.paused); r->set_failed(s.failed);
  });
}
#define EPS_TRAINING_CONTROL(Name) \
void BrpcTrainingServer::Name(google::protobuf::RpcController* c, const v1::TrainingControl* q, \
                               v1::Ack* r, google::protobuf::Closure* done) { \
  Reply(r, done, [&] { \
    impl_->Validate(c, *q); ErrorOnly(impl_->progress.Name(), r); impl_->changed.notify_one(); \
  }); \
}
EPS_TRAINING_CONTROL(Pause)
EPS_TRAINING_CONTROL(Resume)
#undef EPS_TRAINING_CONTROL
void BrpcTrainingServer::Abort(google::protobuf::RpcController* c, const v1::TrainingControl* q,
                               v1::Ack* r, google::protobuf::Closure* done) {
  Reply(r, done, [&] {
    impl_->Validate(c, *q); impl_->progress.Abort(q->reason()); impl_->changed.notify_one();
  });
}
void BrpcTrainingServer::Checkpoint(google::protobuf::RpcController* c, const v1::TrainingControl* q,
                                    v1::TrainingCheckpointReply* r, google::protobuf::Closure* done) {
  Reply(r, done, [&] {
    impl_->Validate(c, *q);
    const auto result = impl_->progress.Checkpoint();
    if (const auto* error = std::get_if<Error>(&result)) { SetError(r->mutable_status(), *error); return; }
    const auto& s = std::get<TrainingProgressCheckpoint>(result);
    r->set_source_run(s.source_run); r->set_epoch(s.epoch);
    r->set_consistency(static_cast<unsigned>(s.consistency)); r->set_max_delay(s.max_delay);
    for (auto count : s.completed) r->add_completed(count);
  });
}
}  // namespace eps::transport
