#include "transport/brpc_synchronous_server.hpp"

#include <condition_variable>
#include <algorithm>
#include <mutex>
#include <optional>
#include <thread>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include "transport/codec.hpp"
#include "transport/rpc_dispatch.hpp"
#include "transport/replicated_rounds.hpp"

namespace eps::transport {
namespace {
template<class T> T Take(std::future<Result<T>> f) {
  auto r = f.get(); if (auto* e = std::get_if<Error>(&r)) throw detail::Failure(*e);
  return std::get<T>(std::move(r));
}
void ValidateParameters(const ParameterTrainingMetadata& m, const SynchronousGradientCheckpoint& c) {
  detail::Require(m.tables.size() == c.tables.size() && m.optimizer_contexts.size() == c.tables.size(),
    "parameter table set differs from gradient checkpoint");
  for (const auto& schema:c.tables) {
    auto table=std::find_if(m.tables.begin(),m.tables.end(),[&](const auto& t){return t.id==schema.table;});
    auto context=std::find_if(m.optimizer_contexts.begin(),m.optimizer_contexts.end(),[&](const auto& t){return t.table==schema.table;});
    detail::Require(table!=m.tables.end() && table->dimension==schema.dimension && table->shard_count==c.shards &&
      table->partition_epoch==c.epoch && context!=m.optimizer_contexts.end() && context->optimizer_sequence==c.next_round,
      "parameter optimizer clock/schema differs from gradient checkpoint");
  }
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
    v1::GradientReply* response;
    google::protobuf::Closure* done;
  };
  CpuParameterService& parameters;
  SynchronousGradientOptions options;
  std::shared_ptr<ReplicatedRoundWriter> replica;
  SynchronousGradientCollector collector;
  GradientLayoutCache layouts;
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<std::optional<Pending>> pending;
  bool stopping = false;
  std::thread pump;

  Impl(CpuParameterService& ps, SynchronousGradientOptions o,std::shared_ptr<ReplicatedRoundWriter> r,
       GradientLayoutCacheOptions cache)
      : parameters(ps), options(std::move(o)),replica(std::move(r)), collector(options,
        [this](auto round, auto tables) { return Apply(round, std::move(tables)); },
        [this](auto progress)->Result<Ack>{if(replica)return parameters.CertifyReadFrontiers(std::move(progress)).get();return Ack{};}),
        layouts(cache), pending(options.workers), pump([this] { Poll(); }) {}
  Result<Ack> Apply(std::uint64_t round, std::vector<GradientTable> tables) {
    if(replica)replica->BeforeRound(round);
    std::vector<RowKey> touched;
    LookupRequest request{{options.run, 1023, round, Phase::lookup}, options.epoch, false, {}};
    std::vector<TensorBlock> gradients;
    for (std::size_t i = 0; i < tables.size(); ++i) {
      auto& table = tables[i];
      if(replica)for(auto key:table.keys)touched.push_back({table.table,key});
      const auto count = table.keys.size();
      request.features.push_back({table.table, static_cast<std::uint32_t>(i), Pooling::sequence,
        std::move(table.keys), {0, count}, {count}});
      gradients.push_back({static_cast<std::uint32_t>(i), {count, table.dimension}, std::move(table.values)});
    }
    auto lookup = Take(parameters.Lookup(std::move(request)));
    Take(parameters.Push({{options.run, 1023, round, Phase::push},std::move(lookup.ticket),std::move(gradients)}));
    if(replica)replica->Commit(parameters,round,touched);
    return Ack{};
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
BrpcSynchronousServer::BrpcSynchronousServer(CpuParameterService& service, SynchronousGradientOptions options,
    std::shared_ptr<ReplicatedRoundWriter> replica,GradientLayoutCacheOptions cache)
    : impl_(std::make_unique<Impl>(service, std::move(options),std::move(replica),cache)) {}
BrpcSynchronousServer::~BrpcSynchronousServer() { Shutdown(); }
void BrpcSynchronousServer::Shutdown() { impl_->Shutdown(); }
Result<Ack> BrpcSynchronousServer::RecoverReplica(std::uint64_t baseline,std::uint64_t next,bool activate) {
  try {
    detail::Require(bool(impl_->replica),"replica recovery requires a configured writer");
    auto metadata=Take(impl_->parameters.TrainingMetadata());
    for(const auto& c:metadata.optimizer_contexts)detail::Require(c.optimizer_sequence==next,"recovered optimizer/group clock differs");
    auto r=impl_->collector.RecoverReplica(baseline,next);if(auto* e=std::get_if<Error>(&r))return *e;
    if(activate){impl_->replica->BeforeRound(next);Take(impl_->parameters.Resume());return impl_->collector.Resume();}
    return Ack{};
  }catch(const detail::Failure& e){return e.error();}
}
Result<Ack> BrpcSynchronousServer::Restore(SynchronousGradientCheckpoint c, RestoredSnapshot expected) {
  try {
    auto m=Take(impl_->parameters.TrainingMetadata());
    const auto& p=m.provenance;
    detail::Require(p.loaded && p.pristine && p.source_run==c.source_run &&
      p.source_run==expected.source_run && p.bytes==expected.bytes && p.checksum==expected.checksum &&
      p.shard==c.shard && p.shard==expected.shard && p.shard_count==c.shards &&
      p.shard_count==expected.shard_count && p.partition_epoch==c.epoch && p.partition_epoch==expected.partition_epoch,
      "loaded parameter snapshot identity differs from group checkpoint");
    ValidateParameters(m,c);
    Take(impl_->parameters.InitializeRequestFrontiers({{1023,c.next_round}}));
    return impl_->collector.Restore(std::move(c));
  } catch(const detail::Failure& e) { return e.error(); }
}
void BrpcSynchronousServer::Submit(google::protobuf::RpcController* c, const v1::GradientContribution* q,
                                  v1::GradientReply* r, google::protobuf::Closure* done) {
  brpc::ClosureGuard guard(done);
  try {
    impl_->Validate(c, *q);
    detail::Require(q->tables_size() <= 256 && q->task().worker() < impl_->options.workers,
      "invalid synchronous worker/table count");
    detail::Require(q->task().run()==impl_->options.run && q->task().epoch()==impl_->options.epoch,
      "gradient run/epoch mismatch",Error::Code::stale_generation);
    detail::Require(q->layout_token().size()<=256 &&
      (q->layout_token().empty() || !q->register_layout()),"invalid gradient layout token request");
    GradientLayoutScope scope{q->task().run(),q->task().epoch(),q->task().worker(),impl_->options.shard};
    std::shared_ptr<const GradientLayoutCache::Layout> cached;
    if(!q->layout_token().empty()) {
      cached=impl_->layouts.Resolve(scope,q->layout_token());
      if(!cached) {r->set_layout_miss(true);return;}
      detail::Require(cached->size()==static_cast<std::size_t>(q->tables_size()),
                      "gradient token table count differs");
    }
    GradientContribution input{{q->task().run(), q->task().epoch(), q->task().worker(), q->task().iteration()},
      q->sample_count(), {}};
    input.completed_reads=q->completed_reads();
    std::uint64_t decoded = 0;
    const auto& attachment = static_cast<brpc::Controller*>(c)->request_attachment();
    std::size_t index=0;
    auto registration=std::make_shared<GradientLayoutCache::Layout>();
    for (const auto& table : q->tables()) {
      // Bound cumulative decoded bytes even when malicious slices overlap.
      std::vector<Key> keys;
      if(cached) {
        const auto& layout=cached->at(index);
        detail::Require(table.table_id()==layout.table && table.dimension()==layout.dimension &&
          !table.has_keys(),"gradient token schema or Key payload differs");
        detail::Require(layout.keys.size()<=kMaxAttachment/sizeof(Key),"cached gradient Keys exceed limit");
        keys=layout.keys;
      } else keys=Unpack<Key>(table.keys(),attachment);
      const auto key_bytes=keys.size()*sizeof(Key);
      detail::Require(key_bytes<=kMaxAttachment-decoded,"gradient decoded Keys exceed limit");
      decoded+=key_bytes;
      detail::Require(table.values().byte_length()<=kMaxAttachment-decoded,
                      "gradient decoded payload exceeds limit");
      decoded+=table.values().byte_length();
      if(q->register_layout())registration->push_back({table.table_id(),table.dimension(),keys});
      input.tables.push_back({table.table_id(),table.dimension(),std::move(keys),
                             Unpack<float>(table.values(),attachment)});
      ++index;
    }
    std::unique_lock lock(impl_->mutex);
    detail::Require(!impl_->stopping, "gradient service shutting down", Error::Code::unavailable);
    auto future = impl_->collector.Submit(std::move(input));
    const bool ready=future.wait_for(std::chrono::seconds(0))==std::future_status::ready;
    if(q->register_layout() && (!ready || std::holds_alternative<Ack>(future.get()))) {
      try {r->set_layout_token(impl_->layouts.Put(scope,std::move(registration)));}
      catch(const std::exception&) {}  // Cache admission cannot undo an accepted update.
    }
    if (ready) {
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
void BrpcSynchronousServer::LayoutStats(google::protobuf::RpcController* c,const v1::TrainingControl* q,
    v1::LookupLayoutStatsReply* r,google::protobuf::Closure* done) {
  Reply(r,done,[&] {
    impl_->Control(c,*q);const auto s=impl_->layouts.Stats();
    r->set_entries(s.entries);r->set_bytes(s.bytes);r->set_hits(s.hits);r->set_misses(s.misses);
    r->set_registrations(s.registrations);r->set_evictions(s.evictions);
    r->set_expirations(s.expirations);r->set_rejected(s.rejected);
  });
}
void BrpcSynchronousServer::Stats(google::protobuf::RpcController* c, const v1::TrainingControl* q,
                                 v1::SynchronousGradientStatsReply* r, google::protobuf::Closure* done) {
  Reply(r, done, [&] {
    impl_->Control(c, *q); auto s = impl_->collector.Stats();
    r->set_next_round(s.next_round); r->set_received_workers(s.received_workers);
    r->set_retained_bytes(s.retained_bytes); r->set_duplicate_submissions(s.duplicate_submissions);
    r->set_applying(s.applying); r->set_failed(s.failed);
    r->set_paused(s.paused);
  });
}
void BrpcSynchronousServer::Abort(google::protobuf::RpcController* c, const v1::TrainingControl* q,
                                 v1::Ack* r, google::protobuf::Closure* done) {
  Reply(r, done, [&] { impl_->Control(c, *q); impl_->collector.Abort(q->reason()); impl_->changed.notify_all(); });
}
void BrpcSynchronousServer::Pause(google::protobuf::RpcController* c,const v1::TrainingControl* q,
                                 v1::Ack* r,google::protobuf::Closure* done) {
  Reply(r,done,[&] {
    impl_->Control(c,*q); auto result=impl_->collector.Pause();
    if (auto* e=std::get_if<Error>(&result)) SetError(r->mutable_status(),*e);
  });
}
void BrpcSynchronousServer::Resume(google::protobuf::RpcController* c,const v1::TrainingControl* q,
                                  v1::Ack* r,google::protobuf::Closure* done) {
  Reply(r,done,[&] {
    impl_->Control(c,*q); std::lock_guard lock(impl_->mutex);
    auto boundary=impl_->collector.Checkpoint();
    if (auto* e=std::get_if<Error>(&boundary)) throw detail::Failure(*e);
    ValidateParameters(Take(impl_->parameters.TrainingMetadata()),std::get<SynchronousGradientCheckpoint>(boundary));
    Take(impl_->parameters.Resume());
    auto result=impl_->collector.Resume();
    if (auto* e=std::get_if<Error>(&result)) SetError(r->mutable_status(),*e);
  });
}
void BrpcSynchronousServer::Checkpoint(google::protobuf::RpcController* c,const v1::TrainingControl* q,
                                      v1::SynchronousGradientCheckpointReply* r,google::protobuf::Closure* done) {
  brpc::ClosureGuard guard(done);
  try {
    impl_->Control(c,*q);
    auto result=impl_->collector.Checkpoint();
    if (auto* e=std::get_if<Error>(&result)) throw detail::Failure(*e);
    const auto checkpoint=std::get<SynchronousGradientCheckpoint>(std::move(result));
    Dispatch<ParameterTrainingMetadata>(c,q,r,guard.release(),
      [&](const auto&){return impl_->parameters.TrainingMetadata();},
      [this,checkpoint](const auto& metadata,auto* out,auto&) {
        auto current=impl_->collector.Checkpoint();
        if (auto* e=std::get_if<Error>(&current)) throw detail::Failure(*e);
        detail::Require(std::get<SynchronousGradientCheckpoint>(current).next_round==checkpoint.next_round,
          "gradient boundary changed during checkpoint inspection");
        ValidateParameters(metadata,checkpoint);
        out->set_source_run(checkpoint.source_run);out->set_epoch(checkpoint.epoch);
        out->set_workers(checkpoint.workers);out->set_shard(checkpoint.shard);out->set_shards(checkpoint.shards);
        out->set_reduction(static_cast<unsigned>(checkpoint.reduction));out->set_next_round(checkpoint.next_round);
        for (const auto& t:checkpoint.tables) {auto* s=out->add_tables();s->set_table_id(t.table);s->set_dimension(t.dimension);}
      },{},CompletionExecutor::Observations());
  } catch(const detail::Failure& e) {SetError(r->mutable_status(),e.error());}
    catch(const std::exception& e) {SetError(r->mutable_status(),{Error::Code::run_failed,e.what()});}
}
}  // namespace eps::transport
