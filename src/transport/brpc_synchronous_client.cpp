#include "eps/brpc_synchronous_gradients.hpp"
#include "transport/rpc_call.hpp"
#include <thread>
#include <atomic>

namespace eps {
namespace t = transport;
struct BrpcSynchronousGradients::Impl {
  brpc::Channel channel;
  int timeout_ms;
  std::string run;
  std::uint64_t epoch;
  GradientLayoutCache layouts;
  std::atomic<bool> cache_enabled{false};
  std::atomic<std::uint64_t> full_requests{0},token_requests{0},fallback_requests{0},
    request_bytes{0},avoided_request_bytes{0};
  struct SubmitCall;
  Impl(const std::string& endpoint, std::string r, std::uint64_t e, int timeout,
       GradientLayoutCacheOptions cache)
      : timeout_ms(timeout), run(std::move(r)), epoch(e), layouts(cache) {
    detail::Require(!run.empty() && run.size() <= 256 && epoch && timeout > 0,
                    "invalid synchronous gradient RPC options");
    brpc::ChannelOptions options; options.protocol = "baidu_std";
    options.timeout_ms = timeout; options.max_retry = 2;
    detail::Require(channel.Init(endpoint.c_str(), &options) == 0, "cannot open gradient RPC channel");
  }
  void Control(v1::TrainingControl& q) const { q.set_run(run); q.set_epoch(epoch); }
};
BrpcSynchronousGradients::BrpcSynchronousGradients(std::string endpoint, std::string run,
    std::uint64_t epoch, int timeout,GradientLayoutCacheOptions cache)
    : impl_(std::make_shared<Impl>(endpoint, std::move(run), epoch, timeout,cache)) {}
BrpcSynchronousGradients::~BrpcSynchronousGradients() = default;
void BrpcSynchronousGradients::SetGradientLayoutCache(bool enabled) {
  impl_->cache_enabled=enabled;if(!enabled)impl_->layouts.Clear();
}
GradientTransportStats BrpcSynchronousGradients::TransportStats() const {
  return {impl_->full_requests.load(),impl_->token_requests.load(),impl_->fallback_requests.load(),
    impl_->request_bytes.load(),impl_->avoided_request_bytes.load(),impl_->layouts.Stats()};
}
std::future<Result<GradientLayoutCacheStats>> BrpcSynchronousGradients::LayoutStats() {
  return t::Call<v1::TrainingControl,v1::LookupLayoutStatsReply,GradientLayoutCacheStats>(impl_,
    &v1::SynchronousGradientService_Stub::LayoutStats,
    [p=impl_](auto& q,auto&){p->Control(q);},[](const auto& r,const auto&){
      return GradientLayoutCacheStats{r.entries(),r.bytes(),r.hits(),r.misses(),r.registrations(),
        r.evictions(),r.expirations(),r.rejected()};
    });
}
struct BrpcSynchronousGradients::Impl::SubmitCall final : google::protobuf::Closure {
  std::shared_ptr<Impl> owner;
  GradientLayoutScope scope;
  std::shared_ptr<const GradientLayoutCache::Layout> layout;
  v1::GradientContribution full,request;
  v1::GradientReply response;
  butil::IOBuf full_attachment;
  brpc::Controller controller;
  std::promise<Result<Ack>> promise;
  std::string token;
  bool use_cache=false,fallback=false;
  std::uint64_t saved_bytes=0;
  std::chrono::steady_clock::time_point deadline;
  void Send() {
    try {
      controller.Reset();request.Clear();response.Clear();saved_bytes=0;
      const auto remaining=std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline-std::chrono::steady_clock::now()).count();
      detail::Require(remaining>0,"gradient completion deadline exhausted",Error::Code::unavailable);
      controller.set_timeout_ms(static_cast<int>(remaining));
      if(token.empty()) {
        request=full;controller.request_attachment().append(full_attachment);++owner->full_requests;
      } else {
        *request.mutable_task()=full.task();request.set_sample_count(full.sample_count());
        request.set_completed_reads(full.completed_reads());request.set_layout_token(token);
        for(const auto& table:full.tables()) {
          auto* out=request.add_tables();out->set_table_id(table.table_id());out->set_dimension(table.dimension());
          auto* values=out->mutable_values();*values=table.values();
          values->set_byte_offset(controller.request_attachment().size());
          detail::Require(full_attachment.append_to(&controller.request_attachment(),
            table.values().byte_length(),table.values().byte_offset())==table.values().byte_length(),
            "cannot retain original gradient values");
        }
        ++owner->token_requests;
        const auto original=full.ByteSizeLong()+full_attachment.size();
        const auto compact=request.ByteSizeLong()+controller.request_attachment().size();
        if(original>compact)saved_bytes=original-compact;
      }
      owner->request_bytes+=request.ByteSizeLong()+controller.request_attachment().size();
      v1::SynchronousGradientService_Stub stub(&owner->channel);
      stub.Submit(&controller,&request,&response,this);
    } catch(const detail::Failure& e) {std::unique_ptr<SubmitCall> self(this);promise.set_value(e.error());}
      catch(const std::exception& e) {std::unique_ptr<SubmitCall> self(this);promise.set_value(Error{Error::Code::run_failed,e.what()});}
  }
  void Run() override {
    if(!controller.Failed() && !t::GetError(response.status()) && response.layout_miss() &&
        !token.empty() && !fallback) {
      if(!response.layout_token().empty()) {
        std::unique_ptr<SubmitCall> self(this);
        promise.set_value(Error{Error::Code::run_failed,"gradient layout miss returned a registered token"});return;
      }
      owner->layouts.Erase(token);token.clear();fallback=true;++owner->fallback_requests;Send();return;
    }
    std::unique_ptr<SubmitCall> self(this);
    try {
      if(controller.Failed()) {promise.set_value(Error{Error::Code::unavailable,controller.ErrorText()});return;}
      if(auto e=t::GetError(response.status())) {promise.set_value(*e);return;}
      detail::Require(!response.layout_miss(),"unresolved gradient layout miss");
      owner->avoided_request_bytes+=saved_bytes;
      if(use_cache && owner->cache_enabled && !response.layout_token().empty()) {
        try {owner->layouts.Put(scope,layout,response.layout_token());}catch(const std::exception&) {}
      }
      promise.set_value(Ack{});
    } catch(const detail::Failure& e) {promise.set_value(e.error());}
      catch(const std::exception& e) {promise.set_value(Error{Error::Code::run_failed,e.what()});}
  }
};
Result<GradientRetryStats> SubmitGradientGroup(const std::vector<BrpcSynchronousGradients*>& shards,
    const std::vector<GradientContribution>& packets,std::chrono::milliseconds window) {
  if(shards.empty() || shards.size()!=packets.size() || window.count()<0 || window>std::chrono::seconds(20))
    return Error{Error::Code::invalid_argument,"invalid synchronous retry group/window"};
  const auto deadline=std::chrono::steady_clock::now()+window;
  std::vector<bool> complete(shards.size(),false);GradientRetryStats stats;
  for(;;) {
    std::vector<std::pair<std::size_t,std::future<Result<Ack>>>> pending;
    for(std::size_t home=0;home<shards.size();++home)if(!complete[home]) {
      if(stats.attempts>=shards.size())++stats.retries;
      ++stats.attempts;pending.emplace_back(home,shards[home]->Submit(packets[home]));
    }
    std::optional<Error> error,fatal;
    for(auto& [home,future]:pending) {
      auto result=future.get();
      if(auto* e=std::get_if<Error>(&result)) {
        error=*e;
        if(e->code!=Error::Code::unavailable && !(e->code==Error::Code::capacity_exceeded &&
            e->message.find("pending gradient response")!=std::string::npos))fatal=*e;
      }else complete[home]=true;
    }
    if(fatal)return *fatal;
    if(!error)return stats;
    if(!window.count() || std::chrono::steady_clock::now()>=deadline)return *error;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}
std::future<Result<Ack>> BrpcSynchronousGradients::Submit(GradientContribution c) {
  auto state=std::make_unique<Impl::SubmitCall>();auto future=state->promise.get_future();
  state->owner=impl_;state->use_cache=impl_->cache_enabled;
  state->deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(impl_->timeout_ms);
  try {
    detail::Require(c.task.run==impl_->run && c.task.epoch==impl_->epoch && c.tables.size()<=256,
                    "invalid gradient RPC identity/table count");
    state->scope={c.task.run,c.task.epoch,c.task.worker,0};
    auto* task=state->full.mutable_task();task->set_run(c.task.run);task->set_epoch(c.task.epoch);
    task->set_worker(c.task.worker);task->set_iteration(c.task.iteration);
    state->full.set_sample_count(c.sample_count);state->full.set_completed_reads(c.completed_reads);
    state->full.set_register_layout(state->use_cache);
    auto layout=std::make_shared<GradientLayoutCache::Layout>();
    for(auto& table:c.tables) {
      auto* out=state->full.add_tables();out->set_table_id(table.table);out->set_dimension(table.dimension);
      t::Pack(table.keys,out->mutable_keys(),state->full_attachment);
      t::Pack(table.values,out->mutable_values(),state->full_attachment);
      if(state->use_cache)layout->push_back({table.table,table.dimension,std::move(table.keys)});
    }
    if(state->use_cache) {state->token=impl_->layouts.Find(state->scope,*layout);state->layout=std::move(layout);}
  } catch(const detail::Failure& e) {state->promise.set_value(e.error());return future;}
    catch(const std::exception& e) {state->promise.set_value(Error{Error::Code::run_failed,e.what()});return future;}
  state.release()->Send();return future;
}
std::future<Result<SynchronousGradientStats>> BrpcSynchronousGradients::Stats() {
  return t::Call<v1::TrainingControl, v1::SynchronousGradientStatsReply, SynchronousGradientStats>(impl_,
    &v1::SynchronousGradientService_Stub::Stats,
    [p=impl_](auto& q, auto&) { p->Control(q); }, [](const auto& r, const auto&) {
      return SynchronousGradientStats{r.next_round(), r.received_workers(), r.retained_bytes(),
        r.duplicate_submissions(), r.applying(), r.failed(), r.paused()};
    });
}
std::future<Result<Ack>> BrpcSynchronousGradients::Abort(std::string reason) {
  return t::Call<v1::TrainingControl, v1::Ack, Ack>(impl_, &v1::SynchronousGradientService_Stub::Abort,
    [p=impl_, reason=std::move(reason)](auto& q, auto&) { p->Control(q); q.set_reason(reason); }, t::AckDecoder);
}
#define EPS_SYNC_CONTROL(Name) \
std::future<Result<Ack>> BrpcSynchronousGradients::Name() { \
  return t::Call<v1::TrainingControl,v1::Ack,Ack>(impl_, &v1::SynchronousGradientService_Stub::Name, \
    [p=impl_](auto& q,auto&) { p->Control(q); }, t::AckDecoder); \
}
EPS_SYNC_CONTROL(Pause)
EPS_SYNC_CONTROL(Resume)
#undef EPS_SYNC_CONTROL
std::future<Result<SynchronousGradientCheckpoint>> BrpcSynchronousGradients::Checkpoint() {
  return t::Call<v1::TrainingControl,v1::SynchronousGradientCheckpointReply,SynchronousGradientCheckpoint>(impl_,
    &v1::SynchronousGradientService_Stub::Checkpoint,
    [p=impl_](auto& q,auto&) { p->Control(q); }, [](const auto& r,const auto&) {
      detail::Require(r.reduction() <= 2 && r.tables_size() <= 256, "invalid synchronous checkpoint reply");
      SynchronousGradientCheckpoint c{r.source_run(),r.epoch(),r.workers(),r.shard(),r.shards(),{},
        static_cast<GradientReduction>(r.reduction()),r.next_round()};
      for (const auto& t:r.tables()) c.tables.push_back({t.table_id(),t.dimension(),{}, {}});
      return c;
    });
}
}  // namespace eps
