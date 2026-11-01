#include "eps/brpc_cache_backend.hpp"
#include <atomic>
#include "transport/rpc_call.hpp"

namespace eps {
namespace t=transport;namespace wire=::eps::v1;
struct BrpcCacheBackend::Impl {
  brpc::Channel channel;int timeout_ms;std::string run;ShardId home;AgentId agent;
  std::atomic<std::uint64_t> sequence{0};
  Impl(std::string endpoint,std::string r,ShardId h,AgentId a,int timeout)
    :timeout_ms(timeout),run(std::move(r)),home(h),agent(a){
    brpc::ChannelOptions options;options.protocol="baidu_std";options.timeout_ms=timeout;options.max_retry=2;
    detail::Require(channel.Init(endpoint.c_str(),&options)==0,"cannot initialize cache agent channel");
  }
  AgentCommand Next(){return {run,home,sequence.fetch_add(1)};}
};
BrpcCacheBackend::BrpcCacheBackend(std::string endpoint,std::string run,ShardId home,AgentId agent,int timeout)
  :impl_(std::make_shared<Impl>(std::move(endpoint),std::move(run),home,agent,timeout)){}
BrpcCacheBackend::~BrpcCacheBackend()=default;
std::future<Result<AgentAdmissionStats>> BrpcCacheBackend::AdmissionStats(){
  return t::Call<wire::StatsRequest,wire::AgentAdmissionStatsReply,AgentAdmissionStats>(
    impl_,&wire::CacheAgentService_Stub::AdmissionStats,[](auto&,auto&){},
    [](const auto& q,const auto&){AgentAdmissionStats out;
#define EPS_STAT(field) out.field=q.field();
      EPS_STAT(active_operations) EPS_STAT(queued_operations) EPS_STAT(pending_request_bytes)
      EPS_STAT(peak_active_operations) EPS_STAT(peak_pending_operations) EPS_STAT(peak_request_bytes)
      EPS_STAT(accepted_requests) EPS_STAT(completed_requests) EPS_STAT(rejected_requests)
      EPS_STAT(queue_wait_nanoseconds) EPS_STAT(dispatch_failures)
      EPS_STAT(active_limit) EPS_STAT(pending_limit) EPS_STAT(request_bytes_limit)
#undef EPS_STAT
      return out;});
}
std::future<Result<Ack>> BrpcCacheBackend::Define(TableDefinition table){
  return t::Call<wire::AgentDefineRequest,wire::Ack,Ack>(impl_,&wire::CacheAgentService_Stub::Define,
    [id=impl_->Next(),table](auto& q,auto&){t::Encode(id,q.mutable_identity());t::Encode(table,q.mutable_table());},t::AckDecoder);
}
std::future<Result<std::uint64_t>> BrpcCacheBackend::FreeSlots(TableId table){
  return t::Call<wire::AgentCapacityRequest,wire::AgentCapacityReply,std::uint64_t>(impl_,&wire::CacheAgentService_Stub::FreeSlots,
    [run=impl_->run,table](auto& q,auto&){q.set_run(run);q.set_table_id(table);},
    [](const auto& q,const auto&){return q.free_slots();});
}
std::future<Result<std::vector<LocatedRow>>> BrpcCacheBackend::Load(MigrationPlan plan,std::vector<RowState> states){
  return t::Call<wire::AgentLoadRequest,wire::LocationsReply,std::vector<LocatedRow>>(impl_,&wire::CacheAgentService_Stub::Load,
    [id=impl_->Next(),plan=std::move(plan),states=std::move(states)](auto& q,auto& a){
      t::Encode(id,q.mutable_identity());t::Encode(plan,q.mutable_plan());
      for(const auto& row:states)t::Encode(row,q.add_states(),a,id.shard);
    },[](const auto& q,const auto&){std::vector<LocatedRow> out;for(const auto& row:q.rows())out.push_back(t::Decode(row));return out;});
}
std::future<Result<std::vector<TensorBlock>>> BrpcCacheBackend::GatherHost(std::vector<AgentWork> work,std::vector<FeatureInput> layouts){
  return t::Call<wire::AgentGatherRequest,wire::AgentGatherReply,std::vector<TensorBlock>>(impl_,&wire::CacheAgentService_Stub::Gather,
    [run=impl_->run,work=std::move(work),layouts=std::move(layouts)](auto& q,auto& a){
      q.set_run(run);for(const auto& w:work)t::Encode(w,q.add_work(),a);
      for(const auto& f:layouts)t::Encode(f,q.add_layouts(),a);
    },[](const auto& q,const auto& a){std::vector<TensorBlock> out;for(const auto& f:q.contributions())out.push_back(t::Decode(f,a));return out;});
}
std::future<Result<Ack>> BrpcCacheBackend::ApplyHost(UpdateContext context,RowGradients gradients){
  return t::Call<wire::AgentApplyRequest,wire::Ack,Ack>(impl_,&wire::CacheAgentService_Stub::Apply,
    [id=impl_->Next(),context,g=std::move(gradients)](auto& q,auto& a){
      t::Encode(id,q.mutable_identity());t::Encode(context,q.mutable_context());
      for(auto row:g.rows){auto* x=q.add_rows();x->set_table_id(row.table);x->set_key(row.key);}
      t::Pack(g.values,q.mutable_gradients(),a);
    },t::AckDecoder);
}
std::future<Result<std::vector<RowState>>> BrpcCacheBackend::ReadBack(MigrationPlan plan){
  return t::Call<wire::AgentPlanRequest,wire::ReadStateReply,std::vector<RowState>>(impl_,&wire::CacheAgentService_Stub::ReadBack,
    [id=impl_->Next(),plan=std::move(plan)](auto& q,auto&){t::Encode(id,q.mutable_identity());t::Encode(plan,q.mutable_plan());},
    [](const auto& q,const auto& a){std::vector<RowState> out;for(const auto& row:q.states())out.push_back(t::Decode(row,a));return out;});
}
std::future<Result<Ack>> BrpcCacheBackend::ApplyCompact(UpdateContext context,CompactGradients p){
  return t::Call<wire::AgentCompactApplyRequest,wire::Ack,Ack>(impl_,&wire::CacheAgentService_Stub::ApplyCompact,
    [id=impl_->Next(),context,p=std::move(p)](auto& q,auto& a){
      t::Encode(id,q.mutable_identity());t::Encode(context,q.mutable_context());q.set_dimension(p.dimension);
      for(auto row:p.rows){auto* x=q.add_rows();x->set_table_id(row.table);x->set_key(row.key);}
      t::Pack(p.source_values,q.mutable_source_values(),a);t::Pack(p.row_offsets,q.mutable_row_offsets(),a);
      t::Pack(p.source_rows,q.mutable_source_rows(),a);t::Pack(p.scales,q.mutable_scales(),a);
    },t::AckDecoder);
}
std::future<Result<Ack>> BrpcCacheBackend::CommitEviction(MigrationPlan plan){
  return t::Call<wire::AgentPlanRequest,wire::Ack,Ack>(impl_,&wire::CacheAgentService_Stub::CommitEviction,
    [id=impl_->Next(),plan=std::move(plan)](auto& q,auto&){t::Encode(id,q.mutable_identity());t::Encode(plan,q.mutable_plan());},t::AckDecoder);
}
}  // namespace eps
