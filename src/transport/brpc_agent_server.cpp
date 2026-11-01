#include "transport/brpc_agent_server.hpp"
#include "transport/agent_dispatch.hpp"
namespace eps::transport {
void BrpcAgentServer::AdmissionStats(google::protobuf::RpcController* raw,const wire::StatsRequest* q,
    wire::AgentAdmissionStatsReply* r,google::protobuf::Closure* done){
  brpc::ClosureGuard complete(done);
  try{
    auto* c=static_cast<brpc::Controller*>(raw);
    detail::Require(c->request_attachment().empty() && q->ByteSizeLong()<=1024,"unexpected Agent statistics payload");
    auto s=admission_.Stats();
#define EPS_STAT(field) r->set_##field(s.field);
    EPS_STAT(active_operations) EPS_STAT(queued_operations) EPS_STAT(pending_request_bytes)
    EPS_STAT(peak_active_operations) EPS_STAT(peak_pending_operations) EPS_STAT(peak_request_bytes)
    EPS_STAT(accepted_requests) EPS_STAT(completed_requests) EPS_STAT(rejected_requests)
    EPS_STAT(queue_wait_nanoseconds) EPS_STAT(dispatch_failures)
    EPS_STAT(active_limit) EPS_STAT(pending_limit) EPS_STAT(request_bytes_limit)
#undef EPS_STAT
  }catch(const detail::Failure& e){SetError(r->mutable_status(),e.error());}
}
void BrpcAgentServer::Define(google::protobuf::RpcController* c,const wire::AgentDefineRequest* q,wire::Ack* r,google::protobuf::Closure* done){
  AdmittedDispatch<Ack>(admission_,c,q,r,done,[this,q](const auto&){return runtime_.Define(Decode(q->identity()),Decode(q->table()));},AckEncode);
}
void BrpcAgentServer::FreeSlots(google::protobuf::RpcController* c,const wire::AgentCapacityRequest* q,wire::AgentCapacityReply* r,google::protobuf::Closure* done){
  AdmittedDispatch<std::uint64_t>(admission_,c,q,r,done,[this,q](const auto&){return runtime_.FreeSlots(q->run(),q->table_id());},
    [](auto slots,auto* out,auto&){out->set_free_slots(slots);});
}
void BrpcAgentServer::Load(google::protobuf::RpcController* c,const wire::AgentLoadRequest* q,wire::LocationsReply* r,google::protobuf::Closure* done){
  AdmittedDispatch<std::vector<LocatedRow>>(admission_,c,q,r,done,[this,q](const auto& a){
    std::vector<RowState> states;for(const auto& row:q->states())states.push_back(Decode(row,a));
    return runtime_.Load(Decode(q->identity()),Decode(q->plan()),std::move(states));
  },[](auto rows,auto* out,auto&){for(const auto& row:rows)Encode(row,out->add_rows());});
}
void BrpcAgentServer::Gather(google::protobuf::RpcController* c,const wire::AgentGatherRequest* q,wire::AgentGatherReply* r,google::protobuf::Closure* done){
  AdmittedDispatch<std::vector<TensorBlock>>(admission_,c,q,r,done,[this,q](const auto& a){
    std::vector<AgentWork> work;for(const auto& w:q->work())work.push_back(Decode(w,a));
    std::vector<FeatureInput> layouts;for(const auto& f:q->layouts())layouts.push_back(Decode(f,a));
    return runtime_.Gather(q->run(),std::move(work),std::move(layouts));
  },[](auto features,auto* out,auto& a){for(const auto& f:features)Encode(f,out->add_contributions(),a);});
}
void BrpcAgentServer::Apply(google::protobuf::RpcController* c,const wire::AgentApplyRequest* q,wire::Ack* r,google::protobuf::Closure* done){
  AdmittedDispatch<Ack>(admission_,c,q,r,done,[this,q](const auto& a){
    RowGradients gradients;for(const auto& row:q->rows())gradients.rows.push_back({row.table_id(),row.key()});
    gradients.values=Unpack<float>(q->gradients(),a);
    return runtime_.Apply(Decode(q->identity()),Decode(q->context()),std::move(gradients));
  },AckEncode);
}
void BrpcAgentServer::ReadBack(google::protobuf::RpcController* c,const wire::AgentPlanRequest* q,wire::ReadStateReply* r,google::protobuf::Closure* done){
  AdmittedDispatch<std::vector<RowState>>(admission_,c,q,r,done,[this,q](const auto&){return runtime_.ReadBack(Decode(q->identity()),Decode(q->plan()));},
    [home=q->identity().home_shard()](auto rows,auto* out,auto& a){for(const auto& row:rows)Encode(row,out->add_states(),a,home);});
}
void BrpcAgentServer::ApplyCompact(google::protobuf::RpcController* c,const wire::AgentCompactApplyRequest* q,wire::Ack* r,google::protobuf::Closure* done){
  AdmittedDispatch<Ack>(admission_,c,q,r,done,[this,q](const auto& a){
    CompactGradients p{{},q->dimension(),Unpack<float>(q->source_values(),a),
      Unpack<std::uint64_t>(q->row_offsets(),a),Unpack<std::uint64_t>(q->source_rows(),a),Unpack<float>(q->scales(),a)};
    for(const auto& row:q->rows())p.rows.push_back({row.table_id(),row.key()});
    return runtime_.ApplyCompact(Decode(q->identity()),Decode(q->context()),std::move(p));
  },AckEncode);
}
void BrpcAgentServer::CommitEviction(google::protobuf::RpcController* c,const wire::AgentPlanRequest* q,wire::Ack* r,google::protobuf::Closure* done){
  AdmittedDispatch<Ack>(admission_,c,q,r,done,[this,q](const auto&){return runtime_.CommitEviction(Decode(q->identity()),Decode(q->plan()));},AckEncode);
}
}  // namespace eps::transport
