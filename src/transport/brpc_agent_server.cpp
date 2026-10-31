#include "transport/brpc_agent_server.hpp"
#include "transport/rpc_dispatch.hpp"
namespace eps::transport {
void BrpcAgentServer::Define(google::protobuf::RpcController* c,const wire::AgentDefineRequest* q,wire::Ack* r,google::protobuf::Closure* done){
  Dispatch<Ack>(c,q,r,done,[&](const auto&){return runtime_.Define(Decode(q->identity()),Decode(q->table()));},AckEncode);
}
void BrpcAgentServer::FreeSlots(google::protobuf::RpcController* c,const wire::AgentCapacityRequest* q,wire::AgentCapacityReply* r,google::protobuf::Closure* done){
  Dispatch<std::uint64_t>(c,q,r,done,[&](const auto&){return runtime_.FreeSlots(q->run(),q->table_id());},
    [](auto slots,auto* out,auto&){out->set_free_slots(slots);});
}
void BrpcAgentServer::Load(google::protobuf::RpcController* c,const wire::AgentLoadRequest* q,wire::LocationsReply* r,google::protobuf::Closure* done){
  Dispatch<std::vector<LocatedRow>>(c,q,r,done,[&](const auto& a){
    std::vector<RowState> states;for(const auto& row:q->states())states.push_back(Decode(row,a));
    return runtime_.Load(Decode(q->identity()),Decode(q->plan()),std::move(states));
  },[](auto rows,auto* out,auto&){for(const auto& row:rows)Encode(row,out->add_rows());});
}
void BrpcAgentServer::Gather(google::protobuf::RpcController* c,const wire::AgentGatherRequest* q,wire::AgentGatherReply* r,google::protobuf::Closure* done){
  Dispatch<std::vector<TensorBlock>>(c,q,r,done,[&](const auto& a){
    std::vector<AgentWork> work;for(const auto& w:q->work())work.push_back(Decode(w,a));
    std::vector<FeatureInput> layouts;for(const auto& f:q->layouts())layouts.push_back(Decode(f,a));
    return runtime_.Gather(q->run(),std::move(work),std::move(layouts));
  },[](auto features,auto* out,auto& a){for(const auto& f:features)Encode(f,out->add_contributions(),a);});
}
void BrpcAgentServer::Apply(google::protobuf::RpcController* c,const wire::AgentApplyRequest* q,wire::Ack* r,google::protobuf::Closure* done){
  Dispatch<Ack>(c,q,r,done,[&](const auto& a){
    RowGradients gradients;for(const auto& row:q->rows())gradients.rows.push_back({row.table_id(),row.key()});
    gradients.values=Unpack<float>(q->gradients(),a);
    return runtime_.Apply(Decode(q->identity()),Decode(q->context()),std::move(gradients));
  },AckEncode);
}
void BrpcAgentServer::ReadBack(google::protobuf::RpcController* c,const wire::AgentPlanRequest* q,wire::ReadStateReply* r,google::protobuf::Closure* done){
  Dispatch<std::vector<RowState>>(c,q,r,done,[&](const auto&){return runtime_.ReadBack(Decode(q->identity()),Decode(q->plan()));},
    [home=q->identity().home_shard()](auto rows,auto* out,auto& a){for(const auto& row:rows)Encode(row,out->add_states(),a,home);});
}
void BrpcAgentServer::CommitEviction(google::protobuf::RpcController* c,const wire::AgentPlanRequest* q,wire::Ack* r,google::protobuf::Closure* done){
  Dispatch<Ack>(c,q,r,done,[&](const auto&){return runtime_.CommitEviction(Decode(q->identity()),Decode(q->plan()));},AckEncode);
}
}  // namespace eps::transport
