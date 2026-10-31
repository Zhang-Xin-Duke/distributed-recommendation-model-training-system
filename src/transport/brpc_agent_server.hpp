#pragma once
#include "embedding_service.pb.h"
#include "eps/agent_runtime.hpp"
namespace eps::transport {
class BrpcAgentServer final : public v1::CacheAgentService {
 public:
  explicit BrpcAgentServer(AgentRuntime& runtime):runtime_(runtime){}
#define EPS_AGENT_RPC(Name,Request,Response) \
  void Name(google::protobuf::RpcController*,const v1::Request*,v1::Response*,google::protobuf::Closure*) override;
  EPS_AGENT_RPC(Define,AgentDefineRequest,Ack)
  EPS_AGENT_RPC(FreeSlots,AgentCapacityRequest,AgentCapacityReply)
  EPS_AGENT_RPC(Load,AgentLoadRequest,LocationsReply)
  EPS_AGENT_RPC(Gather,AgentGatherRequest,AgentGatherReply)
  EPS_AGENT_RPC(Apply,AgentApplyRequest,Ack)
  EPS_AGENT_RPC(ApplyCompact,AgentCompactApplyRequest,Ack)
  EPS_AGENT_RPC(ReadBack,AgentPlanRequest,ReadStateReply)
  EPS_AGENT_RPC(CommitEviction,AgentPlanRequest,Ack)
#undef EPS_AGENT_RPC
 private:AgentRuntime& runtime_;
};
}  // namespace eps::transport
