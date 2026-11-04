#pragma once
#include "embedding_service.pb.h"
#include "eps/replica_log.hpp"
namespace eps::transport {
class BrpcReplicaServer final:public v1::ReplicaService {
 public:
  explicit BrpcReplicaServer(ReplicaLogOptions);
  ~BrpcReplicaServer()override;
#define EPS_REPLICA_RPC(Name,Request,Reply) void Name(google::protobuf::RpcController*,const v1::Request*,v1::Reply*,google::protobuf::Closure*)override;
  EPS_REPLICA_RPC(Head,ReplicaControl,ReplicaHeadReply)
  EPS_REPLICA_RPC(Fence,ReplicaFence,ReplicaHeadReply)
  EPS_REPLICA_RPC(Write,ReplicaChunk,Ack)
  EPS_REPLICA_RPC(Read,ReplicaRead,SnapshotChunkReply)
#undef EPS_REPLICA_RPC
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace eps::transport
