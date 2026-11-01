#pragma once

#include "embedding_service.pb.h"
#include "eps/runtime.hpp"

namespace eps::transport {
class BrpcServer final : public v1::EmbeddingService {
 public:
  BrpcServer(CpuParameterService& service, ShardId shard) : service_(service), shard_(shard) {}
#define EPS_RPC(Name, Request, Response) \
  void Name(google::protobuf::RpcController*, const v1::Request*, v1::Response*, google::protobuf::Closure*) override;
  EPS_RPC(CreateTable, CreateTableRequest, Ack)
  EPS_RPC(Stats, StatsRequest, StatsReply)
  EPS_RPC(Lookup, LookupRequest, LookupReply)
  EPS_RPC(Push, PushRequest, Ack)
  EPS_RPC(Release, ReleaseRequest, Ack)
  EPS_RPC(ReadState, ReadStateRequest, ReadStateReply)
  EPS_RPC(CompleteMigration, MigrationCompletion, Ack)
  EPS_RPC(DrainAndFlush, DrainRequest, SnapshotReply)
  EPS_RPC(ReadCheckpoint, SnapshotReadRequest, SnapshotChunkReply)
  EPS_RPC(ResumeCheckpoint, CheckpointIdentity, Ack)
  EPS_RPC(Locations, RowsRequest, LocationsReply)
  EPS_RPC(MoveRows, MoveRequest, LocationsReply)
#undef EPS_RPC
 private:
  CpuParameterService& service_;
  ShardId shard_;
};
}  // namespace eps::transport
