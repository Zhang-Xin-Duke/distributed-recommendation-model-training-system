#include "transport/brpc_replica_server.hpp"
#include "transport/replica_codec.hpp"
#include "transport/rpc_dispatch.hpp"
#include "common/resumable_actor.hpp"
namespace eps::transport {
struct BrpcReplicaServer::Impl {
  ReplicaLogOptions options;
  ReplicaLog log;
  detail::ResumableActor actor{{64,1,1,8ULL<<20,1ULL<<20,true,16}};
  explicit Impl(ReplicaLogOptions o):options(std::move(o)),log(options){}
};
BrpcReplicaServer::BrpcReplicaServer(ReplicaLogOptions o):impl_(std::make_unique<Impl>(std::move(o))){}
BrpcReplicaServer::~BrpcReplicaServer()=default;
void BrpcReplicaServer::Head(google::protobuf::RpcController* c,const v1::ReplicaControl* q,v1::ReplicaHeadReply* r,google::protobuf::Closure* done) {
  Dispatch<ReplicaHead>(c,q,r,done,[&](const auto&){CheckScope(impl_->options,q->scope());return impl_->actor.Submit<ReplicaHead>([this]{return impl_->log.Head();});},
    [](auto h,auto* out,auto&){EncodeHead(h,out);});
}
void BrpcReplicaServer::Fence(google::protobuf::RpcController* c,const v1::ReplicaFence* q,v1::ReplicaHeadReply* r,google::protobuf::Closure* done) {
  Dispatch<ReplicaHead>(c,q,r,done,[&](const auto&) {
    CheckScope(impl_->options,q->scope());const auto expected=q->expected();const auto primary=q->primary();
    return impl_->actor.Submit<ReplicaHead>([this,expected,primary]{impl_->log.Fence(expected,primary);return impl_->log.Head();});
  },[](auto h,auto* out,auto&){EncodeHead(h,out);});
}
void BrpcReplicaServer::Write(google::protobuf::RpcController* c,const v1::ReplicaChunk* q,v1::Ack* r,google::protobuf::Closure* done) {
  Dispatch<Ack>(c,q,r,done,[&](const auto& attachment) {
    CheckScope(impl_->options,q->scope());detail::Require(attachment.size()==q->bytes() && q->bytes()<=(1<<20),"invalid replica attachment");
    const auto lease=DecodeLease(q->lease());const auto baseline=q->baseline();const auto round=q->round(),total=q->total_bytes(),offset=q->offset();
    auto data=std::make_shared<const std::string>(attachment.to_string());
    return impl_->actor.Submit<Ack>([this,lease,baseline,round,total,offset,data]{impl_->log.Write(lease,baseline,round,total,offset,*data);return Ack{};},
      []{return detail::ActorResources{};},data->size());
  },AckEncode);
}
void BrpcReplicaServer::Read(google::protobuf::RpcController* c,const v1::ReplicaRead* q,v1::SnapshotChunkReply* r,google::protobuf::Closure* done) {
  Dispatch<SnapshotChunk>(c,q,r,done,[&](const auto&) {
    CheckScope(impl_->options,q->scope());const auto lease=DecodeLease(q->lease());const auto baseline=q->baseline();
    const auto round=q->round(),offset=q->offset();const auto bytes=q->max_bytes();
    return impl_->actor.Submit<SnapshotChunk>([this,lease,baseline,round,offset,bytes]{return impl_->log.Read(lease,baseline,round,offset,bytes);});
  },[](auto chunk,auto* out,auto&){out->set_offset(chunk.offset);out->set_total_bytes(chunk.total_bytes);out->set_data(std::move(chunk.data));out->set_eof(chunk.eof);});
}
}  // namespace eps::transport
