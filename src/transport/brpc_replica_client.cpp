#include "eps/brpc_replica.hpp"
#include "transport/rpc_call.hpp"
#include "transport/replica_codec.hpp"
namespace eps {
namespace t=transport;
struct BrpcReplicaLog::Impl {
  brpc::Channel channel;
  int timeout_ms;
  ReplicaLogOptions scope;
  Impl(std::string endpoint,ReplicaLogOptions o,int timeout):timeout_ms(timeout),scope(std::move(o)) {
    brpc::ChannelOptions options;options.protocol="baidu_std";options.timeout_ms=timeout;options.max_retry=2;
    detail::Require(timeout>0 && channel.Init(endpoint.c_str(),&options)==0,"cannot open replica channel");
  }
};
BrpcReplicaLog::BrpcReplicaLog(std::string endpoint,ReplicaLogOptions o,int timeout):impl_(std::make_shared<Impl>(std::move(endpoint),std::move(o),timeout)){}
BrpcReplicaLog::~BrpcReplicaLog()=default;
std::future<Result<ReplicaHead>> BrpcReplicaLog::Head() {
  return t::Call<v1::ReplicaControl,v1::ReplicaHeadReply,ReplicaHead>(impl_,&v1::ReplicaService_Stub::Head,
    [p=impl_](auto& q,auto&){t::EncodeScope(p->scope,q.mutable_scope());},[](const auto& r,const auto&){return t::DecodeHead(r);});
}
std::future<Result<ReplicaHead>> BrpcReplicaLog::Fence(std::uint64_t expected,std::string primary) {
  return t::Call<v1::ReplicaFence,v1::ReplicaHeadReply,ReplicaHead>(impl_,&v1::ReplicaService_Stub::Fence,
    [p=impl_,expected,primary=std::move(primary)](auto& q,auto&){t::EncodeScope(p->scope,q.mutable_scope());q.set_expected(expected);q.set_primary(primary);},
    [](const auto& r,const auto&){return t::DecodeHead(r);});
}
std::future<Result<Ack>> BrpcReplicaLog::Write(ReplicaLease lease,bool baseline,std::uint64_t round,
    std::uint64_t total,std::uint64_t offset,std::string data) {
  return t::Call<v1::ReplicaChunk,v1::Ack,Ack>(impl_,&v1::ReplicaService_Stub::Write,
    [p=impl_,lease=std::move(lease),baseline,round,total,offset,data=std::move(data)](auto& q,auto& a) {
      detail::Require(data.size()<=(1<<20),"replica chunk exceeds limit");t::EncodeScope(p->scope,q.mutable_scope());
      t::EncodeLease(lease,q.mutable_lease());q.set_baseline(baseline);q.set_round(round);q.set_total_bytes(total);q.set_offset(offset);
      q.set_bytes(data.size());a.append(data);
    },t::AckDecoder);
}
std::future<Result<SnapshotChunk>> BrpcReplicaLog::Read(ReplicaLease lease,bool baseline,std::uint64_t round,std::uint64_t offset,std::uint32_t max_bytes) {
  return t::Call<v1::ReplicaRead,v1::SnapshotChunkReply,SnapshotChunk>(impl_,&v1::ReplicaService_Stub::Read,
    [p=impl_,lease=std::move(lease),baseline,round,offset,max_bytes](auto& q,auto&) {
      t::EncodeScope(p->scope,q.mutable_scope());t::EncodeLease(lease,q.mutable_lease());q.set_baseline(baseline);
      q.set_round(round);q.set_offset(offset);q.set_max_bytes(max_bytes);
    },[offset,max_bytes](const auto& r,const auto&) {
      detail::Require(r.offset()==offset && offset<=r.total_bytes() && r.data().size()<=max_bytes &&
        r.data().size()<=r.total_bytes()-offset && r.eof()==(offset+r.data().size()==r.total_bytes()) &&
        (r.eof() || !r.data().empty()),"invalid replica read reply");
      return SnapshotChunk{r.offset(),r.total_bytes(),r.data(),r.eof()};
    });
}
}  // namespace eps
