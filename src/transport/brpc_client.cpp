#include "eps/brpc_client.hpp"

#include <functional>
#include <brpc/channel.h>
#include <brpc/controller.h>
#include "transport/codec.hpp"

namespace eps {
namespace t = transport;
namespace wire = ::eps::v1;
struct BrpcParameterService::Impl {
  brpc::Channel channel;
  int timeout_ms;
  explicit Impl(const std::string& endpoint, int timeout) : timeout_ms(timeout) {
    brpc::ChannelOptions options; options.protocol="baidu_std";
    options.timeout_ms=timeout; options.max_retry=2;
    detail::Require(channel.Init(endpoint.c_str(),&options)==0,"cannot initialize BRPC channel");
  }
};
namespace {
template <typename Req, typename Rsp, typename Out>
struct CallState : google::protobuf::Closure {
  std::shared_ptr<void> keep_alive;
  brpc::Controller controller;
  Req request;
  Rsp response;
  std::promise<Result<Out>> promise;
  std::function<Out(const Rsp&,const butil::IOBuf&)> decode;
  void Run() override {
    std::unique_ptr<CallState> self(this);
    try {
      if(controller.Failed()) promise.set_value(Error{Error::Code::unavailable,controller.ErrorText()});
      else if(auto error=t::GetError(response.status())) promise.set_value(*error);
      else promise.set_value(decode(response,controller.response_attachment()));
    } catch(const detail::Failure& e) { promise.set_value(e.error()); }
      catch(const std::exception& e) { promise.set_value(Error{Error::Code::run_failed,e.what()}); }
  }
};
template <typename Req, typename Rsp, typename Out, typename Impl, typename Build, typename Decode>
std::future<Result<Out>> Call(std::shared_ptr<Impl> impl,
    void (wire::EmbeddingService_Stub::*method)(google::protobuf::RpcController*,const Req*,Rsp*,google::protobuf::Closure*),
    Build build, Decode decode) {
  auto state=std::make_unique<CallState<Req,Rsp,Out>>();
  auto future=state->promise.get_future();
  state->keep_alive=impl;state->decode=std::move(decode);
  state->controller.set_timeout_ms(impl->timeout_ms);
  try { build(state->request,state->controller.request_attachment()); }
  catch(const detail::Failure& e) { state->promise.set_value(e.error());return future; }
  wire::EmbeddingService_Stub stub(&impl->channel);
  auto* raw=state.release();(stub.*method)(&raw->controller,&raw->request,&raw->response,raw);
  return future;
}
auto AckDecoder=[](const wire::Ack&,const butil::IOBuf&) { return Ack{}; };
}
BrpcParameterService::BrpcParameterService(std::string endpoint,int timeout)
    :impl_(std::make_shared<Impl>(endpoint,timeout)) {}
BrpcParameterService::~BrpcParameterService()=default;
std::future<Result<Ack>> BrpcParameterService::CreateTable(TableDefinition d) {
  return Call<wire::CreateTableRequest,wire::Ack,Ack>(impl_,&wire::EmbeddingService_Stub::CreateTable,
    [d](auto& q,auto&) {t::Encode(d,q.mutable_table());},AckDecoder);
}
std::future<Result<LookupReply>> BrpcParameterService::Lookup(LookupRequest r) {
  return Call<wire::LookupRequest,wire::LookupReply,LookupReply>(impl_,&wire::EmbeddingService_Stub::Lookup,
    [r=std::move(r)](auto& q,auto& a) {
      t::Encode(r.id,q.mutable_identity());q.set_partition_epoch(r.partition_epoch);q.set_read_only(r.read_only);
      for(const auto& f:r.features)t::Encode(f,q.add_features(),a);
    },[](const auto& q,const auto& a) {
      detail::Require(q.agent_work_size()==0,"GPU response requires a GPU adapter");
      LookupReply out;out.ticket=q.ticket();
      for(const auto& f:q.cpu_contributions())out.cpu_contributions.push_back(t::Decode(f,a));
      return out;
    });
}
std::future<Result<Ack>> BrpcParameterService::Push(PushRequest r) {
  return Call<wire::PushRequest,wire::Ack,Ack>(impl_,&wire::EmbeddingService_Stub::Push,
    [r=std::move(r)](auto& q,auto& a) {
      t::Encode(r.id,q.mutable_identity());q.set_ticket(r.ticket);
      for(const auto& g:r.feature_gradients)t::Encode(g,q.add_feature_gradients(),a);
    },AckDecoder);
}
std::future<Result<Ack>> BrpcParameterService::Release(ReleaseRequest r) {
  return Call<wire::ReleaseRequest,wire::Ack,Ack>(impl_,&wire::EmbeddingService_Stub::Release,
    [r=std::move(r)](auto& q,auto&) {t::Encode(r.id,q.mutable_identity());q.set_ticket(r.ticket);},AckDecoder);
}
std::future<Result<std::vector<RowState>>> BrpcParameterService::ReadState(std::vector<LocatedRow> rows) {
  return Call<wire::ReadStateRequest,wire::ReadStateReply,std::vector<RowState>>(impl_,&wire::EmbeddingService_Stub::ReadState,
    [rows=std::move(rows)](auto& q,auto&) {for(const auto& row:rows)t::Encode(row,q.add_rows());},
    [](const auto& q,const auto& a) {std::vector<RowState> out;for(const auto& r:q.states())out.push_back(t::Decode(r,a));return out;});
}
std::future<Result<Ack>> BrpcParameterService::CompleteMigration(MigrationCompletion r) {
  return Call<wire::MigrationCompletion,wire::Ack,Ack>(impl_,&wire::EmbeddingService_Stub::CompleteMigration,
    [r=std::move(r)](auto& q,auto&) {
      q.set_plan_id(r.plan_id);q.set_completed(r.completed);for(const auto& row:r.rows)t::Encode(row,q.add_rows());
    },AckDecoder);
}
std::future<Result<SnapshotMetadata>> BrpcParameterService::DrainAndFlush(std::string id) {
  return Call<wire::DrainRequest,wire::SnapshotReply,SnapshotMetadata>(impl_,&wire::EmbeddingService_Stub::DrainAndFlush,
    [id=std::move(id)](auto& q,auto&) {q.set_snapshot_id(id);},
    [](const auto& q,const auto&) {
      SnapshotMetadata out{q.snapshot_id(),q.shard_id(),q.partition_epoch(),
        {q.state_files().begin(),q.state_files().end()},{}};
      for(const auto& c:q.optimizer_contexts())out.optimizer_contexts.push_back({{},q.shard_id(),c.table_id(),
        c.optimizer_sequence(),c.learning_rate(),c.beta1_power(),c.beta2_power()});
      return out;
    });
}
}  // namespace eps
