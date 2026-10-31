#include "transport/brpc_server.hpp"

#include <chrono>
#include <functional>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <bthread/bthread.h>
#include "transport/codec.hpp"

namespace eps::transport {
namespace {
template <typename T, typename Rsp>
struct Pending {
  std::future<Result<T>> future;
  brpc::Controller* controller;
  Rsp* response;
  google::protobuf::Closure* done;
  std::function<void(T,const Rsp*,Rsp*,butil::IOBuf&)> encode;
  static void* Run(void* pointer) {
    std::unique_ptr<Pending> p(static_cast<Pending*>(pointer));
    brpc::ClosureGuard done(p->done);
    try {
      // std::future's blocking wait would pin a native BRPC worker. Yield this
      // lightweight bthread while the bounded shard sequencer does the work.
      while(p->future.wait_for(std::chrono::seconds(0))!=std::future_status::ready)
        bthread_usleep(100);
      auto result=p->future.get();
      if(auto error=std::get_if<Error>(&result)) SetError(p->response->mutable_status(),*error);
      else p->encode(std::get<T>(std::move(result)),p->response,p->response,
                     p->controller->response_attachment());
    } catch(const detail::Failure& e) { SetError(p->response->mutable_status(),e.error()); }
      catch(const std::exception& e) { SetError(p->response->mutable_status(),{Error::Code::run_failed,e.what()}); }
    return nullptr;
  }
};

template <typename T,typename Req,typename Rsp,typename Start,typename EncodeFn>
void Dispatch(google::protobuf::RpcController* raw,const Req* request,Rsp* response,
              google::protobuf::Closure* done,Start start,EncodeFn encode) {
  brpc::ClosureGuard guard(done);
  auto* controller=static_cast<brpc::Controller*>(raw);
  try {
    detail::Require(controller->request_attachment().size()<=kMaxAttachment &&
                    request->ByteSizeLong() <= (1ULL<<20),"RPC payload exceeds limit");
    auto p=std::make_unique<Pending<T,Rsp>>();
    p->future=start(controller->request_attachment());
    p->controller=controller;p->response=response;p->done=done;
    p->encode=[encode](T result,const Rsp*,Rsp* out,butil::IOBuf& a) {encode(std::move(result),out,a);};
    bthread_t id;
    detail::Require(bthread_start_background(&id,nullptr,&Pending<T,Rsp>::Run,p.get())==0,
                    "cannot schedule RPC completion",Error::Code::unavailable);
    guard.release();p.release();
  } catch(const detail::Failure& e) {SetError(response->mutable_status(),e.error());}
    catch(const std::exception& e) {SetError(response->mutable_status(),{Error::Code::run_failed,e.what()});}
}
auto AckEncode=[](Ack,wire::Ack*,butil::IOBuf&) {};
}

void BrpcServer::CreateTable(google::protobuf::RpcController* c,const wire::CreateTableRequest* q,
                           wire::Ack* r,google::protobuf::Closure* done) {
  Dispatch<Ack>(c,q,r,done,[&](const auto&) {return service_.CreateTable(Decode(q->table()));},AckEncode);
}
void BrpcServer::Lookup(google::protobuf::RpcController* c,const wire::LookupRequest* q,
                      wire::LookupReply* r,google::protobuf::Closure* done) {
  Dispatch<LookupReply>(c,q,r,done,[&](const auto& a) {
    detail::Require(q->features_size()<=4096,"too many features");
    LookupRequest request{Decode(q->identity()),q->partition_epoch(),q->read_only(),{}};
    for(const auto& f:q->features())request.features.push_back(Decode(f,a));
    return service_.Lookup(std::move(request));
  },[](LookupReply result,auto* out,auto& a) {
    out->set_ticket(result.ticket);
    for(const auto& f:result.cpu_contributions)Encode(f,out->add_cpu_contributions(),a);
    detail::Require(result.agent_work.empty(),"GPU response encoding is not enabled");
  });
}
void BrpcServer::Push(google::protobuf::RpcController* c,const wire::PushRequest* q,
                    wire::Ack* r,google::protobuf::Closure* done) {
  Dispatch<Ack>(c,q,r,done,[&](const auto& a) {
    detail::Require(q->feature_gradients_size()<=4096,"too many gradient features");
    PushRequest request{Decode(q->identity()),q->ticket(),{}};
    for(const auto& g:q->feature_gradients())request.feature_gradients.push_back(Decode(g,a));
    return service_.Push(std::move(request));
  },AckEncode);
}
void BrpcServer::Release(google::protobuf::RpcController* c,const wire::ReleaseRequest* q,
                       wire::Ack* r,google::protobuf::Closure* done) {
  Dispatch<Ack>(c,q,r,done,[&](const auto&) {
    return service_.Release({Decode(q->identity()),q->ticket()});
  },AckEncode);
}
void BrpcServer::ReadState(google::protobuf::RpcController* c,const wire::ReadStateRequest* q,
                         wire::ReadStateReply* r,google::protobuf::Closure* done) {
  Dispatch<std::vector<RowState>>(c,q,r,done,[&](const auto&) {
    std::vector<LocatedRow> rows;for(const auto& x:q->rows())rows.push_back(Decode(x));
    return service_.ReadState(std::move(rows));
  },[shard=shard_](auto result,auto* out,auto& a) {
    for(const auto& row:result)Encode(row,out->add_states(),a,shard);
  });
}
void BrpcServer::CompleteMigration(google::protobuf::RpcController* c,const wire::MigrationCompletion* q,
                                 wire::Ack* r,google::protobuf::Closure* done) {
  Dispatch<Ack>(c,q,r,done,[&](const auto&) {
    MigrationCompletion completion{q->plan_id(),{},q->completed()};
    for(const auto& row:q->rows())completion.rows.push_back(Decode(row));
    return service_.CompleteMigration(std::move(completion));
  },AckEncode);
}
void BrpcServer::DrainAndFlush(google::protobuf::RpcController* c,const wire::DrainRequest* q,
                             wire::SnapshotReply* r,google::protobuf::Closure* done) {
  Dispatch<SnapshotMetadata>(c,q,r,done,[&](const auto&) {return service_.DrainAndFlush(q->snapshot_id());},
    [](SnapshotMetadata result,auto* out,auto&) {
      out->set_snapshot_id(result.snapshot_id);out->set_shard_id(result.shard);
      out->set_partition_epoch(result.partition_epoch);
      for(const auto& file:result.state_files)out->add_state_files(file);
      for(const auto& context:result.optimizer_contexts) {
        auto* x=out->add_optimizer_contexts();x->set_table_id(context.table);
        x->set_optimizer_sequence(context.optimizer_sequence);x->set_learning_rate(context.learning_rate);
        x->set_beta1_power(context.beta1_power);x->set_beta2_power(context.beta2_power);
      }
    });
}
}  // namespace eps::transport
