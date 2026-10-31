#pragma once

#include <chrono>
#include <functional>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <bthread/bthread.h>
#include "transport/codec.hpp"

namespace eps::transport {
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
inline constexpr auto AckEncode=[](Ack,wire::Ack*,butil::IOBuf&) {};
}  // namespace eps::transport
