#pragma once

#include <functional>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include "transport/codec.hpp"
#include "transport/completion_executor.hpp"

namespace eps::transport {
template <typename T, typename Rsp>
struct Pending final : CompletionExecutor::Task {
  std::shared_ptr<void> completion_owner;
  std::future<Result<T>> future;
  brpc::Controller* controller;
  Rsp* response;
  google::protobuf::Closure* done;
  std::function<void(T,const Rsp*,Rsp*,butil::IOBuf&)> encode;
  void Run() override {
    brpc::ClosureGuard guard(done);
    try {
      auto result=future.get();
      if(auto error=std::get_if<Error>(&result)) SetError(response->mutable_status(),*error);
      else encode(std::get<T>(std::move(result)),response,response,controller->response_attachment());
    } catch(const detail::Failure& e) { SetError(response->mutable_status(),e.error()); }
      catch(const std::exception& e) { SetError(response->mutable_status(),{Error::Code::run_failed,e.what()}); }
  }
};

template <typename T,typename Req,typename Rsp,typename Start,typename EncodeFn>
void Dispatch(google::protobuf::RpcController* raw,const Req* request,Rsp* response,
              google::protobuf::Closure* done,Start start,EncodeFn encode,
              std::shared_ptr<void> completion_owner={},
              CompletionExecutor& completions=CompletionExecutor::Instance()) {
  brpc::ClosureGuard guard(done);
  auto* controller=static_cast<brpc::Controller*>(raw);
  try {
    detail::Require(controller->request_attachment().size()<=kMaxAttachment &&
                    request->ByteSizeLong() <= (1ULL<<20),"RPC payload exceeds limit");
    CompletionExecutor::Reservation reservation(completions);
    auto p=std::make_unique<Pending<T,Rsp>>();
    p->completion_owner=std::move(completion_owner);
    p->controller=controller;p->response=response;p->done=done;
    p->encode=[encode](T result,const Rsp*,Rsp* out,butil::IOBuf& a) {encode(std::move(result),out,a);};
    p->future=start(controller->request_attachment());
    guard.release();reservation.Publish(std::move(p));
  } catch(const detail::Failure& e) {SetError(response->mutable_status(),e.error());}
    catch(const std::exception& e) {SetError(response->mutable_status(),{Error::Code::run_failed,e.what()});}
}
inline constexpr auto AckEncode=[](Ack,wire::Ack*,butil::IOBuf&) {};
}  // namespace eps::transport
