#pragma once

#include <functional>
#include <brpc/channel.h>
#include <brpc/controller.h>
#include "transport/codec.hpp"

namespace eps::transport {
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
      else if(auto error=GetError(response.status())) promise.set_value(*error);
      else promise.set_value(decode(response,controller.response_attachment()));
    } catch(const detail::Failure& e) { promise.set_value(e.error()); }
      catch(const std::exception& e) { promise.set_value(Error{Error::Code::run_failed,e.what()}); }
  }
};
template <typename Req, typename Rsp, typename Out, typename Stub, typename Impl, typename Build, typename Decode>
std::future<Result<Out>> Call(std::shared_ptr<Impl> impl,
    void (Stub::*method)(google::protobuf::RpcController*,const Req*,Rsp*,google::protobuf::Closure*),
    Build build, Decode decode) {
  auto state=std::make_unique<CallState<Req,Rsp,Out>>();
  auto future=state->promise.get_future();
  state->keep_alive=impl;state->decode=std::move(decode);
  state->controller.set_timeout_ms(impl->timeout_ms);
  try { build(state->request,state->controller.request_attachment()); }
  catch(const detail::Failure& e) { state->promise.set_value(e.error());return future; }
  Stub stub(&impl->channel);
  auto* raw=state.release();(stub.*method)(&raw->controller,&raw->request,&raw->response,raw);
  return future;
}
inline constexpr auto AckDecoder=[](const wire::Ack&,const butil::IOBuf&) { return Ack{}; };
}  // namespace eps::transport
