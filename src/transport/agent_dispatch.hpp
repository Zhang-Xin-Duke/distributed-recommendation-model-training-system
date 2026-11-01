#pragma once
#include "common/agent_admission.hpp"
#include "transport/rpc_dispatch.hpp"

namespace eps::transport {
template<typename T,typename Req,typename Rsp,typename Start,typename EncodeFn>
void AdmittedDispatch(detail::AgentAdmissionQueue& admission,
    google::protobuf::RpcController* raw,const Req* request,Rsp* response,
    google::protobuf::Closure* done,Start start,EncodeFn encode){
  brpc::ClosureGuard guard(done);
  auto* controller=static_cast<brpc::Controller*>(raw);
  try{
    auto bytes=request->ByteSizeLong();
    detail::Require(bytes<=(1ULL<<20) && controller->request_attachment().size()<=kMaxAttachment,
      "RPC payload exceeds limit");
    bytes+=controller->request_attachment().size();
    admission.Submit(bytes,[controller,request,response,done,start=std::move(start),encode=std::move(encode)]
        (std::shared_ptr<void> permit) mutable {
      // Request/response storage remains owned by BRPC until done is invoked.
      // Dispatch transfers the permit to the asynchronous completion task.
      if(controller->IsCanceled()){
        brpc::ClosureGuard complete(done);
        SetError(response->mutable_status(),{Error::Code::unavailable,"Agent request canceled before dispatch"});
        return;
      }
      Dispatch<T>(controller,request,response,done,std::move(start),std::move(encode),std::move(permit));
    });
    guard.release();
  }catch(const detail::Failure& e){SetError(response->mutable_status(),e.error());}
   catch(const std::bad_alloc&){SetError(response->mutable_status(),{Error::Code::capacity_exceeded,"Agent admission allocation failed"});}
   catch(const std::exception& e){SetError(response->mutable_status(),{Error::Code::run_failed,e.what()});}
}
}  // namespace eps::transport
