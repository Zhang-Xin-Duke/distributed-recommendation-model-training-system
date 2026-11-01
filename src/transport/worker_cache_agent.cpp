#include "eps/worker_cache_agent.hpp"
#include <brpc/server.h>
#include <butil/endpoint.h>
#include "transport/brpc_agent_server.hpp"

namespace eps {
struct WorkerCacheAgent::Impl {
  Impl(WorkerCacheOptions options,std::shared_ptr<CacheBackend> backend)
      :run(std::move(options.run)),runtime(run,std::move(backend),options.replay_bytes),
       service(runtime,options.admission){
    detail::Require(server.AddService(&service,brpc::SERVER_DOESNT_OWN_SERVICE)==0,
      "cannot register worker cache Agent");
    brpc::ServerOptions settings;settings.max_concurrency=0;
    detail::Require(server.Start(options.listen.c_str(),&settings)==0,"cannot start worker cache Agent");
  }
  ~Impl(){Stop();}
  void Stop(){
    std::call_once(stopped,[this]{server.Stop(0);server.Join();service.CloseAdmission();});
  }
  std::string run;
  AgentRuntime runtime;
  transport::BrpcAgentServer service;
  brpc::Server server;
  std::once_flag stopped;
};
namespace {
template<typename T>std::future<Result<T>> Closed(){
  std::promise<Result<T>> result;result.set_value(Error{Error::Code::unavailable,"worker cache Agent is stopped"});
  return result.get_future();
}
}
WorkerCacheAgent::WorkerCacheAgent(WorkerCacheOptions options,std::shared_ptr<CacheBackend> backend)
    :impl_(std::make_unique<Impl>(std::move(options),std::move(backend))){
  endpoint_=butil::endpoint2str(impl_->server.listen_address()).c_str();
}
WorkerCacheAgent::~WorkerCacheAgent(){Stop();}
std::future<Result<std::vector<TensorBlock>>> WorkerCacheAgent::GatherHost(
    std::vector<AgentWork> work,std::vector<FeatureInput> layouts){
  std::lock_guard lock(mutex_);
  if(!impl_)return Closed<std::vector<TensorBlock>>();
  return impl_->service.GatherLocal(impl_->run,std::move(work),std::move(layouts));
}
std::future<Result<std::vector<DeviceTensorBlock>>> WorkerCacheAgent::GatherDevice(
    std::vector<AgentWork> work,std::vector<FeatureInput> layouts){
  std::lock_guard lock(mutex_);
  if(!impl_)return Closed<std::vector<DeviceTensorBlock>>();
  return impl_->service.GatherDeviceLocal(impl_->run,std::move(work),std::move(layouts));
}
AgentAdmissionStats WorkerCacheAgent::Stats() const {
  std::lock_guard lock(mutex_);return impl_?impl_->service.LocalStats():final_stats_;
}
void WorkerCacheAgent::Stop(){
  std::lock_guard lock(mutex_);if(!impl_)return;
  impl_->Stop();final_stats_=impl_->service.LocalStats();impl_.reset();
}
}  // namespace eps
