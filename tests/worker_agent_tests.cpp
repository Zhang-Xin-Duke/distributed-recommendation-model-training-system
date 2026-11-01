#include "eps/worker_cache_agent.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/agent_runtime.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
using namespace eps;
template<typename T>T Take(std::future<Result<T>> f){auto r=f.get();
  if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<typename T>void ErrorIs(std::future<Result<T>> f,Error::Code code){auto r=f.get();
  Check(std::holds_alternative<Error>(r) && std::get<Error>(r).code==code,"unexpected failure status");}
class PausedHost final:public CacheBackend {
  HostCacheBackend host_{0,1<<20};
  std::promise<void> started_,release_;
  std::atomic<bool> pause_{false};
 public:
  std::future<void> Pause(){pause_=true;return started_.get_future();}
  void Unblock(){release_.set_value();}
  std::future<Result<Ack>> Define(TableDefinition t) override{return host_.Define(t);}
  std::future<Result<std::uint64_t>> FreeSlots(TableId t) override{return host_.FreeSlots(t);}
  std::future<Result<std::vector<LocatedRow>>> Load(MigrationPlan p,std::vector<RowState> s) override{return host_.Load(std::move(p),std::move(s));}
  std::future<Result<std::vector<TensorBlock>>> GatherHost(std::vector<AgentWork> w,std::vector<FeatureInput> f) override{
    if(pause_.exchange(false))return std::async(std::launch::async,[this,w=std::move(w),f=std::move(f)]() mutable {
      started_.set_value();release_.get_future().wait();return host_.GatherHost(std::move(w),std::move(f)).get();});
    return host_.GatherHost(std::move(w),std::move(f));
  }
  std::future<Result<Ack>> ApplyHost(UpdateContext c,RowGradients g) override{return host_.ApplyHost(c,std::move(g));}
  std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan p) override{return host_.ReadBack(std::move(p));}
  std::future<Result<Ack>> CommitEviction(MigrationPlan p) override{return host_.CommitEviction(std::move(p));}
};
template<typename P>AgentAdmissionStats Wait(WorkerCacheAgent& worker,P pred){
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  do{auto s=worker.Stats();if(pred(s))return s;std::this_thread::sleep_for(std::chrono::milliseconds(1));}
  while(std::chrono::steady_clock::now()<deadline);throw std::runtime_error("worker stats deadline");
}
int main(){try{
  auto backend=std::make_shared<PausedHost>();std::weak_ptr<CacheBackend> lifetime=backend;
  WorkerCacheAgent worker({"worker-test","127.0.0.1:0",{1,2,4096}},backend);
  const auto endpoint=worker.endpoint();BrpcCacheBackend remote(endpoint,"worker-test",0,0,5000);
  Take(remote.Define({1,2,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},0,0,42}));
  auto rows=Take(remote.Load({"load",0,{{{1,1},0,Placement::loading,0,1,0}},Placement::gpu,24},
      {{{1,1},{0.25f,0.5f},{0,0},{0,0},0}}));
  Wait(worker,[](auto s){return s.accepted_requests==s.completed_requests;});
  std::vector<AgentWork> work{{8,AccessRoute::resident_gpu,0,rows,{0},{}}};
  std::vector<FeatureInput> layouts{{1,8,Pooling::sum,{1},{0,1},{1}}};
  auto local=Take(worker.GatherHost(work,layouts));
  Check(local.at(0).values==std::vector<float>({0.25f,0.5f}),"local gather differs from RPC-loaded state");
  ErrorIs(worker.GatherDevice(work,layouts),Error::Code::invalid_argument);
  {AgentRuntime runtime("right",backend);
    ErrorIs(runtime.GatherDevice("wrong",{},{}),Error::Code::invalid_argument);}
  Wait(worker,[](auto s){return s.accepted_requests==s.completed_requests;});
  auto oversized=layouts;oversized[0].keys.reserve(10000);
  ErrorIs(worker.GatherHost(work,std::move(oversized)),Error::Code::capacity_exceeded);
  auto entered=backend->Pause();auto first=worker.GatherHost(work,layouts);entered.get();
  auto second=remote.GatherHost(work,layouts);
  Wait(worker,[](auto s){return s.active_operations==1 && s.queued_operations==1;});
  ErrorIs(worker.GatherHost(work,layouts),Error::Code::capacity_exceeded);
  auto closing=std::async(std::launch::async,[&]{worker.Stop();});
  Check(closing.wait_for(std::chrono::milliseconds(30))==std::future_status::timeout,"Stop did not wait for pending local/RPC work");
  backend->Unblock();Check(Take(std::move(first))[0].values==local[0].values,"local pending result lost");
  Check(Take(std::move(second))[0].values==local[0].values,"RPC pending result lost");closing.get();
  const auto stats=worker.Stats();
  Check(stats.active_operations==0 && stats.queued_operations==0 && stats.pending_request_bytes==0 &&
    stats.accepted_requests==stats.completed_requests && stats.rejected_requests==2 && stats.dispatch_failures==0,
    "worker shutdown leaked admission state");
  ErrorIs(worker.GatherHost(work,layouts),Error::Code::unavailable);
  ErrorIs(worker.GatherDevice(work,layouts),Error::Code::unavailable);worker.Stop();
  backend.reset();Check(lifetime.expired(),"stopped worker retained backend");
  WorkerCacheAgent rebound({"new-run",endpoint},std::make_shared<HostCacheBackend>(0,1024));
  BrpcCacheBackend new_remote(endpoint,"new-run",0,0,5000);
  // BRPC shares endpoint sockets across channels. Wait for the old connection's
  // health check to reconnect to the rebound listener before mutating state.
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(std::holds_alternative<Error>(new_remote.AdmissionStats().get())){
    Check(std::chrono::steady_clock::now()<deadline,"rebound endpoint did not reconnect");
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  Take(new_remote.Define({1,2,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},0,0,42}));
  ErrorIs(remote.FreeSlots(1),Error::Code::invalid_argument);rebound.Stop();
  std::cout<<"{\"test\":\"in_process_worker_lifecycle\",\"shared_local_rpc_admission\":true,\"drained_on_stop\":true,\"backend_released\":true,\"endpoint_reused\":true}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<"FAIL worker Agent: "<<e.what()<<'\n';return 1;}}
