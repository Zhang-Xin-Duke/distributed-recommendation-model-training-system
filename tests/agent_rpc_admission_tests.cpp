#include "eps/brpc_cache_backend.hpp"
#include "transport/brpc_agent_server.hpp"
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
using namespace eps;
template<typename T>T Take(std::future<Result<T>> f){auto r=f.get();
  if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}

class PausedBackend final:public CacheBackend {
  struct Held {std::vector<AgentWork> work;std::vector<FeatureInput> layouts;
    std::promise<Result<std::vector<TensorBlock>>> result;};
  HostCacheBackend backend_{0,1<<20};std::mutex mutex_;bool block_=false;
  std::promise<void> entered_;std::unique_ptr<Held> held_;
 public:
  std::atomic<std::uint64_t> gathers=0;
  std::future<void> Pause(){std::lock_guard lock(mutex_);block_=true;
    entered_=std::promise<void>();return entered_.get_future();}
  void Release(){std::unique_ptr<Held> held;
    {std::lock_guard lock(mutex_);block_=false;held=std::move(held_);}
    if(held)held->result.set_value(backend_.GatherHost(std::move(held->work),std::move(held->layouts)).get());
  }
  std::future<Result<Ack>> Define(TableDefinition t) override{return backend_.Define(t);}
  std::future<Result<std::uint64_t>> FreeSlots(TableId t) override{return backend_.FreeSlots(t);}
  std::future<Result<std::vector<LocatedRow>>> Load(MigrationPlan p,std::vector<RowState> s) override{return backend_.Load(std::move(p),std::move(s));}
  std::future<Result<std::vector<TensorBlock>>> GatherHost(std::vector<AgentWork> w,std::vector<FeatureInput> f) override{
    ++gathers;std::lock_guard lock(mutex_);
    if(block_){block_=false;held_=std::make_unique<Held>();held_->work=std::move(w);held_->layouts=std::move(f);
      auto result=held_->result.get_future();entered_.set_value();return result;}
    return backend_.GatherHost(std::move(w),std::move(f));
  }
  std::future<Result<Ack>> ApplyHost(UpdateContext c,RowGradients g) override{return backend_.ApplyHost(c,std::move(g));}
  std::future<Result<Ack>> ApplyCompact(UpdateContext c,CompactGradients g) override{return backend_.ApplyCompact(c,std::move(g));}
  std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan p) override{return backend_.ReadBack(std::move(p));}
  std::future<Result<Ack>> CommitEviction(MigrationPlan p) override{return backend_.CommitEviction(std::move(p));}
};
struct Harness {
  std::shared_ptr<PausedBackend> backend=std::make_shared<PausedBackend>();
  AgentRuntime runtime{"admission-test",backend};
  eps::transport::BrpcAgentServer service;
  brpc::Server server;
  std::string endpoint;
  Harness(AgentAdmissionOptions o):service(runtime,o){
    Check(server.AddService(&service,brpc::SERVER_DOESNT_OWN_SERVICE)==0,"register Agent");
    brpc::ServerOptions opts;opts.max_concurrency=0;
    Check(server.Start("127.0.0.1:0",&opts)==0,"start Agent");
    endpoint=butil::endpoint2str(server.listen_address()).c_str();
  }
  ~Harness(){backend->Release();server.Stop(0);server.Join();}
};
template<typename Pred>AgentAdmissionStats Wait(BrpcCacheBackend& agent,Pred predicate){
  auto end=std::chrono::steady_clock::now()+std::chrono::seconds(8);
  do{auto s=Take(agent.AdmissionStats());if(predicate(s))return s;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));}while(std::chrono::steady_clock::now()<end);
  throw std::runtime_error("admission observation deadline");
}
std::vector<LocatedRow> Setup(BrpcCacheBackend& admin){
  Take(admin.Define({1,2,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},0.25f,0.25f,42}));
  MigrationPlan p{"load",0,{{{1,1},0,Placement::loading,0,1,0}},Placement::gpu,24};
  auto rows=Take(admin.Load(p,{{{1,1},{0.25f,0.5f},{0,0},{0,0},0}}));
  Wait(admin,[](auto s){return s.accepted_requests==s.completed_requests;});
  return rows;
}
auto Gather(BrpcCacheBackend& agent,const std::vector<LocatedRow>& rows,std::size_t keys=1){
  return agent.GatherHost({{8,AccessRoute::resident_gpu,0,rows,std::vector<std::uint64_t>(keys,0),{}}},
    {{1,8,Pooling::sum,std::vector<Key>(keys,1),{0,keys},{keys}}});
}
void Values(std::future<Result<std::vector<TensorBlock>>> f){auto values=Take(std::move(f));
  Check(values.size()==1 && values[0].values==std::vector<float>({0.25f,0.5f}),"queued gather changed result");}
void State(BrpcCacheBackend& admin,std::vector<LocatedRow> rows){
  auto state=Take(admin.ReadBack({"inspect",0,std::move(rows),Placement::cpu,24})).at(0);
  Check(state.weights==std::vector<float>({0.25f,0.5f}) && state.first_moment==std::vector<float>({0,0}) &&
    state.second_moment==std::vector<float>({0,0}) && state.row_updates==0,"rejection changed row state");
}
int main(){try{
  {
    Harness h({4,512,2<<20});BrpcCacheBackend admin(h.endpoint,"admission-test",0,0,15000);
    auto rows=Setup(admin);auto entered=h.backend->Pause();
    std::vector<std::unique_ptr<BrpcCacheBackend>> clients;
    for(int i=0;i<8;++i)clients.push_back(std::make_unique<BrpcCacheBackend>(h.endpoint,"admission-test",0,0,15000));
    std::vector<std::future<Result<std::vector<TensorBlock>>>> pending;
    pending.push_back(Gather(*clients[0],rows));entered.get();
    for(int i=1;i<256;++i)pending.push_back(Gather(*clients[i%8],rows));
    auto waiting=Wait(admin,[](auto s){return s.accepted_requests>=258;});
    Check(waiting.active_operations==4 && waiting.queued_operations==252 &&
      waiting.peak_pending_operations==256 && waiting.rejected_requests==0,"burst escaped bounded queue");
    h.backend->Release();for(auto& f:pending)Values(std::move(f));State(admin,rows);
    auto end=Wait(admin,[](auto s){return s.accepted_requests==s.completed_requests;});
    Check(end.active_operations==0 && end.queued_operations==0 && end.pending_request_bytes==0 &&
      end.peak_active_operations==4 && end.dispatch_failures==0,"burst leaked permits");
    std::cout<<"{\"test\":\"agent_rpc_admission_burst\",\"independent_clients\":8,\"requests\":256,\"active_limit\":4,\"observed_queued\":252,\"rejected\":0,\"state_unchanged\":true}\n";
  }
  {
    Harness h({1,2,4096});BrpcCacheBackend admin(h.endpoint,"admission-test",0,0,15000);
    auto rows=Setup(admin);auto entered=h.backend->Pause();auto first=Gather(admin,rows);entered.get();
    auto oversized=Gather(admin,rows,4096).get();
    Check(std::holds_alternative<Error>(oversized) && std::get<Error>(oversized).code==Error::Code::capacity_exceeded,"byte budget accepted oversized request");
    auto second=Gather(admin,rows);Wait(admin,[](auto s){return s.queued_operations==1;});
    auto excess=Gather(admin,rows).get();
    Check(std::holds_alternative<Error>(excess) && std::get<Error>(excess).code==Error::Code::capacity_exceeded,"count budget accepted excess request");
    Check(h.backend->gathers==1,"rejected request reached backend");
    h.backend->Release();Values(std::move(first));Values(std::move(second));State(admin,rows);
    auto end=Wait(admin,[](auto s){return s.accepted_requests==s.completed_requests;});
    Check(end.rejected_requests==2 && end.peak_pending_operations==2 && end.peak_active_operations==1 &&
      end.peak_request_bytes<=4096 && end.pending_request_bytes==0 && h.backend->gathers==2,"rejection cleanup accounting");
    std::cout<<"{\"test\":\"agent_rpc_admission_limits\",\"count_and_byte_rejections\":2,\"rejected_before_execution\":true,\"state_unchanged\":true}\n";
  }
  return 0;
}catch(const std::exception& e){std::cerr<<"FAIL Agent RPC admission: "<<e.what()<<'\n';return 1;}}
