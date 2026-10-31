#include "eps/cache_backend.hpp"
#include "eps/distributed_client.hpp"
#include "eps/runtime.hpp"
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
using namespace eps;
namespace {
template<typename T>T Take(Result<T> r){if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<typename T>T Take(std::future<Result<T>> f){return Take(f.get());}
struct Rendezvous {
  void Arrive(){
    std::unique_lock lock(mutex);const auto previous=round;
    if(++arrived==2){arrived=0;++round;changed.notify_all();return;}
    if(!changed.wait_for(lock,std::chrono::seconds(3),[&]{return round!=previous;}))
      throw std::runtime_error("independent GPU-owner updates were serialized");
  }
  std::mutex mutex;std::condition_variable changed;int arrived=0,round=0;
};
class GatedBackend final : public CacheBackend {
 public:
  GatedBackend(AgentId id,std::shared_ptr<Rendezvous> gate):backend(id,1<<20),gate(std::move(gate)){}
  std::future<Result<Ack>> Define(TableDefinition t)override{return backend.Define(t);}
  std::future<Result<std::uint64_t>> FreeSlots(TableId t)override{return backend.FreeSlots(t);}
  std::future<Result<std::vector<LocatedRow>>> Load(MigrationPlan p,std::vector<RowState> r)override{return backend.Load(std::move(p),std::move(r));}
  std::future<Result<std::vector<TensorBlock>>> GatherHost(std::vector<AgentWork> w,std::vector<FeatureInput> f)override{return backend.GatherHost(std::move(w),std::move(f));}
  std::future<Result<Ack>> ApplyHost(UpdateContext c,RowGradients g)override{
    return std::async(std::launch::async,[this,c,g=std::move(g)]()mutable->Result<Ack>{gate->Arrive();return backend.ApplyHost(c,std::move(g)).get();});
  }
  std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan p)override{return backend.ReadBack(std::move(p));}
  std::future<Result<Ack>> CommitEviction(MigrationPlan p)override{return backend.CommitEviction(std::move(p));}
  HostCacheBackend backend;std::shared_ptr<Rendezvous> gate;
};
void Check(bool x,const char* why){if(!x)throw std::runtime_error(why);}
}
int main(){
  try{
    RuntimeOptions options;options.run_id="multi-owner";
    auto service=std::make_shared<CpuParameterService>(options),reference=std::make_shared<CpuParameterService>(options);
    auto gate=std::make_shared<Rendezvous>();
    auto a=std::make_shared<GatedBackend>(0,gate),b=std::make_shared<GatedBackend>(1,gate);
    Take(service->AttachAgent(0,a));Take(service->AttachAgent(1,b));
    DistributedClient actual({service}),baseline({reference});
    actual.SetCacheResolver([a,b](auto work,auto layouts)->Result<std::vector<TensorBlock>>{
      std::vector<TensorBlock> out;
      for(auto& task:work){auto parts=Take((task.agent==0?a:b)->GatherHost({task},layouts));
        for(auto& part:parts)out.push_back(std::move(part));}
      return out;
    });
    for(TableId table:{1,2}){
      TableDefinition definition{table,table==1?3U:5U,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},0.25f,0.25f,42};
      Take(actual.CreateTable(definition));Take(baseline.CreateTable(definition));
    }
    auto request=[&](std::uint64_t seq,bool eval){return LookupRequest{{"multi-owner",0,seq,Phase::lookup},1,eval,
      {{1,0,Pooling::mean,{1,1,2},{0,3},{3}},{2,1,Pooling::sequence,{2,1},{0,2},{2}}}};};
    auto warm=Take(actual.Lookup(request(0,true)));Take(actual.Release(warm));
    auto warm_ref=Take(baseline.Lookup(request(0,true)));Take(baseline.Release(warm_ref));
    for(AgentId agent:{0,1})Take(service->MoveRows({"multi-owner",{{1,agent+1},{2,agent+1}},agent,Placement::gpu,{1<<20,1<<20,2,0}}));
    for(std::uint64_t seq=1;seq<=3;++seq){
      auto x=Take(actual.Lookup(request(seq,false))),y=Take(baseline.Lookup(request(seq,false)));
      auto gradients=x.features();for(auto& tensor:gradients)for(auto& v:tensor.values)v=0.1f*seq;
      Take(actual.Push(x,gradients));Take(baseline.Push(y,gradients));
    }
    Check(gate->round==6,"multi-table agent control rounds changed");
    for(TableId table:{1,2})for(Key key:{1,2}){
      auto x=Take(service->ReadState(Take(service->Locations({{table,key}})))).at(0);
      auto y=Take(reference->ReadState(Take(reference->Locations({{table,key}})))).at(0);
      Check(x.row_updates==3 && x.row_updates==y.row_updates,"multi-owner row update count");
      for(std::size_t d=0;d<x.weights.size();++d)
        Check(std::abs(x.weights[d]-y.weights[d])<1e-6f && std::abs(x.first_moment[d]-y.first_moment[d])<1e-6f &&
          std::abs(x.second_moment[d]-y.second_moment[d])<1e-6f,"multi-owner optimizer trajectory mismatch");
    }
    std::cout<<"PASS concurrent independent owners and sequential per-agent multi-table contexts\n";return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL multi-agent: "<<e.what()<<'\n';return 1;}
}
