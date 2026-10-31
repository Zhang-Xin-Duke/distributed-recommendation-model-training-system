#include "eps/brpc_client.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/cache_resolver.hpp"
#include "eps/distributed_client.hpp"
#include "eps/runtime.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
using namespace eps;
namespace {
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<typename T>T Take(Result<T> r){if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<typename T>T Take(std::future<Result<T>> f){return Take(f.get());}
void Near(float a,float b){Check(std::isfinite(a)&&std::abs(a-b)<3e-6f,"automatic RPC numerical mismatch");}
TableDefinition Table(std::uint32_t shards){return {1,2,shards,1,{0.05f,0.9f,0.999f,1e-8f,0.9f,0.999f},-0.1f,0.1f,77};}
std::vector<Key> Keys(std::uint64_t step){return step<4?std::vector<Key>{0,1,2,3}:std::vector<Key>{4,5,6,7};}
LookupRequest Request(const std::string& run,std::uint64_t step){
  const auto keys=Keys(step);
  return {{run,0,step,Phase::lookup},1,false,{
    {1,0,Pooling::mean,keys,{0,4,4},{4,0}},
    {1,1,Pooling::sequence,{keys[0],keys[1],keys[0],keys[3]},{0,4},{4}}}};
}
std::vector<TensorBlock> Gradients(const DistributedBatch& batch){
  auto g=batch.features();for(auto& f:g)for(std::size_t n=0;n<f.values.size();++n)f.values[n]=0.01f*(n+1)*(f.feature+1);return g;
}
void ReferenceStep(DistributedClient& client,std::uint64_t step){
  auto b=Take(client.Lookup(Request("baseline",step)));Take(client.Push(b,Gradients(b)));
}
}
int main(int argc,char** argv){
  try{
    Check(argc==5,"expected two PS endpoints, agent endpoint and fresh/resumed mode");
    const bool resumed=std::string(argv[4])=="resumed";
    const std::string run=resumed?"auto-resumed":"auto-fresh";
    std::vector<std::shared_ptr<ParameterService>> remote{
      std::make_shared<BrpcParameterService>(argv[1]),std::make_shared<BrpcParameterService>(argv[2])};
    DistributedClient client(remote);Take(client.CreateTable(Table(2)));
    auto backend=std::make_shared<BrpcCacheBackend>(argv[3],run,0,0);
    const auto mode=std::getenv("EPS_COALESCE_GATHERS");
    client.SetCacheResolver(MakeCacheResolver({{0,backend}}),mode && std::string(mode)=="1");
    RuntimeOptions options;options.run_id="baseline";auto reference=std::make_shared<CpuParameterService>(options);
    DistributedClient baseline({reference});Take(baseline.CreateTable(Table(1)));
    if(resumed)for(std::uint64_t n=0;n<12;++n)ReferenceStep(baseline,n);
    const auto compare=[&](Key limit){
      for(Key key=0;key<limit;++key){
        auto connection=std::dynamic_pointer_cast<BrpcParameterService>(remote[key%2]);
        auto a=Take(connection->ReadState(Take(connection->Locations({{1,key}})))).at(0);
        auto b=Take(reference->ReadState({{{1,key},0,Placement::cpu,0,0,0}})).at(0);
        Check(a.row_updates==b.row_updates,"automatic RPC update count mismatch");
        for(int d=0;d<2;++d){Near(a.weights[d],b.weights[d]);Near(a.first_moment[d],b.first_moment[d]);Near(a.second_moment[d],b.second_moment[d]);}
      }
    };
    if(resumed)compare(8);
    const std::uint64_t begin=resumed?12:0,end=resumed?16:12;
    for(auto n=begin;n<end;++n){
      auto a=Take(client.Lookup(Request(run,n))),b=Take(baseline.Lookup(Request("baseline",n)));
      for(std::size_t f=0;f<a.features().size();++f)for(std::size_t k=0;k<a.features()[f].values.size();++k)
        Near(a.features()[f].values[k],b.features()[f].values[k]);
      auto gradients=Gradients(a);Take(client.Push(a,gradients));Take(client.Push(a,gradients));Take(baseline.Push(b,gradients));
      compare(n<4?4:8);
    }
    std::uint64_t admitted=0,evicted=0,windows=0;
    for(const auto& shard:remote){
      const auto s=Take(std::dynamic_pointer_cast<BrpcParameterService>(shard)->Stats());
      admitted+=s.policy_admitted_rows;evicted+=s.policy_evicted_rows;windows+=s.policy_windows;
      Check(s.policy_windows==(end-begin)/2 && s.policy_tracked_rows<=16 &&
        s.policy_migration_batches<=s.policy_windows*4 && s.policy_migration_bytes<=s.policy_windows*1024 &&
        s.live_tickets==0,"automatic RPC policy bound/counter mismatch");
    }
    Check(admitted>0 && (resumed || evicted>0),"external service did not execute automatic admission/replacement");
    Check(Take(backend->FreeSlots(1))>=1,"shared-agent slot reserve violated");
    if(!resumed)for(const auto& shard:remote){
      Take(shard->DrainAndFlush("checkpoint"));
      Check(Take(std::dynamic_pointer_cast<BrpcParameterService>(shard)->Stats()).cached_rows==0,"automatic save did not drain cache");
    }
    std::cout<<"PASS automatic two-shard RPC "<<(resumed?"restored optimizer trajectory":"hot-set replacement and drain")
      <<"; admitted="<<admitted<<" evicted="<<evicted<<" windows="<<windows<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL automatic cache RPC: "<<e.what()<<'\n';return 1;}
}
