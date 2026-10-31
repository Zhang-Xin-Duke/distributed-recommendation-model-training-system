#include "eps/cache_backend.hpp"
#include "eps/distributed_client.hpp"
#include "eps/runtime.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace eps;
namespace {
template<typename T>T Take(Result<T> r){
  if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));
}
template<typename T>T Take(std::future<Result<T>> f){return Take(f.get());}
template<typename T>Error Rejected(std::future<Result<T>> f){
  auto r=f.get();if(auto e=std::get_if<Error>(&r))return *e;throw std::runtime_error("expected rejection");
}
void Check(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
void Near(float a,float b){Check(std::isfinite(a)&&std::abs(a-b)<3e-6f,"cached numeric mismatch");}
TableDefinition Table(){return {1,2,1,1,{0.1f,0.9f,0.999f,1e-8f,0.9f,0.999f},0.25f,0.25f,42};}
RuntimeOptions Options(){RuntimeOptions o;o.run_id="cache-test";return o;}
LookupRequest Request(std::uint64_t seq,bool eval=false){
  return {{"cache-test",0,seq,Phase::lookup},1,eval,{
    {1,0,Pooling::mean,{1,1,2,3},{0,3,4},{3,1}},
    {1,1,Pooling::sequence,{3,2,1},{0,2,3},{2,1}}}};
}
MoveRowsRequest Move(std::vector<RowKey> rows,Placement destination){
  return {"cache-test",std::move(rows),0,destination,{1<<20,1<<20,8,0}};
}
RowState State(CpuParameterService& service,Key key){
  auto loc=Take(service.Locations({{1,key}}));return Take(service.ReadState(loc)).at(0);
}
void Compare(CpuParameterService& a,CpuParameterService& b){
  for(Key key=1;key<=3;++key){auto x=State(a,key),y=State(b,key);Check(x.row_updates==y.row_updates,"cached update count");
    for(std::size_t i=0;i<x.weights.size();++i){Near(x.weights[i],y.weights[i]);Near(x.first_moment[i],y.first_moment[i]);Near(x.second_moment[i],y.second_moment[i]);}}
}
}
int main(){
  char path[]="/tmp/eps-cache-XXXXXX";auto* directory=::mkdtemp(path);
  if(!directory)return 1;
  try{
    auto options=Options();options.snapshot_directory=directory;
    auto actual=std::make_shared<CpuParameterService>(options);
    auto reference=std::make_shared<CpuParameterService>(Options());
    auto backend=std::make_shared<HostCacheBackend>(0,1024);
    Take(actual->AttachAgent(0,backend));
    Take(actual->ConfigureAccess(0,AccessRoute::transfer_unique));
    DistributedClient cached({actual}),baseline({reference});
    cached.SetCacheResolver([backend](auto work,auto layouts){return backend->GatherHost(std::move(work),std::move(layouts)).get();});
    Take(cached.CreateTable(Table()));Take(baseline.CreateTable(Table()));
    auto init=Take(cached.Lookup(Request(0,true)));Take(cached.Release(init));
    auto init_ref=Take(baseline.Lookup(Request(0,true)));Take(baseline.Release(init_ref));
    auto loaded=Take(actual->MoveRows(Move({{1,1},{1,3}},Placement::gpu)));
    Check(loaded.size()==2&&loaded[0].generation==1&&loaded[0].placement==Placement::gpu,"cache admission publication");
    for(std::uint64_t seq=1;seq<=5;++seq){
      auto a=Take(cached.Lookup(Request(seq))),b=Take(baseline.Lookup(Request(seq)));
      auto gradients=a.features();
      for(std::size_t i=0;i<a.features().size();++i){
        for(std::size_t j=0;j<a.features()[i].values.size();++j){
          Near(a.features()[i].values[j],b.features()[i].values[j]);gradients[i].values[j]=0.5f*(j+1);
        }
      }
      Take(cached.Push(a,gradients));Take(cached.Push(a,gradients));Take(baseline.Push(b,gradients));Compare(*actual,*reference);
    }
    auto held=Take(actual->Lookup(Request(6,true)));
    Check(Rejected(actual->MoveRows(Move({{1,1}},Placement::cpu))).code==Error::Code::unavailable,"evicted referenced row");
    Take(actual->Release({{"cache-test",0,6,Phase::release},held.ticket}));
    auto retired_work=held.agent_work;
    Take(actual->MoveRows(Move({{1,1},{1,3}},Placement::cpu)));Compare(*actual,*reference);
    Check(Rejected(backend->GatherHost(retired_work,Request(6,true).features)).code==Error::Code::stale_generation,
          "old cache location remained usable");
    auto again=Take(actual->MoveRows(Move({{1,1}},Placement::gpu)));
    Check(again[0].generation==3,"placement generation did not advance");
    auto saved=Take(actual->DrainAndFlush("checkpoint"));
    Check(Take(actual->Stats()).cached_rows==0,"snapshot did not drain cache");
    auto resumed_options=Options();resumed_options.run_id="resumed-cache";
    CpuParameterService resumed(resumed_options);Take(resumed.Restore(saved.state_files[0]));Compare(resumed,*reference);
    Check(Take(backend->FreeSlots(1))==32,"cache capacity not reclaimed after writeback");
    std::filesystem::remove_all(directory);
    std::cout<<"PASS host-reference cache lifecycle, mixed updates, row pins, generations and drain/restore\n";
    return 0;
  }catch(const std::exception& e){std::filesystem::remove_all(directory);std::cerr<<"FAIL cache: "<<e.what()<<'\n';return 1;}
}
