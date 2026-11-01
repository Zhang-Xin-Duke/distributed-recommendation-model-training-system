#include "eps/cache_backend.hpp"
#include "eps/distributed_client.hpp"
#include "eps/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace eps;
namespace {
void Check(bool b,const char* m){if(!b)throw std::runtime_error(m);}
void Near(float a,float b){Check(std::isfinite(a)&&std::abs(a-b)<3e-6f,"bounded optimizer state mismatch");}
template<class T>T Take(Result<T> r){if(auto* e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<class T>T Take(std::future<Result<T>> f){return Take(f.get());}
RowState State(CpuParameterService& s){return Take(s.ReadState(Take(s.Locations({{1,1}})))).at(0);}
void Compare(const RowState& a,const RowState& b){Check(a.row_updates==b.row_updates,"update counter mismatch");for(auto pair:{std::pair{&a.weights,&b.weights},std::pair{&a.first_moment,&b.first_moment},std::pair{&a.second_moment,&b.second_moment}})for(std::size_t i=0;i<pair.first->size();++i)Near(pair.first->at(i),pair.second->at(i));}
}
int main(int argc,char** argv){
 char path[]="/tmp/eps-bounds-XXXXXX";auto* dir=::mkdtemp(path);if(!dir)return 1;
 try{
  RuntimeOptions options;options.run_id="bounds";options.snapshot_directory=dir;
  auto actual=std::make_shared<CpuParameterService>(options);
  auto baseline=std::make_shared<CpuParameterService>(options);
  auto cache=std::make_shared<HostCacheBackend>(0,1<<20);Take(actual->AttachAgent(0,cache));
  DistributedClient client({actual}),reference({baseline});
  client.SetCacheResolver([cache](auto w,auto f){return cache->GatherHost(std::move(w),std::move(f)).get();});
  TableDefinition table{1,2,1,1,{1.f,0.9f,0.999f,1e-8f,0.9f,0.999f},0.5f,0.5f,42};
  table.optimizer.weight_bounds=WeightBounds{-0.1f,0.2f};
  Take(client.CreateTable(table));Take(reference.CreateTable(table));
  auto request=[](std::uint64_t seq){return LookupRequest{{"bounds",0,seq,Phase::lookup},1,false,{{1,0,Pooling::sequence,{1},{0,1},{1}}}};};
  auto init=Take(client.Lookup(request(0))),init_ref=Take(reference.Lookup(request(0)));
  Near(init.features()[0].values[0],0.2f);Take(client.Release(init));Take(reference.Release(init_ref));
  Take(actual->MoveRows({"bounds",{{1,1}},0,Placement::gpu,{1<<20,1<<20,1,0}}));
  float m=0,v=0,w=0.2f,b1=0.9f,b2=0.999f;
  for(std::uint64_t seq=1;seq<=6;++seq){
   float g=seq<=2?10.f:-10.f;
   m=0.9f*m+(1-0.9f)*g;v=0.999f*v+(1-0.999f)*g*g;
   w=std::clamp(w-std::sqrt(1-b2)/(1-b1)*m/(std::sqrt(v)+1e-8f),-0.1f,0.2f);
   auto a=Take(client.Lookup(request(seq))),b=Take(reference.Lookup(request(seq)));auto grad=a.features();std::fill(grad[0].values.begin(),grad[0].values.end(),g);
   Take(client.Push(a,grad));Take(reference.Push(b,grad));auto state=State(*actual);Compare(state,State(*baseline));Near(state.weights[0],w);Near(state.first_moment[0],m);Near(state.second_moment[0],v);
   b1*=0.9f;b2*=0.999f;
  }
  auto bad=table;bad.id=7;bad.optimizer.weight_bounds=WeightBounds{1,-1};
  auto invalid=actual->CreateTable(bad).get();
  Check(std::holds_alternative<Error>(invalid) && std::get<Error>(invalid).message=="invalid optimizer weight bounds","reversed CPU bounds accepted");
  Check(std::holds_alternative<Error>(cache->Define(bad).get()),"cache accepted reversed bounds");
  auto saved=Take(actual->DrainAndFlush("bounded"));
  RuntimeOptions next;next.run_id="resumed-bounds";CpuParameterService resumed(next);Take(resumed.Restore(saved.state_files[0]));Compare(State(resumed),State(*baseline));
  Take(resumed.CreateTable(table)); // Recreate must match the restored immutable bounds.
  Check(argc==2,"legacy fixture path required");
  RuntimeOptions legacy_options;legacy_options.run_id="legacy-restored";
  CpuParameterService legacy(legacy_options);Take(legacy.Restore(argv[1]));
  auto legacy_table=table;legacy_table.optimizer.weight_bounds.reset();legacy_table.optimizer.learning_rate=0.1f;legacy_table.initialization_min=legacy_table.initialization_max=0.25f;
  Take(legacy.CreateTable(legacy_table));Near(State(legacy).weights[0],0.25f);
  std::filesystem::remove_all(dir);std::cout<<"PASS bounded initialization, Adam moments, mixed-owner updates and snapshot metadata\n";
 }catch(const std::exception& e){std::filesystem::remove_all(dir);std::cerr<<"FAIL bounds: "<<e.what()<<'\n';return 1;}
}
