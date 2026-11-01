#include "eps/runtime.hpp"
#include "eps/distributed_client.hpp"
#include "eps/cache_resolver.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <tuple>
using namespace eps;
namespace {
void Check(bool x,const char* why){if(!x)throw std::runtime_error(why);}
template<class T>T Get(Result<T> r){if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<class T>T Get(std::future<Result<T>> r){return Get(r.get());}
auto Phases(const RuntimeStats& s){return std::tuple(s.lookup_prepare_nanoseconds,s.gradient_prepare_nanoseconds,
 s.gpu_packet_prepare_nanoseconds,s.update_dispatch_nanoseconds,s.cpu_optimizer_nanoseconds,
 s.agent_wait_nanoseconds,s.migration_nanoseconds,s.cache_maintenance_nanoseconds);}
TableDefinition Table(){return {1,2,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},.25,.25,42};}
LookupRequest Request(std::string run,unsigned seq,bool readonly=false){return {{run,0,seq,Phase::lookup},1,readonly,{{1,0,Pooling::mean,{1,1,2},{0,3},{3}}}};}
}
int main(){
 try {
  RuntimeOptions o;o.run_id="phase-cpu";auto cpu=std::make_shared<CpuParameterService>(o);Get(cpu->CreateTable(Table()));
  auto first=Get(cpu->Lookup(Request(o.run_id,0)));auto after=Get(cpu->Stats());
  Check(after.lookup_prepare_nanoseconds>0 && after.gradient_prepare_nanoseconds==0 && after.agent_wait_nanoseconds==0,"CPU lookup phase attribution changed");
  Get(cpu->Lookup(Request(o.run_id,0)));Check(Phases(Get(cpu->Stats()))==Phases(after),"duplicate Lookup repeated timed preparation");
  PushRequest push{{o.run_id,0,0,Phase::push},first.ticket,first.cpu_contributions};
  for(auto& f:push.feature_gradients)for(auto& v:f.values)v=.01f;
  Get(cpu->Push(push));after=Get(cpu->Stats());
  Check(after.gradient_prepare_nanoseconds>0 && after.cpu_optimizer_nanoseconds>0 &&
   after.gpu_packet_prepare_nanoseconds==0 && after.update_dispatch_nanoseconds==0 && after.migration_nanoseconds==0,"CPU update appeared as GPU packet work");
  Get(cpu->Push(push));Check(Phases(Get(cpu->Stats()))==Phases(after),"duplicate Push repeated timed update work");
  auto root=std::filesystem::temp_directory_path()/("eps-phases-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  o.run_id="phase-owner";o.snapshot_directory=root;auto actual=std::make_shared<CpuParameterService>(o);
  auto agent=std::make_shared<HostCacheBackend>(0,4096);Get(actual->AttachAgent(0,agent));
  DistributedClient client({actual});client.SetCacheResolver(MakeCacheResolver({{0,agent}}));Get(client.CreateTable(Table()));
  auto init=Get(client.Lookup(Request(o.run_id,0,true)));Get(client.Release(init));
  auto before=Get(actual->Stats());
  Get(actual->MoveRows({o.run_id,{{1,1}},0,Placement::gpu,{4096,4096,8,1}}));
  after=Get(actual->Stats());Check(after.migration_nanoseconds>0 && after.agent_wait_nanoseconds>before.agent_wait_nanoseconds,"migration/owner wait was not measured");
  auto batch=Get(client.Lookup(Request(o.run_id,1)));auto gradients=batch.features();for(auto& f:gradients)for(auto& v:f.values)v=.01f;
  before=Get(actual->Stats());Get(client.Push(batch,gradients));after=Get(actual->Stats());
  Check(after.gpu_packet_prepare_nanoseconds>0 && after.update_dispatch_nanoseconds>0 &&
   after.agent_wait_nanoseconds>before.agent_wait_nanoseconds && after.cpu_optimizer_nanoseconds>0,"mixed update phases were not recorded");
  before=after;Get(client.Push(batch,gradients));Check(Phases(Get(actual->Stats()))==Phases(before),"cached duplicate changed phase counters");
  CachePolicyOptions policy;policy.mode=CacheAdmissionMode::frequency;policy.agents={0};policy.window_requests=1;
  policy.minimum_accesses=1;policy.budget={4096,4096,8,1};Get(actual->ConfigureCachePolicy(policy));
  auto input=Request(o.run_id,2,true);input.features[0].keys={3,3,3};
  auto future=Get(client.Lookup(std::move(input)));Get(client.Release(future));after=Get(actual->Stats());
  Check(after.cache_maintenance_nanoseconds>0 && after.policy_admitted_rows>0 && after.policy_windows==1,"automatic maintenance phase was not recorded");
  Get(actual->DrainAndFlush("checkpoint"));Check(Get(actual->Stats()).live_tickets==0,"phase instrumentation changed ticket lifecycle");
  std::filesystem::remove_all(root);
  std::cout<<"PASS sequencer phase times: CPU/mixed ownership, migration, maintenance and duplicate exclusion\n";return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL phases: "<<e.what()<<'\n';return 1;}
}
