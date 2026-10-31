#include "eps/cache_policy.hpp"
#include "eps/cache_resolver.hpp"
#include "eps/distributed_client.hpp"
#include "eps/runtime.hpp"
#include "cache/reuse_policy.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unistd.h>

using namespace eps;
namespace {
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<typename T>T Take(Result<T> r){
  if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
template<typename T>T Take(std::future<Result<T>> r){return Take(r.get());}
void Near(float a,float b){Check(std::isfinite(a)&&std::abs(a-b)<3e-6f,"automatic-cache numeric mismatch");}
class FaultBackend final:public CacheBackend {
 public:
  explicit FaultBackend(std::shared_ptr<HostCacheBackend> backend):backend_(std::move(backend)){}
  bool fail_load=false;
  std::future<Result<Ack>> Define(TableDefinition t)override{return backend_->Define(t);}
  std::future<Result<std::uint64_t>> FreeSlots(TableId t)override{return backend_->FreeSlots(t);}
  std::future<Result<std::vector<LocatedRow>>> Load(MigrationPlan p,std::vector<RowState> s)override{
    if(fail_load){fail_load=false;std::promise<Result<std::vector<LocatedRow>>> result;
      result.set_value(Error{Error::Code::capacity_exceeded,"injected full agent"});return result.get_future();}
    return backend_->Load(std::move(p),std::move(s));
  }
  std::future<Result<std::vector<TensorBlock>>> GatherHost(std::vector<AgentWork> w,std::vector<FeatureInput> f)override{
    return backend_->GatherHost(std::move(w),std::move(f));}
  std::future<Result<Ack>> ApplyHost(UpdateContext c,RowGradients g)override{return backend_->ApplyHost(c,std::move(g));}
  std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan p)override{return backend_->ReadBack(std::move(p));}
  std::future<Result<Ack>> CommitEviction(MigrationPlan p)override{return backend_->CommitEviction(std::move(p));}
 private:std::shared_ptr<HostCacheBackend> backend_;
};
TableDefinition Table(){return {1,2,1,1,{0.1f,0.9f,0.999f,1e-8f,0.9f,0.999f},-0.1f,0.1f,42};}
CachePolicyOptions Policy(){
  CachePolicyOptions o;o.mode=CacheAdmissionMode::reuse_cost;o.agents={0};
  o.window_requests=2;o.minimum_accesses=2;o.minimum_residence_windows=2;
  o.saved_microseconds_per_access=100;o.load_microseconds_per_byte=0;
  o.writeback_microseconds_per_byte=0;o.learn_migration_costs=false;
  o.budget={1024,1024,4,1};return o;
}
LookupRequest Request(const std::string& run,std::uint64_t seq,const std::vector<Key>& keys,bool eval=false){
  return {{run,0,seq,Phase::lookup},1,eval,{
    {1,0,Pooling::mean,keys,{0,keys.size(),keys.size()},{keys.size(),0}},
    {1,1,Pooling::sequence,keys,{0,keys.size()},{keys.size()}}}};
}
struct Harness {
  std::string run="policy";
  std::shared_ptr<CpuParameterService> actual,reference;
  std::unique_ptr<DistributedClient> client,baseline;
  std::map<AgentId,std::shared_ptr<CacheBackend>> agents;
  std::set<Key> known;
  std::uint64_t baseline_sequence=0;
  Harness(std::uint64_t bytes=96,bool fault=false,const std::vector<AgentId>& ids={0},
          const std::filesystem::path& directory=std::filesystem::temp_directory_path()) {
    RuntimeOptions o;o.run_id=run;o.snapshot_directory=directory;actual=std::make_shared<CpuParameterService>(o);
    o.run_id="baseline";reference=std::make_shared<CpuParameterService>(o);
    for(auto id:ids){
      auto host=std::make_shared<HostCacheBackend>(id,bytes);
      agents[id]=fault?std::shared_ptr<CacheBackend>(std::make_shared<FaultBackend>(host)):host;
      Take(actual->AttachAgent(id,agents.at(id)));
    }
    client=std::make_unique<DistributedClient>(std::vector<std::shared_ptr<ParameterService>>{actual});
    client->SetCacheResolver(MakeCacheResolver(agents));
    baseline=std::make_unique<DistributedClient>(std::vector<std::shared_ptr<ParameterService>>{reference});
    Take(client->CreateTable(Table()));Take(baseline->CreateTable(Table()));
  }
  void Compare(){
    for(auto key:known){
      auto a=Take(actual->ReadState(Take(actual->Locations({{1,key}})))).at(0);
      auto b=Take(reference->ReadState(Take(reference->Locations({{1,key}})))).at(0);
      Check(a.row_updates==b.row_updates,"automatic cache changed row update count");
      for(std::size_t d=0;d<a.weights.size();++d){Near(a.weights[d],b.weights[d]);
        Near(a.first_moment[d],b.first_moment[d]);Near(a.second_moment[d],b.second_moment[d]);}
    }
  }
  void Train(std::uint64_t seq,const std::vector<Key>& keys,bool compare=true){
    known.insert(keys.begin(),keys.end());
    auto a=Take(client->Lookup(Request(run,seq,keys)));
    auto b=Take(baseline->Lookup(Request("baseline",baseline_sequence++,keys)));
    auto gradients=a.features();
    for(std::size_t f=0;f<gradients.size();++f)for(std::size_t n=0;n<gradients[f].values.size();++n){
      Near(a.features()[f].values[n],b.features()[f].values[n]);gradients[f].values[n]=0.1f*(n+1)*(f+1);
    }
    Take(client->Push(a,gradients));Take(baseline->Push(b,gradients));
    const auto before=Take(actual->Stats());Take(client->Push(a,gradients));
    const auto after=Take(actual->Stats());
    Check(before.policy_windows==after.policy_windows && before.policy_migration_batches==after.policy_migration_batches,
      "duplicate Push advanced cache policy");
    if(compare)Compare();
  }
  LocatedRow Location(Key key){return Take(actual->Locations({{1,key}})).at(0);}
};
void PolicyMath(){
  auto o=Policy();o.max_tracked_rows=3;o.window_requests=1;o.minimum_accesses=1;
  detail::ReusePolicy p(o);
  p.Observe({{1,1},{1,1},{1,1}});p.CompleteRequest();
  Check(p.ExpectedAccesses({1,1})==1,"duplicate positions inflated reuse");
  p.CompleteRequest();Check(p.ExpectedAccesses({1,1})==0.5,"statistics did not decay");
  p.Observe({{1,2},{1,3},{1,4}});Check(p.tracked_rows()==3 && p.ExpectedAccesses({1,1})==0,"unbounded history");
  auto resident=LocatedRow{{1,2},0,Placement::gpu,0,0,1};p.Admitted(resident);
  Check(!p.CanReplace(resident),"minimum residence ignored");p.CompleteRequest();p.CompleteRequest();
  Check(p.CanReplace(resident),"expired residence blocked replacement");
  ++resident.generation;p.SynchronizeResidents({resident});Check(!p.CanReplace(resident),"new generation inherited old age");
  o.learn_migration_costs=true;o.cost_average_alpha=1;o.minimum_accesses=1;
  detail::ReusePolicy cost(o);cost.Observe({{1,9}});cost.CompleteRequest();
  Check(cost.AdmissionValue({1,9},32)>0,"positive reuse rejected");
  cost.ObserveMigration(Placement::gpu,32,1000);
  Check(cost.AdmissionValue({1,9},32)==0,"migration measurement ignored");
  o.window_decay=std::numeric_limits<double>::quiet_NaN();bool rejected=false;
  try{detail::ReusePolicy invalid(o);}catch(const std::exception&){rejected=true;}
  Check(rejected,"nonfinite policy accepted");
}
void HotSetAndRestore(const std::filesystem::path& directory){
  Harness h(96,false,{0},directory);auto o=Policy();Take(h.actual->ConfigureCachePolicy(o));
  h.Train(0,{1,1,2});h.Train(1,{1,1,2});
  Check(h.Location(1).placement==Placement::gpu && h.Location(2).placement==Placement::gpu,"hot rows not admitted");
  Check(Take(h.agents.at(0)->FreeSlots(1))==1,"reserved slot consumed");
  const auto old=h.Location(1);
  auto held=Take(h.client->Lookup(Request(h.run,2,{1},true)));
  for(std::uint64_t i=3;i<=6;++i)h.Train(i,{3,3,4},false);
  Check(h.Location(1).placement==Placement::gpu && h.Location(1).generation==old.generation,"pinned row replaced");
  Take(h.client->Release(held));h.Train(7,{3,4});
  for(std::uint64_t i=8;i<12;++i)h.Train(i,{3,3,4});
  Check(h.Location(1).placement==Placement::cpu && h.Location(2).placement==Placement::cpu,"cold rows retained forever");
  Check(h.Location(3).placement==Placement::gpu && h.Location(4).placement==Placement::gpu,"new hot set not admitted");
  const auto stats=Take(h.actual->Stats());
  Check(stats.policy_admitted_rows>=4 && stats.policy_evicted_rows>=2 && stats.policy_pinned_skips>0,"missing policy evidence");
  Check(stats.policy_failures==0 && stats.policy_migration_batches<=stats.policy_windows*o.budget.max_inflight_batches &&
    stats.policy_migration_bytes<=stats.policy_windows*o.budget.network_bytes,"window budgets exceeded");
  auto saved=Take(h.actual->DrainAndFlush("automatic-policy-test-"+std::to_string(::getpid())));
  Check(Take(h.actual->Stats()).cached_rows==0,"save failed to drain automatic cache");
  h.Compare();
  RuntimeOptions options;h.run="policy-restored";options.run_id=h.run;options.snapshot_directory=directory;
  h.actual=std::make_shared<CpuParameterService>(options);Take(h.actual->Restore(saved.state_files.at(0)));
  auto backend=std::make_shared<HostCacheBackend>(0,96);h.agents={{0,backend}};
  Take(h.actual->AttachAgent(0,backend));Take(h.actual->ConfigureCachePolicy(o));
  h.client=std::make_unique<DistributedClient>(std::vector<std::shared_ptr<ParameterService>>{h.actual});
  h.client->SetCacheResolver(MakeCacheResolver(h.agents));Take(h.client->CreateTable(Table()));
  h.Compare();h.Train(0,{1,3,3,4});h.Train(1,{1,3,4});h.Compare();
  std::filesystem::remove_all(std::filesystem::path(saved.state_files[0]).parent_path());
}
void BoundsAndFailures(){
  Harness bounded(1024);auto o=Policy();o.mode=CacheAdmissionMode::frequency;
  o.window_requests=1;o.minimum_accesses=1;o.max_tracked_rows=3;o.budget={128,31,4,0};
  Take(bounded.actual->ConfigureCachePolicy(o));bounded.Train(0,{1,2,3,4,5,6});
  auto stats=Take(bounded.actual->Stats());
  Check(stats.policy_tracked_rows==3 && stats.cached_rows==0 && stats.policy_budget_skips>0,"tracking/byte bounds ignored");
  o.agents={99};Check(std::holds_alternative<Error>(bounded.actual->ConfigureCachePolicy(o).get()),"unknown agent configured");
  bounded.Train(1,{1,2});Check(Take(bounded.actual->Stats()).cached_rows==0,"failed configuration replaced old policy");

  Harness failure(1024,true);o=Policy();o.mode=CacheAdmissionMode::frequency;o.window_requests=1;
  o.minimum_accesses=1;o.budget.reserve_slots=0;Take(failure.actual->ConfigureCachePolicy(o));
  auto fault=std::dynamic_pointer_cast<FaultBackend>(failure.agents.at(0));fault->fail_load=true;
  failure.Train(0,{1,1});Check(Take(failure.actual->Stats()).cached_rows==0 &&
    Take(failure.actual->Stats()).policy_failures==1,"admission failure did not preserve CPU fallback");
  failure.Train(1,{1,1});Check(failure.Location(1).placement==Placement::gpu,"admission did not retry later");

  Harness replaced(64,true);o.budget.reserve_slots=1;o.minimum_residence_windows=0;
  Take(replaced.actual->ConfigureCachePolicy(o));replaced.Train(0,{1});
  auto full=std::dynamic_pointer_cast<FaultBackend>(replaced.agents.at(0));full->fail_load=true;
  replaced.Train(1,{2});Check(replaced.Location(1).placement==Placement::cpu &&
    replaced.Location(2).placement==Placement::cpu,"failed replacement lost CPU ownership");
  Check(Take(replaced.actual->Stats()).policy_evicted_rows==1 &&
    Take(replaced.actual->Stats()).policy_failures==1,"replacement failure was not accounted");
  replaced.Train(2,{2});Check(replaced.Location(2).placement==Placement::gpu,"failed replacement did not recover");

  Harness duplicate(1024);o=Policy();o.mode=CacheAdmissionMode::frequency;o.window_requests=1;
  o.minimum_accesses=2;o.window_decay=1;o.budget.reserve_slots=0;
  Take(duplicate.actual->ConfigureCachePolicy(o));
  auto lookup=Request(duplicate.run,0,std::vector<Key>(100,7),true);
  auto a=Take(duplicate.actual->Lookup(lookup));Take(duplicate.actual->Lookup(lookup));
  Take(duplicate.actual->Release({{duplicate.run,0,0,Phase::release},a.ticket}));
  Check(duplicate.Location(7).placement==Placement::cpu,"occurrences/retry counted as repeated requests");
  auto b=Take(duplicate.actual->Lookup(Request(duplicate.run,1,{7},true)));
  Take(duplicate.actual->Release({{duplicate.run,0,1,Phase::release},b.ticket}));
  Check(duplicate.Location(7).placement==Placement::gpu,"read-only completion did not trigger policy");
}
void MultiAgentBudget(){
  Harness h(160,false,{3,9});auto o=Policy();o.mode=CacheAdmissionMode::frequency;o.agents={3,9};
  o.window_requests=1;o.minimum_accesses=1;o.budget={128,256,1,0};
  Take(h.actual->ConfigureCachePolicy(o));h.Train(0,{0,1,2,3,4,5,6,7});
  auto stats=Take(h.actual->Stats());
  Check(stats.policy_admitted_rows==4 && stats.policy_migration_batches==1 && stats.policy_migration_bytes==128,
    "budget not shared across agents");
  Check(h.Location(0).agent==3 && h.Location(0).placement==Placement::gpu && h.Location(1).placement==Placement::cpu,
    "agent routing mismatch");
  h.Train(1,{0,1,2,3,4,5,6,7});
  Check(h.Location(1).agent==9 && h.Location(1).placement==Placement::gpu,"second agent not admitted next window");
  Check(Take(h.actual->Stats()).cached_rows==8,"distributed policy lost cached rows");
}
void AtomicReserve(){
  HostCacheBackend backend(0,96);Take(backend.Define(Table()));
  const auto load=[&](Key first){
    MigrationPlan plan{"reserve",0,{},Placement::gpu,64,1};std::vector<RowState> rows;
    for(Key key=first;key<first+2;++key){
      plan.rows.push_back({{1,key},0,Placement::loading,0,0,1});
      rows.push_back({{1,key},{0.1f,0.2f},{0,0},{0,0},0});
    }
    return backend.Load(std::move(plan),std::move(rows));
  };
  auto first=load(1),second=load(3);Take(std::move(first));auto result=second.get();
  Check(std::holds_alternative<Error>(result) && std::get<Error>(result).code==Error::Code::capacity_exceeded &&
    Take(backend.FreeSlots(1))==1,"concurrent load consumed reserved slot");
}
void MultiTableBudget(){
  RuntimeOptions options;options.run_id="tables";
  auto actual=std::make_shared<CpuParameterService>(options);
  options.run_id="tables-ref";auto reference=std::make_shared<CpuParameterService>(options);
  auto backend=std::make_shared<HostCacheBackend>(0,512);Take(actual->AttachAgent(0,backend));
  DistributedClient client({actual}),baseline({reference});
  client.SetCacheResolver(MakeCacheResolver({{0,backend}}));
  auto second=Table();second.id=2;second.dimension=4;
  for(auto table:{Table(),second}){Take(client.CreateTable(table));Take(baseline.CreateTable(table));}
  auto o=Policy();o.mode=CacheAdmissionMode::frequency;o.window_requests=1;o.minimum_accesses=1;
  o.budget={96,96,2,0};Take(actual->ConfigureCachePolicy(o));
  for(std::uint64_t step=0;step<2;++step){
    LookupRequest q{{"tables",0,step,Phase::lookup},1,false,{
      {1,0,Pooling::sum,{1,2},{0,2},{2}},{2,1,Pooling::sum,{3},{0,1},{1}}}};
    auto a=Take(client.Lookup(q));q.id.run="tables-ref";auto b=Take(baseline.Lookup(q));
    auto gradients=a.features();for(auto& f:gradients)for(auto& v:f.values)v=0.5f;
    Take(client.Push(a,gradients));Take(baseline.Push(b,gradients));
    if(!step){const auto s=Take(actual->Stats());Check(s.policy_admitted_rows==2 &&
      s.policy_migration_bytes==64 && s.policy_budget_skips>0,"budget not shared across table layouts");}
    for(auto row:std::vector<RowKey>{{1,1},{1,2},{2,3}}){
      auto x=Take(actual->ReadState(Take(actual->Locations({row})))).at(0);
      auto y=Take(reference->ReadState(Take(reference->Locations({row})))).at(0);
      Check(x.row_updates==y.row_updates,"multi-table policy changed update count");
      for(std::size_t d=0;d<x.weights.size();++d){Near(x.weights[d],y.weights[d]);
        Near(x.first_moment[d],y.first_moment[d]);Near(x.second_moment[d],y.second_moment[d]);}
    }
  }
  Check(Take(actual->Stats()).policy_migration_bytes==120 &&
    Take(actual->Locations({{2,3}})).at(0).placement==Placement::gpu,"different layout was not admitted next window");
}
}
int main(){
  char path[]="/tmp/eps-auto-cache-XXXXXX";const auto* directory=::mkdtemp(path);
  if(!directory)return 1;
  try{PolicyMath();HotSetAndRestore(directory);BoundsAndFailures();MultiAgentBudget();MultiTableBudget();AtomicReserve();
    std::filesystem::remove_all(directory);
    std::cout<<"PASS automatic cache hot-set replacement, pins, budgets, bounded history, retries, fallback and optimizer restore\n";
    return 0;
  }catch(const std::exception& e){std::filesystem::remove_all(directory);std::cerr<<"FAIL automatic cache: "<<e.what()<<'\n';return 1;}
}
