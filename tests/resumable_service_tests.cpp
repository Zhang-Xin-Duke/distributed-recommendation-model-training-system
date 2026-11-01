#include "eps/runtime.hpp"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#ifdef EPS_HAS_CUDA
#include "eps/cuda_cache_backend.hpp"
#endif

using namespace eps;
using namespace std::chrono_literals;
namespace {
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class T>T Get(std::future<Result<T>> future) {
  Check(future.wait_for(5s)==std::future_status::ready,"service future did not complete");
  auto result=future.get();if(auto error=std::get_if<Error>(&result))throw std::runtime_error(error->message);
  return std::get<T>(std::move(result));
}
template<class T>Error ErrorOf(std::future<Result<T>> future) {
  Check(future.wait_for(5s)==std::future_status::ready,"service failure did not complete");
  auto result=future.get();Check(std::holds_alternative<Error>(result),"expected service failure");return std::get<Error>(result);
}
struct Gate {
  std::promise<void> entered,release;
  std::future<void> started=entered.get_future();
  std::shared_future<void> open=release.get_future().share();
  std::atomic<bool> opened{false};
  std::optional<Error> error;
  void Open(){if(!opened.exchange(true))release.set_value();}
};
struct GateHandle {
  std::shared_ptr<Gate> gate;
  ~GateHandle(){gate->Open();}
  void Started(){Check(gate->started.wait_for(5s)==std::future_status::ready,"agent dispatch did not start");}
  void Open(){gate->Open();}
};
class DelayedBackend final:public CacheBackend {
 public:
  explicit DelayedBackend(AgentId id) {
    threads_.reserve(16);
    const char* backend=std::getenv("EPS_RESUMABLE_BACKEND");
    if(backend && std::string(backend)=="cuda") {
#ifdef EPS_HAS_CUDA
      CudaCacheOptions options;options.agent=id;options.rows_per_table=32;options.state_bytes=1<<20;
      backend_=std::make_shared<CudaCacheBackend>(options);
#else
      throw std::runtime_error("resumable CUDA test requires EPS_BUILD_CUDA");
#endif
    }else backend_=std::make_shared<HostCacheBackend>(id,1<<20);
  }
  ~DelayedBackend(){for(auto& thread:threads_)thread.join();}
  GateHandle Arm(std::string method,std::optional<Error> error={}) {
    std::lock_guard lock(mutex_);Check(!gate_,"test gate already armed");
    method_=std::move(method);gate_=std::make_shared<Gate>();gate_->error=std::move(error);return {gate_};
  }
  std::future<Result<Ack>> Define(TableDefinition t) override{return backend_->Define(t);}
  std::future<Result<std::uint64_t>> FreeSlots(TableId t) override{return backend_->FreeSlots(t);}
  std::future<Result<std::vector<LocatedRow>>> Load(MigrationPlan p,std::vector<RowState> s) override {
    return Dispatch<std::vector<LocatedRow>>("load",[this,p=std::move(p),s=std::move(s)]() mutable{return backend_->Load(std::move(p),std::move(s));});
  }
  std::future<Result<std::vector<TensorBlock>>> GatherHost(std::vector<AgentWork> w,std::vector<FeatureInput> f) override {
    return backend_->GatherHost(std::move(w),std::move(f));
  }
  std::future<Result<Ack>> ApplyHost(UpdateContext c,RowGradients g) override {
    return Dispatch<Ack>("apply",[this,c,g=std::move(g)]() mutable{return backend_->ApplyHost(c,std::move(g));});
  }
  std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan p) override {
    return Dispatch<std::vector<RowState>>("readback",[this,p=std::move(p)]() mutable{return backend_->ReadBack(std::move(p));});
  }
  std::future<Result<Ack>> CommitEviction(MigrationPlan p) override{return backend_->CommitEviction(std::move(p));}
 private:
  template<class T,class Fn>std::future<Result<T>> Dispatch(const std::string& method,Fn fn) {
    std::shared_ptr<Gate> gate;
    {std::lock_guard lock(mutex_);if(gate_ && method_==method)gate=std::exchange(gate_,{});}
    if(!gate)return fn();
    auto promise=std::make_shared<std::promise<Result<T>>>();auto future=promise->get_future();
    threads_.emplace_back([gate,promise,fn=std::move(fn)]() mutable {
      gate->open.wait();
      try{promise->set_value(gate->error?Result<T>(*gate->error):fn().get());}
      catch(const std::exception& e){promise->set_value(Error{Error::Code::run_failed,e.what()});}
    });
    gate->entered.set_value();return future;
  }
  std::shared_ptr<CacheBackend> backend_;std::mutex mutex_;std::string method_;
  std::shared_ptr<Gate> gate_;std::vector<std::thread> threads_;
};
TableDefinition Table(TableId id){return {id,2,1,1,{.01f,.9f,.999f,1e-8f,.9f,.999f},.25,.25,42};}
LookupRequest Lookup(std::string run,unsigned learner,TableId table,Key key,bool read_only=true) {
  return {{run,learner,0,Phase::lookup},1,read_only,{{table,0,Pooling::mean,{key},{0,1},{1}}}};
}
PushRequest Push(std::string run,unsigned learner,std::string ticket,float gradient=.03f) {
  return {{run,learner,0,Phase::push},std::move(ticket),{{0,{1,2},{gradient,gradient}}}};
}
void Release(CpuParameterService& s,std::string run,unsigned learner,std::string ticket) {
  Get(s.Release({{run,learner,0,Phase::release},std::move(ticket)}));
}
void Seed(CpuParameterService& s,const std::string& run,TableId table,Key key,unsigned learner) {
  auto reply=Get(s.Lookup(Lookup(run,learner,table,key)));Release(s,run,learner,reply.ticket);
}
void Match(const RowState& a,const RowState& b) {
  Check(a.weights==b.weights && a.first_moment==b.first_moment && a.second_moment==b.second_moment &&
    a.row_updates==b.row_updates,"resumption changed optimizer state or update count");
}
void UpdateOrderingAndSave() {
  RuntimeOptions options;options.run_id="resumable-update";
  auto root=std::filesystem::temp_directory_path()/("eps-resumable-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  options.snapshot_directory=root;CpuParameterService service(options);
  auto backend=std::make_shared<DelayedBackend>(0);Get(service.AttachAgent(0,backend));
  Get(service.CreateTable(Table(1)));Get(service.CreateTable(Table(2)));Seed(service,options.run_id,1,1,0);
  Get(service.MoveRows({options.run_id,{{1,1}},0,Placement::gpu,{4096,4096,4,0}}));
  auto first=Get(service.Lookup(Lookup(options.run_id,1,1,1,false)));
  auto gate=backend->Arm("apply");auto update=service.Push(Push(options.run_id,1,first.ticket));gate.Started();
  auto dependent=service.Lookup(Lookup(options.run_id,3,1,1));
  auto independent=Get(service.Lookup(Lookup(options.run_id,2,1,2,false)));
  Check(dependent.wait_for(0s)!=std::future_status::ready,"dependent lookup observed a pending update");
  auto same_table=service.Push(Push(options.run_id,2,independent.ticket,.02f));
  auto other=Get(service.Lookup(Lookup(options.run_id,4,2,9,false)));Get(service.Push(Push(options.run_id,4,other.ticket)));
  Check(same_table.wait_for(0s)!=std::future_status::ready,"same-table optimizer context overtook a pending update");
  auto during=Get(service.Stats());Check(during.waiting_operations==1 && during.active_operations>=2 &&
    during.active_payload_bytes>0 && during.dependency_payload_bytes>0,"pending service state was not observable");
  auto duplicate=service.Push(Push(options.run_id,1,first.ticket));
  auto dependent_release=service.Release({{options.run_id,3,0,Phase::release},"3/0"});
  auto save=service.DrainAndFlush("checked");auto after_barrier=service.Lookup(Lookup(options.run_id,5,1,20));
  Check(save.wait_for(30ms)!=std::future_status::ready,"save crossed an outstanding update");
  gate.Open();Get(std::move(update));Get(std::move(same_table));Get(std::move(duplicate));
  Get(std::move(dependent));Get(std::move(dependent_release));Get(std::move(save));
  Check(ErrorOf(std::move(after_barrier)).code==Error::Code::unavailable,"lookup crossed the save admission boundary");
  auto after=Get(service.Stats());Check(after.updates==3 && after.duplicate_requests==1 && after.live_tickets==0 &&
    after.waiting_operations==0 && after.active_payload_bytes==0,"update retry or cleanup changed after resumption");
  RuntimeOptions baseline_options;baseline_options.run_id="resumable-reference";CpuParameterService baseline(baseline_options);
  Get(baseline.CreateTable(Table(1)));auto a=Get(baseline.Lookup(Lookup(baseline_options.run_id,1,1,1,false)));
  Get(baseline.Push(Push(baseline_options.run_id,1,a.ticket)));auto b=Get(baseline.Lookup(Lookup(baseline_options.run_id,2,1,2,false)));
  Get(baseline.Push(Push(baseline_options.run_id,2,b.ticket,.02f)));
  auto locations=Get(service.Locations({{1,1},{1,2}}));auto expected_locations=Get(baseline.Locations({{1,1},{1,2}}));
  auto actual=Get(service.ReadState(locations)),expected=Get(baseline.ReadState(expected_locations));
  Match(actual[0],expected[0]);Match(actual[1],expected[1]);std::filesystem::remove_all(root);
}

void MigrationExclusion() {
  RuntimeOptions options;options.run_id="resumable-move";CpuParameterService service(options);
  auto backend=std::make_shared<DelayedBackend>(0);Get(service.AttachAgent(0,backend));Get(service.CreateTable(Table(1)));
  Seed(service,options.run_id,1,1,0);
  auto load_gate=backend->Arm("load");auto load=service.MoveRows({options.run_id,{{1,1}},0,Placement::gpu,{4096,4096,4,0}});load_gate.Started();
  auto conflict=service.Lookup(Lookup(options.run_id,1,1,1));
  auto location=service.Locations({{1,1}});
  auto other=Get(service.Lookup(Lookup(options.run_id,2,1,2)));Release(service,options.run_id,2,other.ticket);
  Check(conflict.wait_for(0s)!=std::future_status::ready && location.wait_for(0s)!=std::future_status::ready,"load published rows before owner acknowledgement");
  load_gate.Open();auto loaded=Get(std::move(load));Check(loaded[0].placement==Placement::gpu,"load did not publish residency");
  auto read=Get(std::move(conflict));Release(service,options.run_id,1,read.ticket);
  Check(Get(std::move(location))[0].generation==loaded[0].generation,"location inspection returned an intermediate generation");
  auto write_gate=backend->Arm("readback");auto writeback=service.MoveRows({options.run_id,{{1,1}},0,Placement::cpu,{4096,4096,4,0}});write_gate.Started();
  auto pending=service.Lookup(Lookup(options.run_id,3,1,1));
  auto independent=Get(service.Lookup(Lookup(options.run_id,4,1,2)));Release(service,options.run_id,4,independent.ticket);
  Check(pending.wait_for(0s)!=std::future_status::ready,"writeback released row exclusion early");
  write_gate.Open();auto moved=Get(std::move(writeback));Check(moved[0].placement==Placement::cpu &&
    moved[0].generation==loaded[0].generation+1,"writeback publication changed generation");
  auto result=Get(std::move(pending));Check(result.agent_work.empty() && result.cpu_contributions[0].values==std::vector<float>({.25f,.25f}),"lookup observed stale GPU placement or CPU weights");
  Release(service,options.run_id,3,result.ticket);
}

void FailureFencesOtherContinuations() {
  RuntimeOptions options;options.run_id="resumable-failure";CpuParameterService service(options);
  auto a=std::make_shared<DelayedBackend>(0),b=std::make_shared<DelayedBackend>(1);
  Get(service.AttachAgent(0,a));Get(service.AttachAgent(1,b));Get(service.CreateTable(Table(1)));Get(service.CreateTable(Table(2)));
  Seed(service,options.run_id,1,1,0);Seed(service,options.run_id,2,2,1);
  Get(service.MoveRows({options.run_id,{{1,1}},0,Placement::gpu,{4096,4096,4,0}}));
  Get(service.MoveRows({options.run_id,{{2,2}},1,Placement::gpu,{4096,4096,4,0}}));
  auto x=Get(service.Lookup(Lookup(options.run_id,2,1,1,false))),y=Get(service.Lookup(Lookup(options.run_id,3,2,2,false)));
  auto failed=a->Arm("apply",Error{Error::Code::unavailable,"injected owner loss"}),held=b->Arm("apply");
  auto first=service.Push(Push(options.run_id,2,x.ticket)),second=service.Push(Push(options.run_id,3,y.ticket));failed.Started();held.Started();
  failed.Open();Check(ErrorOf(std::move(first)).code==Error::Code::run_failed,"ambiguous owner failure did not fence the run");
  held.Open();Check(ErrorOf(std::move(second)).code==Error::Code::run_failed,"another continuation acknowledged after the run was fenced");
  Check(Get(service.Stats()).updates==0,"failed updates became committed lifecycle completions");
  Release(service,options.run_id,2,x.ticket);Release(service,options.run_id,3,y.ticket);
  Check(Get(service.Stats()).live_tickets==0,"known tickets could not be released after fencing");
}
void AutomaticPolicyKeepsReadProgress() {
  RuntimeOptions options;options.run_id="resumable-automatic";CpuParameterService service(options);
  auto backend=std::make_shared<DelayedBackend>(0);Get(service.AttachAgent(0,backend));Get(service.CreateTable(Table(1)));
  Seed(service,options.run_id,1,1,0);Get(service.MoveRows({options.run_id,{{1,1}},0,Placement::gpu,{4096,4096,4,0}}));
  CachePolicyOptions policy;policy.mode=CacheAdmissionMode::frequency;policy.agents={0};policy.window_requests=1;
  policy.minimum_accesses=1000;policy.budget={4096,4096,4,0};Get(service.ConfigureCachePolicy(policy));
  auto first=Get(service.Lookup(Lookup(options.run_id,1,1,1,false)));
  auto gate=backend->Arm("apply");auto update=service.Push(Push(options.run_id,1,first.ticket));gate.Started();
  auto unrelated=Get(service.Lookup(Lookup(options.run_id,2,1,9)));
  Check(update.wait_for(0s)!=std::future_status::ready,"delayed automatic-cache update finished before its owner");
  gate.Open();Get(std::move(update));Release(service,options.run_id,2,unrelated.ticket);
  auto stats=Get(service.Stats());Check(stats.updates==1 && stats.policy_windows==2 &&
    stats.policy_failures==0 && stats.live_tickets==0,"exclusive policy window changed request completion or cleanup");
}
void OwnershipInvalidatesQueuedPlan() {
  RuntimeOptions options;options.run_id="resumable-plan-epoch";CpuParameterService service(options);
  auto backend=std::make_shared<DelayedBackend>(0);Get(service.AttachAgent(0,backend));Get(service.CreateTable(Table(1)));
  Seed(service,options.run_id,1,1,0);Seed(service,options.run_id,1,3,1);
  Get(service.MoveRows({options.run_id,{{1,3}},0,Placement::gpu,{4096,4096,4,0}}));
  auto training=Get(service.Lookup(Lookup(options.run_id,2,1,3,false)));
  auto load_gate=backend->Arm("load");auto move=service.MoveRows({options.run_id,{{1,1}},0,Placement::gpu,{4096,4096,4,0}});load_gate.Started();
  // Queue a read for the location this in-flight move will publish. Its first
  // plan sees CPU ownership; after load it must acquire the new agent stream.
  auto read=service.ReadState({{{1,1},0,Placement::gpu,0,1,0}});
  auto read_gate=backend->Arm("readback");auto update=service.Push(Push(options.run_id,2,training.ticket));
  auto marker=Get(service.Lookup(Lookup(options.run_id,3,1,8)));Release(service,options.run_id,3,marker.ticket);
  load_gate.Open();Get(std::move(move));read_gate.Started();
  Check(update.wait_for(30ms)!=std::future_status::ready,"queued plan failed to acquire the newly published agent control stream");
  read_gate.Open();auto states=Get(std::move(read));Get(std::move(update));
  Check(states.size()==1 && states[0].weights==std::vector<float>({.25f,.25f}),"readback changed after dependency invalidation");
  auto stats=Get(service.Stats());Check(stats.dependency_rebuilds>0 && stats.dependency_prepare_nanoseconds>0 &&
    stats.dependency_payload_bytes==0,"service did not expose/reclaim dependency plans");
}
void DestructionDrainsAcceptedContinuations() {
  RuntimeOptions options;options.run_id="resumable-shutdown";
  auto root=std::filesystem::temp_directory_path()/("eps-shutdown-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  options.snapshot_directory=root;
  auto service=std::make_unique<CpuParameterService>(options);
  auto backend=std::make_shared<DelayedBackend>(0);Get(service->AttachAgent(0,backend));
  Get(service->CreateTable(Table(1)));Seed(*service,options.run_id,1,1,0);
  Get(service->MoveRows({options.run_id,{{1,1}},0,Placement::gpu,{4096,4096,4,0}}));
  auto request=Get(service->Lookup(Lookup(options.run_id,1,1,1,false)));
  auto gate=backend->Arm("apply");auto update=service->Push(Push(options.run_id,1,request.ticket));gate.Started();
  auto checkpoint=service->DrainAndFlush("accepted-before-shutdown");
  std::promise<void> starting,finished;auto begun=starting.get_future(),closed=finished.get_future();
  std::thread closer([&]{starting.set_value();service.reset();finished.set_value();});
  begun.wait();const bool premature=closed.wait_for(30ms)==std::future_status::ready;
  gate.Open();closer.join();
  Check(!premature,"service destroyed an accepted suspended update");
  Get(std::move(update));auto saved=Get(std::move(checkpoint));
  Check(saved.state_files.size()==1 && std::filesystem::exists(saved.state_files.front()),"shutdown dropped an accepted checkpoint");
  std::filesystem::remove_all(root);
}
}
int main() {
  try{UpdateOrderingAndSave();MigrationExclusion();FailureFencesOtherContinuations();AutomaticPolicyKeepsReadProgress();OwnershipInvalidatesQueuedPlan();DestructionDrainsAcceptedContinuations();
    std::cout<<"PASS resumable service: independent progress, ordered Adam, migration publication, save barrier, retry and failure fencing\n";return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL resumable service: "<<e.what()<<'\n';return 1;}
}
