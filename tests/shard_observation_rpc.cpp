#include "eps/brpc_client.hpp"
#include "transport/brpc_server.hpp"
#include "transport/completion_executor.hpp"
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
using namespace eps;
using namespace std::chrono_literals;
void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class T>T Get(std::future<Result<T>> future){
  Check(future.wait_for(5s)==std::future_status::ready,"RPC test future did not complete");
  auto value=future.get();if(auto e=std::get_if<Error>(&value))throw std::runtime_error(e->message);
  return std::get<T>(std::move(value));
}
class PausedOwner final:public CacheBackend {
  HostCacheBackend backend_{0,1<<20};std::mutex mutex_;
  struct Held{UpdateContext context;RowGradients gradients;std::promise<Result<Ack>> promise;};
  std::unique_ptr<Held> held_;bool pause_=true;
 public:
  std::promise<void> entered;
  void Open(){std::unique_ptr<Held> held;{std::lock_guard lock(mutex_);pause_=false;held=std::move(held_);}
    if(held)held->promise.set_value(backend_.ApplyHost(held->context,std::move(held->gradients)).get());}
  std::future<Result<Ack>> Define(TableDefinition t) override{return backend_.Define(t);}
  std::future<Result<std::uint64_t>> FreeSlots(TableId t) override{return backend_.FreeSlots(t);}
  std::future<Result<std::vector<LocatedRow>>> Load(MigrationPlan p,std::vector<RowState> s) override{return backend_.Load(std::move(p),std::move(s));}
  std::future<Result<std::vector<TensorBlock>>> GatherHost(std::vector<AgentWork> w,std::vector<FeatureInput> f) override{return backend_.GatherHost(std::move(w),std::move(f));}
  std::future<Result<Ack>> ApplyHost(UpdateContext c,RowGradients g) override{
    std::lock_guard lock(mutex_);if(!pause_)return backend_.ApplyHost(c,std::move(g));
    Check(!held_,"test owner received concurrent held updates");held_=std::make_unique<Held>();held_->context=c;held_->gradients=std::move(g);
    auto future=held_->promise.get_future();entered.set_value();return future;
  }
  std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan p) override{return backend_.ReadBack(std::move(p));}
  std::future<Result<Ack>> CommitEviction(MigrationPlan p) override{return backend_.CommitEviction(std::move(p));}
};
TableDefinition Table(){return {1,2,1,1,{.01f,.9f,.999f,1e-8f,.9f,.999f},.25f,.25f,42};}
LookupRequest Read(const std::string& run,unsigned learner,bool read_only=false){return {{run,learner,0,Phase::lookup},1,read_only,{{1,0,Pooling::mean,{1},{0,1},{1}}}};}
PushRequest Write(const std::string& run,unsigned learner,std::string ticket){return {{run,learner,0,Phase::push},std::move(ticket),{{0,{1,2},{.03f,.03f}}}};}
struct Harness {
  CpuParameterService parameters;
  std::shared_ptr<PausedOwner> owner=std::make_shared<PausedOwner>();
  eps::transport::BrpcServer adapter;
  brpc::Server server;
  Harness():parameters([]{RuntimeOptions o;o.run_id="observed-rpc";o.max_active_operations=1;return o;}()),adapter(parameters,0){
    Get(parameters.AttachAgent(0,owner));Get(parameters.CreateTable(Table()));
    auto seed=Get(parameters.Lookup(Read("observed-rpc",10,true)));Get(parameters.Release({{"observed-rpc",10,0,Phase::release},seed.ticket}));
    Get(parameters.MoveRows({"observed-rpc",{{1,1}},0,Placement::gpu,{4096,4096,4,0}}));
    Check(server.AddService(&adapter,brpc::SERVER_DOESNT_OWN_SERVICE)==0,"register shard");
    brpc::ServerOptions options;options.max_concurrency=0;Check(server.Start("127.0.0.1:0",&options)==0,"start shard");
  }
  ~Harness(){owner->Open();server.Stop(0);server.Join();}
};
int main(){try{
  Harness h;BrpcParameterService client(butil::endpoint2str(h.server.listen_address()).c_str());
  auto entered=h.owner->entered.get_future();std::vector<LookupReply> tickets;
  for(unsigned i=0;i<4;++i)tickets.push_back(Get(client.Lookup(Read("observed-rpc",i))));
  std::vector<std::future<Result<Ack>>> updates;
  for(unsigned i=0;i<4;++i)updates.push_back(client.Push(Write("observed-rpc",i,tickets[i].ticket)));
  Check(entered.wait_for(3s)==std::future_status::ready,"held GPU-owner update did not start");
  auto deadline=std::chrono::steady_clock::now()+3s;
  while(eps::transport::CompletionExecutor::Instance().Snapshot().running<4 && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(1ms);
  const bool saturated=eps::transport::CompletionExecutor::Instance().Snapshot().running==4;
  auto begin=std::chrono::steady_clock::now();auto observation=client.Stats();const bool responded=observation.wait_for(1s)==std::future_status::ready;
  auto latency=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  h.owner->Open();for(auto& update:updates)Get(std::move(update));auto during=Get(std::move(observation));
  Check(saturated && responded,"statistics RPC blocked behind saturated data completion waiters");
  Check(during.active_operations==1 && during.waiting_operations==1 && during.live_tickets==4,"statistics observed after held updates completed");
  auto after=Get(client.Stats());Check(after.updates==4 && after.live_tickets==0 && after.active_operations==0 && after.completed_observations>=1,"observed RPC updates failed to drain");
  RuntimeOptions o;o.run_id="observation-reference";CpuParameterService reference(o);Get(reference.CreateTable(Table()));
  for(unsigned i=0;i<4;++i){auto t=Get(reference.Lookup(Read(o.run_id,i)));Get(reference.Push(Write(o.run_id,i,t.ticket)));}
  auto actual=Get(h.parameters.ReadState(Get(h.parameters.Locations({{1,1}})))).at(0);
  auto expected=Get(reference.ReadState(Get(reference.Locations({{1,1}})))).at(0);
  Check(actual.weights==expected.weights && actual.first_moment==expected.first_moment && actual.second_moment==expected.second_moment && actual.row_updates==expected.row_updates,"observation changed full optimizer state");
  std::cout<<"{\"test\":\"shard_observation_rpc\",\"held_data_completion_threads\":4,\"stats_latency_ms\":"<<latency<<",\"acknowledged_updates\":4,\"optimizer_state_exact\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
