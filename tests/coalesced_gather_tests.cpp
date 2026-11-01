#include "eps/cache_resolver.hpp"
#include "eps/runtime.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>

using namespace eps;
namespace {
template<typename T>T Take(Result<T> r) {
  if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
template<typename T>T Take(std::future<Result<T>> f){return Take(f.get());}
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void Near(float a,float b) {
  Check(std::isfinite(a) && std::abs(a-b)<5e-6f,"coalesced gather numerical mismatch");
}
struct Gate {
  void Arrive() {
    std::unique_lock lock(mutex);
    if(++arrived==3){ready.notify_all();return;}
    if(!ready.wait_for(lock,std::chrono::seconds(3),[&]{return arrived==3;}))
      throw std::runtime_error("GPU gathers were not dispatched concurrently");
  }
  std::mutex mutex;std::condition_variable ready;int arrived=0;
};
class DelayedLookup final : public ParameterService {
 public:
  DelayedLookup(std::shared_ptr<ParameterService> target,std::shared_future<void> signal)
      :target_(std::move(target)),signal_(std::move(signal)){}
  std::future<Result<Ack>> CreateTable(TableDefinition t)override{return target_->CreateTable(t);}
  std::future<Result<LookupReply>> Lookup(LookupRequest r)override {
    return std::async(std::launch::async,[target=target_,signal=signal_,r=std::move(r)]()mutable->Result<LookupReply> {
      if(signal.wait_for(std::chrono::seconds(3))!=std::future_status::ready)
        return Error{Error::Code::unavailable,"client waited for slow shard before starting ready-shard gather"};
      return target->Lookup(std::move(r)).get();
    });
  }
  std::future<Result<Ack>> Push(PushRequest r)override{return target_->Push(std::move(r));}
  std::future<Result<Ack>> Release(ReleaseRequest r)override{return target_->Release(std::move(r));}
  std::future<Result<std::vector<RowState>>> ReadState(std::vector<LocatedRow> r)override{return target_->ReadState(std::move(r));}
  std::future<Result<Ack>> CompleteMigration(MigrationCompletion r)override{return target_->CompleteMigration(std::move(r));}
  std::future<Result<SnapshotMetadata>> DrainAndFlush(std::string r)override{return target_->DrainAndFlush(std::move(r));}
  std::future<Result<SnapshotChunk>> ReadCheckpoint(CheckpointIdentity i,std::uint64_t o,std::uint32_t n)override{return target_->ReadCheckpoint(std::move(i),o,n);}
  std::future<Result<Ack>> ResumeCheckpoint(CheckpointIdentity i)override{return target_->ResumeCheckpoint(std::move(i));}
 private:
  std::shared_ptr<ParameterService> target_;std::shared_future<void> signal_;
};
class ObservedBackend final : public CacheBackend {
 public:
  explicit ObservedBackend(AgentId id):backend(id,1<<20){}
  std::future<Result<Ack>> Define(TableDefinition t)override{return backend.Define(t);}
  std::future<Result<std::uint64_t>> FreeSlots(TableId t)override{return backend.FreeSlots(t);}
  std::future<Result<std::vector<LocatedRow>>> Load(MigrationPlan p,std::vector<RowState> s)override {
    return backend.Load(std::move(p),std::move(s));
  }
  std::future<Result<std::vector<TensorBlock>>> GatherHost(
      std::vector<AgentWork> work,std::vector<FeatureInput> layouts)override {
    ++calls;
    if(on_gather){auto signal=std::move(on_gather);signal->set_value();}
    if(held) {
      auto future=held->get_future();dispatched->set_value();return future;
    }
    if(fail) {
      std::promise<Result<std::vector<TensorBlock>>> result;
      result.set_value(Error{Error::Code::unavailable,"injected destination failure"});return result.get_future();
    }
    std::set<std::tuple<std::uint32_t,AccessRoute>> groups;
    std::uint64_t sent=0;
    for(const auto& layout:layouts) {
      sent+=layout.keys.size();
      for(std::size_t i=0;i<layout.keys.size();++i) {
        bool used=false;
        for(const auto& task:work)if(task.feature==layout.feature)
          used|=task.occurrence_to_unique.at(i)!=UINT64_MAX;
        Check(used,"destination received an unreferenced input ID");
      }
    }
    received_keys+=sent;
    for(const auto& task:work)
      Check(groups.emplace(task.feature,task.route).second,"same feature/route was not coalesced across shards");
    auto barrier=gate;
    return std::async(std::launch::async,[this,barrier,work=std::move(work),layouts=std::move(layouts)]()mutable {
      if(barrier)barrier->Arrive();
      return backend.GatherHost(std::move(work),std::move(layouts)).get();
    });
  }
  std::future<Result<Ack>> ApplyHost(UpdateContext c,RowGradients g)override {
    return backend.ApplyHost(c,std::move(g));
  }
  std::future<Result<std::vector<RowState>>> ReadBack(MigrationPlan p)override {
    return backend.ReadBack(std::move(p));
  }
  std::future<Result<Ack>> CommitEviction(MigrationPlan p)override {
    return backend.CommitEviction(std::move(p));
  }
  HostCacheBackend backend;std::atomic<int> calls{0};std::shared_ptr<Gate> gate;
  std::atomic<std::uint64_t> received_keys{0};
  bool fail=false;
  std::shared_ptr<std::promise<Result<std::vector<TensorBlock>>>> held;
  std::shared_ptr<std::promise<void>> dispatched;
  std::shared_ptr<std::promise<void>> on_gather;
};
TableDefinition Table(TableId id,std::uint32_t shards) {
  return {id,id==1?3U:5U,shards,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},-0.5f,0.5f,42};
}
LookupRequest Request(std::uint32_t learner,std::uint64_t sequence,bool eval) {
  return {{"coalesced",learner,sequence,Phase::lookup},1,eval,{
    {1,10,Pooling::mean,{1,2,1,3,8,9,4},{0,4,4,7},{4,0,3}},
    {1,11,Pooling::sequence,{6,1,8,3,12,2},{0,2,6},{2,4}},
    {1,12,Pooling::sum,{5,9,2,5},{0,2,4},{2,2}},
    {2,13,Pooling::sequence,{1,2,7,6},{0,0,4},{0,4}},
    {2,14,Pooling::mean,{7,8},{0,2},{2}}}};
}
void Same(const DistributedBatch& a,const DistributedBatch& b) {
  Check(a.features().size()==b.features().size(),"feature count");
  for(std::size_t i=0;i<a.features().size();++i) {
    const auto& x=a.features()[i];const auto& y=b.features()[i];
    Check(x.feature==y.feature && x.shape==y.shape && x.values.size()==y.values.size(),"feature layout");
    for(std::size_t j=0;j<x.values.size();++j)Near(x.values[j],y.values[j]);
  }
}
}
int main() {
  try {
    std::vector<std::shared_ptr<CpuParameterService>> homes;
    std::vector<std::shared_ptr<ParameterService>> shards;
    std::map<AgentId,std::shared_ptr<CacheBackend>> endpoints;
    std::vector<std::shared_ptr<ObservedBackend>> workers;
    for(AgentId a=0;a<3;++a) {
      auto worker=std::make_shared<ObservedBackend>(a);workers.push_back(worker);endpoints[a]=worker;
    }
    for(ShardId s=0;s<2;++s) {
      RuntimeOptions o;o.run_id="coalesced";o.shard_id=s;o.shard_count=2;o.max_live_tickets=1;
      auto home=std::make_shared<CpuParameterService>(o);
      for(const auto& [a,worker]:endpoints)Take(home->AttachAgent(a,worker));
      homes.push_back(home);shards.push_back(home);
    }
    RuntimeOptions ref_options;ref_options.run_id="coalesced";
    auto reference=std::make_shared<CpuParameterService>(ref_options);
    DistributedClient actual(shards),baseline({reference});
    auto resolver=MakeCacheResolver(endpoints);actual.SetCacheResolver(resolver,true);
    for(TableId t:{1,2}){Take(actual.CreateTable(Table(t,2)));Take(baseline.CreateTable(Table(t,1)));}
    for(auto& home:homes)Take(home->ConfigureAccess(0,AccessRoute::transfer_unique));
    auto warm=Take(actual.Lookup(Request(0,0,true)));Take(actual.Release(warm));
    auto warm_ref=Take(baseline.Lookup(Request(0,0,true)));Take(baseline.Release(warm_ref));
    // Materialize all resident rows; some do not occur in the original warmup.
    for(ShardId s=0;s<2;++s) {
      auto init=Request(20+s,0,true);init.features={{1,0,Pooling::sequence,{},{0},{}}};
      for(Key k=1;k<=6;++k)if(k%2==s)init.features[0].keys.push_back(k);
      init.features[0].offsets.push_back(init.features[0].keys.size());
      init.features[0].valid_lengths.push_back(init.features[0].keys.size());
      auto reply=Take(homes[s]->Lookup(init));Take(homes[s]->Release({{init.id.run,init.id.learner,0,Phase::release},reply.ticket}));
      // Table 2 has only resident keys present in the test request.
      for(TableId t:{1,2})for(Key k=1;k<=6;++k)if(k%2==s && (t==1 || k==1 || k==2 || k==6))
        Take(homes[s]->MoveRows({"coalesced",{{t,k}},AgentId((k/2)%3),Placement::gpu,{1<<20,1<<20,2,0}}));
    }
    // ConfigureAccess is per shard: select CPU route for all cold rows on one
    // shard and transferred vectors on the other. Both contain resident rows.
    Take(homes[0]->ConfigureAccess(0,AccessRoute::cpu_pool));
    for(std::uint64_t seq=1;seq<=3;++seq) {
      auto gate=std::make_shared<Gate>();
      std::vector<int> before;
      std::uint64_t before_keys=0;
      for(auto& worker:workers)before_keys+=worker->received_keys;
      for(auto& worker:workers){before.push_back(worker->calls);worker->gate=gate;}
      auto a=Take(actual.Lookup(Request(0,seq,false))),b=Take(baseline.Lookup(Request(0,seq,false)));
      Same(a,b);
      for(std::size_t i=0;i<workers.size();++i) {
        Check(workers[i]->calls==before[i]+1,"lookup did not issue exactly one gather per destination");
        workers[i]->gate.reset();
      }
      std::uint64_t sent_keys=0;for(auto& worker:workers)sent_keys+=worker->received_keys;
      // There are 23 input occurrences; CPU-only cold rows remain on their
      // shard. Each remaining occurrence appears in one destination layout.
      Check(sent_keys-before_keys==19,"destination projection did not send exactly the referenced positions");
      auto gradients=a.features();
      for(auto& f:gradients)for(std::size_t j=0;j<f.values.size();++j)f.values[j]=0.03f*(1+j%7)*seq;
      Take(actual.Push(a,gradients));Take(baseline.Push(b,gradients));
    }
    const std::map<TableId,std::vector<Key>> touched{{1,{1,2,3,4,5,6,8,9,12}},{2,{1,2,6,7,8}}};
    for(const auto& [t,keys]:touched)for(Key k:keys) {
      auto x=Take(homes[k%2]->ReadState(Take(homes[k%2]->Locations({{t,k}})))).at(0);
      auto y=Take(reference->ReadState(Take(reference->Locations({{t,k}})))).at(0);
      Check(x.row_updates==y.row_updates,"optimizer update count");
      for(std::size_t d=0;d<x.weights.size();++d) {
        Near(x.weights[d],y.weights[d]);Near(x.first_moment[d],y.first_moment[d]);Near(x.second_moment[d],y.second_moment[d]);
      }
    }
    auto signal=std::make_shared<std::promise<void>>();
    auto delayed=std::make_shared<DelayedLookup>(homes[1],signal->get_future().share());
    DistributedClient streamed({homes[0],delayed});
    for(TableId t:{1,2})Take(streamed.CreateTable(Table(t,2)));
    streamed.SetCacheResolver(resolver); // Default: process a ready shard before waiting for the next.
    workers[0]->on_gather=signal;
    std::vector<int> before_stream;for(auto& worker:workers)before_stream.push_back(worker->calls);
    auto streaming=Take(streamed.Lookup(Request(50,0,true)));
    auto streaming_reference=Take(baseline.Lookup(Request(50,0,true)));Same(streaming,streaming_reference);
    for(std::size_t i=0;i<workers.size();++i)
      Check(workers[i]->calls==before_stream[i]+2,"streaming mode did not retain per-shard gathers");
    Take(streamed.Release(streaming));Take(baseline.Release(streaming_reference));
    auto clean=[&]{for(auto& home:homes)Check(Take(home->Stats()).live_tickets==0,"failed lookup leaked a ticket");};
    for(int mode=0;mode<3;++mode) {
      DistributedClient failing(shards);for(TableId t:{1,2})Take(failing.CreateTable(Table(t,2)));
      if(mode==1)failing.SetCacheResolver([](auto,auto)->Result<std::vector<TensorBlock>> {
        return Error{Error::Code::unavailable,"injected gather failure"};
      });
      if(mode==2)failing.SetCacheResolver([](auto,auto)->Result<std::vector<TensorBlock>> {
        return std::vector<TensorBlock>{{10,{1,1},{0}}};
      });
      Check(std::holds_alternative<Error>(failing.Lookup(Request(30+mode,0,true))),"invalid resolver accepted");clean();
    }
    // One shard rejects while the other issues a ticket; the accepted side must
    // be released even though there will be no usable DistributedBatch.
    LookupRequest held_request{{"coalesced",99,0,Phase::lookup},1,true,{{1,0,Pooling::sequence,{2},{0,1},{1}}}};
    auto held=Take(homes[0]->Lookup(held_request));
    DistributedClient blocked(shards);for(TableId t:{1,2})Take(blocked.CreateTable(Table(t,2)));blocked.SetCacheResolver(resolver);
    auto rejected=blocked.Lookup(Request(40,0,true));
    Check(std::holds_alternative<Error>(rejected) && std::get<Error>(rejected).code==Error::Code::capacity_exceeded,"partial rejection lost its error");
    Check(Take(homes[1]->Stats()).live_tickets==0,"accepted shard ticket leaked after peer rejection");
    Take(homes[0]->Release({{"coalesced",99,0,Phase::release},held.ticket}));clean();
    auto before=workers[0]->calls.load();
    AgentWork known{};known.agent=0;AgentWork unknown{};unknown.agent=99;
    auto missing=resolver({known,unknown},{});
    Check(std::holds_alternative<Error>(missing) && workers[0]->calls==before,"resolver dispatched before validating endpoints");
    // Use a promise-backed future (not std::async's joining destructor) to prove
    // a fast destination error cannot return while another gather is in flight.
    workers[0]->fail=true;
    workers[1]->held=std::make_shared<std::promise<Result<std::vector<TensorBlock>>>>();
    workers[1]->dispatched=std::make_shared<std::promise<void>>();
    auto dispatched=workers[1]->dispatched->get_future();
    AgentWork slow{};slow.agent=1;
    known.feature=slow.feature=0;known.occurrence_to_unique=slow.occurrence_to_unique={0};
    FeatureInput dummy{1,0,Pooling::sequence,{1},{0,1},{1}};
    auto waiting=std::async(std::launch::async,[&]{return resolver({known,slow},{dummy});});
    const bool launched=dispatched.wait_for(std::chrono::seconds(3))==std::future_status::ready;
    const bool waited=waiting.wait_for(std::chrono::milliseconds(10))==std::future_status::timeout;
    workers[1]->held->set_value(std::vector<TensorBlock>{});
    auto failed=waiting.get();
    Check(launched && waited && std::holds_alternative<Error>(failed),"resolver returned before outstanding reads finished");
    std::cout<<"PASS coalesced two-shard/three-agent gathers, mixed routes, parallel dispatch, optimizer trajectory and failure cleanup\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL coalesced gather: "<<e.what()<<'\n';return 1;}
}
