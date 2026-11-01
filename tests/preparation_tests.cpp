#include "eps/entropy_adapter.hpp"
#include "eps/runtime.hpp"
#include "access/prepare_queue.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <latch>
#include <stdexcept>
#include <thread>
using namespace eps;
using namespace std::chrono_literals;
namespace {
void Check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class T>T Get(Result<T> result){if(auto e=std::get_if<Error>(&result))throw std::runtime_error(e->message);return std::get<T>(std::move(result));}
template<class T>T Get(std::future<Result<T>> result){return Get(result.get());}
template<class T>bool ErrorIs(const Result<T>& r,Error::Code c){auto e=std::get_if<Error>(&r);return e && e->code==c;}
template<class F>void Until(F predicate){const auto end=std::chrono::steady_clock::now()+5s;while(!predicate()){if(std::chrono::steady_clock::now()>end)throw std::runtime_error("state transition timed out");std::this_thread::yield();}}
TableDefinition Table(unsigned shards,unsigned dim=2){return {1,dim,shards,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},0.25f,0.25f,42};}
std::vector<entropy::Feature> Features(){return {
 {1,0,Pooling::mean,{1,0,1,2,0,3},{0,3,3,4,6},0},
 {1,1,Pooling::sequence,{0,2,1,0,3},{0,2,5},0},
 {1,2,Pooling::sum,{0,3,1,3,0},{0,0,3,5},0}};}
std::vector<TensorBlock> Gradients(const entropy::PullResult& batch){auto out=batch.features;for(auto& f:out)for(std::size_t i=0;i<f.values.size();++i)f.values[i]=0.01f*(i+1)*(f.feature+1);return out;}
struct Gate {
 std::promise<void> entered;std::promise<void> release;
 std::shared_future<void> opened;
 Gate():opened(release.get_future().share()){}
 void Open(){release.set_value();}
};
void QueueBoundsAndCancellation(){
 auto gate=std::make_shared<Gate>();std::atomic<int> calls=0;
 entropy::PreparationOptions limits;limits.max_pending_batches=2;
 detail::PrepareQueue queue(2,limits,[&](auto input)->Result<detail::PreparedAdapterInput>{
  if(calls.fetch_add(1)==0){gate->entered.set_value();gate->opened.wait();}
  return detail::PrepareAdapterInput(std::move(input),2,limits.lookup_limits);
 });
 auto first=Get(queue.Submit(Features()));gate->entered.get_future().wait();
 auto second=Get(queue.Submit(Features()));
 Check(ErrorIs(queue.Submit(Features()),Error::Code::capacity_exceeded),"job count was not bounded");
 Get(queue.Cancel(second));auto third=Get(queue.Submit(Features()));Get(queue.Cancel(first));
 auto s=queue.Stats();Check(s.pending==1 && s.reserved_jobs==2 && s.reserved_bytes>0,"running cancellation released reservation early");
 Check(ErrorIs(queue.Submit(Features()),Error::Code::capacity_exceeded),"cancelled running work did not hold its slot");
 gate->Open();auto prepared=Get(queue.Take(third));
 Check(prepared.lookup.features()[0].keys==std::vector<Key>({1,1,2,3}),"padding compaction changed input");
 s=queue.Stats();Check(s.pending==0 && s.reserved_jobs==0 && s.reserved_bytes==0 && s.cancelled==2 && s.consumed==1,"queue did not reclaim its reservations");
 Check(s.peak_reserved_bytes<=limits.payload_bytes && s.prepare_nanoseconds>0,"invalid preparation accounting");
 Get(queue.CloseIfEmpty());Check(ErrorIs(queue.Submit(Features()),Error::Code::unavailable),"closed queue accepted input");
 entropy::PreparationOptions tiny;tiny.payload_bytes=32;
 detail::PrepareQueue bytes(2,tiny);Check(ErrorIs(bytes.Submit(Features()),Error::Code::capacity_exceeded),"byte budget was ignored");
 auto input=Features();input[0].keys.reserve(4096);tiny.lookup_limits.max_input_bytes=2048;tiny.payload_bytes=1<<20;
 detail::PrepareQueue capacity(2,tiny);Check(ErrorIs(capacity.Submit(std::move(input)),Error::Code::capacity_exceeded),"unused input capacity bypassed the budget");
}
void QueueClaimAndFailure(){
 auto gate=std::make_shared<Gate>();entropy::PreparationOptions options;
 detail::PrepareQueue queue(1,options,[&](auto input)->Result<detail::PreparedAdapterInput>{
  gate->entered.set_value();gate->opened.wait();return detail::PrepareAdapterInput(std::move(input),1,options.lookup_limits);
 });
 auto token=Get(queue.Submit(Features()));gate->entered.get_future().wait();
 auto consumer=std::async(std::launch::async,[&]{return queue.Take(token);});
 Until([&]{return queue.Stats().claimed==1;});
 const auto cancel=queue.Cancel(token);gate->Open();Get(consumer.get());
 Check(ErrorIs(cancel,Error::Code::unavailable),"cancelled an input already claimed by Pull");
 std::atomic<int> calls=0;
 detail::PrepareQueue failure(1,options,[&](auto input)->Result<detail::PreparedAdapterInput>{
  if(calls.fetch_add(1)==0)return Error{Error::Code::invalid_argument,"injected pure-preparation failure"};
  return detail::PrepareAdapterInput(std::move(input),1,options.lookup_limits);
 });
 auto bad=Get(failure.Submit(Features())),good=Get(failure.Submit(Features()));
 Check(ErrorIs(failure.Take(good),Error::Code::unavailable),"FIFO input ordering was ignored");
 Check(ErrorIs(failure.Take(bad),Error::Code::invalid_argument),"worker failure was hidden");Get(failure.Take(good));
 Check(failure.Stats().failed==1 && failure.Stats().reserved_bytes==0,"failed input leaked a reservation");
 auto closing_gate=std::make_shared<Gate>();
 auto closing=std::make_unique<detail::PrepareQueue>(1,options,[&](auto input)->Result<detail::PreparedAdapterInput>{
  closing_gate->entered.set_value();closing_gate->opened.wait();return detail::PrepareAdapterInput(std::move(input),1,options.lookup_limits);
 });
 Get(closing->Submit(Features()));closing_gate->entered.get_future().wait();closing->Close();
 Check(closing->Stats().pending==0 && closing->Stats().reserved_jobs==1,"close reclaimed running buffers early");
 auto destroy=std::async(std::launch::async,[&]{closing.reset();});
 Check(destroy.wait_for(0s)!=std::future_status::ready,"destruction did not join the worker");closing_gate->Open();destroy.get();
}
struct BlockingService final : ParameterService {
 std::shared_ptr<CpuParameterService> inner;std::shared_ptr<Gate> gate;std::atomic<int> lookups=0;
 BlockingService(std::shared_ptr<CpuParameterService> s,std::shared_ptr<Gate> g):inner(std::move(s)),gate(std::move(g)){}
 std::future<Result<Ack>> CreateTable(TableDefinition t)override{return inner->CreateTable(t);}
 std::future<Result<LookupReply>> Lookup(LookupRequest r)override{
  if(lookups.fetch_add(1)==0)return std::async(std::launch::async,[this,r=std::move(r)]()mutable{gate->entered.set_value();gate->opened.wait();return inner->Lookup(std::move(r)).get();});
  return inner->Lookup(std::move(r));
 }
 std::future<Result<Ack>> Push(PushRequest r)override{return inner->Push(std::move(r));}
 std::future<Result<Ack>> Release(ReleaseRequest r)override{return inner->Release(std::move(r));}
 std::future<Result<std::vector<RowState>>> ReadState(std::vector<LocatedRow> r)override{return inner->ReadState(std::move(r));}
 std::future<Result<Ack>> CompleteMigration(MigrationCompletion r)override{return inner->CompleteMigration(std::move(r));}
 std::future<Result<SnapshotMetadata>> DrainAndFlush(std::string s)override{return inner->DrainAndFlush(std::move(s));}
};
void RpcOverlapAndReadBoundary(){
 RuntimeOptions options;options.run_id="prepared-overlap";auto core=std::make_shared<CpuParameterService>(options);auto gate=std::make_shared<Gate>();
 auto blocked=std::make_shared<BlockingService>(core,gate);
 entropy::TableAdapter adapter({options.run_id,0,1,1,16},{blocked});Get(adapter.CreateTable(Table(1)));
 const auto first=Get(adapter.PrepareAsync(Features()));
 Check(Get(core->Stats()).lookups==0,"preparation read server weights early");
 auto pull=std::async(std::launch::async,[&]{return adapter.PullPrepared(first);});gate->entered.get_future().wait();
 auto submit=std::async(std::launch::async,[&]{return adapter.PrepareAsync(Features());});
 const bool overlapped=submit.wait_for(5s)==std::future_status::ready;
 if(!overlapped){gate->Open();(void)pull.get();(void)submit.get();throw std::runtime_error("input preparation blocked behind the RPC mutex");}
 const auto second=Get(submit.get());Until([&]{return adapter.PreparationStatus().completed==2;});
 Check(blocked->lookups==1 && Get(core->Stats()).lookups==0,"future preparation issued parameter RPCs");
 gate->Open();auto batch=Get(pull.get());
 Check(ErrorIs(adapter.PullPrepared(second),Error::Code::capacity_exceeded),"live-batch bound consumed a future input");
 Get(adapter.Push(batch.handle,Gradients(batch)));Get(adapter.Join());
 auto next=Get(adapter.PullPrepared(second));Check(next.handle==1,"preparation assigned training sequence IDs");
 Check(next.features[0].values[0]<batch.features[0].values[0],"prepared input captured weights before the previous Push");
 Get(adapter.Release(next.handle));Get(adapter.Join());Check(Get(core->Stats()).live_tickets==0,"overlap test leaked a ticket");
}
void NumericalTrajectoryAndSave(){
 const auto root=std::filesystem::temp_directory_path()/("eps-prepare-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 RuntimeOptions ref;ref.run_id="prepared-reference";ref.snapshot_directory=root/"reference";
 auto baseline=std::make_shared<CpuParameterService>(ref);entropy::TableAdapter synchronous({ref.run_id},{baseline});Get(synchronous.CreateTable(Table(1)));
 std::vector<std::shared_ptr<CpuParameterService>> homes;std::vector<std::shared_ptr<ParameterService>> endpoints;
 for(unsigned i=0;i<3;++i){auto o=ref;o.run_id="prepared-three";o.shard_id=i;o.shard_count=3;o.snapshot_directory=root/"actual";auto h=std::make_shared<CpuParameterService>(o);homes.push_back(h);endpoints.push_back(h);}
 entropy::TableAdapter asynchronous({"prepared-three"},endpoints);Get(asynchronous.CreateTable(Table(3)));
 auto current=Get(asynchronous.PrepareAsync(Features()));
 auto cancelled=Get(asynchronous.PrepareAsync(Features()));Get(asynchronous.CancelPrepared(cancelled));
 Check(ErrorIs(asynchronous.Pull(Features()),Error::Code::unavailable),"synchronous Pull overtook queued input");
 Check(ErrorIs(asynchronous.Save("pending"),Error::Code::unavailable),"save passed pending preparation");Get(asynchronous.Join());
 for(unsigned step=0;step<5;++step){
  auto next=Get(asynchronous.PrepareAsync(Features()));
  auto a=Get(asynchronous.PullPrepared(current)),b=Get(synchronous.Pull(Features()));
  Check(a.handle==step,"cancelled token left a hole in request IDs");
  Check(a.features.size()==b.features.size(),"prepared feature count changed");
  for(unsigned f=0;f<a.features.size();++f){Check(a.features[f].shape==b.features[f].shape,"prepared shape changed");for(unsigned k=0;k<a.features[f].values.size();++k)Check(std::abs(a.features[f].values[k]-b.features[f].values[k])<1e-6f,"prepared feature differs from one-shard reference");}
  if(step==0){Check(a.features[0].values==std::vector<float>({.25,.25,0,0,.25,.25,.25,.25}),"mean padding/empty bags changed");Check(a.features[1].values==std::vector<float>({0,0,.25,.25,.25,.25,0,0,.25,.25}),"sequence positions changed");}
  Get(asynchronous.Push(a.handle,Gradients(a)));Get(synchronous.Push(b.handle,Gradients(b)));Get(asynchronous.Join());
  for(Key key=1;key<=3;++key){auto h=homes[key%3];auto state=Get(h->ReadState(Get(h->Locations({{1,key}})))).at(0);auto expected=Get(baseline->ReadState(Get(baseline->Locations({{1,key}})))).at(0);
   Check(state.weights==expected.weights && state.first_moment==expected.first_moment && state.second_moment==expected.second_moment && state.row_updates==expected.row_updates,"prepared optimizer trajectory changed");}
  current=next;
 }
 Check(ErrorIs(asynchronous.Save("future"),Error::Code::unavailable),"save ignored next input");Get(asynchronous.CancelPrepared(current));
 auto files=Get(asynchronous.Save("complete"));Check(files.size()==3,"save did not reach all shards");Check(ErrorIs(asynchronous.PrepareAsync(Features()),Error::Code::unavailable),"save did not close preparation admission");
 std::filesystem::remove_all(root);
}
void PostLookupCleanupAndPreparedValidation(){
 RuntimeOptions options;options.run_id="prepared-cleanup";auto core=std::make_shared<CpuParameterService>(options);
 entropy::TableAdapter adapter({options.run_id},{core});Get(adapter.CreateTable(Table(1,256)));
 std::vector<entropy::Feature> padded{{1,0,Pooling::sequence,std::vector<Key>(100000,0),{0,100000},0}};
 auto first=Get(adapter.PrepareAsync(std::move(padded))),future=Get(adapter.PrepareAsync(Features()));(void)future;
 Check(ErrorIs(adapter.PullPrepared(first),Error::Code::capacity_exceeded),"oversized reconstructed output passed");
 auto stats=Get(core->Stats());Check(stats.lookups==1 && stats.live_tickets==0 && stats.rows==0,"post-Lookup output failure orphaned a ticket");
 Check(adapter.PreparationStatus().pending==0,"post-Lookup failure retained future inputs");
 RuntimeOptions o;o.run_id="prepared-validation";auto service=std::make_shared<CpuParameterService>(o);DistributedClient client({service});Get(client.CreateTable(Table(1)));
 std::vector<FeatureInput> input{{1,1,Pooling::sum,{1,2},{0,2},{2}}};
 auto wrong=Get(PrepareLookupInputs(input,2));Check(ErrorIs(client.LookupPrepared({o.run_id,0,0,Phase::lookup},1,true,std::move(wrong)),Error::Code::invalid_argument),"wrong prepared topology passed");
 Check(Get(service->Stats()).lookups==0,"invalid prepared topology reached the service");
 auto prepared=Get(PrepareLookupInputs(input,1));Check(prepared.payload_bytes()<=Get(EstimatePreparedLookupBytes(input,1)),"retained prepared buffers exceed estimate");
 auto batch=Get(client.LookupPrepared({o.run_id,0,0,Phase::lookup},1,true,std::move(prepared)));Get(client.Release(batch));
 Check(ErrorIs(client.LookupPrepared({o.run_id,0,1,Phase::lookup},1,true,std::move(prepared)),Error::Code::invalid_argument),"moved-from prepared input was reused");
 input[0].valid_lengths={1};Check(std::holds_alternative<Error>(PrepareLookupInputs(input,1)),"invalid original mean lengths passed");
 {
  entropy::TableAdapter abandoned({o.run_id,1},{service});Get(abandoned.CreateTable(Table(1)));Get(abandoned.Pull(Features()));
 }
 Check(Get(service->Stats()).live_tickets==0,"adapter destruction did not release known tickets");
}
}
int main(){try{QueueBoundsAndCancellation();QueueClaimAndFailure();RpcOverlapAndReadBoundary();NumericalTrajectoryAndSave();PostLookupCleanupAndPreparedValidation();
 std::cout<<"PASS bounded preparation: FIFO/cancellation, RPC overlap, update-read boundary, three-shard optimizer replay, save and ticket cleanup\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL preparation: "<<e.what()<<'\n';return 1;}}
