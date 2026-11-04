#include "eps/entropy_adapter.hpp"
#include "eps/runtime.hpp"
#include <iostream>
#include <optional>
#include <stdexcept>
using namespace eps;
template<class T>T Take(Result<T> r){if(auto* e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<class T>T Take(std::future<Result<T>> f){return Take(f.get());}
template<class T>std::future<Result<T>> Ready(Result<T> r){std::promise<Result<T>> p;p.set_value(std::move(r));return p.get_future();}
void Check(bool v,const char* why){if(!v)throw std::runtime_error(why);}
template<class T>void Rejected(Result<T> r){Check(std::holds_alternative<Error>(r),"expected rejection");}
class FaultService:public ParameterService {
 public:
  std::shared_ptr<CpuParameterService> target;
  std::string lookup_fault,release_fault;unsigned lookups_left=0,releases_left=0,release_calls=0;
  std::optional<LookupRequest> delayed;
  explicit FaultService(std::shared_ptr<CpuParameterService> p):target(std::move(p)){}
  std::future<Result<Ack>> CreateTable(TableDefinition t)override{return target->CreateTable(t);}
  std::future<Result<LookupReply>> Lookup(LookupRequest r)override{
    if(lookups_left){--lookups_left;delayed=r;
      if(lookup_fault=="throw")throw std::runtime_error("injected dispatch failure");
      if(lookup_fault=="after")Take(target->Lookup(r));
      return Ready<LookupReply>(Error{Error::Code::unavailable,"injected lookup failure"});
    }return target->Lookup(std::move(r));
  }
  std::future<Result<Ack>> Push(PushRequest r)override{return target->Push(std::move(r));}
  std::future<Result<Ack>> Release(ReleaseRequest r)override{
    ++release_calls;
    if(releases_left){--releases_left;if(release_fault=="after")Take(target->Release(r));
      return Ready<Ack>(Error{Error::Code::unavailable,"injected release failure"});}
    return target->Release(std::move(r));
  }
  std::future<Result<std::vector<RowState>>> ReadState(std::vector<LocatedRow> r)override{return target->ReadState(std::move(r));}
  std::future<Result<Ack>> CompleteMigration(MigrationCompletion r)override{return target->CompleteMigration(std::move(r));}
  std::future<Result<SnapshotMetadata>> DrainAndFlush(std::string r)override{return target->DrainAndFlush(std::move(r));}
  std::future<Result<SnapshotChunk>> ReadCheckpoint(CheckpointIdentity i,std::uint64_t o,std::uint32_t n)override{return target->ReadCheckpoint(std::move(i),o,n);}
  std::future<Result<Ack>> ResumeCheckpoint(CheckpointIdentity i)override{return target->ResumeCheckpoint(std::move(i));}
};
TableDefinition Table(unsigned shards=2){return {1,2,shards,1,{.01f,.9f,.999f,1e-8f,.9f,.999f},.25f,.25f,42};}
std::vector<entropy::Feature> Features(){return {{1,0,Pooling::sequence,{0,1},{0,2},{}}};}
struct Harness {
  std::vector<std::shared_ptr<CpuParameterService>> homes;
  std::vector<std::shared_ptr<FaultService>> faults;
  std::unique_ptr<entropy::TableAdapter> adapter;
  Harness(){
    std::vector<std::shared_ptr<ParameterService>> endpoints;
    for(unsigned i=0;i<2;++i){RuntimeOptions o;o.run_id="read-cleanup";o.shard_id=i;o.shard_count=2;
      homes.push_back(std::make_shared<CpuParameterService>(o));faults.push_back(std::make_shared<FaultService>(homes.back()));endpoints.push_back(faults.back());}
    adapter=std::make_unique<entropy::TableAdapter>(entropy::AdapterOptions{"read-cleanup"},endpoints);Take(adapter->CreateTable(Table()));
  }
  void Unpinned(){for(auto& h:homes){auto s=Take(h->Stats());Check(s.live_tickets==0 && s.ticket_payload_bytes==0 && s.updates==0,"read cleanup left pins or mutated parameters");}}
};
void PartialLookups(){
  for(const auto& mode:{"before","after","throw"})for(const auto& release:{"before","after"}){
    Harness h;auto& fault=*h.faults[1];fault.lookup_fault=mode;fault.lookups_left=1;fault.release_fault=release;fault.releases_left=2;
    Rejected(h.adapter->Pull(Features(),true));Check(h.adapter->ReadLifecycleHealthy(),"confirmed cancellation closed adapter");
    h.Unpinned();Check(fault.release_calls==3 && h.faults[0]->release_calls==1,"cleanup did not retry only unacknowledged shards");
    Rejected(h.homes[1]->Lookup(*fault.delayed).get());h.Unpinned();
    auto retry=Take(h.adapter->Pull(Features(),true));Check(retry.handle==1,"failed read identity reused");
    Take(h.adapter->Release(retry.handle));Take(h.adapter->Join());h.Unpinned();
    Check(Take(h.adapter->CompletedReadFrontier())==2,"canceled read missing from certified frontier");
  }
}
void PartialReleases(){
  for(const auto& mode:{"before","after"}){
    Harness h;auto batch=Take(h.adapter->Pull(Features(),true));auto& fault=*h.faults[1];fault.release_fault=mode;fault.releases_left=2;
    Take(h.adapter->Release(batch.handle));Check(h.adapter->ReadLifecycleHealthy(),"confirmed release closed adapter");
    Check(fault.release_calls==3 && h.faults[0]->release_calls==1,"release retried an acknowledged shard");h.Unpinned();
    auto next=Take(h.adapter->Pull(Features(),true));Take(h.adapter->Release(next.handle));h.Unpinned();
  }
}
void PreparedInputOrder(){
  Harness h;auto first=Take(h.adapter->PrepareAsync(Features()));Take(h.adapter->PrepareAsync(Features()));
  h.faults[1]->lookup_fault="after";h.faults[1]->lookups_left=1;
  Rejected(h.adapter->PullPrepared(first,true));h.Unpinned();
  Check(!h.adapter->ReadLifecycleHealthy() && h.adapter->PreparationStatus().pending==0,
    "failed prepared input allowed a future batch to overtake it");
  Rejected(h.adapter->CompletedReadFrontier());Rejected(h.adapter->PrepareAsync(Features()));
}
void UnknownCleanup(){
  for(bool lookup:{false,true}){
    Harness h;auto& fault=*h.faults[1];
    if(lookup){fault.lookup_fault="after";fault.lookups_left=1;}
    std::optional<entropy::PullResult> batch;
    if(!lookup)batch=Take(h.adapter->Pull(Features(),true));
    fault.release_fault="before";fault.releases_left=3;
    if(lookup)Rejected(h.adapter->Pull(Features(),true));else Rejected(h.adapter->Release(batch->handle));
    Check(!h.adapter->ReadLifecycleHealthy(),"unknown cleanup permitted more reads");
    Rejected(h.adapter->CompletedReadFrontier());Rejected(h.adapter->Join());Rejected(h.adapter->Pull(Features(),true));
    Check(fault.release_calls==3,"cleanup exceeded retry bound");
    for(auto& home:h.homes)Take(home->Release({{"read-cleanup",0,0,Phase::release},"",true,1}));
    h.Unpinned();Check(!h.adapter->ReadLifecycleHealthy(),"failed client implicitly reopened");
  }
}
void Tombstones(){
  RuntimeOptions o;o.run_id="cancellation";o.request_history=8;CpuParameterService s(o);Take(s.CreateTable(Table(1)));
  auto id=[](std::uint64_t n){return RequestId{"cancellation",0,n,Phase::release};};
  Rejected(s.Release({id(0),"",true,2}).get());
  Rejected(s.Release({id(0),"wrong",true,1}).get());
  for(unsigned n=0;n<40;++n)Take(s.Release({id(n),"",true,1}));
  Take(s.Release({id(0),"",true,1}));
  auto late=id(0);late.phase=Phase::lookup;
  Rejected(s.Lookup({late,1,true,{{1,0,Pooling::sequence,{0},{0,1},{1}}}}).get());
  Rejected(s.Release({id(1000),"",true,1}).get());
  Check(Take(s.Stats()).rows==0 && Take(s.Stats()).live_tickets==0,"unseen cancellation initialized rows");
  auto request=id(40);request.phase=Phase::lookup;
  auto batch=Take(s.Lookup({request,1,false,{{1,0,Pooling::sequence,{0},{0,1},{1}}}}));request.phase=Phase::push;
  Take(s.Push({request,batch.ticket,{{0,{1,2},{1,1}}}}));request.phase=Phase::release;
  Rejected(s.Release({request,"",true,1}).get());Check(Take(s.Stats()).updates==1,"cancel undid a completed update");
  RuntimeOptions exclusive;exclusive.run_id="exclusive";exclusive.exclusive_update_learner=1023;CpuParameterService training(exclusive);
  Rejected(training.Release({{"exclusive",1023,0,Phase::release},"",true,1}).get());
}
int main(){try{Tombstones();PartialLookups();PartialReleases();PreparedInputOrder();UnknownCleanup();
 std::cout<<"{\"test\":\"read_lifecycle_cleanup\",\"lost_lookup_reply\":true,\"late_lookup_fenced\":true,\"partial_release\":true,\"uncertain_cleanup_closed\":true}\n";
}catch(const std::exception& e){std::cerr<<"FAIL read cleanup: "<<e.what()<<'\n';return 1;}}
