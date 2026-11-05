#include "eps/replica_log.hpp"
#include "eps/runtime.hpp"
#include <atomic>
#include <fstream>
#include <iostream>
#include <thread>
using namespace eps;
namespace {
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F>void Reject(F operation){bool failed=false;try{operation();}catch(const std::exception&){failed=true;}Check(failed,"invalid replica operation accepted");}
ReplicatedUpdate Image(std::uint64_t round) {
  return {"replica-test",0,1,1,round,{{{},0,1,round+1,.02f,.81f,.998f}},
    {{{1,UINT64_C(9007199254740993)},{1.25f,-0.f},{.2f,.3f},{.4f,.5f},round+1}},{{0,round+1},{1023,round+1}}};
}
void Lifecycle() {
  const auto root=std::filesystem::temp_directory_path()/("eps-replica-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  ReplicaLogOptions options{root,"replica-test","model-schema-v1",0,1,1,0,4096,16384};
  const auto baseline=EncodeReplicatedUpdate(Image(0));const auto record0=baseline,record1=EncodeReplicatedUpdate(Image(1));
  const auto decoded=DecodeReplicatedUpdate(record0);
  Check(decoded.rows[0].weights==Image(0).rows[0].weights && decoded.rows[0].row.key==UINT64_C(9007199254740993) &&
        decoded.requests[1].next==1,"replica image codec changed state/IDs/clocks");
  auto broken=record0;broken[20]^=1;Reject([&]{DecodeReplicatedUpdate(broken);});
  Reject([&]{EncodeReplicatedUpdate(Image(0),32);});
  ReplicaLease first;
  {
    ReplicaLog log(options);first=log.Fence(0,"primary-a");
    Reject([&]{ReplicaLog second_writer(options);});
    Check(first.term==1 && log.Fence(0,"primary-a")==first,"promotion retry changed term");
    Reject([&]{log.Write({1,"other"},true,0,baseline.size(),0,baseline);});
    log.Write(first,true,0,baseline.size(),0,baseline.substr(0,40));
    Check(!log.Head().baseline,"partial baseline was published");
    log.Write(first,true,0,baseline.size(),0,baseline.substr(0,40));
    log.Write(first,true,0,baseline.size(),40,baseline.substr(40));
    Check(log.Head().baseline && log.Read(first,true,0,0,1<<20).data==baseline,"baseline did not round trip");
    log.Write(first,false,0,record0.size(),0,record0.substr(0,70));
    Check(log.Head().next_round==0,"partial round advanced durable clock");
  }
  {
    ReplicaLog log(options);Check(log.Head().lease==first && log.Head().next_round==0,"reopen lost lease or published partial round");
    log.Write(first,false,0,record0.size(),0,record0);
    Check(log.Head().next_round==1,"complete record did not advance clock");
    log.Write(first,false,0,record0.size(),0,record0);
    auto changed=Image(0);changed.rows[0].weights[0]=7;
    Reject([&]{log.Write(first,false,0,record0.size(),0,EncodeReplicatedUpdate(changed));});
    log.Write(first,false,1,record1.size(),0,record1.substr(0,60));
    const auto second=log.Fence(1,"primary-b");
    Reject([&]{log.Write(first,false,1,record1.size(),60,record1.substr(60));});
    Reject([&]{log.Read(first,true,0,0,512);});
    log.Write(second,false,1,record1.size(),0,record1);
    Check(log.Head().next_round==2 && log.Read(second,false,1,0,1<<20).data==record1,"promotion lost completed records");
    Reject([&]{log.Write(second,false,3,record1.size(),0,record1);});
    Reject([&]{log.Write(second,false,2,4097,0,std::string(8,'x'));});
  }
  {ReplicaLog log(options);Check(log.Head().next_round==2 && log.Head().lease.term==2,"durable head lost on reopen");}
  {std::ofstream marker(root/"retired.json");marker<<"{}";}
  Reject([&]{ReplicaLog retired(options);});std::filesystem::remove(root/"retired.json");
  auto mismatch=options;mismatch.binding="another-model";Reject([&]{ReplicaLog different(mismatch);});
  {
    std::fstream file(root/"round-0.epsr",std::ios::in|std::ios::out|std::ios::binary);file.seekp(30);file.put('!');
  }
  Reject([&]{ReplicaLog corrupt(options);});
  std::filesystem::remove_all(root);
}
void ConcurrentPromotion() {
  const auto root=std::filesystem::temp_directory_path()/("eps-replica-cas-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  {
    ReplicaLog log({root,"replica-test","schema",0,1,1,0,4096,16384});std::atomic<unsigned> granted=0;
    std::vector<std::jthread> threads;
    for(unsigned w=0;w<8;++w)threads.emplace_back([&,w]{try{log.Fence(0,"primary-"+std::to_string(w));++granted;}catch(const std::exception&) {}});
    threads.clear();Check(granted==1 && log.Head().lease.term==1,"competing managers both acquired primary term");
  }
  std::filesystem::remove_all(root);
}
void RequestProgress() {
  RuntimeOptions options;options.run_id="replica-requests";options.exclusive_update_learner=1023;
  CpuParameterService service(options);
  auto take=[]<class T>(std::future<Result<T>> future)->T {
    auto r=future.get();if(auto* e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));
  };
  take(service.CreateTable({1,2,1,1,{.02f,.9f,.999f,1e-8f,.9f,.999f},0,0,42}));
  take(service.InitializeRequestFrontiers({{7,50000}}));
  auto query=[&](std::uint64_t n){return LookupRequest{{options.run_id,7,n,Phase::lookup},1,true,{{1,0,Pooling::sequence,{1},{0,1},{1}}}};};
  auto ticket=take(service.Lookup(query(50001)));
  Check(std::holds_alternative<Error>(service.CertifyReadFrontiers({{7,50002}}).get()),"certification discarded a live read ticket");
  take(service.Release({{options.run_id,7,50001,Phase::release},ticket.ticket}));
  take(service.CertifyReadFrontiers({{7,50002}}));
  auto metadata=take(service.TrainingMetadata());
  Check(metadata.requests.size()==1 && metadata.requests[0].next==50002,"retained-gradient read certificate did not close a recovered hole");
  Check(std::holds_alternative<Error>(service.Lookup(query(50001)).get()),"retired read request became new after recovery");
  ticket=take(service.Lookup(query(50002)));
  take(service.Release({{options.run_id,7,50002,Phase::release},ticket.ticket}));
  Check(take(service.TrainingMetadata()).requests[0].next==50003,"large restored stream could not continue");
}
}
int main(){try{Lifecycle();ConcurrentPromotion();RequestProgress();std::cout<<"PASS replica log: exact state codec, chunk retries, partial recovery, fsynced publication, durable term CAS, fencing, corruption, limits and restored/certified request frontiers\n";}
 catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
