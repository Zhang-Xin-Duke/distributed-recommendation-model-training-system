#include "eps/entropy_adapter.hpp"
#include "eps/runtime.hpp"
#include "eps/cache_backend.hpp"
#include <filesystem>
#include <atomic>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace eps;
namespace {
void Check(bool x,const char* m){if(!x)throw std::runtime_error(m);}
template<class T>T Take(Result<T> r){if(auto* e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<class T>T Take(std::future<Result<T>> f){return Take(f.get());}
template<class T>void Rejected(Result<T> r){Check(std::holds_alternative<Error>(r),"expected checkpoint rejection");}
template<class T>void Rejected(std::future<Result<T>> f){Rejected(f.get());}
struct Temp {std::filesystem::path path;Temp(){char x[]="/tmp/eps-periodic-XXXXXX";auto* p=::mkdtemp(x);Check(p,"mkdtemp failed");path=p;}~Temp(){std::error_code e;std::filesystem::remove_all(path,e);}};
using Homes=std::vector<std::shared_ptr<CpuParameterService>>;
Homes Make(const std::string& run,const std::filesystem::path& path){Homes out;for(unsigned i=0;i<2;++i){RuntimeOptions o;o.run_id=run;o.shard_id=i;o.shard_count=2;o.snapshot_directory=path;out.push_back(std::make_shared<CpuParameterService>(o));}return out;}
std::vector<std::shared_ptr<ParameterService>> Base(const Homes& h){return {h.begin(),h.end()};}
class RetryResume final:public ParameterService {
 public:
  std::shared_ptr<ParameterService> target;std::atomic<bool> fail=false;
  explicit RetryResume(std::shared_ptr<ParameterService> t):target(std::move(t)){}
  std::future<Result<Ack>> CreateTable(TableDefinition t)override{return target->CreateTable(t);}
  std::future<Result<LookupReply>> Lookup(LookupRequest r)override{return target->Lookup(std::move(r));}
  std::future<Result<Ack>> Push(PushRequest r)override{return target->Push(std::move(r));}
  std::future<Result<Ack>> Release(ReleaseRequest r)override{return target->Release(std::move(r));}
  std::future<Result<std::vector<RowState>>> ReadState(std::vector<LocatedRow> r)override{return target->ReadState(std::move(r));}
  std::future<Result<Ack>> CompleteMigration(MigrationCompletion r)override{return target->CompleteMigration(std::move(r));}
  std::future<Result<SnapshotMetadata>> DrainAndFlush(std::string id)override{return target->DrainAndFlush(std::move(id));}
  std::future<Result<SnapshotChunk>> ReadCheckpoint(CheckpointIdentity i,std::uint64_t o,std::uint32_t n)override{return target->ReadCheckpoint(std::move(i),o,n);}
  std::future<Result<Ack>> ResumeCheckpoint(CheckpointIdentity i)override{
    if(fail.exchange(false)){std::promise<Result<Ack>> p;p.set_value(Error{Error::Code::unavailable,"injected resume reply failure"});return p.get_future();}
    return target->ResumeCheckpoint(std::move(i));
  }
};
TableDefinition Definition(){TableDefinition t{1,4,2,1,{.1f,.9f,.999f,1e-8f,.9f,.999f},.25f,.25f,42};t.optimizer.weight_bounds=WeightBounds{-.2f,.2f};return t;}
std::vector<entropy::Feature> Features(){return {{1,0,Pooling::mean,{1,1,2,3},{0,3,4},{}},{1,1,Pooling::sequence,{3,4,0},{0,3},Key(0)}};}
void Compare(const Homes& a,const Homes& b){for(Key k=1;k<=4;++k){auto s=k%2;auto x=Take(a[s]->ReadState(Take(a[s]->Locations({{1,k}})))).at(0);auto y=Take(b[s]->ReadState(Take(b[s]->Locations({{1,k}})))).at(0);Check(x.weights==y.weights && x.first_moment==y.first_moment && x.second_moment==y.second_moment && x.row_updates==y.row_updates,"save/resume changed optimizer trajectory");}}
void Periodic(){
 Temp temp;auto actual=Make("periodic",temp.path/"server"),reference=Make("reference",temp.path/"reference");
 auto cache=std::make_shared<HostCacheBackend>(0,1<<20);for(auto& h:actual)Take(h->AttachAgent(0,cache));
 auto retry=std::make_shared<RetryResume>(actual[0]);auto endpoints=Base(actual);endpoints[0]=retry;
 entropy::TableAdapter adapter({"periodic"},endpoints),baseline({"reference"},Base(reference));
 adapter.SetCacheResolver([cache](auto w,auto f){return cache->GatherHost(std::move(w),std::move(f)).get();});
 Take(adapter.CreateTable(Definition()));Take(baseline.CreateTable(Definition()));
 auto init=Take(adapter.Pull(Features(),true));Take(adapter.Release(init.handle));
 for(unsigned i=0;i<2;++i)Take(actual[i]->MoveRows({"periodic",{{1,Key(2+i)}},0,Placement::gpu,{1<<20,1<<20,1,0}}));
 Homes restored;std::unique_ptr<entropy::TableAdapter> recovered;std::uint64_t last_handle=0,last_prepared=0;bool had_prepared=false;
 for(unsigned step=0;step<8;++step){
  auto token=Take(adapter.PrepareAsync(Features()));if(had_prepared)Check(token>last_prepared,"preparation token reused after resume");had_prepared=true;last_prepared=token;
  auto a=Take(adapter.PullPrepared(token));Check(a.handle>last_handle,"request handle reused after resume");last_handle=a.handle;
  auto b=Take(baseline.Pull(Features()));auto gradients=a.features;
  for(auto& g:gradients)for(auto& x:g.values)x=(step<4?.1f:-.2f)*(g.feature+1);
  Take(adapter.Push(a.handle,gradients));Take(baseline.Push(b.handle,gradients));
  if(recovered){auto c=Take(recovered->Pull(Features()));Take(recovered->Push(c.handle,gradients));Compare(restored,reference);}
  Compare(actual,reference);
  if(step!=2 && step!=5)continue;
  auto id="step-"+std::to_string(step);auto metadata=Take(adapter.Save(id));
  Rejected(adapter.Pull(Features()));Rejected(adapter.PrepareAsync(Features()));Rejected(adapter.Resume("stale"));
  Rejected(actual[0]->ResumeCheckpoint({"wrong-run",1,id}));Rejected(actual[0]->ResumeCheckpoint({"periodic",2,id}));
  Rejected(actual[0]->ReadCheckpoint({"periodic",1,id},0,(1<<20)+1));
  std::vector<std::filesystem::path> copies;
  for(unsigned shard=0;shard<2;++shard){
   std::string bytes;std::uint64_t offset=0;
   for(;;){auto chunk=Take(adapter.ReadCheckpoint(id,shard,offset,7));Check(chunk.offset==offset,"chunk offset changed");bytes+=chunk.data;offset+=chunk.data.size();if(chunk.eof){Check(offset==chunk.total_bytes,"chunk size changed");break;}}
   std::ifstream input(metadata[shard].state_files[0],std::ios::binary);std::string original{std::istreambuf_iterator<char>(input),{}};Check(bytes==original,"export differs from frozen file");
   Rejected(adapter.ReadCheckpoint(id,shard,offset+1,7));
   auto path=temp.path/(id+"-shard-"+std::to_string(shard));std::ofstream(path,std::ios::binary).write(bytes.data(),bytes.size());copies.push_back(path);
  }
  if(step==5){
   Rejected(actual[0]->ResumeCheckpoint({"periodic",1,"step-2"}));
   restored=Make("recovered",temp.path/"restored");for(unsigned i=0;i<2;++i)Take(restored[i]->Restore(copies[i]));Compare(restored,reference);
   recovered=std::make_unique<entropy::TableAdapter>(entropy::AdapterOptions{"recovered"},Base(restored));Take(recovered->CreateTable(Definition()));
  }
  if(step==2){retry->fail=true;Rejected(adapter.Resume(id));Rejected(adapter.PrepareAsync(Features()));}
  Take(adapter.Resume(id));Take(adapter.Resume(id));Rejected(adapter.ReadCheckpoint(id,0,0,7));
  Rejected(adapter.PullPrepared(token));
 }
}
void PartialSave(){
 Temp temp;auto homes=Make("partial",temp.path);entropy::TableAdapter adapter({"partial"},Base(homes));Take(adapter.CreateTable(Definition()));
 Take(adapter.Save("first"));Take(adapter.Resume("first"));
 auto held=Take(homes[0]->Lookup({{"partial",99,0,Phase::lookup},1,true,{{1,0,Pooling::sum,{2},{0,1},{1}}}}));
 Rejected(adapter.Save("incomplete"));Rejected(adapter.Resume("incomplete"));Rejected(adapter.Resume("first"));
 Take(homes[0]->Release({{"partial",99,0,Phase::release},held.ticket}));
 Rejected(homes[0]->ResumeCheckpoint({"partial",1,"incomplete"}));Rejected(homes[0]->ResumeCheckpoint({"partial",1,"first"}));
 Rejected(adapter.PrepareAsync(Features()));
}
}
int main(){try{Periodic();PartialSave();std::cout<<"PASS repeated checkpoints, chunked export, monotonic identities, exact restore trajectory and partial-save fencing\n";}catch(const std::exception& e){std::cerr<<"FAIL periodic checkpoint: "<<e.what()<<'\n';return 1;}}
