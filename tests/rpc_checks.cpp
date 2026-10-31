#include "eps/brpc_client.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/distributed_client.hpp"
#include "eps/runtime.hpp"
#include "transport/codec.hpp"
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace eps;
namespace {
template<typename T>T Take(Result<T> r) {
  if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
template<typename T>T Take(std::future<Result<T>> f){return Take(f.get());}
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void Near(float a,float b){Check(std::isfinite(a)&&std::abs(a-b)<2e-6f,"distributed numeric mismatch");}
TableDefinition Table(std::uint32_t count){
  return {1,2,count,1,{0.1f,0.9f,0.999f,1e-8f,0.9f,0.999f},0.25f,0.25f,42};
}
LookupRequest Request(std::string run,std::uint64_t seq,bool read_only=false) {
  return {{run,0,seq,Phase::lookup},1,read_only,{
    {1,0,Pooling::sum,{1,1,2,5},{0,3,4,4},{3,1,0}},
    {1,1,Pooling::mean,{3,4,3,6},{0,2,4},{2,2}},
    {1,2,Pooling::sequence,{5,2,5},{0,2,3},{2,1}}}};
}
std::vector<TensorBlock> Gradients(const DistributedBatch& batch){
  auto out=batch.features();
  for(auto& f:out)for(std::size_t i=0;i<f.values.size();++i)f.values[i]=0.25f*(i+1)*(f.feature+1);
  return out;
}
void Train(DistributedClient& client,const std::string& run,std::uint64_t seq){
  auto batch=Take(client.Lookup(Request(run,seq)));
  auto gradients=Gradients(batch);Take(client.Push(batch,gradients));
  Take(client.Push(batch,gradients)); // retained retry must not apply twice
}
void CompareState(const std::vector<std::shared_ptr<ParameterService>>& remote,
                  CpuParameterService& reference){
  for(Key key=1;key<=6;++key){
    auto expected=Take(reference.ReadState({{{1,key},0,Placement::cpu,0,0,0}})).at(0);
    const auto home=static_cast<ShardId>(key%remote.size());
    auto connection=std::dynamic_pointer_cast<BrpcParameterService>(remote[home]);
    auto location=Take(connection->Locations({{1,key}}));
    auto actual=Take(remote[home]->ReadState(location)).at(0);
    Check(actual.row_updates==expected.row_updates,"distributed update count mismatch");
    for(std::size_t i=0;i<actual.weights.size();++i){
      Near(actual.weights[i],expected.weights[i]);Near(actual.first_moment[i],expected.first_moment[i]);
      Near(actual.second_moment[i],expected.second_moment[i]);
    }
  }
}
void MalformedAttachment(const std::string& endpoint){
  brpc::Channel channel;Check(channel.Init(endpoint.c_str(),nullptr)==0,"raw channel failed");
  eps::v1::EmbeddingService_Stub stub(&channel);
  eps::v1::LookupRequest q;eps::v1::LookupReply r;brpc::Controller c;c.set_timeout_ms(2000);
  q.mutable_identity()->set_run("wire-test");q.mutable_identity()->set_phase(eps::v1::LOOKUP);
  q.set_partition_epoch(1);auto* f=q.add_features();f->set_table_id(1);f->set_pooling(eps::v1::SUM);
  f->mutable_keys()->set_element_type(eps::v1::ELEMENT_UINT64);
  f->mutable_keys()->set_element_count(UINT64_MAX); // multiplication/offset attack
  stub.Lookup(&c,&q,&r,nullptr);
  Check(!c.Failed()&&r.status().code()==eps::v1::Status::INVALID_ARGUMENT,"malformed attachment was accepted");
}
}
int main(int argc,char** argv){
  try{
    Check(argc==4 || argc==5,"expected two endpoints, fresh/resumed mode, optional cache-agent endpoint");
    const bool caching=argc==5;
    const bool resumed=std::string(argv[3])=="resumed";
    const std::string run=resumed?"wire-resumed":"wire-test";
    std::vector<std::shared_ptr<ParameterService>> remote={
      std::make_shared<BrpcParameterService>(argv[1]),std::make_shared<BrpcParameterService>(argv[2])};
    DistributedClient client(remote);Take(client.CreateTable(Table(2)));
    if(caching){
      auto reader=std::make_shared<BrpcCacheBackend>(argv[4],run,0,0);
      client.SetCacheResolver([reader](auto work,auto layouts){return reader->GatherHost(std::move(work),std::move(layouts)).get();});
    }
    if(resumed){
      auto packet=Request("wire-test",0,true);
      // Only shard-0-owned keys; rejection must be about the old run, not routing.
      packet.features={{1,0,Pooling::sum,{2},{0,1},{1}}};
      auto old=remote[0]->Lookup(std::move(packet)).get();
      Check(std::holds_alternative<Error>(old),"packet from old run was accepted");
    }
    RuntimeOptions options;options.run_id="baseline";
    auto reference=std::make_shared<CpuParameterService>(options);
    DistributedClient baseline({reference});Take(baseline.CreateTable(Table(1)));
    if(resumed){Train(baseline,"baseline",0);Train(baseline,"baseline",1);}
    if(caching){
      if(!resumed){
        auto warm=Take(client.Lookup(Request(run,0,true)));Take(client.Release(warm));
        auto ref=Take(baseline.Lookup(Request("baseline",0,true)));Take(baseline.Release(ref));
      }
      for(std::size_t shard=0;shard<remote.size();++shard){
        auto connection=std::dynamic_pointer_cast<BrpcParameterService>(remote[shard]);
        auto rows=shard?std::vector<RowKey>{{1,3},{1,5}}:std::vector<RowKey>{{1,2}};
        auto moved=Take(connection->MoveRows({run,rows,0,Placement::gpu,{1<<20,1<<20,8,0}}));
        Check(moved.size()==rows.size()&&moved[0].placement==Placement::gpu,"remote cache admission failed");
      }
    }
    const std::uint64_t count=resumed?1:2;
    for(std::uint64_t i=0;i<count;++i){
      const auto seq=i+(caching&&!resumed?1:0);
      auto batch=Take(client.Lookup(Request(run,seq)));
      auto single=Take(baseline.Lookup(Request("baseline",resumed?2:seq)));
      if(caching&&i==0){
        auto connection=std::dynamic_pointer_cast<BrpcParameterService>(remote[0]);
        auto blocked=connection->MoveRows({run,{{1,2}},0,Placement::cpu,{1<<20,1<<20,8,0}}).get();
        Check(std::holds_alternative<Error>(blocked)&&std::get<Error>(blocked).code==Error::Code::unavailable,
          "remote migration reclaimed a referenced row");
      }
      for(std::size_t fi=0;fi<batch.features().size();++fi){
        const auto& a=batch.features()[fi];const auto& b=single.features()[fi];
        Check(a.shape==b.shape,"distributed output shape mismatch");
        for(std::size_t j=0;j<a.values.size();++j)Near(a.values[j],b.values[j]);
      }
      auto grads=Gradients(batch);Take(client.Push(batch,grads));Take(client.Push(batch,grads));
      Take(baseline.Push(single,grads));CompareState(remote,*reference);
    }
    auto eval=Take(client.Lookup(Request(run,count+(caching&&!resumed?1:0),true)));Take(client.Release(eval));
    if(!resumed){
      MalformedAttachment(argv[1]);
      for(auto& shard:remote){auto saved=Take(shard->DrainAndFlush("checkpoint"));
        Check(saved.state_files.size()==1,"missing checkpoint shard");}
    }
    std::cout<<"PASS two-shard "<<(resumed?"restore trajectory":"lookup, mean, sequence, updates and retry")<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL RPC: "<<e.what()<<'\n';return 1;}
}
