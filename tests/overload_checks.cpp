#include "eps/brpc_client.hpp"
#include "eps/brpc_cache_backend.hpp"
#include <iostream>
#include <stdexcept>
using namespace eps;
template<typename T>T Take(std::future<Result<T>> f){auto r=f.get();if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
void Check(bool x,const char* why){if(!x)throw std::runtime_error(why);}
int main(int argc,char** argv){
  try{
    Check(argc==3,"expected parameter and agent endpoints");
    BrpcParameterService service(argv[1]);BrpcCacheBackend agent(argv[2],"overload-test",0,0);
    Take(service.CreateTable({1,128,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},0.25f,0.25f,42}));
    auto request=[](std::uint32_t learner){return LookupRequest{{"overload-test",learner,0,Phase::lookup},1,true,
      {{1,0,Pooling::sum,{1},{0,1},{1}}}};};
    auto initial=Take(service.Lookup(request(999)));auto release=request(999).id;release.phase=Phase::release;
    Take(service.Release({release,initial.ticket}));
    Take(service.MoveRows({"overload-test",{{1,1}},0,Placement::gpu,{1<<20,1<<20,1,0}}));
    std::vector<std::future<Result<LookupReply>>> pending;
    for(std::uint32_t i=0;i<64;++i)pending.push_back(service.Lookup(request(i)));
    std::vector<std::pair<std::uint32_t,LookupReply>> accepted;std::vector<std::uint32_t> rejected;
    for(std::uint32_t i=0;i<64;++i){auto result=pending[i].get();
      if(auto e=std::get_if<Error>(&result)){Check(e->code==Error::Code::capacity_exceeded,"overload returned an unexpected error");rejected.push_back(i);}
      else accepted.emplace_back(i,std::get<LookupReply>(std::move(result)));}
    Check(accepted.size()==32 && rejected.size()==32 && Take(service.Stats()).live_tickets==32,"configured ticket bound was not enforced");
    auto features=Take(agent.GatherHost(accepted[0].second.agent_work,request(accepted[0].first).features));
    Check(features.size()==1 && features[0].values==std::vector<float>(128,0.25f),"overload corrupted cached reads");
    auto held=service.MoveRows({"overload-test",{{1,1}},0,Placement::cpu,{1<<20,1<<20,1,0}}).get();
    Check(std::holds_alternative<Error>(held) && std::get<Error>(held).code==Error::Code::unavailable,"overload bypassed row references");
    std::vector<std::future<Result<Ack>>> releases;
    for(auto& [learner,reply]:accepted){auto id=request(learner).id;id.phase=Phase::release;releases.push_back(service.Release({id,reply.ticket}));}
    for(auto& f:releases)Take(std::move(f));
    auto retried=Take(service.Lookup(request(rejected[0])));auto id=request(rejected[0]).id;id.phase=Phase::release;
    Take(service.Release({id,retried.ticket}));
    Check(Take(service.Stats()).live_tickets==0,"overload leaked lookup references");
    auto cpu=Take(service.MoveRows({"overload-test",{{1,1}},0,Placement::cpu,{1<<20,1<<20,1,0}}));
    auto state=Take(service.ReadState(cpu)).at(0);
    Check(state.row_updates==0 && state.weights==std::vector<float>(128,0.25f),"read-only overload changed parameter state");
    std::cout<<"{\"test\":\"rpc_cache_backpressure\",\"configured_live_ticket_limit\":32,\"offered\":64,\"admitted\":32,\"rejected\":32,\"retry_after_release\":true,\"state_preserved\":true}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL overload: "<<e.what()<<'\n';return 1;}
}
