#include "eps/agent_runtime.hpp"
#include <iostream>
#include <stdexcept>
using namespace eps;
template<typename T>T Take(std::future<Result<T>> f){auto r=f.get();if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
void Check(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
int main(){
  try{
    auto backend=std::make_shared<HostCacheBackend>(0,64);AgentRuntime runtime("agent-test",backend);
    TableDefinition table{1,2,1,1,{0.1f,0.9f,0.999f,1e-8f,0.9f,0.999f},0,0,42};
    auto id=[](std::uint64_t n){return AgentCommand{"agent-test",0,n};};
    Take(runtime.Define(id(0),table));Take(runtime.Define(id(0),table));
    MigrationPlan load{"one",0,{{{1,1},0,Placement::loading,0,1,0}},Placement::gpu,24};
    RowState state{{1,1},{0,0},{0,0},{0,0},0};
    auto first=Take(runtime.Load(id(1),load,{state}));auto second=Take(runtime.Load(id(1),load,{state}));
    Check(first[0].slot==second[0].slot && Take(runtime.FreeSlots("agent-test",1))==1,"load retry allocated twice");
    UpdateContext context{{"agent-test",0,0,Phase::push},0,1,0,0.1f,0.9f,0.999f};
    Take(runtime.Apply(id(2),context,{{{1,1}},{1,1}}));Take(runtime.Apply(id(2),context,{{{1,1}},{1,1}}));
    auto retired=runtime.Load(id(1),load,{state}).get();
    Check(std::holds_alternative<Error>(retired)&&std::get<Error>(retired).code==Error::Code::retired_request,"old load replayed after update");
    CompactGradients compact{{{1,1}},2,{0.5f,0.25f},{0,2},{0,0},{1,1}};
    context.original_request.sequence=1;context.optimizer_sequence=1;context.beta1_power*=0.9f;context.beta2_power*=0.999f;
    Take(runtime.ApplyCompact(id(3),context,compact));Take(runtime.ApplyCompact(id(3),context,compact));
    MigrationPlan evict{"two",0,first,Placement::cpu,24};
    auto current=Take(runtime.ReadBack(id(4),evict));auto replay=Take(runtime.ReadBack(id(4),evict));
    Check(current[0].row_updates==2 && current[0].weights==replay[0].weights,"cached update/readback replay mismatch");
    Take(runtime.CommitEviction(id(5),evict));Take(runtime.CommitEviction(id(5),evict));
    Check(Take(runtime.FreeSlots("agent-test",1))==2,"commit did not reclaim slots");
    Take(runtime.Define(id(7),table));
    auto skipped=runtime.Define(id(6),table).get();
    Check(std::holds_alternative<Error>(skipped)&&std::get<Error>(skipped).code==Error::Code::retired_request,
      "forward skip did not fence earlier commands");
    std::cout<<"PASS agent control replay, retired command rejection and acknowledged eviction\n";return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL agent: "<<e.what()<<'\n';return 1;}
}
