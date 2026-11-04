#include "eps/entropy_adapter.hpp"
#include "eps/runtime.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace eps;
namespace {
template<typename T>T Take(Result<T> r){if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<typename T>T Take(std::future<Result<T>> r){return Take(r.get());}
void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
}
int main(){
  try {
    RuntimeOptions options;options.run_id="adapter-check";
    auto shard=std::make_shared<CpuParameterService>(options);
    entropy::TableAdapter adapter({"adapter-check",0,1,2,4},{shard});
    Take(adapter.CreateTable({1,2,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},0.25f,0.25f,42}));
    std::vector<entropy::Feature> features{
      {1,0,Pooling::mean,{1,0,1,2,0},{0,3,4,5},0},
      {1,1,Pooling::sequence,{0,2,1,0},{0,2,4},0}};
    auto pulled=Take(adapter.Pull(features));
    Check(std::holds_alternative<Error>(adapter.CompletedReadFrontier()),"certified reads with a live batch");
    Check(pulled.features[0].shape==std::vector<std::uint64_t>({3,2}),"mean adapter shape");
    Check(pulled.features[0].values==std::vector<float>({0.25,0.25,0.25,0.25,0,0}),"padding changed mean denominator");
    Check(pulled.features[1].values==std::vector<float>({0,0,0.25,0.25,0.25,0.25,0,0}),"sequence padding positions");
    Check(std::holds_alternative<Error>(adapter.Join()),"Join passed with a live backward");
    auto gradients=pulled.features;
    gradients[0].values={2,2,3,3,99,99};gradients[1].values={99,99,5,5,7,7,99,99};
    Take(adapter.Push(pulled.handle,gradients));Take(adapter.Push(pulled.handle,gradients));Take(adapter.Join());
    Check(Take(adapter.CompletedReadFrontier())==1,"read frontier did not advance after completion");
    auto loc=Take(shard->Locations({{1,1},{1,2}}));auto states=Take(shard->ReadState(loc));
    // Key 1 receives (2 / two valid IDs) twice + 7 = 9; key 2 receives 3 + 5 = 8.
    Check(std::abs(states[0].first_moment[0]-0.9f)<1e-6f,"mean gradient divided incorrectly");
    Check(std::abs(states[1].first_moment[0]-0.8f)<1e-6f,"sequence gradient mapping changed");
    Check(states[0].row_updates==1 && states[1].row_updates==1,"duplicate Push reapplied");
    Check(Take(shard->Stats()).rows==2,"padding created a parameter row");
    auto eval=Take(adapter.Pull(features,true));Take(adapter.Release(eval.handle));Take(adapter.Release(eval.handle));
    Check(std::holds_alternative<Error>(adapter.Push(eval.handle,gradients)),"Push after Release succeeded");
    Take(adapter.Join());
    std::cout<<"PASS embedding adapter padding, mean/sequence gradients, completion and replay\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL adapter: "<<e.what()<<'\n';return 1;}
}
