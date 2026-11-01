#include "store/snapshot.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace eps;
int main(int argc,char** argv){
 try{
  if(argc<7)throw std::runtime_error("expected snapshot A, snapshot B, shard, count, epoch, selected table IDs");
  const auto shard=std::stoul(argv[3]),count=std::stoul(argv[4]);const auto epoch=std::stoull(argv[5]);
  auto a=detail::ReadSnapshot(argv[1],"snapshot-comparison-a",shard,count,epoch,16ULL<<30);
  auto b=detail::ReadSnapshot(argv[2],"snapshot-comparison-b",shard,count,epoch,16ULL<<30);
  double maximum=0;std::uint64_t rows=0;
  auto compare=[&](float x,float y){if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("non-finite restored state");maximum=std::max(maximum,double(std::abs(x-y)));};
  for(int i=6;i<argc;++i){
   auto id=std::stoul(argv[i]);auto& x=a->Table(id);auto& y=b->Table(id);
   if(x.definition!=y.definition || x.optimizer_sequence!=y.optimizer_sequence || x.keys.size()!=y.keys.size())throw std::runtime_error("table configuration/clock mismatch");
   compare(x.beta1_power,y.beta1_power);compare(x.beta2_power,y.beta2_power);
   for(auto key:x.keys){auto at=y.index.find(key);if(at==y.index.end())throw std::runtime_error("restored key missing");auto p=x.At(x.index.at(key)),q=y.At(at->second);
    if(*p.updates!=*q.updates)throw std::runtime_error("restored row counter mismatch");
    for(std::uint32_t d=0;d<x.definition.dimension;++d){compare(p.weights[d],q.weights[d]);compare(p.first[d],q.first[d]);compare(p.second[d],q.second[d]);}++rows;
   }
  }
  if(maximum>1e-5)throw std::runtime_error("restored parameter/Adam state differs");
  std::cout<<"{\"checked_rows\":"<<rows<<",\"max_state_absolute_error\":"<<maximum<<"}\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
