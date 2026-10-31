#include "eps/compact_gradients.hpp"
#include "access/batch_access.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace eps;
namespace {
void Check(bool x,const char* why){if(!x)throw std::runtime_error(why);}
template<typename F>void Reject(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}Check(rejected,"malformed compact gradient accepted");}
}
int main(){
  try{
    detail::ShardStore store(0,1,1,1<<24);
    store.Create({1,128,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},0.25f,0.25f,42});
    std::vector<detail::PreparedFeature> features;
    for(auto f:std::vector<FeatureInput>{{1,0,Pooling::mean,{1,1,2,3},{0,3,4},{3,1}},
        {1,1,Pooling::sum,{3,1},{0,2},{2}},{1,2,Pooling::sequence,{2,1},{0,2},{2}}})
      features.push_back(detail::PrepareFeature(std::move(f),store));
    std::vector<TensorBlock> gradients;
    for(const auto& f:features){auto g=f.contribution;for(std::size_t i=0;i<g.values.size();++i)g.values[i]=std::sin(float(i+f.input.feature));gradients.push_back(std::move(g));}
    auto original=detail::ReduceGradients(features,gradients,store).at(1);
    auto compact=detail::BuildCompactGradients(features,gradients,{{1,3},{1,1}},128);
    auto expanded=ExpandCompactGradients(compact);
    for(std::size_t r=0;r<expanded.rows.size();++r){
      auto it=std::find(original.rows.begin(),original.rows.end(),expanded.rows[r]);Check(it!=original.rows.end(),"missing compact target");
      const auto index=std::distance(original.rows.begin(),it);
      for(int d=0;d<128;++d)Check(std::abs(expanded.values[r*128+d]-original.values[index*128+d])<1e-6f,"compact mean/sequence/cross-feature gradient changed");
    }
    FeatureInput pooled{1,3,Pooling::mean,{},{0,64},{64}};
    std::vector<RowKey> rows;for(Key key=1;key<=64;++key){pooled.keys.push_back(key);rows.push_back({1,key});}
    auto f=detail::PrepareFeature(pooled,store);auto g=f.contribution;std::fill(g.values.begin(),g.values.end(),1);
    auto compressed=detail::BuildCompactGradients({f},{g},rows,128);
    Check(CompactGradientBytes(compressed)<rows.size()*128*4/8,"long pooled bag did not reduce gradient payload");
    auto bad=compact;bad.source_rows[0]=UINT64_MAX;Reject([&]{ExpandCompactGradients(bad);});
    bad=compact;bad.row_offsets.back()++;Reject([&]{ExpandCompactGradients(bad);});
    bad=compact;bad.rows[1]=bad.rows[0];Reject([&]{ExpandCompactGradients(bad);});
    bad=compact;bad.scales[0]=std::numeric_limits<float>::quiet_NaN();Reject([&]{ExpandCompactGradients(bad);});
    bad=compact;bad.source_values[0]=std::numeric_limits<float>::infinity();Reject([&]{ExpandCompactGradients(bad);});
    std::cout<<"PASS compact gradient ordering, pooling, sequence, payload reduction and validation\n";return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL compact gradients: "<<e.what()<<'\n';return 1;}
}
