#include "eps/cuda_cache_backend.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace eps;
namespace {
template<typename T>T Take(std::future<Result<T>> f){
  auto r=f.get();if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void Near(float a,float b){Check(std::isfinite(a)&&std::abs(a-b)<1e-5f,"CUDA/host reference mismatch");}
TableDefinition Table(TableId id,std::uint32_t dimension){
  return {id,dimension,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},-0.1f,0.1f,42};
}
std::vector<RowState> States(){
  std::vector<RowState> rows;
  for(TableId table:{1,2})for(Key key=1;key<=(table==1?4:2);++key){
    const auto dim=table==1?128:64;
    RowState row{{table,key},std::vector<float>(dim),std::vector<float>(dim),std::vector<float>(dim),3};
    for(int d=0;d<dim;++d){
      row.weights[d]=0.05f*std::sin(float(key*7+d));row.first_moment[d]=0.001f*(d%5);
      row.second_moment[d]=0.002f*(1+d%3);
    }rows.push_back(std::move(row));
  }return rows;
}
MigrationPlan Plan(const std::vector<RowState>& rows){
  MigrationPlan p{"gpu-test",0,{},Placement::gpu,0};
  for(const auto& row:rows)p.rows.push_back({row.row,0,Placement::loading,0,1,0});return p;
}
std::vector<FeatureInput> Layouts(){return {
  {1,0,Pooling::mean,{1,1,999,2},{0,3,4},{3,1}},
  {1,1,Pooling::sequence,{3,1,3},{0,2,3},{2,1}},
  {2,2,Pooling::sum,{1,2},{0,2,2},{2,0}}};}
std::vector<AgentWork> Work(const std::vector<LocatedRow>& locations){
  auto find=[&](TableId table,Key key){return *std::find_if(locations.begin(),locations.end(),
    [&](const auto& loc){return loc.row==RowKey{table,key};});};
  return {{0,AccessRoute::resident_gpu,0,{find(1,1),find(1,2)},{0,0,UINT64_MAX,1},{}},
          {1,AccessRoute::resident_gpu,0,{find(1,3),find(1,1)},{0,1,0},{}},
          {2,AccessRoute::resident_gpu,0,{find(2,1),find(2,2)},{0,1},{}}};
}
void Compare(const std::vector<TensorBlock>& a,const std::vector<TensorBlock>& b){
  Check(a.size()==b.size(),"CUDA output count");
  for(std::size_t i=0;i<a.size();++i){Check(a[i].shape==b[i].shape,"CUDA output layout");
    for(std::size_t j=0;j<a[i].values.size();++j)Near(a[i].values[j],b[i].values[j]);}
}
void CompareStates(const std::vector<RowState>& a,const std::vector<RowState>& b){
  Check(a.size()==b.size(),"CUDA state count");
  for(std::size_t i=0;i<a.size();++i){Check(a[i].row==b[i].row&&a[i].row_updates==b[i].row_updates,"CUDA state identity/counter");
    for(std::size_t j=0;j<a[i].weights.size();++j){Near(a[i].weights[j],b[i].weights[j]);
      Near(a[i].first_moment[j],b[i].first_moment[j]);Near(a[i].second_moment[j],b[i].second_moment[j]);}}
}
void AtomicReserve(){
  CudaCacheOptions options;options.rows_per_table=3;options.state_bytes=96;
  options.transfer_bytes=1<<20;options.outstanding_output_bytes=1<<20;
  CudaCacheBackend gpu(options);Take(gpu.Define(Table(99,2)));
  const auto load=[&](Key first,std::size_t count){
    std::vector<RowState> states;
    for(std::size_t n=0;n<count;++n)states.push_back({{99,first+n},{0.1f,0.2f},{0,0},{0,0},0});
    auto plan=Plan(states);plan.reserve_slots=1;
    return gpu.Load(std::move(plan),std::move(states));
  };
  auto first=load(1,2),second=load(3,1);Take(std::move(first));auto rejected=second.get();
  Check(std::holds_alternative<Error>(rejected) &&
    std::get<Error>(rejected).code==Error::Code::capacity_exceeded && Take(gpu.FreeSlots(99))==1,
    "CUDA concurrent load consumed slot reserve");
}
}
int main(){
  try{
    AtomicReserve();
    CudaCacheOptions options;options.rows_per_table=8;options.state_bytes=1<<20;
    options.transfer_bytes=1<<20;options.outstanding_output_bytes=1<<20;
    auto gpu=std::make_shared<CudaCacheBackend>(options);
    HostCacheBackend host(0,1<<20);
    for(auto table:{Table(1,128),Table(2,64)}){Take(gpu->Define(table));Take(host.Define(table));}
    auto rows=States();auto plan=Plan(rows);
    auto gl=Take(gpu->Load(plan,rows)),hl=Take(host.Load(plan,rows));
    auto gw=Work(gl),hw=Work(hl);auto layout=Layouts();
    Compare(Take(gpu->GatherHost(gw,layout)),Take(host.GatherHost(hw,layout)));
    // CPU-owned unique vectors use the same pooling kernels without entering
    // the resident cache or carrying optimizer state across the link.
    auto cold_gpu=gw,cold_host=hw;
    for(auto* work:{&cold_gpu,&cold_host})for(auto& task:*work){
      task.route=AccessRoute::transfer_unique;
      for(auto& loc:task.rows){
        loc.placement=Placement::cpu;
        const auto& state=*std::find_if(rows.begin(),rows.end(),[&](const auto& r){return r.row==loc.row;});
        task.transferred_weights.insert(task.transferred_weights.end(),state.weights.begin(),state.weights.end());
      }
    }
    Compare(Take(gpu->GatherHost(cold_gpu,layout)),Take(host.GatherHost(cold_host,layout)));
    auto device=Take(gpu->GatherDevice(gw,layout));
    auto before=Take(host.GatherHost(hw,layout));
    for(std::size_t i=0;i<device.size();++i){
      device[i].tensor->WaitOnStream(nullptr);auto copy=device[i].tensor->CopyToHost();
      Check(copy.size()==before[i].values.size(),"device result dimensions");
      for(std::size_t j=0;j<copy.size();++j)Near(copy[j],before[i].values[j]);
    }
    float b1=0.9f,b2=0.999f;
    for(std::uint64_t step=0;step<7;++step){
      UpdateContext c{{"cuda-test",0,step,Phase::push},0,1,step,0.01f,b1,b2};
      RowGradients g;for(Key key=1;key<=4;++key){g.rows.push_back({1,key});
        for(int d=0;d<128;++d)g.values.push_back(0.03f*std::cos(float(key+d+step)));}
      Take(gpu->ApplyHost(c,g));Take(host.ApplyHost(c,g));
      b1*=0.9f;b2*=0.999f;
      Compare(Take(gpu->GatherHost(gw,layout)),Take(host.GatherHost(hw,layout)));
    }
    MigrationPlan gp{"read",0,gl,Placement::cpu,0},hp{"read",0,hl,Placement::cpu,0};
    CompactGradients compact{{{1,1},{1,2},{1,3},{1,4}},128,{},{0,3,4,6,7},
      {0,1,0,2,1,2,0},{1.0f/3,0.5f,1.0f/3,1,0.25f,0.25f,1}};
    for(int source=0;source<3;++source)for(int d=0;d<128;++d)compact.source_values.push_back(0.01f*std::sin(float(source*11+d)));
    for(std::uint64_t step=7;step<9;++step){
      UpdateContext context{{"cuda-test",0,step,Phase::push},0,1,step,0.01f,b1,b2};
      Take(gpu->ApplyCompact(context,compact));Take(host.ApplyCompact(context,compact));
      b1*=0.9f;b2*=0.999f;
    }
    std::reverse(gp.rows.begin(),gp.rows.end());std::reverse(hp.rows.begin(),hp.rows.end());
    CompareStates(Take(gpu->ReadBack(gp)),Take(host.ReadBack(hp)));
    auto original=Take(gpu->ReadBack(gp));
    UpdateContext bad{{"cuda-test",0,9,Phase::push},0,1,9,0.01f,b1,b2};
    auto compact_overflow=compact;
    std::fill(compact_overflow.source_values.begin(),compact_overflow.source_values.end(),std::numeric_limits<float>::max());
    Check(std::holds_alternative<Error>(gpu->ApplyCompact(bad,compact_overflow).get()),"CUDA compact overflow accepted");
    CompareStates(original,Take(gpu->ReadBack(gp)));
    RowGradients overflow{{{1,1}},std::vector<float>(128,1)};
    overflow.values.back()=std::numeric_limits<float>::max();
    Check(std::holds_alternative<Error>(gpu->ApplyHost(bad,overflow).get()),"CUDA overflow accepted");
    CompareStates(original,Take(gpu->ReadBack(gp)));
    for(std::size_t i=0;i<device.size();++i){auto copy=device[i].tensor->CopyToHost();
      for(std::size_t j=0;j<copy.size();++j)Near(copy[j],before[i].values[j]);}
    Take(gpu->CommitEviction(gp));Check(Take(gpu->FreeSlots(1))==8,"CUDA slot capacity not reclaimed");
    for(auto& row:rows)row=*std::find_if(original.begin(),original.end(),[&](const auto& x){return x.row==row.row;});
    for(auto& location:plan.rows)location.generation+=2;
    auto reloaded=Take(gpu->Load(plan,rows));
    Check(std::holds_alternative<Error>(gpu->GatherHost(gw,layout).get()),"stale CUDA slot remained usable after reuse");
    auto final_work=Work(reloaded);Take(gpu->GatherHost(final_work,layout));
    gpu.reset(); // Owning device results outlive the backend and its staging buffers.
    for(std::size_t i=0;i<device.size();++i){auto copy=device[i].tensor->CopyToHost();
      for(std::size_t j=0;j<copy.size();++j)Near(copy[j],before[i].values[j]);}
    std::cout<<"PASS real CUDA multi-table gather, optimizer states, atomic rejection, slot reuse and device-result lifetime\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL CUDA: "<<e.what()<<'\n';return 1;}
}
