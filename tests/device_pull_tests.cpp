#include "eps/entropy_adapter.hpp"
#include "eps/cache_resolver.hpp"
#include "eps/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <unistd.h>
#ifdef EPS_DEVICE_CUDA
#include "eps/cuda_cache_backend.hpp"
#include "eps/worker_cache_agent.hpp"
#include "eps/brpc_cache_backend.hpp"
std::vector<float> DeviceModelGradient(const eps::DeviceTensorBlock&);
#endif
using namespace eps;
template<typename T>T Take(Result<T> r){if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<typename T>T Take(std::future<Result<T>> f){return Take(f.get());}
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void Near(const std::vector<float>& a,const std::vector<float>& b){Check(a.size()==b.size(),"device comparison size");
  for(std::size_t i=0;i<a.size();++i)Check(std::isfinite(a[i]) && std::abs(a[i]-b[i])<5e-6f,"device/host trajectory difference");}

// Explicit CPU protocol reference. CUDA builds use real device results instead.
class ReferenceTensor final:public DeviceTensor {
  std::vector<float> values_;
 public:
  explicit ReferenceTensor(std::vector<float> v):values_(std::move(v)){}
  void* data()const override{return const_cast<float*>(values_.data());}
  std::uint64_t elements()const override{return values_.size();}
  int device()const override{return -1;}
  void WaitOnStream(void*)const override{}
  std::vector<float> CopyToHost()const override{return values_;}
};
Result<std::vector<DeviceTensorBlock>> ReferenceAssembly(DeviceAssemblyRequest request){
  std::map<std::uint32_t,std::size_t> indices;
  for(std::size_t i=0;i<request.host.size();++i)indices[request.host[i].feature]=i;
  for(const auto& source:request.device){auto& target=request.host.at(indices.at(source.block.feature));
    const auto values=source.block.tensor->CopyToHost();const auto dim=target.shape[1];
    for(std::size_t i=0;i<values.size();++i){const auto p=source.projected?source.positions.at(i/dim)*dim+i%dim:i;target.values.at(p)+=values[i];}
  }
  std::vector<DeviceTensorBlock> output;
  for(std::size_t i=0;i<request.host.size();++i){auto& host=request.host[i];const auto& layout=request.layouts[i];
    if(layout.pooling==Pooling::mean)for(std::size_t j=0;j<host.values.size();++j)
      if(layout.valid_lengths[j/host.shape[1]])host.values[j]/=layout.valid_lengths[j/host.shape[1]];
    for(const auto& projection:request.projections)if(projection.feature==host.feature){
      std::vector<float> padded(projection.rows*host.shape[1],0);
      for(std::size_t row=0;row<projection.positions.size();++row)
        std::copy_n(host.values.data()+row*host.shape[1],host.shape[1],padded.data()+projection.positions[row]*host.shape[1]);
      host.shape[0]=projection.rows;host.values=std::move(padded);
    }
    output.push_back({host.feature,host.shape,std::make_shared<ReferenceTensor>(std::move(host.values))});
  }
  return output;
}
std::vector<entropy::Feature> Features(){return {
  {1,10,Pooling::mean,{1,2,1,0,3,4,0},{0,4,4,7},0},
  {1,11,Pooling::sequence,{0,4,1,0,5,2,0},{0,2,7},0},
  {2,12,Pooling::sum,{2,1,2,3},{0,2,4},0},
  {2,13,Pooling::sequence,{0,0},{0,0,2},0},
  {2,14,Pooling::mean,{},{0,0},0},
  {2,15,Pooling::sequence,{},{0,0},0},
  {2,16,Pooling::mean,{0,0},{0,2},0}};}
struct Temporary {
  std::filesystem::path path;
  Temporary(){char name[]="/tmp/eps-device-pull-XXXXXX";auto* p=mkdtemp(name);Check(p,"temporary directory");path=p;}
  ~Temporary(){std::error_code ignored;std::filesystem::remove_all(path,ignored);}
};
void ResolverFailureDrains(){
  auto held=std::make_shared<std::promise<Result<std::vector<DeviceTensorBlock>>>>();
  std::promise<void> dispatched;auto signal=dispatched.get_future();
  auto resolver=MakeDeviceCacheResolver({
    {0,[](auto,auto){std::promise<Result<std::vector<DeviceTensorBlock>>> p;
      p.set_value(Error{Error::Code::unavailable,"injected native gather failure"});return p.get_future();}},
    {1,[held,&dispatched](auto,auto){auto f=held->get_future();dispatched.set_value();return f;}}});
  std::vector<AgentWork> work{{10,AccessRoute::resident_gpu,0,{{{1,1},0,Placement::gpu,0,1,0}},{0,UINT64_MAX},{}},
    {10,AccessRoute::resident_gpu,1,{{{1,2},0,Placement::gpu,1,1,0}},{UINT64_MAX,0},{}}};
  auto result=std::async(std::launch::async,[&]{return resolver(work,{{1,10,Pooling::sequence,{1,2},{0,2},{2}}});});
  signal.get();const auto before=result.wait_for(std::chrono::milliseconds(30));
  held->set_value(std::vector<DeviceTensorBlock>{});
  Check(before==std::future_status::timeout && std::holds_alternative<Error>(result.get()),"failed native resolver returned before other device read completed");
}
int main(){try{
  ResolverFailureDrains();
  Temporary temporary;std::string run="device-pull";
  std::vector<std::shared_ptr<CpuParameterService>> homes,refs;
  std::vector<std::shared_ptr<ParameterService>> shards,ref_shards;
  std::map<AgentId,std::shared_ptr<CacheBackend>> backends;
  std::map<AgentId,NativeDeviceGather> native;
#ifdef EPS_DEVICE_CUDA
  std::map<AgentId,std::shared_ptr<CudaCacheBackend>> cuda;
  std::map<AgentId,std::shared_ptr<WorkerCacheAgent>> workers;
  for(AgentId a:{0,1}){CudaCacheOptions o;o.agent=a;o.state_bytes=1<<20;o.rows_per_table=32;o.transfer_bytes=1<<20;o.outstanding_output_bytes=1<<20;
    cuda[a]=std::make_shared<CudaCacheBackend>(o);workers[a]=std::make_shared<WorkerCacheAgent>(WorkerCacheOptions{run},cuda[a]);}
  native[0]=[w=workers[0]](auto work,auto layouts){return w->GatherDevice(std::move(work),std::move(layouts));};
  backends[1]=std::make_shared<BrpcCacheBackend>(workers[1]->endpoint(),run,99,1);
#else
  for(AgentId a:{0,1})backends[a]=std::make_shared<HostCacheBackend>(a,1<<20);
  native[0]=[b=backends[0]](auto work,auto layouts){return std::async(std::launch::async,[b,work=std::move(work),layouts=std::move(layouts)]()mutable->Result<std::vector<DeviceTensorBlock>>{
    auto parts=b->GatherHost(std::move(work),std::move(layouts)).get();if(auto e=std::get_if<Error>(&parts))return *e;
    std::vector<DeviceTensorBlock> output;
    for(auto& block:std::get<std::vector<TensorBlock>>(parts))output.push_back({block.feature,block.shape,std::make_shared<ReferenceTensor>(std::move(block.values))});
    return output;
  });};
#endif
  for(ShardId s=0;s<2;++s){RuntimeOptions options;options.run_id=run;options.shard_id=s;options.shard_count=2;
    options.snapshot_directory=temporary.path/"actual";homes.push_back(std::make_shared<CpuParameterService>(options));shards.push_back(homes.back());
    options.snapshot_directory=temporary.path/"reference";refs.push_back(std::make_shared<CpuParameterService>(options));ref_shards.push_back(refs.back());
    for(AgentId a:{0,1}){
#ifdef EPS_DEVICE_CUDA
      Take(homes.back()->AttachAgent(a,std::make_shared<BrpcCacheBackend>(workers[a]->endpoint(),run,s,a)));
#else
      Take(homes.back()->AttachAgent(a,backends[a]));
#endif
    }
  }
  entropy::TableAdapter actual({run},shards),reference({run},ref_shards);
  for(TableId t:{1,2}){TableDefinition definition{t,t==1?3U:5U,2,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},-0.25f,0.25f,42};
    Take(actual.CreateTable(definition));Take(reference.CreateTable(definition));}
  auto warm=Take(actual.Pull(Features(),true));Take(actual.Release(warm.handle));
  auto warm_ref=Take(reference.Pull(Features(),true));Take(reference.Release(warm_ref.handle));
  auto migrate=[&]{for(auto [row,agent]:std::vector<std::pair<RowKey,AgentId>>{{{1,1},0},{{1,4},1},{{2,2},0}})
    Take(homes[row.key%2]->MoveRows({run,{row},agent,Placement::gpu,{1<<20,1<<20,8,0}}));};
  migrate();Take(homes[1]->ConfigureAccess(0,AccessRoute::transfer_unique));
  auto resolver=MakeDeviceCacheResolver(native,{{1,backends.at(1)}});
  std::uint64_t native_parts=0,remote_parts=0;
  actual.SetDevicePath([&](auto work,auto layouts)->Result<ResolvedDeviceFeatures>{
      auto result=resolver(std::move(work),std::move(layouts));
      if(auto p=std::get_if<ResolvedDeviceFeatures>(&result)){native_parts+=p->device.size();remote_parts+=p->host.size();}return result;
    },
#ifdef EPS_DEVICE_CUDA
    [b=cuda[0]](auto request){return b->AssembleFeatures(std::move(request)).get();}
#else
    ReferenceAssembly
#endif
  );
  auto state_check=[&]{for(TableId t:{1,2})for(Key k=1;k<=(t==1?5:3);++k){
    const auto home=k%2;auto a=Take(homes[home]->ReadState(Take(homes[home]->Locations({{t,k}})))).at(0);
    auto b=Take(refs[home]->ReadState(Take(refs[home]->Locations({{t,k}})))).at(0);
    Check(a.row_updates==b.row_updates,"device update counter mismatch");Near(a.weights,b.weights);Near(a.first_moment,b.first_moment);Near(a.second_moment,b.second_moment);
  }};
  std::vector<DeviceTensorBlock> retained;std::vector<TensorBlock> retained_host;
  for(int step=0;step<5;++step){
    entropy::DevicePullResult batch;
    if(step%2){auto token=Take(actual.PrepareAsync(Features()));batch=Take(actual.PullPreparedDevice(token));}
    else batch=Take(actual.PullDevice(Features()));
    auto baseline=Take(reference.Pull(Features()));Check(batch.features.size()==baseline.features.size(),"device feature count");
    std::vector<TensorBlock> gradients,reference_gradients;
    for(std::size_t i=0;i<batch.features.size();++i){const auto& block=batch.features[i];const auto& host=baseline.features[i];
      Check(block.feature==host.feature && block.shape==host.shape,"complete device output shape");Near(block.tensor->CopyToHost(),host.values);
      auto reference_gradient=host;for(auto& x:reference_gradient.values)x*=2;
#ifdef EPS_DEVICE_CUDA
      TensorBlock gradient{block.feature,block.shape,DeviceModelGradient(block)};
#else
      auto gradient=reference_gradient;
#endif
      Near(gradient.values,reference_gradient.values);gradients.push_back(std::move(gradient));reference_gradients.push_back(std::move(reference_gradient));
    }
    if(!step){retained=batch.features;retained_host=baseline.features;}
    Take(actual.Push(batch.handle,gradients));Take(actual.Push(batch.handle,gradients));Take(reference.Push(baseline.handle,reference_gradients));
    Take(actual.Join());state_check();
    if(step==1){Take(actual.Save("middle"));Take(reference.Save("middle"));
      for(auto& home:homes)Check(Take(home->Stats()).cached_rows==0,"device save did not write back cache");
      Take(actual.Resume("middle"));Take(reference.Resume("middle"));migrate();}
  }
  for(std::size_t i=0;i<retained.size();++i)Near(retained[i].tensor->CopyToHost(),retained_host[i].values);
  retained.clear();Take(actual.Save("final"));Take(reference.Save("final"));state_check();
  for(auto& home:homes){auto stats=Take(home->Stats());Check(stats.live_tickets==0 && stats.cached_rows==0,"device lifecycle leaked tickets/cache");}
  Check(native_parts>0 && remote_parts>0,"mixed native and remote paths not exercised");
  Take(actual.Resume("final"));for(auto& home:homes)Take(home->ConfigureAccess(0,AccessRoute::cpu_pool));
#ifdef EPS_DEVICE_CUDA
  CudaCacheOptions tiny;tiny.transfer_bytes=1<<20;tiny.outstanding_output_bytes=1;
  auto failing=std::make_shared<CudaCacheBackend>(tiny);
  actual.SetDevicePath({},[failing](auto request){return failing->AssembleFeatures(std::move(request)).get();});
#else
  actual.SetDevicePath({},[](auto)->Result<std::vector<DeviceTensorBlock>>{return Error{Error::Code::capacity_exceeded,"injected assembly budget failure"};});
#endif
  auto failed=actual.PullDevice(Features());Check(std::holds_alternative<Error>(failed) && std::get<Error>(failed).code==Error::Code::capacity_exceeded,"assembly failure not surfaced");
  for(auto& home:homes)Check(Take(home->Stats()).live_tickets==0,"failed assembly retained lookup tickets");
#ifdef EPS_DEVICE_CUDA
  Check(Take(cuda[0]->MemoryStats()).outstanding_output_bytes==0,"assembled device outputs leaked");
  Check(Take(failing->MemoryStats()).outstanding_output_bytes==0,"failed assembly output charge leaked");
  for(auto& [id,worker]:workers)worker->Stop();
  constexpr const char* backend="cuda";
#else
  constexpr const char* backend="cpu_protocol_reference";
#endif
  std::cout<<"{\"test\":\"distributed_device_pull\",\"backend\":\""<<backend<<"\",\"shards\":2,\"training_steps\":5,\"full_optimizer_rows_per_step\":8,\"features_per_step\":7,\"prepared_inputs\":2,\"native_parts\":"<<native_parts<<",\"remote_parts\":"<<remote_parts<<",\"mid_run_save_resume\":true,\"assembly_failure_tickets_released\":true,\"resolver_failure_drained\":true}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<"FAIL device Pull: "<<e.what()<<'\n';return 1;}}
