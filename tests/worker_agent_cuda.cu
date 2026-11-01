#include "eps/worker_cache_agent.hpp"
#include "eps/cuda_cache_backend.hpp"
#include "eps/brpc_cache_backend.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace eps;
template<typename T>T Take(std::future<Result<T>> f){auto r=f.get();
  if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void CUDA(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void Near(const std::vector<float>& a,const std::vector<float>& b){Check(a.size()==b.size(),"CUDA result count");
  for(std::size_t i=0;i<a.size();++i)Check(std::isfinite(a[i]) && std::abs(a[i]-b[i])<1e-6f,"CUDA worker/reference difference");}
__global__ void Model(const float* features,float* output,std::uint64_t count){
  const auto i=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<count)output[i]=2*features[i]+1;
}
struct ModelStream {
  cudaStream_t stream=nullptr;
  float* output=nullptr;
  ModelStream(){CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));CUDA(cudaMalloc(&output,1024));}
  ~ModelStream(){if(stream)cudaStreamSynchronize(stream);if(output)cudaFree(output);if(stream)cudaStreamDestroy(stream);}
  std::vector<float> Run(const DeviceTensorBlock& input){
    const auto n=input.tensor->elements();Check(n<=256,"model scratch bound");
    input.tensor->WaitOnStream(stream);
    if(n)Model<<<(n+255)/256,256,0,stream>>>(static_cast<const float*>(input.tensor->data()),output,n);
    CUDA(cudaGetLastError());std::vector<float> result(n);
    if(n)CUDA(cudaMemcpyAsync(result.data(),output,n*sizeof(float),cudaMemcpyDeviceToHost,stream));
    CUDA(cudaStreamSynchronize(stream));return result;
  }
};
int main(){try{
  CudaCacheOptions options;options.rows_per_table=4;options.state_bytes=1<<20;
  options.transfer_bytes=1<<20;options.outstanding_output_bytes=160;
  auto backend=std::make_shared<CudaCacheBackend>(options);std::weak_ptr<CacheBackend> lifetime=backend;
  auto worker=std::make_unique<WorkerCacheAgent>(WorkerCacheOptions{"worker-cuda"},backend);
  BrpcCacheBackend control(worker->endpoint(),"worker-cuda",0,0,10000);
  TableDefinition table{1,4,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},0,0,42};
  Take(control.Define(table));HostCacheBackend reference(0,1<<20);Take(reference.Define(table));
  std::vector<RowState> initial{{{1,1},{0.1f,0.2f,0.3f,0.4f},{0,0,0,0},{0,0,0,0},0},
    {{1,2},{0.5f,0.6f,0.7f,0.8f},{0,0,0,0},{0,0,0,0},0}};
  MigrationPlan load{"load",0,{{{1,1},0,Placement::loading,0,1,0},{{1,2},0,Placement::loading,0,1,0}},Placement::gpu,96};
  auto rows=Take(control.Load(load,initial));auto host_rows=Take(reference.Load(load,initial));
  auto work=[&](const auto& locations){return std::vector<AgentWork>{
    {10,AccessRoute::resident_gpu,0,locations,{0,0,1},{}},
    {11,AccessRoute::resident_gpu,0,locations,{1,0},{}}};};
  std::vector<FeatureInput> layouts{{1,10,Pooling::sum,{1,1,2},{0,2,3,3},{2,1,0}},
    {1,11,Pooling::sequence,{2,1},{0,2},{2}}};
  auto expected=Take(reference.GatherHost(work(host_rows),layouts));
  auto first=Take(worker->GatherDevice(work(rows),layouts));ModelStream model;
  for(std::size_t f=0;f<first.size();++f){
    Check(first[f].feature==expected[f].feature && first[f].shape==expected[f].shape,"native device feature layout");
    auto wanted=expected[f].values;for(auto& value:wanted)value=2*value+1;
    Near(model.Run(first[f]),wanted);
  }
  auto second=Take(worker->GatherDevice(work(rows),layouts));
  auto rejected=worker->GatherDevice(work(rows),layouts).get();
  Check(std::holds_alternative<Error>(rejected) && std::get<Error>(rejected).code==Error::Code::capacity_exceeded,
    "native device output budget not enforced");
  Check(Take(backend->MemoryStats()).outstanding_output_bytes==160,"device output ownership accounting");
  second.clear();Check(Take(backend->MemoryStats()).outstanding_output_bytes==80,"device output not reclaimed");
  RowGradients gradients{{{1,1},{1,2}},{0.2f,0.4f,0.6f,0.8f,1,1.2f,1.4f,1.6f}};
  UpdateContext update{{"worker-cuda",0,0,Phase::push},0,1,0,0.01f,0.9f,0.999f};
  Take(control.ApplyHost(update,gradients));Take(reference.ApplyHost(update,gradients));
  MigrationPlan inspect{"inspect",0,rows,Placement::cpu,96};auto actual=Take(control.ReadBack(inspect));
  auto wanted=Take(reference.ReadBack({"inspect-host",0,host_rows,Placement::cpu,96}));
  for(std::size_t i=0;i<actual.size();++i){Check(actual[i].row_updates==1,"GPU worker update count");
    Near(actual[i].weights,wanted[i].weights);Near(actual[i].first_moment,wanted[i].first_moment);Near(actual[i].second_moment,wanted[i].second_moment);}
  auto current=Take(worker->GatherHost(work(rows),layouts));auto host=Take(reference.GatherHost(work(host_rows),layouts));
  for(std::size_t f=0;f<host.size();++f)Near(current[f].values,host[f].values);
  Take(control.CommitEviction(inspect));Check(Take(control.FreeSlots(1))==4,"worker eviction did not free slots");
  for(auto& loc:load.rows)loc.generation+=2;
  Take(control.Load(load,initial));
  Check(std::holds_alternative<Error>(worker->GatherDevice(work(rows),layouts).get()),"native device path accepted stale location");
  worker->Stop();auto stats=worker->Stats();
  Check(stats.accepted_requests==stats.completed_requests && stats.active_operations==0 &&
    stats.queued_operations==0 && stats.pending_request_bytes==0,"GPU worker admission not drained");
  worker.reset();backend.reset();Check(lifetime.expired(),"worker CUDA backend not destroyed");
  for(std::size_t f=0;f<first.size();++f){auto values=expected[f].values;for(auto& value:values)value=2*value+1;
    Near(model.Run(first[f]),values);}
  first.clear();
  std::cout<<"{\"test\":\"native_worker_cuda_stream\",\"model_stream\":\"nondefault\",\"remote_control_local_device_reads\":true,\"full_optimizer_rows_checked\":2,\"output_budget_enforced\":true,\"result_outlives_worker\":true}\n";
  return 0;
}catch(const std::exception& e){std::cerr<<"FAIL CUDA worker Agent: "<<e.what()<<'\n';return 1;}}
