#include "eps/runtime_c.h"
#include "eps/entropy_adapter.hpp"
#include "eps/brpc_client.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/cache_resolver.hpp"
#include "eps/worker_cache_agent.hpp"
#include "common/error.hpp"
#include <nlohmann/json.hpp>
#include <cstring>
#include <mutex>
#ifdef EPS_HAS_CUDA
#include "eps/cuda_cache_backend.hpp"
#include <cuda_runtime.h>
#endif
using namespace eps;
namespace {
using Json=nlohmann::json;
template<typename T>T Take(Result<T> r){if(auto e=std::get_if<Error>(&r))throw detail::Failure(*e);return std::get<T>(std::move(r));}
template<typename T>T Take(std::future<Result<T>> r){return Take(r.get());}
template<typename F>int Boundary(EpsStatus* status,F fn) noexcept {
  if(status){status->code=0;status->message[0]=0;}
  auto fail=[&](Error::Code code,const char* message){const int result=static_cast<int>(code)+1;
    if(status){status->code=result;std::strncpy(status->message,message,sizeof(status->message)-1);status->message[sizeof(status->message)-1]=0;}return result;};
  try{fn();return 0;}
  catch(const detail::Failure& e){return fail(e.error().code,e.what());}
  catch(const std::bad_alloc&){return fail(Error::Code::capacity_exceeded,"native runtime allocation failed");}
  catch(const std::exception& e){return fail(Error::Code::run_failed,e.what());}
  catch(...){return fail(Error::Code::run_failed,"unknown native runtime failure");}
}
Json Parse(const char* text){detail::Require(text && strnlen(text,(1<<20)+1)<=(1<<20),"native JSON exceeds limit");return Json::parse(text);}
TableDefinition Table(const Json& t,std::uint32_t shards,std::uint64_t epoch){
  detail::Require(t.value("dtype",std::string("float32"))=="float32" && t.at("optimizer").at("kind")=="sparse_lazy_adam","unsupported native table format");
  const auto& o=t.at("optimizer");const auto& i=t.at("initialization");
  TableDefinition d{t.at("id"),t.at("dimension"),shards,epoch,
    {o.at("learning_rate"),o.at("beta1"),o.at("beta2"),o.at("epsilon"),o.at("initial_beta1_power"),o.at("initial_beta2_power")},i.at("min"),i.at("max"),i.at("seed")};
  if(o.contains("weight_bounds")){const auto& b=o.at("weight_bounds");detail::Require(b.is_array() && b.size()==2,"invalid weight bounds");d.optimizer.weight_bounds=WeightBounds{b[0],b[1]};}return d;
}
struct Runtime {
  std::mutex mutex;
  std::vector<std::shared_ptr<BrpcParameterService>> connections;
  std::shared_ptr<CacheBackend> backend;
  std::shared_ptr<WorkerCacheAgent> worker;
  std::unique_ptr<entropy::TableAdapter> adapter;
  int device=-1;
  std::uint64_t epoch=1;
  explicit Runtime(const Json& config){
    const auto run=config.at("run").get<std::string>();epoch=config.value("epoch",std::uint64_t(1));
    std::vector<std::shared_ptr<ParameterService>> shards;
    for(const auto& endpoint:config.at("endpoints")){auto connection=std::make_shared<BrpcParameterService>(endpoint.get<std::string>());connections.push_back(connection);shards.push_back(connection);}
    adapter=std::make_unique<entropy::TableAdapter>(entropy::AdapterOptions{run,config.value("learner",0U),epoch,config.value("max_live",128U),4096},shards);
    std::map<AgentId,std::shared_ptr<CacheBackend>> remote;
    if(config.contains("agents"))for(const auto& a:config.at("agents")){AgentId id=a.at("id");detail::Require(!remote.contains(id),"duplicate native Agent");remote[id]=std::make_shared<BrpcCacheBackend>(a.at("endpoint").get<std::string>(),run,0,id);}
#ifdef EPS_HAS_CUDA
    if(config.contains("cuda")){
      const auto& c=config.at("cuda");CudaCacheOptions options;options.agent=c.value("agent",0U);options.device=c.value("device",0);
      options.state_bytes=c.value("state_bytes",options.state_bytes);options.rows_per_table=c.value("rows_per_table",options.rows_per_table);
      options.transfer_bytes=c.value("transfer_bytes",options.transfer_bytes);options.outstanding_output_bytes=c.value("outstanding_output_bytes",options.outstanding_output_bytes);
      options.output_alignment=256;
      detail::Require(!remote.contains(options.agent),"local Agent duplicates remote Agent");device=options.device;
      auto cuda=std::make_shared<CudaCacheBackend>(options);backend=cuda;
      worker=std::make_shared<WorkerCacheAgent>(WorkerCacheOptions{run,c.value("listen",std::string("127.0.0.1:0"))},cuda);
      auto host_endpoints=remote;host_endpoints[options.agent]=std::make_shared<BrpcCacheBackend>(worker->endpoint(),run,0,options.agent);
      adapter->SetCacheResolver(MakeCacheResolver(std::move(host_endpoints)),true);
      adapter->SetDevicePath(MakeDeviceCacheResolver({{options.agent,[w=worker](auto work,auto layouts){return w->GatherDevice(std::move(work),std::move(layouts));}}},remote),
        [cuda](auto request){return cuda->AssembleFeatures(std::move(request)).get();});
    }else
#endif
    {detail::Require(!config.contains("cuda"),"CUDA native runtime is not available");if(!remote.empty())adapter->SetCacheResolver(MakeCacheResolver(std::move(remote)),true);}
    for(const auto& definition:config.at("tables"))Take(adapter->CreateTable(Table(definition,shards.size(),epoch)));
  }
  ~Runtime(){adapter.reset();worker.reset();backend.reset();}
};
Runtime& Get(void* pointer){detail::Require(pointer,"null native runtime");return *static_cast<Runtime*>(pointer);}
void Ready(Runtime& r){detail::Require(bool(r.adapter),"native runtime is closed",Error::Code::unavailable);}
std::vector<entropy::Feature> Features(const EpsFeature* input,std::uint64_t count){
  detail::Require(input && count>0 && count<=1024,"invalid native feature count");std::vector<entropy::Feature> result;result.reserve(count);std::uint64_t bytes=0;
  for(std::uint64_t i=0;i<count;++i){const auto& f=input[i];detail::Require(f.pooling<=2 && f.key_count<=(16ULL<<20)/8 && f.offset_count<=(16ULL<<20)/8 &&
      (!f.key_count || f.keys) && f.offset_count && f.offsets,"invalid native feature input");
    bytes+=(f.key_count+f.offset_count)*8;detail::Require(bytes<=(16ULL<<20),"native feature input exceeds budget");
    entropy::Feature output{f.table,f.feature,static_cast<Pooling>(f.pooling),{},{},std::nullopt};
    if(f.key_count)output.keys.assign(f.keys,f.keys+f.key_count);output.offsets.assign(f.offsets,f.offsets+f.offset_count);
    if(f.has_padding)output.padding_key=f.padding_key;result.push_back(std::move(output));}
  return result;
}
struct BufferOwner {std::vector<float> host;std::shared_ptr<DeviceTensor> device;};
void ReleaseBuffer(void* p){delete static_cast<BufferOwner*>(p);}
int WaitBuffer(void* p,void* stream,EpsStatus* status){return Boundary(status,[&]{auto& b=*static_cast<BufferOwner*>(p);if(b.device)b.device->WaitOnStream(stream);});}
struct BatchOwner {
  std::vector<EpsBuffer> buffers;
  ~BatchOwner(){for(auto& b:buffers)if(b.owner)b.release(b.owner);}
};
void ReleaseBatch(void* p){delete static_cast<BatchOwner*>(p);}
template<typename Pull>void Pack(Runtime& runtime,Pull result,EpsBatch* output){
  try{
    auto owner=std::make_unique<BatchOwner>();owner->buffers.reserve(result.features.size());
    for(auto& block:result.features){auto buffer=std::make_unique<BufferOwner>();void* data;int device=-1;
      if constexpr(std::is_same_v<Pull,entropy::DevicePullResult>){buffer->device=std::move(block.tensor);data=buffer->device->data();device=buffer->device->device();}
      else{buffer->host=std::move(block.values);data=buffer->host.data();}
      owner->buffers.push_back({block.feature,block.shape.at(0),block.shape.at(1),data,device,buffer.get(),ReleaseBuffer,WaitBuffer});buffer.release();}
    *output={result.handle,owner->buffers.size(),owner->buffers.data(),owner.get(),ReleaseBatch};owner.release();
  }catch(...){(void)runtime.adapter->Release(result.handle);throw;}
}
int Create(const char* text,void** output,EpsStatus* status){if(output)*output=nullptr;return Boundary(status,[&]{detail::Require(output,"null runtime output");*output=new Runtime(Parse(text));});}
void Destroy(void* p){try{delete static_cast<Runtime*>(p);}catch(...){}}
int Pull(void* p,const EpsFeature* features,std::uint64_t count,int32_t read_only,int32_t device,EpsBatch* output,EpsStatus* status){
  if(output)*output={};return Boundary(status,[&]{detail::Require(output,"null batch output");auto& r=Get(p);std::lock_guard lock(r.mutex);Ready(r);auto input=Features(features,count);
    if(device){detail::Require(r.device>=0,"runtime has no CUDA device");Pack(r,Take(r.adapter->PullDevice(std::move(input),read_only)),output);}
    else Pack(r,Take(r.adapter->Pull(std::move(input),read_only)),output);});
}
int Prepare(void* p,const EpsFeature* features,std::uint64_t count,std::uint64_t* token,EpsStatus* status){return Boundary(status,[&]{detail::Require(token,"null token output");auto& r=Get(p);std::lock_guard lock(r.mutex);Ready(r);*token=Take(r.adapter->PrepareAsync(Features(features,count)));});}
int PullPrepared(void* p,std::uint64_t token,int32_t read_only,int32_t device,EpsBatch* output,EpsStatus* status){
  if(output)*output={};return Boundary(status,[&]{detail::Require(output,"null prepared output");auto& r=Get(p);std::lock_guard lock(r.mutex);Ready(r);
    if(device){detail::Require(r.device>=0,"runtime has no CUDA device");Pack(r,Take(r.adapter->PullPreparedDevice(token,read_only)),output);}
    else Pack(r,Take(r.adapter->PullPrepared(token,read_only)),output);});
}
int Push(void* p,std::uint64_t handle,const EpsBuffer* input,std::uint64_t count,void* stream,EpsStatus* status){return Boundary(status,[&]{
  auto& r=Get(p);std::lock_guard lock(r.mutex);Ready(r);detail::Require(input && count>0 && count<=1024,"invalid native gradient count");std::vector<TensorBlock> gradients;gradients.reserve(count);std::uint64_t bytes=0;
  for(std::uint64_t i=0;i<count;++i){const auto& b=input[i];detail::Require(b.columns>0 && b.columns<=(64ULL<<20)/4 && b.rows<=(64ULL<<20)/(4*b.columns),"invalid native gradient shape");
    const auto n=b.rows*b.columns;bytes+=n*4;detail::Require(bytes<=(64ULL<<20) && (!n || b.data),"native gradient budget/pointer invalid");
    gradients.push_back({b.feature,{b.rows,b.columns},std::vector<float>(n)});if(!n)continue;
    if(b.device<0)std::memcpy(gradients.back().values.data(),b.data,n*4);
    else{
#ifdef EPS_HAS_CUDA
      detail::Require(b.device==r.device,"gradient device mismatch");auto check=[](cudaError_t e){detail::Require(e==cudaSuccess,cudaGetErrorString(e),Error::Code::run_failed);};
      check(cudaSetDevice(b.device));check(cudaMemcpyAsync(gradients.back().values.data(),b.data,n*4,cudaMemcpyDeviceToHost,static_cast<cudaStream_t>(stream)));
      check(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
#else
      detail::Require(false,"device gradient is not supported");
#endif
    }
  }Take(r.adapter->Push(handle,std::move(gradients)));
});}
void ReleaseText(void* p){delete static_cast<std::string*>(p);}
int Command(void* p,const char* command,const char* arguments,EpsText* output,EpsStatus* status){if(output)*output={};return Boundary(status,[&]{
  detail::Require(command && output,"invalid native control output");auto& r=Get(p);std::lock_guard lock(r.mutex);Ready(r);const auto args=Parse(arguments);const std::string method(command);Json result;
  if(method=="join")Take(r.adapter->Join());
  else if(method=="create_table")Take(r.adapter->CreateTable(Table(args,r.connections.size(),r.epoch)));
  else if(method=="release")Take(r.adapter->Release(args.at("handle")));
  else if(method=="cancel_prepared")Take(r.adapter->CancelPrepared(args.at("token")));
  else if(method=="save"){auto saved=Take(r.adapter->Save(args.at("id")));result=Json::array();for(const auto& s:saved)for(const auto& file:s.state_files)result.push_back(file);}
  else if(method=="resume")Take(r.adapter->Resume(args.at("id")));
  else if(method=="endpoint")result=r.worker?r.worker->endpoint():std::string();
  else if(method=="stats"){result=Json::array();for(auto& c:r.connections){auto s=Take(c->Stats());result.push_back({{"updates",s.updates},{"live_tickets",s.live_tickets},{"cached_rows",s.cached_rows}});}}
  else if(method=="close"){Take(r.adapter->Join());r.adapter.reset();r.worker.reset();r.backend.reset();}
  else if(method=="abort"){r.adapter.reset();r.worker.reset();r.backend.reset();}
  else detail::Require(false,"unknown native control command");
  auto owner=std::make_unique<std::string>(result.dump());*output={owner->data(),owner->size(),owner.get(),ReleaseText};owner.release();
});}
const EpsRuntimeApi api{EPS_RUNTIME_ABI,sizeof(EpsRuntimeApi),Create,Destroy,Pull,Prepare,PullPrepared,Push,Command};
}
extern "C" EPS_RUNTIME_EXPORT const EpsRuntimeApi* eps_runtime_api_v1(){return &api;}
