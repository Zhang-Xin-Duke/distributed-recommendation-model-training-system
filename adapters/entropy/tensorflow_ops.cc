#include "eps/runtime_c.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/cancellation.h"
#include "xla/stream_executor/stream.h"
#include <dlfcn.h>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <cstring>
#include <cstdio>
using namespace tensorflow;
namespace {
struct Failure {absl::Status status;};
void Require(bool ok,const char* message){if(!ok)throw Failure{errors::InvalidArgument(message)};}
void Status(absl::Status status){if(!status.ok())throw Failure{std::move(status)};}
void Native(int result,const EpsStatus& status){if(!result)return;
  switch(status.code){
    case 1:throw Failure{errors::InvalidArgument(status.message)};
    case 2:throw Failure{errors::Unavailable(status.message)};
    case 3:case 4:throw Failure{errors::FailedPrecondition(status.message)};
    case 5:throw Failure{errors::ResourceExhausted(status.message)};
    default:throw Failure{errors::Internal(status.message)};
  }
}
const EpsRuntimeApi* Load(const std::string& path){
#ifdef __APPLE__
  throw Failure{errors::FailedPrecondition("Native EPS TensorFlow runtime requires the Linux static transport build; use eps_tensorflow on macOS")};
#endif
  static std::mutex mutex;static std::map<std::string,const EpsRuntimeApi*> loaded;
  std::lock_guard lock(mutex);if(loaded.count(path))return loaded.at(path);
  // Keep the module loaded for process lifetime: native buffers can outlive
  // their runtime resource and still need its release function.
  void* library=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
  if(!library)throw Failure{errors::FailedPrecondition("cannot load EPS runtime: ",dlerror())};
  auto entry=reinterpret_cast<EpsRuntimeEntry>(dlsym(library,"eps_runtime_api_v1"));
  Require(entry,"EPS runtime entry point missing");const auto* api=entry();
  Require(api && api->abi_version==EPS_RUNTIME_ABI && api->struct_bytes>=sizeof(EpsRuntimeApi),"EPS runtime ABI mismatch");
  loaded[path]=api;return api;
}
struct NativeContext {
  const EpsRuntimeApi* api;void* runtime=nullptr;
  std::string configuration,library;
  NativeContext(const EpsRuntimeApi* a,std::string config,std::string path):api(a),configuration(std::move(config)),library(std::move(path)){
    EpsStatus status{};Native(api->create(configuration.c_str(),&runtime,&status),status);
  }
  ~NativeContext(){api->destroy(runtime);}
};
std::shared_ptr<NativeContext> Context(const EpsRuntimeApi* api,const std::string& config,const std::string& path,const std::string& key){
  if(key.empty())return std::make_shared<NativeContext>(api,config,path);
  static std::mutex mutex;static std::map<std::string,std::weak_ptr<NativeContext>> contexts;
  std::lock_guard lock(mutex);
  for(auto it=contexts.begin();it!=contexts.end();)if(it->second.expired())it=contexts.erase(it);else ++it;
  if(auto existing=contexts[key].lock()){
    Require(existing->configuration==config && existing->library==path,"shared EPS context configuration mismatch");return existing;
  }
  auto created=std::make_shared<NativeContext>(api,config,path);contexts[key]=created;return created;
}
class RuntimeResource final:public ResourceBase {
 public:
  std::shared_ptr<NativeContext> context;
  const EpsRuntimeApi* api;void* runtime;std::string configuration,library,context_key;
  RuntimeResource(const EpsRuntimeApi* a,std::string config,std::string path,std::string key):
    context(Context(a,config,path,key)),api(a),runtime(context->runtime),configuration(std::move(config)),library(std::move(path)),context_key(std::move(key)){}
  std::string DebugString()const override{return "EPS native parameter runtime";}
};
class CreateKernel final:public ResourceOpKernel<RuntimeResource> {
  std::string configuration_,library_,context_key_;
 public:
  explicit CreateKernel(OpKernelConstruction* c):ResourceOpKernel(c){OP_REQUIRES_OK(c,c->GetAttr("configuration",&configuration_));OP_REQUIRES_OK(c,c->GetAttr("library",&library_));OP_REQUIRES_OK(c,c->GetAttr("context_key",&context_key_));}
 private:
  absl::Status CreateResource(RuntimeResource** r)override{
    *r=nullptr;
    try{*r=new RuntimeResource(Load(library_),configuration_,library_,context_key_);return absl::OkStatus();}
    catch(const Failure& e){return e.status;}catch(const std::exception& e){return errors::Internal(e.what());}
  }
  absl::Status VerifyResource(RuntimeResource* r)override{return r->configuration==configuration_ && r->library==library_ && r->context_key==context_key_?absl::OkStatus():errors::FailedPrecondition("shared EPS resource configuration mismatch");}
};
class IoPool {
  std::mutex mutex_;std::condition_variable changed_;std::deque<std::function<void()>> queue_;bool stop_=false;std::vector<std::thread> workers_;
  IoPool(){try{for(int i=0;i<4;++i)workers_.emplace_back([this]{for(;;){std::function<void()> job;
    {std::unique_lock lock(mutex_);changed_.wait(lock,[&]{return stop_ || !queue_.empty();});if(stop_ && queue_.empty())return;job=std::move(queue_.front());queue_.pop_front();}job();}});
    }catch(...){Close();throw;}}
  void Close(){{std::lock_guard lock(mutex_);stop_=true;}changed_.notify_all();for(auto& w:workers_)w.join();}
 public:
  static IoPool& Instance(){static IoPool pool;return pool;}
  ~IoPool(){Close();}
  bool Submit(std::function<void()> job){{std::lock_guard lock(mutex_);if(stop_ || queue_.size()>=128)return false;queue_.push_back(std::move(job));}changed_.notify_one();return true;}
};
void Cancelled(OpKernelContext* c){if(c->cancellation_manager() && c->cancellation_manager()->IsCancelled())throw Failure{errors::Cancelled("EPS graph operation cancelled")};}
class AsyncKernel:public AsyncOpKernel {
 public:
  explicit AsyncKernel(OpKernelConstruction* c):AsyncOpKernel(c){}
  void ComputeAsync(OpKernelContext* c,DoneCallback done) final{
    RuntimeResource* raw=nullptr;auto status=LookupResource(c,HandleFromInput(c,0),&raw);
    if(!status.ok()){c->CtxFailure(status);done();return;}
    try{
      auto resource=std::shared_ptr<RuntimeResource>(raw,[](auto* p){p->Unref();});
      if(!IoPool::Instance().Submit([this,c,done,resource]{
        try{Cancelled(c);Run(c,*resource);}
        catch(const Failure& e){c->CtxFailure(e.status);}
        catch(const std::bad_alloc&){c->CtxFailure(errors::ResourceExhausted("EPS operation allocation failed"));}
        catch(const std::exception& e){c->CtxFailure(errors::Internal(e.what()));}
        done();
      })){c->CtxFailure(errors::ResourceExhausted("EPS TensorFlow I/O queue is full"));done();}
    }catch(const std::exception& e){c->CtxFailure(errors::Internal(e.what()));done();}
  }
  virtual void Run(OpKernelContext*,RuntimeResource&)=0;
};
void* Stream(OpKernelContext* c){auto* context=c->op_device_context();Require(context && context->stream(),"missing TensorFlow device stream");return context->stream()->platform_specific_handle().stream;}
int Device(OpKernelContext* c){const auto* info=c->device()->tensorflow_accelerator_device_info();Require(info,"missing TensorFlow GPU information");return info->gpu_id;}
std::uint64_t Scalar(OpKernelContext* c,int input){Require(TensorShapeUtils::IsScalar(c->input(input).shape()),"EPS handle/token must be scalar");return c->input(input).scalar<uint64>()();}
void OutputId(OpKernelContext* c,std::uint64_t value){Tensor* output=nullptr;Status(c->allocate_output(0,TensorShape({}),&output));output->scalar<uint64>()()=value;}
struct Schema {
  std::vector<int64_t> tables,features,modes,padding_keys,dimensions;std::vector<bool> padding;
  void Read(OpKernelConstruction* c){
    Status(c->GetAttr("tables",&tables));Status(c->GetAttr("features",&features));Status(c->GetAttr("modes",&modes));
    Status(c->GetAttr("padding",&padding));Status(c->GetAttr("padding_keys",&padding_keys));Status(c->GetAttr("dimensions",&dimensions));
    int n;Status(c->GetAttr("N",&n));Require(n>0 && tables.size()==n && features.size()==n && modes.size()==n && padding.size()==n && padding_keys.size()==n && dimensions.size()==n,"EPS schema count mismatch");
    for(int i=0;i<n;++i)Require(tables[i]>=0 && tables[i]<=UINT32_MAX && features[i]>=0 && features[i]<=UINT32_MAX && modes[i]>=0 && modes[i]<=2 && dimensions[i]>0,"invalid EPS feature schema");
  }
  std::vector<EpsFeature> Inputs(OpKernelContext* c)const{
    OpInputList keys,offsets;Status(c->input_list("keys",&keys));Status(c->input_list("offsets",&offsets));
    Require(keys.size()==tables.size() && offsets.size()==tables.size(),"EPS input list count mismatch");std::vector<EpsFeature> result;
    for(std::size_t i=0;i<tables.size();++i){Require(keys[i].dims()==1 && offsets[i].dims()==1,"EPS IDs and offsets must be vectors");
      result.push_back({uint32_t(tables[i]),uint32_t(features[i]),uint32_t(modes[i]),keys[i].flat<uint64>().data(),uint64_t(keys[i].NumElements()),offsets[i].flat<uint64>().data(),uint64_t(offsets[i].NumElements()),padding[i],uint64_t(padding_keys[i])});}
    return result;
  }
};
struct TextOwner {EpsText value{};~TextOwner(){if(value.owner)value.release(value.owner);}};
void Control(RuntimeResource& r,const char* method,const std::string& arguments,EpsText* output){EpsStatus status{};Native(r.api->command(r.runtime,method,arguments.c_str(),output,&status),status);}
struct BatchOwner {
  RuntimeResource& runtime;EpsBatch value{};bool published=false;
  ~BatchOwner(){if(value.owner){if(!published){TextOwner ignored;EpsStatus status{};char args[64];std::snprintf(args,sizeof(args),"{\"handle\":%llu}",static_cast<unsigned long long>(value.handle));runtime.api->command(runtime.runtime,"release",args,&ignored.value,&status);}value.release(value.owner);}}
};
class ForeignBuffer final:public TensorBuffer {
  EpsBuffer buffer_;
 public:
  explicit ForeignBuffer(EpsBuffer buffer):TensorBuffer(buffer.data),buffer_(buffer){}
  ~ForeignBuffer()override{if(buffer_.owner)buffer_.release(buffer_.owner);}
  size_t size()const override{return buffer_.rows*buffer_.columns*sizeof(float);}
  TensorBuffer* root_buffer()override{return this;}
  bool OwnsMemory()const override{return false;}
  void FillAllocationDescription(AllocationDescription* description)const override{description->set_requested_bytes(size());description->set_allocator_name("eps_native_device");}
};
void Publish(OpKernelContext* c,BatchOwner& batch,const std::vector<int64_t>& dimensions,bool gpu){
  Require(batch.value.count==dimensions.size(),"native EPS output feature count mismatch");
  OutputId(c,batch.value.handle);
  for(std::size_t i=0;i<dimensions.size();++i){auto& b=batch.value.buffers[i];
    Require(b.columns==dimensions[i] && b.rows<=INT64_MAX && b.columns>0 && b.rows<=(64ULL<<20)/(4*b.columns),"native EPS output shape mismatch");
    TensorShape shape({int64_t(b.rows),int64_t(b.columns)});
    if(gpu){Require(b.device==Device(c),"native EPS output is on another GPU");EpsStatus status{};Native(b.wait(b.owner,Stream(c),&status),status);
      core::RefCountPtr<TensorBuffer> buffer(new ForeignBuffer(b));b.owner=nullptr;
      Tensor output(DT_FLOAT,std::move(shape),std::move(buffer));c->set_output(i+1,std::move(output));
    }else{Require(b.device<0,"CPU kernel received a GPU output");Tensor* output=nullptr;Status(c->allocate_output(i+1,shape,&output));if(b.rows)std::memcpy(output->flat<float>().data(),b.data,b.rows*b.columns*4);}
  }
  Cancelled(c);batch.published=true;
}
template<bool GPU>class PullKernel final:public AsyncKernel {
  Schema schema_;bool read_only_;
 public:
  explicit PullKernel(OpKernelConstruction* c):AsyncKernel(c){try{schema_.Read(c);Status(c->GetAttr("read_only",&read_only_));}catch(const Failure& e){c->CtxFailure(e.status);}}
  void Run(OpKernelContext* c,RuntimeResource& r)override{auto features=schema_.Inputs(c);BatchOwner batch{r};EpsStatus status{};
    Native(r.api->pull(r.runtime,features.data(),features.size(),read_only_,GPU,&batch.value,&status),status);Publish(c,batch,schema_.dimensions,GPU);}
};
class PrepareKernel final:public AsyncKernel {
  Schema schema_;
 public:
  explicit PrepareKernel(OpKernelConstruction* c):AsyncKernel(c){try{schema_.Read(c);}catch(const Failure& e){c->CtxFailure(e.status);}}
  void Run(OpKernelContext* c,RuntimeResource& r)override{auto features=schema_.Inputs(c);Tensor* output=nullptr;Status(c->allocate_output(0,TensorShape({}),&output));uint64_t token=0;EpsStatus status{};
    Native(r.api->prepare(r.runtime,features.data(),features.size(),&token,&status),status);
    try{Cancelled(c);}catch(...){TextOwner ignored;Control(r,"cancel_prepared",std::string("{\"token\":")+std::to_string(token)+"}",&ignored.value);throw;}
    output->scalar<uint64>()()=token;
  }
};
template<bool GPU>class PreparedKernel final:public AsyncKernel {
  std::vector<int64_t> dimensions_;bool read_only_;
 public:
  explicit PreparedKernel(OpKernelConstruction* c):AsyncKernel(c){OP_REQUIRES_OK(c,c->GetAttr("dimensions",&dimensions_));OP_REQUIRES_OK(c,c->GetAttr("read_only",&read_only_));}
  void Run(OpKernelContext* c,RuntimeResource& r)override{BatchOwner batch{r};EpsStatus status{};Native(r.api->pull_prepared(r.runtime,Scalar(c,1),read_only_,GPU,&batch.value,&status),status);Publish(c,batch,dimensions_,GPU);}
};
template<bool GPU>class PushKernel final:public AsyncKernel {
  std::vector<int64_t> features_;
 public:
  explicit PushKernel(OpKernelConstruction* c):AsyncKernel(c){OP_REQUIRES_OK(c,c->GetAttr("features",&features_));}
  void Run(OpKernelContext* c,RuntimeResource& r)override{OpInputList gradients;Status(c->input_list("gradients",&gradients));Require(gradients.size()==features_.size(),"EPS gradient count mismatch");
    std::vector<EpsBuffer> buffers;for(std::size_t i=0;i<features_.size();++i){const auto& tensor=gradients[i];Require(tensor.dims()==2 && features_[i]>=0 && features_[i]<=UINT32_MAX,"invalid EPS gradient schema");
      buffers.push_back({uint32_t(features_[i]),uint64_t(tensor.dim_size(0)),uint64_t(tensor.dim_size(1)),const_cast<float*>(tensor.flat<float>().data()),GPU?Device(c):-1,nullptr,nullptr,nullptr});}
    const auto handle=Scalar(c,1);OutputId(c,handle);EpsStatus status{};Native(r.api->push(r.runtime,handle,buffers.data(),buffers.size(),GPU?Stream(c):nullptr,&status),status);
  }
};
class CommandKernel final:public AsyncKernel {
  std::string method_;
 public:
  explicit CommandKernel(OpKernelConstruction* c):AsyncKernel(c){OP_REQUIRES_OK(c,c->GetAttr("method",&method_));}
  void Run(OpKernelContext* c,RuntimeResource& r)override{Require(TensorShapeUtils::IsScalar(c->input(1).shape()),"EPS command arguments must be scalar");TextOwner text;
    Control(r,method_.c_str(),std::string(c->input(1).scalar<tstring>()()),&text.value);Tensor* output=nullptr;Status(c->allocate_output(0,TensorShape({}),&output));output->scalar<tstring>()()=std::string(text.value.data,text.value.size);}
};
class IdCommandKernel final:public AsyncKernel {
  std::string method_;
 public:
  explicit IdCommandKernel(OpKernelConstruction* c):AsyncKernel(c){OP_REQUIRES_OK(c,c->GetAttr("method",&method_));OP_REQUIRES(c,method_=="release" || method_=="cancel_prepared",errors::InvalidArgument("invalid EPS ID command"));}
  void Run(OpKernelContext* c,RuntimeResource& r)override{auto id=Scalar(c,1);OutputId(c,id);TextOwner ignored;
    Control(r,method_.c_str(),std::string(method_=="release"?"{\"handle\":":"{\"token\":")+std::to_string(id)+"}",&ignored.value);}
};
absl::Status Shapes(shape_inference::InferenceContext* c){std::vector<int64_t> dims;TF_RETURN_IF_ERROR(c->GetAttr("dimensions",&dims));int n;TF_RETURN_IF_ERROR(c->GetAttr("N",&n));
  if(dims.size()!=n)return errors::InvalidArgument("EPS dimension count mismatch");c->set_output(0,c->Scalar());for(int i=0;i<n;++i){if(dims[i]<=0)return errors::InvalidArgument("invalid EPS dimension");c->set_output(i+1,c->Matrix(c->UnknownDim(),dims[i]));}return absl::OkStatus();}
}  // namespace
REGISTER_OP("EpsRuntime").Output("resource: resource").Attr("configuration: string").Attr("library: string").Attr("context_key: string = ''").Attr("container: string = ''").Attr("shared_name: string = ''").SetIsStateful().SetShapeFn([](auto* c){c->set_output(0,c->Scalar());return absl::OkStatus();});
#define EPS_SCHEMA .Attr("N: int >= 1").Attr("tables: list(int)").Attr("features: list(int)").Attr("modes: list(int)").Attr("padding: list(bool)").Attr("padding_keys: list(int)").Attr("dimensions: list(int)")
REGISTER_OP("EpsPull").Input("resource: resource").Input("keys: N * uint64").Input("offsets: N * uint64").Output("handle: uint64").Output("outputs: N * float") EPS_SCHEMA .Attr("read_only: bool = false").SetIsStateful().SetShapeFn(Shapes);
REGISTER_OP("EpsPrepare").Input("resource: resource").Input("keys: N * uint64").Input("offsets: N * uint64").Output("token: uint64") EPS_SCHEMA .SetIsStateful().SetShapeFn([](auto* c){c->set_output(0,c->Scalar());return absl::OkStatus();});
#undef EPS_SCHEMA
REGISTER_OP("EpsPullPrepared").Input("resource: resource").Input("token: uint64").Output("handle: uint64").Output("outputs: N * float").Attr("N: int >= 1").Attr("dimensions: list(int)").Attr("read_only: bool = false").SetIsStateful().SetShapeFn(Shapes);
REGISTER_OP("EpsPush").Input("resource: resource").Input("handle: uint64").Input("gradients: N * float").Output("ack: uint64").Attr("N: int >= 1").Attr("features: list(int)").SetIsStateful().SetShapeFn([](auto* c){c->set_output(0,c->Scalar());return absl::OkStatus();});
REGISTER_OP("EpsCommand").Input("resource: resource").Input("arguments: string").Output("result: string").Attr("method: string").SetIsStateful().SetShapeFn([](auto* c){c->set_output(0,c->Scalar());return absl::OkStatus();});
REGISTER_OP("EpsIdCommand").Input("resource: resource").Input("id: uint64").Output("ack: uint64").Attr("method: string").SetIsStateful().SetShapeFn([](auto* c){c->set_output(0,c->Scalar());return absl::OkStatus();});
REGISTER_KERNEL_BUILDER(Name("EpsRuntime").Device(DEVICE_CPU),CreateKernel);
REGISTER_KERNEL_BUILDER(Name("EpsRuntime").Device(DEVICE_GPU).HostMemory("resource"),CreateKernel);
REGISTER_KERNEL_BUILDER(Name("EpsPull").Device(DEVICE_CPU),PullKernel<false>);
REGISTER_KERNEL_BUILDER(Name("EpsPullPrepared").Device(DEVICE_CPU),PreparedKernel<false>);
REGISTER_KERNEL_BUILDER(Name("EpsPrepare").Device(DEVICE_CPU),PrepareKernel);
REGISTER_KERNEL_BUILDER(Name("EpsPush").Device(DEVICE_CPU),PushKernel<false>);
REGISTER_KERNEL_BUILDER(Name("EpsCommand").Device(DEVICE_CPU),CommandKernel);
REGISTER_KERNEL_BUILDER(Name("EpsIdCommand").Device(DEVICE_CPU),IdCommandKernel);
REGISTER_KERNEL_BUILDER(Name("EpsPull").Device(DEVICE_GPU).HostMemory("resource").HostMemory("keys").HostMemory("offsets").HostMemory("handle"),PullKernel<true>);
REGISTER_KERNEL_BUILDER(Name("EpsPullPrepared").Device(DEVICE_GPU).HostMemory("resource").HostMemory("token").HostMemory("handle"),PreparedKernel<true>);
REGISTER_KERNEL_BUILDER(Name("EpsPrepare").Device(DEVICE_GPU).HostMemory("resource").HostMemory("keys").HostMemory("offsets").HostMemory("token"),PrepareKernel);
REGISTER_KERNEL_BUILDER(Name("EpsPush").Device(DEVICE_GPU).HostMemory("resource").HostMemory("handle").HostMemory("ack"),PushKernel<true>);
REGISTER_KERNEL_BUILDER(Name("EpsCommand").Device(DEVICE_GPU).HostMemory("resource").HostMemory("arguments").HostMemory("result"),CommandKernel);
REGISTER_KERNEL_BUILDER(Name("EpsIdCommand").Device(DEVICE_GPU).HostMemory("resource").HostMemory("id").HostMemory("ack"),IdCommandKernel);
