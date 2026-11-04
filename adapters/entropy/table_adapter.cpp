#include "eps/entropy_adapter.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>
#include "common/error.hpp"
#include "access/prepare_queue.hpp"

namespace eps::entropy {
namespace {
template<typename T>T Take(Result<T> r) {
  if (auto e=std::get_if<Error>(&r)) throw detail::Failure(*e);
  return std::get<T>(std::move(r));
}
}
struct TableAdapter::Impl {
  Impl(AdapterOptions o, std::vector<std::shared_ptr<ParameterService>> s)
      : options(std::move(o)), shards(std::move(s)), client(shards), preparation(shards.size(),options.preparation) {
    detail::Require(!options.run.empty() && options.max_live_batches>0 &&
      options.completion_history>0,"invalid adapter options");
  }
  ~Impl() {
    preparation.Close();
    for(auto& [handle,state]:live)try{(void)client.Release(state.batch);}catch(...){}
  }
  struct Batch {
    DistributedBatch batch;
    std::vector<std::vector<std::uint64_t>> positions;
    std::vector<TensorBlock> outputs;
    bool read_only;
  };
  void Finish(std::uint64_t handle, bool pushed) {
    // The map entry was reserved before Pull publishes a live ticket.
    completed.at(handle)=pushed;
    live.erase(handle);
    while(completed.size()>options.completion_history) {
      auto it=std::find_if(completed.begin(),completed.end(),[&](const auto& x){return !live.contains(x.first);});
      if(it==completed.end()) break;
      completed.erase(it);
    }
  }
  void Ready() {
    detail::Require(!closed,"adapter input is closed",Error::Code::unavailable);
    detail::Require(live.size()<options.max_live_batches,"adapter live-batch limit reached",Error::Code::capacity_exceeded);
  }
  template<bool Device=false>
  std::conditional_t<Device,DevicePullResult,PullResult> Dispatch(detail::PreparedAdapterInput prepared,bool read_only,bool allow_retry=true) {
    detail::Require(next!=UINT64_MAX,"adapter request sequence exhausted",Error::Code::capacity_exceeded);
    const auto handle=next;
    Batch batch;batch.read_only=read_only;batch.positions=std::move(prepared.positions);
    completed.emplace(handle,false);
    decltype(live)::iterator entry;
    try {entry=live.emplace(handle,std::move(batch)).first;}
    catch(...){completed.erase(handle);throw;}
    ++next;bool looked_up=false;
    try {
      if constexpr(Device){
        std::vector<SequenceProjection> projections;
        for(std::size_t fi=0;fi<prepared.pooling.size();++fi)if(prepared.pooling[fi]==Pooling::sequence)
          projections.push_back({prepared.lookup.features()[fi].feature,prepared.original_sizes[fi],entry->second.positions[fi]});
        entry->second.batch=Take(client.LookupPreparedDevice({options.run,options.learner,handle,Phase::lookup},
          options.partition_epoch,read_only,std::move(prepared.lookup),std::move(projections)));
      }else entry->second.batch=Take(client.LookupPrepared({options.run,options.learner,handle,Phase::lookup},
        options.partition_epoch,read_only,std::move(prepared.lookup)));
      looked_up=true;auto& state=entry->second;state.outputs=state.batch.features();
      if constexpr(Device){
        state.outputs.clear();
        for(const auto& output:state.batch.device_features())state.outputs.push_back({output.feature,output.shape,{}});
        return DevicePullResult{handle,state.batch.device_features()};
      }else{
      for(std::size_t fi=0;fi<prepared.pooling.size();++fi)if(prepared.pooling[fi]==Pooling::sequence) {
        const auto& compact=state.batch.features()[fi];const auto dim=compact.shape[1];
        detail::Require(prepared.original_sizes[fi]<=(64ULL<<20)/(dim*4),"padded sequence output exceeds limit",Error::Code::capacity_exceeded);
        TensorBlock padded{compact.feature,{prepared.original_sizes[fi],dim},std::vector<float>(prepared.original_sizes[fi]*dim,0)};
        for(std::size_t i=0;i<state.positions[fi].size();++i)
          std::copy_n(compact.values.data()+i*dim,dim,padded.values.data()+state.positions[fi][i]*dim);
        state.outputs[fi]=std::move(padded);
      }
      return PullResult{handle,state.outputs};
      }
    }catch(...) {
      // Output reconstruction/allocation can fail after successful RPCs. Release
      // those known tickets before destroying the batch and closing admission.
      bool retryable=false;
      try{throw;}catch(const detail::Failure& e){retryable=allow_retry && !looked_up &&
        e.error().code==Error::Code::unavailable && preparation.Empty();}catch(...){}
      bool released=true;
      if(looked_up)try{released=std::holds_alternative<Ack>(client.Release(entry->second.batch));}catch(...){released=false;}
      if(!retryable || !released || !client.Healthy()){closed=true;preparation.Close();}
      live.erase(handle);completed.erase(handle);throw;
    }
  }
  AdapterOptions options;
  std::vector<std::shared_ptr<ParameterService>> shards;
  DistributedClient client;
  detail::PrepareQueue preparation;
  std::mutex mutex;
  std::uint64_t next=0;
  std::map<std::uint64_t,Batch> live;
  std::map<std::uint64_t,bool> completed;
  bool closed=false;
  bool read_stream_restored=false;
  std::optional<std::string> saved_checkpoint,resumed_checkpoint;
};
TableAdapter::TableAdapter(AdapterOptions o,std::vector<std::shared_ptr<ParameterService>> s)
    : impl_(std::make_unique<Impl>(std::move(o),std::move(s))) {}
TableAdapter::~TableAdapter()=default;
Result<Ack> TableAdapter::CreateTable(TableDefinition table) {
  std::lock_guard lock(impl_->mutex);
  if(impl_->closed)return Error{Error::Code::unavailable,"adapter input is closed"};
  return impl_->client.CreateTable(table);
}
void TableAdapter::SetCacheResolver(DistributedClient::CacheResolver resolver,bool coalesce) {
  std::lock_guard lock(impl_->mutex);
  impl_->client.SetCacheResolver(std::move(resolver),coalesce);
}
void TableAdapter::SetDevicePath(DistributedClient::DeviceResolver resolver,DistributedClient::DeviceAssembler assembler){
  std::lock_guard lock(impl_->mutex);impl_->client.SetDevicePath(std::move(resolver),std::move(assembler));
}
Result<DevicePullResult> TableAdapter::PullDevice(std::vector<Feature> features,bool read_only){
  std::lock_guard lock(impl_->mutex);
  try{
    impl_->Ready();
    detail::Require(impl_->preparation.Empty(),"consume or cancel prepared inputs before synchronous Pull",Error::Code::unavailable);
    return impl_->Dispatch<true>(Take(detail::PrepareAdapterInput(std::move(features),impl_->shards.size(),
      impl_->options.preparation.lookup_limits)),read_only);
  }catch(const detail::Failure& e){return e.error();}
   catch(const std::bad_alloc&){return Error{Error::Code::capacity_exceeded,"device Pull allocation failed"};}
   catch(const std::exception& e){impl_->closed=true;impl_->preparation.Close();return Error{Error::Code::run_failed,e.what()};}
}
Result<PullResult> TableAdapter::Pull(std::vector<Feature> features,bool read_only) {
  std::lock_guard lock(impl_->mutex);
  try {
    impl_->Ready();
    detail::Require(impl_->preparation.Empty(),"consume or cancel prepared inputs before synchronous Pull",Error::Code::unavailable);
    return impl_->Dispatch(Take(detail::PrepareAdapterInput(std::move(features),impl_->shards.size(),
      impl_->options.preparation.lookup_limits)),read_only);
  }catch(const detail::Failure& e){return e.error();}
  catch(const std::bad_alloc&){return Error{Error::Code::capacity_exceeded,"adapter Pull allocation failed"};}
  catch(const std::exception& e){impl_->closed=true;impl_->preparation.Close();return Error{Error::Code::run_failed,e.what()};}
}
Result<std::uint64_t> TableAdapter::PrepareAsync(std::vector<Feature> features) {
  return impl_->preparation.Submit(std::move(features));
}
Result<PullResult> TableAdapter::PullPrepared(std::uint64_t token,bool read_only) {
  std::lock_guard lock(impl_->mutex);
  try {
    impl_->Ready();
    return impl_->Dispatch(Take(impl_->preparation.Take(token)),read_only,false);
  }catch(const detail::Failure& e){return e.error();}
  catch(const std::bad_alloc&){return Error{Error::Code::capacity_exceeded,"prepared Pull allocation failed"};}
  catch(const std::exception& e){impl_->closed=true;impl_->preparation.Close();return Error{Error::Code::run_failed,e.what()};}
}
Result<DevicePullResult> TableAdapter::PullPreparedDevice(std::uint64_t token,bool read_only){
  std::lock_guard lock(impl_->mutex);
  try{impl_->Ready();return impl_->Dispatch<true>(Take(impl_->preparation.Take(token)),read_only,false);}
  catch(const detail::Failure& e){return e.error();}
  catch(const std::bad_alloc&){return Error{Error::Code::capacity_exceeded,"prepared device Pull allocation failed"};}
  catch(const std::exception& e){impl_->closed=true;impl_->preparation.Close();return Error{Error::Code::run_failed,e.what()};}
}
Result<Ack> TableAdapter::CancelPrepared(std::uint64_t token) {return impl_->preparation.Cancel(token);}
PreparationStats TableAdapter::PreparationStatus() const {return impl_->preparation.Stats();}
Result<Ack> TableAdapter::Push(std::uint64_t handle,std::vector<TensorBlock> gradients) {
  std::lock_guard lock(impl_->mutex);
  try {
    if(!impl_->live.contains(handle)) {
      detail::Require(impl_->completed.contains(handle) && impl_->completed.at(handle),
        "unknown or released adapter batch",Error::Code::retired_request);
      return Ack{};
    }
    auto& state=impl_->live.at(handle);
    detail::Require(!state.read_only && gradients.size()==state.outputs.size(),"invalid adapter Push");
    std::map<std::uint32_t,TensorBlock> by_id;
    for(auto& g:gradients) detail::Require(by_id.emplace(g.feature,std::move(g)).second,"duplicate gradient feature");
    gradients.clear();
    for(std::size_t fi=0;fi<state.outputs.size();++fi) {
      const auto& output=state.outputs[fi];
      detail::Require(by_id.contains(output.feature),"missing adapter gradient");
      auto g=std::move(by_id.at(output.feature));
      detail::Require(g.shape==output.shape && g.values.size()==output.shape[0]*output.shape[1],"adapter gradient shape mismatch");
      const auto& compact=state.batch.features()[fi];
      if(compact.shape!=g.shape) {
        TensorBlock selected{g.feature,compact.shape,{}}; selected.values.reserve(compact.shape[0]*compact.shape[1]);
        const auto dim=compact.shape[1];
        for(auto pos:state.positions[fi])
          selected.values.insert(selected.values.end(),g.values.begin()+pos*dim,g.values.begin()+(pos+1)*dim);
        g=std::move(selected);
      }
      gradients.push_back(std::move(g));
    }
    auto pushed=impl_->client.Push(state.batch,std::move(gradients));
    if(auto error=std::get_if<Error>(&pushed)) {
      impl_->closed=true;impl_->preparation.Close();return *error;
    }
    impl_->Finish(handle,true);return Ack{};
  } catch(const detail::Failure& e) {return e.error();}
}
Result<Ack> TableAdapter::Release(std::uint64_t handle) {
  std::lock_guard lock(impl_->mutex);
  try {
    if(!impl_->live.contains(handle)) {
      detail::Require(impl_->completed.contains(handle),"retired adapter batch",Error::Code::retired_request);return Ack{};
    }
    Take(impl_->client.Release(impl_->live.at(handle).batch));
    impl_->Finish(handle,false);return Ack{};
  } catch(const detail::Failure& e) {return e.error();}
}
Result<Ack> TableAdapter::Join() {
  std::lock_guard lock(impl_->mutex);
  if(!impl_->client.Healthy())return Error{Error::Code::run_failed,"read or update cleanup is uncertain; recover the job before Join"};
  if(!impl_->live.empty())return Error{Error::Code::unavailable,"complete Push/Release before Join"};
  return Ack{};
}
bool TableAdapter::ReadLifecycleHealthy() const {
  std::lock_guard lock(impl_->mutex);return !impl_->closed && impl_->client.Healthy();
}
Result<std::uint64_t> TableAdapter::CompletedReadFrontier() const {
  std::lock_guard lock(impl_->mutex);
  if(impl_->closed || !impl_->client.Healthy() || !impl_->live.empty())return Error{Error::Code::unavailable,"finish all read lifecycles before certifying progress"};
  return impl_->next;
}
Result<Ack> TableAdapter::RestoreReadFrontier(std::uint64_t next) {
  std::lock_guard lock(impl_->mutex);
  if(impl_->closed || impl_->read_stream_restored || impl_->next || !impl_->live.empty() ||
      !impl_->completed.empty() || !impl_->preparation.Empty() || next==UINT64_MAX)
    return Error{Error::Code::invalid_argument,"read-stream recovery requires a fresh adapter"};
  impl_->next=next;impl_->read_stream_restored=true;return Ack{};
}
Result<std::vector<SnapshotMetadata>> TableAdapter::Save(std::string id) {
  std::lock_guard lock(impl_->mutex);
  try {
    detail::Require(!impl_->closed,"adapter input is closed",Error::Code::unavailable);
    detail::Require(impl_->live.empty(),"complete all batches before saving",Error::Code::unavailable);
    Take(impl_->preparation.CloseIfEmpty());
    impl_->closed=true;
    impl_->saved_checkpoint.reset();impl_->resumed_checkpoint.reset();
    std::vector<std::future<Result<SnapshotMetadata>>> futures;
    for(auto& shard:impl_->shards)futures.push_back(shard->DrainAndFlush(id));
    std::vector<SnapshotMetadata> result;
    for(auto& f:futures)result.push_back(Take(f.get()));
    impl_->saved_checkpoint=id;
    return result;
  } catch(const detail::Failure& e) {return e.error();}
}
Result<SnapshotChunk> TableAdapter::ReadCheckpoint(std::string id,std::uint32_t shard,
    std::uint64_t offset,std::uint32_t max_bytes) {
  std::lock_guard lock(impl_->mutex);
  try {
    detail::Require(impl_->closed && impl_->saved_checkpoint==id,
      "export requires the adapter's successful frozen save",Error::Code::unavailable);
    detail::Require(shard<impl_->shards.size(),"checkpoint shard is out of range");
    return Take(impl_->shards[shard]->ReadCheckpoint(
      {impl_->options.run,impl_->options.partition_epoch,std::move(id)},offset,max_bytes).get());
  }catch(const detail::Failure& e){return e.error();}
}
Result<Ack> TableAdapter::Resume(std::string id) {
  std::lock_guard lock(impl_->mutex);
  try {
    if(!impl_->closed && impl_->resumed_checkpoint==id)return Ack{};
    detail::Require(impl_->closed && impl_->saved_checkpoint==id,
      "resume requires the adapter's successful frozen save",Error::Code::unavailable);
    std::vector<std::future<Result<Ack>>> futures;
    for(auto& shard:impl_->shards)futures.push_back(shard->ResumeCheckpoint(
      {impl_->options.run,impl_->options.partition_epoch,id}));
    for(auto& f:futures)Take(f.get());
    Take(impl_->preparation.Reopen());
    impl_->resumed_checkpoint=std::move(id);impl_->saved_checkpoint.reset();impl_->closed=false;
    return Ack{};
  }catch(const detail::Failure& e){return e.error();}
}
}  // namespace eps::entropy
