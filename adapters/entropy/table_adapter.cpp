#include "eps/entropy_adapter.hpp"

#include <algorithm>
#include <map>
#include <set>
#include "common/error.hpp"

namespace eps::entropy {
namespace {
template<typename T>T Take(Result<T> r) {
  if (auto e=std::get_if<Error>(&r)) throw detail::Failure(*e);
  return std::get<T>(std::move(r));
}
}
struct TableAdapter::Impl {
  Impl(AdapterOptions o, std::vector<std::shared_ptr<ParameterService>> s)
      : options(std::move(o)), shards(std::move(s)), client(shards) {
    detail::Require(!options.run.empty() && options.max_live_batches>0 &&
      options.completion_history>0,"invalid adapter options");
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
  AdapterOptions options;
  std::vector<std::shared_ptr<ParameterService>> shards;
  DistributedClient client;
  std::mutex mutex;
  std::uint64_t next=0;
  std::map<std::uint64_t,Batch> live;
  std::map<std::uint64_t,bool> completed;
  bool closed=false;
};
TableAdapter::TableAdapter(AdapterOptions o,std::vector<std::shared_ptr<ParameterService>> s)
    : impl_(std::make_unique<Impl>(std::move(o),std::move(s))) {}
TableAdapter::~TableAdapter()=default;
Result<Ack> TableAdapter::CreateTable(TableDefinition table) {
  return impl_->client.CreateTable(table);
}
void TableAdapter::SetCacheResolver(DistributedClient::CacheResolver resolver) {
  impl_->client.SetCacheResolver(std::move(resolver));
}
Result<PullResult> TableAdapter::Pull(std::vector<Feature> features,bool read_only) {
  std::lock_guard lock(impl_->mutex);
  try {
    detail::Require(!impl_->closed,"adapter input is closed",Error::Code::unavailable);
    detail::Require(impl_->live.size()<impl_->options.max_live_batches,
      "adapter live-batch limit reached",Error::Code::capacity_exceeded);
    detail::Require(!features.empty(),"empty adapter Pull");
    LookupRequest request{{impl_->options.run,impl_->options.learner,impl_->next,Phase::lookup},
      impl_->options.partition_epoch,read_only,{}};
    Impl::Batch batch; batch.read_only=read_only;
    std::vector<std::uint64_t> original_sizes;
    for (const auto& f:features) {
      detail::Require(!f.offsets.empty() && f.offsets.front()==0 &&
        f.offsets.back()==f.keys.size() && std::is_sorted(f.offsets.begin(),f.offsets.end()),
        "invalid adapter row splits");
      FeatureInput compact{f.table,f.feature,f.pooling,{},{0},{}};
      std::vector<std::uint64_t> positions;
      for(std::size_t b=0;b+1<f.offsets.size();++b) {
        const auto start=compact.keys.size();
        for(auto i=f.offsets[b];i<f.offsets[b+1];++i)
          if(!f.padding_key || f.keys[i]!=*f.padding_key) {
            compact.keys.push_back(f.keys[i]); positions.push_back(i);
          }
        compact.valid_lengths.push_back(compact.keys.size()-start);
        compact.offsets.push_back(compact.keys.size());
      }
      original_sizes.push_back(f.keys.size());
      batch.positions.push_back(std::move(positions)); request.features.push_back(std::move(compact));
    }
    const auto handle=impl_->next;
    // Reserve bookkeeping before dispatch so allocation failures cannot orphan a
    // server ticket. A post-dispatch failure closes this adapter run.
    auto [entry,inserted]=impl_->live.emplace(handle,std::move(batch));
    impl_->completed.emplace(handle,false);
    ++impl_->next;
    try {
      entry->second.batch=Take(impl_->client.Lookup(std::move(request)));
      auto& state=entry->second;
      state.outputs=state.batch.features();
      for(std::size_t fi=0;fi<features.size();++fi) if(features[fi].pooling==Pooling::sequence) {
        const auto& compact=state.batch.features()[fi];const auto dim=compact.shape[1];
        detail::Require(original_sizes[fi]<=(64ULL<<20)/(dim*4),"padded sequence output exceeds limit");
        TensorBlock padded{compact.feature,{original_sizes[fi],dim},std::vector<float>(original_sizes[fi]*dim,0)};
        for(std::size_t i=0;i<state.positions[fi].size();++i)
          std::copy_n(compact.values.data()+i*dim,dim,padded.values.data()+state.positions[fi][i]*dim);
        state.outputs[fi]=std::move(padded);
      }
      return PullResult{handle,state.outputs};
    } catch(...) {
      impl_->closed=true;impl_->live.erase(handle);impl_->completed.erase(handle);throw;
    }
  } catch(const detail::Failure& e) { return e.error(); }
    catch(const std::exception& e) { impl_->closed=true;return Error{Error::Code::run_failed,e.what()}; }
}
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
      detail::Require(g.shape==output.shape && g.values.size()==output.values.size(),"adapter gradient shape mismatch");
      const auto& compact=state.batch.features()[fi];
      if(compact.shape!=g.shape) {
        TensorBlock selected{g.feature,compact.shape,{}}; selected.values.reserve(compact.values.size());
        const auto dim=compact.shape[1];
        for(auto pos:state.positions[fi])
          selected.values.insert(selected.values.end(),g.values.begin()+pos*dim,g.values.begin()+(pos+1)*dim);
        g=std::move(selected);
      }
      gradients.push_back(std::move(g));
    }
    Take(impl_->client.Push(state.batch,std::move(gradients)));
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
  if(!impl_->live.empty())return Error{Error::Code::unavailable,"complete Push/Release before Join"};
  return Ack{};
}
Result<std::vector<SnapshotMetadata>> TableAdapter::Save(std::string id) {
  std::lock_guard lock(impl_->mutex);
  try {
    detail::Require(impl_->live.empty(),"complete all batches before saving",Error::Code::unavailable);
    impl_->closed=true;
    std::vector<std::future<Result<SnapshotMetadata>>> futures;
    for(auto& shard:impl_->shards)futures.push_back(shard->DrainAndFlush(id));
    std::vector<SnapshotMetadata> result;
    for(auto& f:futures)result.push_back(Take(f.get()));
    return result;
  } catch(const detail::Failure& e) {return e.error();}
}
}  // namespace eps::entropy
