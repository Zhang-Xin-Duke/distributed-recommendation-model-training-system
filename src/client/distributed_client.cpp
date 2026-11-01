#include "eps/distributed_client.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <map>
#include <tuple>
#include <unordered_set>
#include "common/error.hpp"

namespace eps {
namespace {
template <typename T>T Take(std::future<Result<T>>& future) {
  auto r=future.get();if(auto error=std::get_if<Error>(&r))throw detail::Failure(*error);
  return std::get<T>(std::move(r));
}
template <typename T>T Take(Result<T> result) {
  if(auto e=std::get_if<Error>(&result))throw detail::Failure(*e);
  return std::get<T>(std::move(result));
}
}
DistributedClient::DistributedClient(std::vector<std::shared_ptr<ParameterService>> shards)
    :shards_(std::move(shards)) {
  detail::Require(!shards_.empty() && shards_.size()<=1024,"invalid shard endpoints");
  for(const auto& s:shards_)detail::Require(bool(s),"null parameter-service endpoint");
}
Result<Ack> DistributedClient::CreateTable(TableDefinition table) {
  std::lock_guard lock(mutex_);
  try {
    detail::Require(!failed_,"client run has failed",Error::Code::run_failed);
    detail::Require(table.shard_count==shards_.size(),"table/client shard-count mismatch");
    std::vector<std::future<Result<Ack>>> futures;futures.reserve(shards_.size());
    for(auto& s:shards_)futures.push_back(s->CreateTable(table));
    for(auto& f:futures)Take(f);
    tables_[table.id]=table;return Ack{};
  }catch(const detail::Failure& e){return e.error();}
}
Result<DistributedBatch> DistributedClient::Lookup(LookupRequest request) {
  auto prepared=PrepareLookupInputs(std::move(request.features),shards_.size());
  if(auto error=std::get_if<Error>(&prepared))return *error;
  return LookupPrepared(std::move(request.id),request.partition_epoch,request.read_only,
    std::get<PreparedLookup>(std::move(prepared)));
}
Result<DistributedBatch> DistributedClient::LookupPrepared(RequestId id,std::uint64_t epoch,
    bool read_only,PreparedLookup prepared) {
  std::lock_guard lock(mutex_);
  bool sent=false;
  DistributedBatch out;
  std::vector<std::future<Result<LookupReply>>> futures;
  std::vector<LookupReply> replies(shards_.size());
  std::size_t collected=0;
  std::optional<Error> lookup_failure;
  auto collect=[&](std::size_t until) {
    for(;collected<std::min(until,futures.size());++collected) {
      try {
        auto result=futures[collected].get();
        if(auto error=std::get_if<Error>(&result)) {
          if(!lookup_failure)lookup_failure=*error;
        }else {
          replies[collected]=std::get<LookupReply>(std::move(result));
          out.shards_[collected].ticket=replies[collected].ticket;
        }
      }catch(const std::exception& e) {
        if(!lookup_failure)lookup_failure=Error{Error::Code::unavailable,e.what()};
      }
    }
  };
  auto cleanup=[&] {
    collect(futures.size());
    // Resolve/gather completions have finished before we can reach cleanup.
    // Unknown tickets after transport failure remain the run controller's job.
    std::vector<std::future<Result<Ack>>> releases;
    for(std::size_t i=0;i<out.shards_.size();++i)if(!out.shards_[i].ticket.empty()) {
      try {
        auto release_id=id;release_id.phase=Phase::release;
        releases.push_back(shards_[i]->Release({release_id,out.shards_[i].ticket}));
      }catch(const std::exception&) {}
    }
    for(auto& release:releases)try{(void)release.get();}catch(const std::exception&){}
    if(sent)failed_=true;
  };
  try {
    detail::Require(!failed_,"client run has failed",Error::Code::run_failed);
    detail::Require(id.phase==Phase::lookup && prepared.shard_count_==shards_.size() &&
      !prepared.original_.empty() && prepared.parts_.size()==shards_.size(),"invalid prepared lookup");
    for(const auto& f:prepared.original_)
      detail::Require(tables_.contains(f.table),"unknown prepared table");
    out.original_={id,epoch,read_only,std::move(prepared.original_)};
    const auto& request=out.original_;out.shards_.resize(shards_.size());
    std::vector<LookupRequest> requests;requests.reserve(shards_.size());
    for(std::size_t shard=0;shard<shards_.size();++shard) {
      out.shards_[shard].positions=std::move(prepared.parts_[shard].positions);
      requests.push_back({id,epoch,read_only,std::move(prepared.parts_[shard].features)});
    }
    for(const auto& f:request.features) {
      const auto dim=tables_.at(f.table).dimension;
      const auto rows=f.pooling==Pooling::sequence?f.keys.size():f.valid_lengths.size();
      detail::Require(rows <= (64ULL<<20)/(sizeof(float)*dim),"client feature output exceeds limit");
      out.features_.push_back({f.feature,{rows,dim},std::vector<float>(rows*dim,0)});
    }
    futures.reserve(shards_.size());
    sent=true;
    for(std::size_t i=0;i<shards_.size();++i)futures.push_back(shards_[i]->Lookup(requests[i]));
    if(coalesce_cache_)collect(futures.size());
    if(lookup_failure)throw detail::Failure(*lookup_failure);
    using Group=std::tuple<AgentId,std::uint32_t,AccessRoute>;
    std::map<Group,AgentWork> work;
    std::unordered_map<std::uint32_t,std::size_t> feature_indices;
    for(std::size_t fi=0;fi<request.features.size();++fi)
      feature_indices.emplace(request.features[fi].feature,fi);
    for(std::size_t shard=0;shard<shards_.size();++shard) {
      if(!coalesce_cache_)collect(shard+1);
      if(lookup_failure)throw detail::Failure(*lookup_failure);
      auto& reply=replies[shard];
      detail::Require(!reply.ticket.empty(),"server returned an empty lookup ticket");
      detail::Require(reply.cpu_contributions.size()==request.features.size(),"server feature count mismatch");
      if(coalesce_cache_)for(auto& task:reply.agent_work) {
        auto found=feature_indices.find(task.feature);
        detail::Require(found!=feature_indices.end(),"unknown cached feature");
        const auto fi=found->second;
        const auto& positions=out.shards_[shard].positions[fi];
        const auto dim=tables_.at(request.features[fi].table).dimension;
        detail::Require(task.occurrence_to_unique.size()==positions.size(),"cached occurrence map size mismatch");
        detail::Require(task.route==AccessRoute::resident_gpu || task.route==AccessRoute::transfer_unique,
                        "invalid cached access route");
        detail::Require(task.route==AccessRoute::transfer_unique ?
          task.transferred_weights.size()==task.rows.size()*dim : task.transferred_weights.empty(),
          "cached transferred weight size mismatch");
        auto [it,inserted]=work.try_emplace({task.agent,task.feature,task.route});
        auto& merged=it->second;
        if(inserted) {
          merged.feature=task.feature;merged.route=task.route;merged.agent=task.agent;
          merged.occurrence_to_unique.assign(request.features[fi].keys.size(),UINT64_MAX);
        }
        const auto base=merged.rows.size();
        merged.rows.insert(merged.rows.end(),task.rows.begin(),task.rows.end());
        merged.transferred_weights.insert(merged.transferred_weights.end(),
          task.transferred_weights.begin(),task.transferred_weights.end());
        for(std::size_t i=0;i<positions.size();++i) {
          const auto unique=task.occurrence_to_unique[i];
          if(unique==UINT64_MAX)continue;
          detail::Require(unique<task.rows.size() &&
            merged.occurrence_to_unique[positions[i]]==UINT64_MAX,"invalid/duplicate cached occurrence");
          merged.occurrence_to_unique[positions[i]]=base+unique;
        }
      }
      if(!coalesce_cache_ && !reply.agent_work.empty()) {
        detail::Require(bool(resolver_),"cached rows require a cache resolver");
        auto parts=Take(resolver_(std::move(reply.agent_work),requests[shard].features));
        for(const auto& part:parts) {
          auto target=std::find_if(reply.cpu_contributions.begin(),reply.cpu_contributions.end(),
              [&](const auto& value){return value.feature==part.feature;});
          detail::Require(target!=reply.cpu_contributions.end() && target->shape==part.shape &&
              target->values.size()==part.values.size(),"cached shard contribution layout mismatch");
          for(std::size_t i=0;i<part.values.size();++i)target->values[i]+=part.values[i];
        }
      }
      reply.agent_work.clear();
      for(std::size_t fi=0;fi<request.features.size();++fi) {
        const auto& f=request.features[fi];const auto dim=tables_.at(f.table).dimension;
        const auto& src=reply.cpu_contributions[fi];auto& dst=out.features_[fi];
        detail::Require(src.feature==dst.feature,"server feature order mismatch");
        if(f.pooling==Pooling::sequence) {
          const auto& positions=out.shards_[shard].positions[fi];
          detail::Require(src.shape==std::vector<std::uint64_t>({positions.size(),dim}) &&
            src.values.size()==positions.size()*dim,"server sequence layout mismatch");
          for(std::size_t i=0;i<positions.size();++i)
            std::copy_n(src.values.data()+i*dim,dim,dst.values.data()+positions[i]*dim);
        }else {
          detail::Require(src.shape==dst.shape && src.values.size()==dst.values.size(),"server pooling layout mismatch");
          for(std::size_t i=0;i<dst.values.size();++i)dst.values[i]+=src.values[i];
        }
      }
      reply.cpu_contributions.clear();
    }
    if(!work.empty()) {
      detail::Require(bool(resolver_),"cached rows require a cache resolver");
      std::vector<AgentWork> tasks;tasks.reserve(work.size());
      for(auto& [group,task]:work)tasks.push_back(std::move(task));
      auto parts=Take(resolver_(std::move(tasks),request.features));
      for(const auto& part:parts) {
        auto found=feature_indices.find(part.feature);
        detail::Require(found!=feature_indices.end(),"unknown cached contribution");
        auto& dst=out.features_[found->second];
        detail::Require(part.shape==dst.shape && part.values.size()==dst.values.size(),
                        "cached contribution layout mismatch");
        for(std::size_t i=0;i<dst.values.size();++i)dst.values[i]+=part.values[i];
      }
    }
    for(std::size_t fi=0;fi<request.features.size();++fi) {
      const auto& f=request.features[fi];auto& tensor=out.features_[fi];
      if(f.pooling==Pooling::mean)for(std::size_t b=0;b<f.valid_lengths.size();++b)
        if(f.valid_lengths[b])for(std::uint64_t d=0;d<tensor.shape[1];++d)
          tensor.values[b*tensor.shape[1]+d]/=f.valid_lengths[b];
    }
    return out;
  }catch(const detail::Failure& e){cleanup();return e.error();}
  catch(const std::bad_alloc&){cleanup();return Error{Error::Code::capacity_exceeded,"client lookup allocation failed"};}
  catch(const std::exception& e){cleanup();return Error{Error::Code::unavailable,e.what()};}
}
Result<Ack> DistributedClient::Push(const DistributedBatch& batch,std::vector<TensorBlock> gradients) {
  std::lock_guard lock(mutex_);bool sent=false;
  try {
    detail::Require(!failed_,"client run has failed",Error::Code::run_failed);
    detail::Require(batch.shards_.size()==shards_.size() && !batch.original_.read_only,"invalid training batch");
    detail::Require(gradients.size()==batch.features_.size(),"gradient feature count mismatch");
    std::unordered_map<std::uint32_t,const TensorBlock*> by_id;
    for(const auto& g:gradients)detail::Require(by_id.emplace(g.feature,&g).second,"duplicate gradient feature");
    for(const auto& f:batch.features_) {
      detail::Require(by_id.contains(f.feature),"missing feature gradient");const auto& g=*by_id.at(f.feature);
      detail::Require(g.shape==f.shape && g.values.size()==f.values.size(),"gradient shape mismatch");
      for(auto x:g.values)detail::Require(std::isfinite(x),"non-finite gradient");
    }
    std::vector<PushRequest> requests;requests.reserve(shards_.size());
    for(std::size_t shard=0;shard<shards_.size();++shard) {
      auto id=batch.original_.id;id.phase=Phase::push;
      PushRequest request{id,batch.shards_[shard].ticket,{}};
      for(std::size_t fi=0;fi<batch.original_.features.size();++fi) {
        const auto& feature=batch.original_.features[fi];auto g=*by_id.at(feature.feature);
        if(feature.pooling==Pooling::sequence) {
          const auto& positions=batch.shards_[shard].positions[fi];const auto dim=g.shape[1];
          TensorBlock part{g.feature,{positions.size(),dim},{}};part.values.reserve(positions.size()*dim);
          for(auto pos:positions)part.values.insert(part.values.end(),g.values.begin()+pos*dim,g.values.begin()+(pos+1)*dim);
          g=std::move(part);
        }
        request.feature_gradients.push_back(std::move(g));
      }
      requests.push_back(std::move(request));
    }
    std::vector<std::future<Result<Ack>>> futures;futures.reserve(shards_.size());sent=true;
    for(std::size_t i=0;i<shards_.size();++i)futures.push_back(shards_[i]->Push(std::move(requests[i])));
    std::optional<Error> failure;
    for(auto& f:futures){auto result=f.get();if(auto error=std::get_if<Error>(&result))failure=*error;}
    if(failure)throw detail::Failure(*failure);
    return Ack{};
  }catch(const detail::Failure& e){if(sent)failed_=true;return e.error();}
}
Result<Ack> DistributedClient::Release(const DistributedBatch& batch) {
  std::lock_guard lock(mutex_);
  try {
    detail::Require(batch.shards_.size()==shards_.size(),"batch/client topology mismatch");
    auto id=batch.original_.id;id.phase=Phase::release;
    std::vector<std::future<Result<Ack>>> futures;futures.reserve(shards_.size());
    for(std::size_t i=0;i<shards_.size();++i)futures.push_back(shards_[i]->Release({id,batch.shards_[i].ticket}));
    for(auto& f:futures)Take(f);
    return Ack{};
  }catch(const detail::Failure& e){failed_=true;return e.error();}
}
void DistributedClient::SetCacheResolver(CacheResolver resolver,bool coalesce){
  std::lock_guard lock(mutex_);resolver_=std::move(resolver);coalesce_cache_=coalesce;
}
}  // namespace eps
