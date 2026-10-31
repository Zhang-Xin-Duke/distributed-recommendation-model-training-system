#include "eps/distributed_client.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
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
  std::lock_guard lock(mutex_);
  bool sent=false;
  try {
    detail::Require(!failed_,"client run has failed",Error::Code::run_failed);
    detail::Require(request.id.phase==Phase::lookup && !request.features.empty(),"invalid lookup");
    std::unordered_set<std::uint32_t> feature_ids;
    for(const auto& f:request.features) {
      detail::Require(tables_.contains(f.table) && feature_ids.insert(f.feature).second,"unknown/duplicate feature");
      detail::Require(f.pooling==Pooling::sum || f.pooling==Pooling::mean || f.pooling==Pooling::sequence,"invalid pooling");
      detail::Require(!f.offsets.empty() && f.offsets.front()==0 && f.offsets.back()==f.keys.size() &&
        std::is_sorted(f.offsets.begin(),f.offsets.end()) && f.offsets.size()==f.valid_lengths.size()+1,"invalid feature layout");
      for(std::size_t b=0;b<f.valid_lengths.size();++b)
        detail::Require(f.valid_lengths[b]==f.offsets[b+1]-f.offsets[b],"client requires original unpadded bag lengths");
    }
    DistributedBatch out;out.original_=request;out.shards_.resize(shards_.size());
    std::vector<LookupRequest> requests(shards_.size(),request);
    for(std::size_t shard=0;shard<shards_.size();++shard) {
      auto& part=out.shards_[shard];part.positions.resize(request.features.size());
      for(std::size_t fi=0;fi<request.features.size();++fi) {
        const auto& f=request.features[fi];auto& dst=requests[shard].features[fi];
        dst.keys.clear();dst.offsets={0};
        for(std::size_t b=0;b<f.valid_lengths.size();++b) {
          for(auto pos=f.offsets[b];pos<f.offsets[b+1];++pos)if(f.keys[pos]%shards_.size()==shard) {
            dst.keys.push_back(f.keys[pos]);part.positions[fi].push_back(pos);
          }
          dst.offsets.push_back(dst.keys.size());
        }
      }
    }
    for(const auto& f:request.features) {
      const auto dim=tables_.at(f.table).dimension;
      const auto rows=f.pooling==Pooling::sequence?f.keys.size():f.valid_lengths.size();
      detail::Require(rows <= (64ULL<<20)/(sizeof(float)*dim),"client feature output exceeds limit");
      out.features_.push_back({f.feature,{rows,dim},std::vector<float>(rows*dim,0)});
    }
    std::vector<std::future<Result<LookupReply>>> futures;futures.reserve(shards_.size());
    sent=true;
    for(std::size_t i=0;i<shards_.size();++i)futures.push_back(shards_[i]->Lookup(std::move(requests[i])));
    std::optional<Error> failure;
    for(std::size_t shard=0;shard<shards_.size();++shard) {
      auto result=futures[shard].get();
      if(auto error=std::get_if<Error>(&result)){failure=*error;continue;}
      auto reply=std::get<LookupReply>(std::move(result));out.shards_[shard].ticket=reply.ticket;
      detail::Require(reply.agent_work.empty(),"GPU work requires the GPU adapter");
      detail::Require(reply.cpu_contributions.size()==request.features.size(),"server feature count mismatch");
      for(std::size_t fi=0;fi<request.features.size();++fi) {
        const auto& f=request.features[fi];const auto dim=tables_.at(f.table).dimension;
        const auto& src=reply.cpu_contributions[fi];auto& dst=out.features_[fi];
        detail::Require(src.feature==dst.feature,"server feature order mismatch");
        if(f.pooling==Pooling::sequence) {
          const auto& positions=out.shards_[shard].positions[fi];
          detail::Require(src.values.size()==positions.size()*dim,"server sequence layout mismatch");
          for(std::size_t i=0;i<positions.size();++i)
            std::copy_n(src.values.data()+i*dim,dim,dst.values.data()+positions[i]*dim);
        }else {
          detail::Require(src.shape==dst.shape && src.values.size()==dst.values.size(),"server pooling layout mismatch");
          for(std::size_t i=0;i<dst.values.size();++i)dst.values[i]+=src.values[i];
        }
      }
    }
    if(failure) {
      for(std::size_t i=0;i<shards_.size();++i)if(!out.shards_[i].ticket.empty()) {
        auto id=request.id;id.phase=Phase::release;
        auto released=shards_[i]->Release({id,out.shards_[i].ticket});released.wait();
      }
      throw detail::Failure(*failure);
    }
    for(std::size_t fi=0;fi<request.features.size();++fi) {
      const auto& f=request.features[fi];auto& tensor=out.features_[fi];
      if(f.pooling==Pooling::mean)for(std::size_t b=0;b<f.valid_lengths.size();++b)
        if(f.valid_lengths[b])for(std::uint64_t d=0;d<tensor.shape[1];++d)
          tensor.values[b*tensor.shape[1]+d]/=f.valid_lengths[b];
    }
    return out;
  }catch(const detail::Failure& e){if(sent)failed_=true;return e.error();}
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
}  // namespace eps
