#include "eps/cache_resolver.hpp"

#include <algorithm>
#include <optional>
#include "common/error.hpp"

namespace eps {
namespace {
struct Projection {
  Pooling pooling;
  std::uint64_t original_rows,bags;
  std::vector<std::uint64_t> positions;
};
struct Destination {
  std::vector<AgentWork> work;
  std::vector<FeatureInput> layouts;
  std::map<std::uint32_t,Projection> projections;
};
void Project(Destination& destination,const std::vector<FeatureInput>& layouts) {
  for(const auto& f:layouts) {
    std::vector<AgentWork*> tasks;
    for(auto& task:destination.work)if(task.feature==f.feature)tasks.push_back(&task);
    if(tasks.empty())continue;
    detail::Require(!destination.projections.contains(f.feature) && !f.offsets.empty() &&
      f.offsets.front()==0 && f.offsets.back()==f.keys.size() &&
      std::is_sorted(f.offsets.begin(),f.offsets.end()) &&
      f.valid_lengths.size()+1==f.offsets.size(),"invalid destination feature layout");
    std::vector<unsigned char> selected(f.keys.size(),0);
    for(auto* task:tasks) {
      detail::Require(task->occurrence_to_unique.size()==f.keys.size(),"destination occurrence map size mismatch");
      for(std::size_t i=0;i<f.keys.size();++i)
        if(task->occurrence_to_unique[i]!=UINT64_MAX)selected[i]=1;
    }
    Projection projection{f.pooling,f.keys.size(),f.offsets.size()-1,{}};
    projection.positions.reserve(std::count(selected.begin(),selected.end(),1));
    FeatureInput local{f.table,f.feature,f.pooling,{},{0},f.valid_lengths};
    local.keys.reserve(projection.positions.capacity());local.offsets.reserve(f.offsets.size());
    for(std::size_t bag=0;bag+1<f.offsets.size();++bag) {
      for(auto i=f.offsets[bag];i<f.offsets[bag+1];++i)if(selected[i]) {
        local.keys.push_back(f.keys[i]);projection.positions.push_back(i);
      }
      local.offsets.push_back(local.keys.size());
    }
    for(auto* task:tasks) {
      std::vector<std::uint64_t> map;map.reserve(projection.positions.size());
      for(auto position:projection.positions)map.push_back(task->occurrence_to_unique[position]);
      task->occurrence_to_unique=std::move(map);
    }
    destination.layouts.push_back(std::move(local));
    destination.projections.emplace(f.feature,std::move(projection));
  }
  for(const auto& task:destination.work)
    detail::Require(destination.projections.contains(task.feature),"unknown destination feature");
}
TensorBlock Restore(TensorBlock part,const Destination& destination) {
  auto it=destination.projections.find(part.feature);
  detail::Require(it!=destination.projections.end() && part.shape.size()==2 &&
                  part.shape[1]>0 && part.shape[1]<=(64ULL<<20)/sizeof(float),
                  "invalid destination contribution");
  const auto& projection=it->second;const auto dim=part.shape[1];
  const auto expected=projection.pooling==Pooling::sequence?projection.positions.size():projection.bags;
  detail::Require(expected<=(64ULL<<20)/(sizeof(float)*dim) && part.shape[0]==expected &&
                  part.values.size()==expected*dim,"destination contribution shape mismatch");
  if(projection.pooling!=Pooling::sequence)return part;
  detail::Require(projection.original_rows<=(64ULL<<20)/(sizeof(float)*dim),
                  "restored sequence exceeds client output limit",Error::Code::capacity_exceeded);
  TensorBlock output{part.feature,{projection.original_rows,dim},
                     std::vector<float>(projection.original_rows*dim,0)};
  for(std::size_t i=0;i<projection.positions.size();++i)
    std::copy_n(part.values.data()+i*dim,dim,output.values.data()+projection.positions[i]*dim);
  return output;
}
}  // namespace
DistributedClient::CacheResolver MakeCacheResolver(
    std::map<AgentId,std::shared_ptr<CacheBackend>> agents) {
  return [agents=std::move(agents)](std::vector<AgentWork> work,
      std::vector<FeatureInput> layouts)->Result<std::vector<TensorBlock>> {
    std::map<AgentId,Destination> groups;
    for(auto& task:work) {
      auto endpoint=agents.find(task.agent);
      if(endpoint==agents.end() || !endpoint->second)
        return Error{Error::Code::invalid_argument,"unknown worker cache endpoint"};
      groups[task.agent].work.push_back(std::move(task));
    }
    try {
      for(auto& [id,destination]:groups)Project(destination,layouts);
    }catch(const detail::Failure& e){return e.error();}
    catch(const std::bad_alloc&){return Error{Error::Code::capacity_exceeded,"destination projection allocation failed"};}
    struct Pending {
      std::future<Result<std::vector<TensorBlock>>> result;
      const Destination* destination;
    };
    std::vector<Pending> pending;
    pending.reserve(groups.size());
    std::optional<Error> failure;
    // Dispatch without waiting; backend futures own their input payloads.
    try {
      for(auto& [id,destination]:groups)
        pending.push_back({agents.at(id)->GatherHost(std::move(destination.work),
            std::move(destination.layouts)),&destination});
    }catch(const std::exception& e) {
      failure=Error{Error::Code::unavailable,e.what()};
    }
    std::vector<TensorBlock> output;
    for(auto& completion:pending) {
      try {
        auto result=completion.result.get();
        if(auto error=std::get_if<Error>(&result)) {
          if(!failure)failure=*error;
        }else if(!failure) {
          for(auto& part:std::get<std::vector<TensorBlock>>(result))
            output.push_back(Restore(std::move(part),*completion.destination));
        }
      }catch(const detail::Failure& e) {
        if(!failure)failure=e.error();
      }catch(const std::exception& e) {
        if(!failure)failure=Error{Error::Code::unavailable,e.what()};
      }
    }
    if(failure)return *failure;
    return output;
  };
}
}  // namespace eps
