#include "eps/cache_resolver.hpp"

#include <optional>

namespace eps {
DistributedClient::CacheResolver MakeCacheResolver(
    std::map<AgentId,std::shared_ptr<CacheBackend>> agents) {
  return [agents=std::move(agents)](std::vector<AgentWork> work,
      std::vector<FeatureInput> layouts)->Result<std::vector<TensorBlock>> {
    std::map<AgentId,std::vector<AgentWork>> groups;
    for(auto& task:work) {
      auto endpoint=agents.find(task.agent);
      if(endpoint==agents.end() || !endpoint->second)
        return Error{Error::Code::invalid_argument,"unknown worker cache endpoint"};
      groups[task.agent].push_back(std::move(task));
    }
    std::vector<std::future<Result<std::vector<TensorBlock>>>> pending;
    pending.reserve(groups.size());
    std::optional<Error> failure;
    // Dispatch without waiting; backend futures own their input payloads.
    try {
      for(auto& [id,tasks]:groups)
        pending.push_back(agents.at(id)->GatherHost(std::move(tasks),layouts));
    }catch(const std::exception& e) {
      failure=Error{Error::Code::unavailable,e.what()};
    }
    std::vector<TensorBlock> output;
    for(auto& future:pending) {
      try {
        auto result=future.get();
        if(auto error=std::get_if<Error>(&result)) {
          if(!failure)failure=*error;
        }else if(!failure) {
          for(auto& part:std::get<std::vector<TensorBlock>>(result))
            output.push_back(std::move(part));
        }
      }catch(const std::exception& e) {
        if(!failure)failure=Error{Error::Code::unavailable,e.what()};
      }
    }
    if(failure)return *failure;
    return output;
  };
}
}  // namespace eps
