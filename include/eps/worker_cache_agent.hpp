#pragma once
#include <mutex>
#include "eps/agent_admission.hpp"
#include "eps/cache_agent.hpp"

namespace eps {
struct WorkerCacheOptions {
  std::string run;
  std::string listen="127.0.0.1:0";
  AgentAdmissionOptions admission;
  std::uint64_t replay_bytes=64ULL<<20;
  // Stable model identity; run identifies this particular owner incarnation.
  // Empty preserves the existing shared run identity.
  std::string training_run;
};

// Owned by the training process. PS control RPCs and native feature reads use
// the same admission/ordering boundary. Callers retain ordinary lookup tickets
// until backward/update or explicit release; this owner does not replace them.
class WorkerCacheAgent {
 public:
  WorkerCacheAgent(WorkerCacheOptions,std::shared_ptr<CacheBackend>);
  ~WorkerCacheAgent();
  WorkerCacheAgent(const WorkerCacheAgent&)=delete;
  const std::string& endpoint() const {return endpoint_;}
  std::future<Result<std::vector<TensorBlock>>> GatherHost(std::vector<AgentWork>,std::vector<FeatureInput>);
  std::future<Result<std::vector<DeviceTensorBlock>>> GatherDevice(std::vector<AgentWork>,std::vector<FeatureInput>);
  AgentAdmissionStats Stats() const;
  // Stop input, drain admitted RPCs/local reads, then release owned runtime and
  // backend references. Independent device results may outlive this object.
  void Stop();
 private:
  struct Impl;
  std::string endpoint_;
  mutable std::mutex mutex_;
  std::unique_ptr<Impl> impl_;
  AgentAdmissionStats final_stats_;
};
}  // namespace eps
