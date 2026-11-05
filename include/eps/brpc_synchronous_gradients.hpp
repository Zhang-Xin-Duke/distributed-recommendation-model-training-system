#pragma once

#include "eps/synchronous_gradients.hpp"
#include "eps/gradient_layout_cache.hpp"

namespace eps {
struct GradientTransportStats {
  std::uint64_t full_requests=0,token_requests=0,fallback_requests=0,
    request_bytes=0,avoided_request_bytes=0;
  GradientLayoutCacheStats cache;
};
class BrpcSynchronousGradients {
 public:
  BrpcSynchronousGradients(std::string endpoint, std::string run,
                           std::uint64_t epoch = 1, int timeout_ms = 35000,
                           GradientLayoutCacheOptions cache = {});
  ~BrpcSynchronousGradients();
  std::future<Result<Ack>> Submit(GradientContribution contribution);
  std::future<Result<SynchronousGradientStats>> Stats();
  std::future<Result<Ack>> Abort(std::string reason);
  std::future<Result<Ack>> Pause();
  std::future<Result<Ack>> Resume();
  std::future<Result<SynchronousGradientCheckpoint>> Checkpoint();
  // Optional immutable Key metadata only; never reuse gradient values.
  void SetGradientLayoutCache(bool enabled);
  GradientTransportStats TransportStats() const;
  std::future<Result<GradientLayoutCacheStats>> LayoutStats();
 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
struct GradientRetryStats { std::uint64_t attempts=0,retries=0; };
// Blocking framework-worker helper; never call from a transport completion.
// Retains original gradients until every shard acknowledges the same task.
Result<GradientRetryStats> SubmitGradientGroup(const std::vector<BrpcSynchronousGradients*>&,
    const std::vector<GradientContribution>&,std::chrono::milliseconds retry_window=std::chrono::milliseconds(0));
}  // namespace eps
