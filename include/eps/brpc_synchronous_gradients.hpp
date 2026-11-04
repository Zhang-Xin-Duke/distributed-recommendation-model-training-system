#pragma once

#include "eps/synchronous_gradients.hpp"

namespace eps {
class BrpcSynchronousGradients {
 public:
  BrpcSynchronousGradients(std::string endpoint, std::string run,
                           std::uint64_t epoch = 1, int timeout_ms = 35000);
  ~BrpcSynchronousGradients();
  std::future<Result<Ack>> Submit(GradientContribution contribution);
  std::future<Result<SynchronousGradientStats>> Stats();
  std::future<Result<Ack>> Abort(std::string reason);
 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace eps
