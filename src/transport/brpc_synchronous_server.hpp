#pragma once

#include "embedding_service.pb.h"
#include "eps/runtime.hpp"
#include "eps/synchronous_gradients.hpp"

namespace eps::transport {
class BrpcSynchronousServer final : public v1::SynchronousGradientService {
 public:
  // RuntimeOptions::exclusive_update_learner must be 1023 for this mode.
  BrpcSynchronousServer(CpuParameterService& parameters, SynchronousGradientOptions options);
  ~BrpcSynchronousServer() override;
  void Shutdown();
  void Submit(google::protobuf::RpcController*, const v1::GradientContribution*,
              v1::Ack*, google::protobuf::Closure*) override;
  void Stats(google::protobuf::RpcController*, const v1::TrainingControl*,
             v1::SynchronousGradientStatsReply*, google::protobuf::Closure*) override;
  void Abort(google::protobuf::RpcController*, const v1::TrainingControl*,
             v1::Ack*, google::protobuf::Closure*) override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace eps::transport
