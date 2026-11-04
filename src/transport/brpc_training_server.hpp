#pragma once

#include "embedding_service.pb.h"
#include "eps/training_progress.hpp"

namespace eps::transport {
// Begin can wait for another worker. Pending calls are polled by one event
// loop; they never occupy the parameter service's blocking completion pool.
class BrpcTrainingServer final : public v1::TrainingService {
 public:
  explicit BrpcTrainingServer(TrainingProgressOptions options,
                              std::size_t max_pending = 4096);
  ~BrpcTrainingServer() override;
  Result<Ack> Restore(TrainingProgressCheckpoint checkpoint);
  Result<Ack> RecoverSynchronousRound(std::uint64_t next);
  // Call before Server::Stop/Join so outstanding Begin callbacks can drain.
  void Shutdown();
#define EPS_TRAINING_RPC(Name, Request, Response) \
  void Name(google::protobuf::RpcController*, const v1::Request*, v1::Response*, google::protobuf::Closure*) override;
  EPS_TRAINING_RPC(Begin, TrainingTask, TrainingPermitReply)
  EPS_TRAINING_RPC(Complete, TrainingTask, Ack)
  EPS_TRAINING_RPC(Stats, TrainingControl, TrainingStatsReply)
  EPS_TRAINING_RPC(Pause, TrainingControl, Ack)
  EPS_TRAINING_RPC(Checkpoint, TrainingControl, TrainingCheckpointReply)
  EPS_TRAINING_RPC(Resume, TrainingControl, Ack)
  EPS_TRAINING_RPC(Abort, TrainingControl, Ack)
#undef EPS_TRAINING_RPC
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace eps::transport
