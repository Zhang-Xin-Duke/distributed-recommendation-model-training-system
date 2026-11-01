#pragma once

#include <memory>
#include <string>
#include "eps/parameter_service.hpp"
#include "eps/runtime.hpp"

namespace eps {
class BrpcParameterService final : public ParameterService {
 public:
  explicit BrpcParameterService(std::string endpoint, int timeout_ms = 5000);
  ~BrpcParameterService() override;
  std::future<Result<Ack>> CreateTable(TableDefinition) override;
  std::future<Result<LookupReply>> Lookup(LookupRequest) override;
  std::future<Result<Ack>> Push(PushRequest) override;
  std::future<Result<Ack>> Release(ReleaseRequest) override;
  std::future<Result<std::vector<RowState>>> ReadState(std::vector<LocatedRow>) override;
  std::future<Result<Ack>> CompleteMigration(MigrationCompletion) override;
  std::future<Result<SnapshotMetadata>> DrainAndFlush(std::string) override;
  std::future<Result<SnapshotChunk>> ReadCheckpoint(CheckpointIdentity,
      std::uint64_t offset, std::uint32_t max_bytes) override;
  std::future<Result<Ack>> ResumeCheckpoint(CheckpointIdentity) override;
  std::future<Result<std::vector<LocatedRow>>> Locations(std::vector<RowKey>);
  std::future<Result<std::vector<LocatedRow>>> MoveRows(MoveRowsRequest);
  std::future<Result<RuntimeStats>> Stats();
 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace eps
