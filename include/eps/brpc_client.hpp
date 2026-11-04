#pragma once

#include <memory>
#include <string>
#include "eps/parameter_service.hpp"
#include "eps/runtime.hpp"
#include "eps/lookup_layout_cache.hpp"

namespace eps {
struct LookupTransportStats {
  std::uint64_t full_requests = 0, token_requests = 0, fallback_requests = 0,
    request_bytes = 0, avoided_request_bytes = 0;
  LookupLayoutCacheStats cache;
};
class BrpcParameterService final : public ParameterService {
 public:
  explicit BrpcParameterService(std::string endpoint, int timeout_ms = 5000, LookupLayoutCacheOptions = {});
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
  std::future<Result<RestoredSnapshot>> RestoredFrom();
  // Configure before training; values/tickets are never cached. Disabled by default.
  void SetLookupLayoutCache(bool enabled);
  LookupTransportStats LookupTransport() const;
  std::future<Result<LookupLayoutCacheStats>> LookupLayoutStats();
 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};
}  // namespace eps
