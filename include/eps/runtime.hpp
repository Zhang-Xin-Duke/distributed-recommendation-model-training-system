#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "eps/parameter_service.hpp"
#include "eps/cache_backend.hpp"
#include "eps/access_planner.hpp"

namespace eps {

struct RuntimeOptions {
  std::string run_id;
  ShardId shard_id = 0;
  std::uint32_t shard_count = 1;
  std::uint64_t partition_epoch = 1;
  std::uint64_t state_bytes = 1ULL << 30;
  std::uint64_t max_request_bytes = 16ULL << 20;
  std::uint64_t ticket_payload_bytes = 64ULL << 20;
  std::uint64_t queue_payload_bytes = 64ULL << 20;
  std::uint32_t max_live_tickets = 1024;
  std::uint32_t request_history = 4096;
  std::uint32_t max_queued_operations = 1024;
  bool concurrent_agent_updates = true;
  std::filesystem::path snapshot_directory = "snapshots";
};

struct RuntimeStats {
  std::uint64_t lookups = 0;
  std::uint64_t updates = 0;
  std::uint64_t duplicate_requests = 0;
  std::uint64_t rows = 0;
  std::uint64_t state_bytes = 0;
  std::uint64_t live_tickets = 0;
  std::uint64_t ticket_payload_bytes = 0;
  std::uint64_t queued_payload_bytes = 0;
  std::uint64_t cached_rows = 0;
  std::uint64_t migration_bytes = 0;
  std::uint64_t executed_operations = 0;
  std::uint64_t queue_wait_nanoseconds = 0;
  std::uint64_t executor_busy_nanoseconds = 0;
  std::uint64_t rejected_operations = 0;
  std::uint64_t peak_queued_operations = 0;
};

// A bounded asynchronous shard sequencer. Transport callbacks never execute
// parameter work, and independent server shards progress independently.
class CpuParameterService final : public ParameterService {
 public:
  explicit CpuParameterService(RuntimeOptions options);
  ~CpuParameterService() override;
  CpuParameterService(const CpuParameterService&) = delete;
  CpuParameterService& operator=(const CpuParameterService&) = delete;

  std::future<Result<Ack>> CreateTable(TableDefinition table) override;
  std::future<Result<LookupReply>> Lookup(LookupRequest request) override;
  std::future<Result<Ack>> Push(PushRequest request) override;
  std::future<Result<Ack>> Release(ReleaseRequest request) override;
  std::future<Result<std::vector<RowState>>> ReadState(
      std::vector<LocatedRow> rows) override;
  std::future<Result<Ack>> CompleteMigration(
      MigrationCompletion completion) override;
  std::future<Result<SnapshotMetadata>> DrainAndFlush(
      std::string snapshot_id) override;

  // Restore requires a fresh service and a new run ID. Resume reopens admission
  // after the framework has completed its coordinated save boundary.
  std::future<Result<Ack>> Restore(std::filesystem::path snapshot_file);
  std::future<Result<Ack>> Resume();
  std::future<Result<RuntimeStats>> Stats();
  std::future<Result<Ack>> AttachAgent(AgentId id,std::shared_ptr<CacheBackend> backend);
  std::future<Result<Ack>> ConfigureAccess(AgentId agent,AccessRoute fallback,
                                         std::shared_ptr<AccessPlanner> planner={});
  std::future<Result<std::vector<LocatedRow>>> Locations(std::vector<RowKey> rows);
  std::future<Result<std::vector<LocatedRow>>> MoveRows(MoveRowsRequest request);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace eps
