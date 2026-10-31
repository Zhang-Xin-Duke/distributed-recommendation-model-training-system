#pragma once

#include <future>
#include <vector>

#include "eps/types.hpp"

namespace eps {

// Design contract. Futures must not block the transport's completion threads.
// A client groups rows by home shard; lookup tickets are shard-local.
class ParameterService {
 public:
  virtual ~ParameterService() = default;

  virtual std::future<Result<Ack>> CreateTable(TableDefinition table) = 0;
  virtual std::future<Result<LookupReply>> Lookup(LookupRequest request) = 0;
  virtual std::future<Result<Ack>> Push(PushRequest request) = 0;
  // Read-only release or a framework-coordinated abandoned lookup.
  virtual std::future<Result<Ack>> Release(ReleaseRequest request) = 0;

  // Maintenance reads require quiescent rows at the stated generation.
  virtual std::future<Result<std::vector<RowState>>> ReadState(
      std::vector<LocatedRow> rows) = 0;
  virtual std::future<Result<Ack>> CompleteMigration(
      MigrationCompletion completion) = 0;

  // The caller first closes input and reaches the framework save boundary.
  virtual std::future<Result<SnapshotMetadata>> DrainAndFlush(
      std::string snapshot_id) = 0;
};

}  // namespace eps
