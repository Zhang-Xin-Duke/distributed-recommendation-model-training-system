#pragma once

#include <memory>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "eps/parameter_service.hpp"
#include "eps/prepared_lookup.hpp"
#include "eps/device_features.hpp"

namespace eps {
class DistributedBatch {
 public:
  const std::vector<TensorBlock>& features() const { return features_; }
  // Device batches retain compact host shape metadata, without host values.
  const std::vector<DeviceTensorBlock>& device_features() const {return device_features_;}
 private:
  friend class DistributedClient;
  LookupRequest original_;
  std::vector<TensorBlock> features_;
  std::vector<DeviceTensorBlock> device_features_;
  struct Part {
    std::string ticket;
    std::vector<std::vector<std::uint64_t>> positions;
  };
  std::vector<Part> shards_;
};

// Fan-out/fan-in client. It preserves global bag denominators and sequence
// positions; each server still owns an independent asynchronous update stream.
class DistributedClient {
 public:
  explicit DistributedClient(std::vector<std::shared_ptr<ParameterService>> shards);
  Result<Ack> CreateTable(TableDefinition table);
  Result<DistributedBatch> Lookup(LookupRequest request);
  Result<DistributedBatch> LookupPrepared(RequestId id,std::uint64_t partition_epoch,
      bool read_only,PreparedLookup prepared);
  Result<DistributedBatch> LookupDevice(LookupRequest request);
  Result<DistributedBatch> LookupPreparedDevice(RequestId id,std::uint64_t partition_epoch,
      bool read_only,PreparedLookup prepared,std::vector<SequenceProjection> projections={});
  Result<Ack> Push(const DistributedBatch& batch, std::vector<TensorBlock> gradients);
  Result<Ack> Release(const DistributedBatch& batch);
  // Receives shard-local work/layouts by default. With coalesce=true it receives
  // all shards' work remapped to ORIGINAL layouts and occurrence positions.
  // Returns unnormalized sums or sequence contributions in the supplied layout. It must finish all submitted reads before returning
  // (including errors), so lookup-failure cleanup may safely release row pins.
  using CacheResolver=std::function<Result<std::vector<TensorBlock>>(
      std::vector<AgentWork>,std::vector<FeatureInput>)>;
  void SetCacheResolver(CacheResolver resolver,bool coalesce=false);
  using DeviceResolver=std::function<Result<ResolvedDeviceFeatures>(std::vector<AgentWork>,std::vector<FeatureInput>)>;
  // Must finish submitted device reads before returning, including errors.
  using DeviceAssembler=std::function<Result<std::vector<DeviceTensorBlock>>(DeviceAssemblyRequest)>;
  void SetDevicePath(DeviceResolver,DeviceAssembler);
 private:
  Result<DistributedBatch> LookupImpl(RequestId,std::uint64_t,bool,PreparedLookup,
      bool device,std::vector<SequenceProjection>);
  std::vector<std::shared_ptr<ParameterService>> shards_;
  std::unordered_map<TableId, TableDefinition> tables_;
  // One client may be used by multiple graph threads. The transport calls are
  // asynchronous across shards; this lock protects its metadata and failure state.
  std::mutex mutex_;
  bool failed_ = false;
  bool coalesce_cache_ = false;
  CacheResolver resolver_;
  DeviceResolver device_resolver_;
  DeviceAssembler device_assembler_;
};
}  // namespace eps
