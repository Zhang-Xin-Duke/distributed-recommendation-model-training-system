#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace eps {

using TableId = std::uint32_t;
using ShardId = std::uint32_t;
using AgentId = std::uint32_t;
using Key = std::uint64_t;

struct Error {
  enum class Code { invalid_argument, unavailable, stale_generation,
                    retired_request, capacity_exceeded, run_failed };
  Code code;
  std::string message;
};

template <typename T>
using Result = std::variant<T, Error>;

struct Ack {};

struct CheckpointIdentity {
  std::string run;
  std::uint64_t partition_epoch;
  std::string snapshot_id;
};
struct SnapshotChunk {
  std::uint64_t offset;
  std::uint64_t total_bytes;
  std::string data;
  bool eof;
};

struct RestoredSnapshot {
  bool loaded=false, pristine=false;
  std::string run, source_run;
  std::uint64_t bytes=0, checksum=0;
  ShardId shard=0;
  std::uint32_t shard_count=0;
  std::uint64_t partition_epoch=0;
};

enum class Phase { lookup, push, release, maintenance };
enum class Pooling { sum, mean, sequence };
enum class Placement { cpu, loading, gpu, draining };
enum class AccessRoute { cpu_pool, transfer_unique, resident_gpu };

struct RequestId {
  std::string run;
  std::uint32_t learner;
  std::uint64_t sequence;
  Phase phase;
  bool operator==(const RequestId&) const = default;
};
struct RequestFrontier { std::uint32_t learner; std::uint64_t next; };

struct RowKey {
  TableId table;
  Key key;
  bool operator==(const RowKey&) const = default;
};

struct WeightBounds {
  float minimum;
  float maximum;
  bool operator==(const WeightBounds&) const = default;
};

struct OptimizerConfig {
  // Initial implementation: unshared FP32 sparse LazyAdam states.
  float learning_rate;
  float beta1;
  float beta2;
  float epsilon;
  float initial_beta1_power;
  float initial_beta2_power;
  std::optional<WeightBounds> weight_bounds = std::nullopt;
  bool operator==(const OptimizerConfig&) const = default;
};

struct TableDefinition {
  TableId id;
  std::uint32_t dimension;
  std::uint32_t shard_count;
  std::uint64_t partition_epoch;
  OptimizerConfig optimizer;
  float initialization_min;
  float initialization_max;
  std::uint64_t initialization_seed;
  bool operator==(const TableDefinition&) const = default;
};

struct FeatureInput {
  TableId table;
  std::uint32_t feature;
  Pooling pooling;
  std::vector<Key> keys;
  // Offsets delimit bags/sequences; padding keys are removed by the adapter.
  std::vector<std::uint64_t> offsets;
  // Original lengths before shard/CPU/GPU partitioning; used once for mean.
  std::vector<std::uint64_t> valid_lengths;
};

struct TensorBlock {
  std::uint32_t feature;
  std::vector<std::uint64_t> shape;
  std::vector<float> values;
};

// Adapter-defined holder for an existing framework tensor on the GPU.
struct DeviceTensor;
struct DeviceTensorBlock {
  std::uint32_t feature;
  std::vector<std::uint64_t> shape;
  std::shared_ptr<DeviceTensor> tensor;
};

struct LocatedRow {
  RowKey row;
  ShardId home;
  Placement placement;
  AgentId agent;
  std::uint64_t generation;
  std::uint64_t slot;
};

struct AgentWork {
  std::uint32_t feature;
  AccessRoute route;
  AgentId agent;
  std::vector<LocatedRow> rows;
  std::vector<std::uint64_t> occurrence_to_unique;
  // CPU-backed unique vectors, or an empty payload for resident rows.
  std::vector<float> transferred_weights;
};

struct LookupRequest {
  RequestId id;
  std::uint64_t partition_epoch;
  bool read_only;
  std::vector<FeatureInput> features;
};

struct LookupReply {
  // Opaque shard-local ticket; owns occurrence maps and row references.
  std::string ticket;
  std::vector<TensorBlock> cpu_contributions;
  std::vector<AgentWork> agent_work;
};

struct PushRequest {
  RequestId id;
  std::string ticket;
  // Gradients of completed feature tensors, before service-side unpooling.
  std::vector<TensorBlock> feature_gradients;
};

struct ReleaseRequest {
  RequestId id;
  std::string ticket;
  // Retire a failed read by identity, including a Lookup that has not arrived.
  bool cancel_unseen = false;
  std::uint64_t partition_epoch = 0;
};

struct RowState {
  RowKey row;
  std::vector<float> weights;
  std::vector<float> first_moment;
  std::vector<float> second_moment;
  std::uint64_t row_updates;
};

struct UpdateContext {
  RequestId original_request;
  ShardId shard;
  TableId table;
  std::uint64_t optimizer_sequence;
  float learning_rate;
  float beta1_power;
  float beta2_power;
};

struct RowGradients {
  std::vector<RowKey> rows;
  // Row-major unique-key gradients, reduced within one original Push only.
  std::vector<float> values;
};

// CSR-style references from each unique parameter row to the feature-gradient
// rows contributing to it. Repeated IDs retain repeated, ordered references.
struct CompactGradients {
  std::vector<RowKey> rows;
  std::uint32_t dimension;
  std::vector<float> source_values;
  std::vector<std::uint64_t> row_offsets;
  std::vector<std::uint64_t> source_rows;
  std::vector<float> scales;
};

struct DeviceRowGradients {
  std::vector<RowKey> rows;
  std::shared_ptr<DeviceTensor> values;
};

struct MigrationBudget {
  std::uint64_t network_bytes;
  std::uint64_t staging_bytes;
  std::uint32_t max_inflight_batches;
  std::uint64_t reserve_slots;
};

struct MigrationPlan {
  std::string id;
  AgentId agent;
  std::vector<LocatedRow> rows;
  Placement destination;
  std::uint64_t estimated_bytes;
  // Enforced atomically by the destination backend when slots are allocated.
  std::uint64_t reserve_slots = 0;
};

struct MigrationCompletion {
  std::string plan_id;
  std::vector<LocatedRow> rows;
  bool completed;
};

struct MoveRowsRequest {
  std::string run;
  std::vector<RowKey> rows;
  AgentId agent;
  Placement destination;
  MigrationBudget budget;
};

struct AgentCommand {
  std::string run;
  ShardId shard;
  std::uint64_t sequence;
};

struct SnapshotMetadata {
  std::string snapshot_id;
  ShardId shard;
  std::uint64_t partition_epoch;
  std::vector<std::string> state_files;
  std::vector<UpdateContext> optimizer_contexts;
};

}  // namespace eps
