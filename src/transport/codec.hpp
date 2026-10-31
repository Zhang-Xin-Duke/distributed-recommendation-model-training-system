#pragma once

#include <bit>
#include <cstring>
#include <optional>
#include <span>
#include <type_traits>

#include <butil/iobuf.h>
#include "embedding_service.pb.h"
#include "common/error.hpp"
#include "eps/types.hpp"

namespace eps::transport {
namespace wire = ::eps::v1;
constexpr std::uint64_t kMaxAttachment = 16ULL << 20;

template <typename T> constexpr wire::ElementType Element() {
  if constexpr (std::is_same_v<T, float>) return wire::ELEMENT_FLOAT32;
  else { static_assert(std::is_same_v<T, std::uint64_t>); return wire::ELEMENT_UINT64; }
}
template <typename T>
void Pack(const std::vector<T>& values, wire::PayloadSlice* slice, butil::IOBuf& out) {
  static_assert(std::endian::native == std::endian::little,
                "transport currently requires a little-endian host");
  const auto bytes = values.size() * sizeof(T);
  detail::Require(bytes <= kMaxAttachment && out.size() <= kMaxAttachment - bytes,
                  "response attachment exceeds limit", Error::Code::capacity_exceeded);
  slice->set_byte_offset(out.size()); slice->set_byte_length(bytes);
  slice->set_element_count(values.size()); slice->set_element_type(Element<T>());
  if (bytes) out.append(values.data(), bytes);
}
template <typename T>
std::vector<T> Unpack(const wire::PayloadSlice& slice, const butil::IOBuf& in) {
  detail::Require(in.size() <= kMaxAttachment && slice.element_type() == Element<T>() &&
    slice.element_count() <= kMaxAttachment / sizeof(T) &&
    slice.byte_length() == slice.element_count() * sizeof(T) &&
    slice.byte_offset() <= in.size() && slice.byte_length() <= in.size() - slice.byte_offset(),
    "invalid attachment slice");
  std::vector<T> values(slice.element_count());
  if (!values.empty()) in.copy_to(values.data(), slice.byte_length(), slice.byte_offset());
  return values;
}
inline void Encode(const RequestId& x, wire::RequestIdentity* y) {
  y->set_run(x.run); y->set_learner(x.learner); y->set_sequence(x.sequence);
  y->set_phase(static_cast<wire::Phase>(static_cast<int>(x.phase) + 1));
}
inline RequestId Decode(const wire::RequestIdentity& x) {
  detail::Require(x.phase() >= wire::LOOKUP && x.phase() <= wire::MAINTENANCE,
                  "invalid request phase");
  return {x.run(), x.learner(), x.sequence(), static_cast<Phase>(x.phase() - 1)};
}
inline void Encode(const OptimizerConfig& x, wire::OptimizerConfig* y) {
  y->set_learning_rate(x.learning_rate); y->set_beta1(x.beta1); y->set_beta2(x.beta2);
  y->set_epsilon(x.epsilon); y->set_initial_beta1_power(x.initial_beta1_power);
  y->set_initial_beta2_power(x.initial_beta2_power);
}
inline OptimizerConfig Decode(const wire::OptimizerConfig& x) {
  return {x.learning_rate(),x.beta1(),x.beta2(),x.epsilon(),
          x.initial_beta1_power(),x.initial_beta2_power()};
}
inline void Encode(const TableDefinition& x, wire::TableDefinition* y) {
  y->set_table_id(x.id); y->set_dimension(x.dimension); y->set_shard_count(x.shard_count);
  y->set_partition_epoch(x.partition_epoch); Encode(x.optimizer,y->mutable_optimizer());
  y->set_initialization_min(x.initialization_min); y->set_initialization_max(x.initialization_max);
  y->set_initialization_seed(x.initialization_seed);
}
inline TableDefinition Decode(const wire::TableDefinition& x) {
  return {x.table_id(),x.dimension(),x.shard_count(),x.partition_epoch(),Decode(x.optimizer()),
          x.initialization_min(),x.initialization_max(),x.initialization_seed()};
}
inline void Encode(const LocatedRow& x, wire::LocatedRow* y) {
  y->set_table_id(x.row.table); y->set_key(x.row.key); y->set_home_shard(x.home);
  y->set_placement(static_cast<wire::Placement>(static_cast<int>(x.placement) + 1));
  y->set_agent_id(x.agent); y->set_generation(x.generation); y->set_slot(x.slot);
}
inline LocatedRow Decode(const wire::LocatedRow& x) {
  detail::Require(x.placement() >= wire::CPU && x.placement() <= wire::DRAINING,
                  "invalid placement");
  return {{x.table_id(),x.key()},x.home_shard(),static_cast<Placement>(x.placement()-1),
          x.agent_id(),x.generation(),x.slot()};
}
inline void Encode(const TensorBlock& x, wire::TensorBlock* y, butil::IOBuf& attachment) {
  y->set_feature_id(x.feature);
  for(auto d:x.shape) y->add_shape(d);
  Pack(x.values,y->mutable_values(),attachment);
}
inline TensorBlock Decode(const wire::TensorBlock& x, const butil::IOBuf& attachment) {
  return {x.feature_id(),{x.shape().begin(),x.shape().end()},Unpack<float>(x.values(),attachment)};
}
inline void Encode(const FeatureInput& x, wire::FeatureInput* y, butil::IOBuf& attachment) {
  y->set_table_id(x.table); y->set_feature_id(x.feature);
  y->set_pooling(static_cast<wire::Pooling>(static_cast<int>(x.pooling) + 1));
  Pack(x.keys,y->mutable_keys(),attachment);
  Pack(x.offsets,y->mutable_offsets(),attachment);
  Pack(x.valid_lengths,y->mutable_original_valid_lengths(),attachment);
}
inline FeatureInput Decode(const wire::FeatureInput& x, const butil::IOBuf& attachment) {
  detail::Require(x.pooling() >= wire::SUM && x.pooling() <= wire::SEQUENCE,
                  "invalid pooling");
  return {x.table_id(),x.feature_id(),static_cast<Pooling>(x.pooling()-1),
    Unpack<Key>(x.keys(),attachment),Unpack<std::uint64_t>(x.offsets(),attachment),
    Unpack<std::uint64_t>(x.original_valid_lengths(),attachment)};
}
inline void Encode(const RowState& x, wire::RowState* y, butil::IOBuf& attachment, ShardId shard) {
  Encode(LocatedRow{x.row,shard,Placement::cpu,0,0,0},y->mutable_row());
  Pack(x.weights,y->mutable_weights(),attachment);
  Pack(x.first_moment,y->mutable_first_moment(),attachment);
  Pack(x.second_moment,y->mutable_second_moment(),attachment);
  y->set_row_updates(x.row_updates);
}
inline RowState Decode(const wire::RowState& x, const butil::IOBuf& attachment) {
  return {{x.row().table_id(),x.row().key()},Unpack<float>(x.weights(),attachment),
    Unpack<float>(x.first_moment(),attachment),Unpack<float>(x.second_moment(),attachment),x.row_updates()};
}
inline void SetError(wire::Status* status, const Error& e) {
  status->set_code(static_cast<wire::Status_Code>(static_cast<int>(e.code)+1));
  status->set_message(e.message);
}
inline std::optional<Error> GetError(const wire::Status& status) {
  if(status.code()==wire::Status::OK) return std::nullopt;
  if(status.code()<wire::Status::INVALID_ARGUMENT || status.code()>wire::Status::RUN_FAILED)
    return Error{Error::Code::run_failed,"unknown response status"};
  return Error{static_cast<Error::Code>(status.code()-1),status.message()};
}
}  // namespace eps::transport
