#include "eps/brpc_client.hpp"

#include <functional>
#include <brpc/channel.h>
#include <brpc/controller.h>
#include "transport/rpc_call.hpp"

namespace eps {
namespace t = transport;
namespace wire = ::eps::v1;
struct BrpcParameterService::Impl {
  brpc::Channel channel;
  int timeout_ms;
  explicit Impl(const std::string& endpoint, int timeout) : timeout_ms(timeout) {
    brpc::ChannelOptions options; options.protocol="baidu_std";
    options.timeout_ms=timeout; options.max_retry=2;
    detail::Require(channel.Init(endpoint.c_str(),&options)==0,"cannot initialize BRPC channel");
  }
};
using transport::Call;
using transport::AckDecoder;

BrpcParameterService::BrpcParameterService(std::string endpoint,int timeout)
    :impl_(std::make_shared<Impl>(endpoint,timeout)) {}
BrpcParameterService::~BrpcParameterService()=default;
std::future<Result<RuntimeStats>> BrpcParameterService::Stats(){
  return Call<wire::StatsRequest,wire::StatsReply,RuntimeStats>(impl_,&wire::EmbeddingService_Stub::Stats,
    [](auto&,auto&){},[](const auto& q,const auto&){RuntimeStats s;
#define EPS_STAT(name) s.name=q.name();
      EPS_STAT(lookups) EPS_STAT(updates) EPS_STAT(duplicate_requests) EPS_STAT(rows)
      EPS_STAT(state_bytes) EPS_STAT(live_tickets) EPS_STAT(ticket_payload_bytes)
      EPS_STAT(queued_payload_bytes) EPS_STAT(cached_rows) EPS_STAT(migration_bytes)
      EPS_STAT(executed_operations) EPS_STAT(queue_wait_nanoseconds) EPS_STAT(executor_busy_nanoseconds)
      EPS_STAT(rejected_operations) EPS_STAT(peak_queued_operations)
      EPS_STAT(expanded_gradient_bytes) EPS_STAT(transferred_gradient_bytes) EPS_STAT(compact_update_commands)
      EPS_STAT(policy_windows) EPS_STAT(policy_tracked_rows) EPS_STAT(policy_admitted_rows)
      EPS_STAT(policy_evicted_rows) EPS_STAT(policy_pinned_skips) EPS_STAT(policy_budget_skips)
      EPS_STAT(policy_migration_batches) EPS_STAT(policy_migration_bytes) EPS_STAT(policy_failures)
      EPS_STAT(lookup_prepare_nanoseconds) EPS_STAT(gradient_prepare_nanoseconds) EPS_STAT(gpu_packet_prepare_nanoseconds) EPS_STAT(update_dispatch_nanoseconds) EPS_STAT(cpu_optimizer_nanoseconds) EPS_STAT(agent_wait_nanoseconds) EPS_STAT(migration_nanoseconds) EPS_STAT(cache_maintenance_nanoseconds)
      EPS_STAT(active_operations) EPS_STAT(waiting_operations) EPS_STAT(active_payload_bytes) EPS_STAT(dependency_payload_bytes)
#undef EPS_STAT
      return s;});
}
std::future<Result<std::vector<LocatedRow>>> BrpcParameterService::Locations(std::vector<RowKey> rows){
  return Call<wire::RowsRequest,wire::LocationsReply,std::vector<LocatedRow>>(impl_,&wire::EmbeddingService_Stub::Locations,
    [rows=std::move(rows)](auto& q,auto&){for(auto row:rows){auto* r=q.add_rows();r->set_table_id(row.table);r->set_key(row.key);}},
    [](const auto& q,const auto&){std::vector<LocatedRow> out;for(const auto& row:q.rows())out.push_back(t::Decode(row));return out;});
}
std::future<Result<std::vector<LocatedRow>>> BrpcParameterService::MoveRows(MoveRowsRequest r){
  return Call<wire::MoveRequest,wire::LocationsReply,std::vector<LocatedRow>>(impl_,&wire::EmbeddingService_Stub::MoveRows,
    [r=std::move(r)](auto& q,auto&){
      q.set_run(r.run);q.set_agent_id(r.agent);q.set_destination(static_cast<wire::Placement>(static_cast<int>(r.destination)+1));
      q.set_network_bytes(r.budget.network_bytes);q.set_staging_bytes(r.budget.staging_bytes);
      q.set_max_inflight_batches(r.budget.max_inflight_batches);q.set_reserve_slots(r.budget.reserve_slots);
      for(auto row:r.rows){auto* x=q.add_rows();x->set_table_id(row.table);x->set_key(row.key);}
    },[](const auto& q,const auto&){std::vector<LocatedRow> out;for(const auto& row:q.rows())out.push_back(t::Decode(row));return out;});
}
std::future<Result<Ack>> BrpcParameterService::CreateTable(TableDefinition d) {
  return Call<wire::CreateTableRequest,wire::Ack,Ack>(impl_,&wire::EmbeddingService_Stub::CreateTable,
    [d](auto& q,auto&) {t::Encode(d,q.mutable_table());},AckDecoder);
}
std::future<Result<LookupReply>> BrpcParameterService::Lookup(LookupRequest r) {
  return Call<wire::LookupRequest,wire::LookupReply,LookupReply>(impl_,&wire::EmbeddingService_Stub::Lookup,
    [r=std::move(r)](auto& q,auto& a) {
      t::Encode(r.id,q.mutable_identity());q.set_partition_epoch(r.partition_epoch);q.set_read_only(r.read_only);
      for(const auto& f:r.features)t::Encode(f,q.add_features(),a);
    },[](const auto& q,const auto& a) {
      LookupReply out;out.ticket=q.ticket();
      for(const auto& f:q.cpu_contributions())out.cpu_contributions.push_back(t::Decode(f,a));
      for(const auto& work:q.agent_work())out.agent_work.push_back(t::Decode(work,a));
      return out;
    });
}
std::future<Result<Ack>> BrpcParameterService::Push(PushRequest r) {
  return Call<wire::PushRequest,wire::Ack,Ack>(impl_,&wire::EmbeddingService_Stub::Push,
    [r=std::move(r)](auto& q,auto& a) {
      t::Encode(r.id,q.mutable_identity());q.set_ticket(r.ticket);
      for(const auto& g:r.feature_gradients)t::Encode(g,q.add_feature_gradients(),a);
    },AckDecoder);
}
std::future<Result<Ack>> BrpcParameterService::Release(ReleaseRequest r) {
  return Call<wire::ReleaseRequest,wire::Ack,Ack>(impl_,&wire::EmbeddingService_Stub::Release,
    [r=std::move(r)](auto& q,auto&) {t::Encode(r.id,q.mutable_identity());q.set_ticket(r.ticket);},AckDecoder);
}
std::future<Result<std::vector<RowState>>> BrpcParameterService::ReadState(std::vector<LocatedRow> rows) {
  return Call<wire::ReadStateRequest,wire::ReadStateReply,std::vector<RowState>>(impl_,&wire::EmbeddingService_Stub::ReadState,
    [rows=std::move(rows)](auto& q,auto&) {for(const auto& row:rows)t::Encode(row,q.add_rows());},
    [](const auto& q,const auto& a) {std::vector<RowState> out;for(const auto& r:q.states())out.push_back(t::Decode(r,a));return out;});
}
std::future<Result<Ack>> BrpcParameterService::CompleteMigration(MigrationCompletion r) {
  return Call<wire::MigrationCompletion,wire::Ack,Ack>(impl_,&wire::EmbeddingService_Stub::CompleteMigration,
    [r=std::move(r)](auto& q,auto&) {
      q.set_plan_id(r.plan_id);q.set_completed(r.completed);for(const auto& row:r.rows)t::Encode(row,q.add_rows());
    },AckDecoder);
}
std::future<Result<SnapshotMetadata>> BrpcParameterService::DrainAndFlush(std::string id) {
  return Call<wire::DrainRequest,wire::SnapshotReply,SnapshotMetadata>(impl_,&wire::EmbeddingService_Stub::DrainAndFlush,
    [id=std::move(id)](auto& q,auto&) {q.set_snapshot_id(id);},
    [](const auto& q,const auto&) {
      SnapshotMetadata out{q.snapshot_id(),q.shard_id(),q.partition_epoch(),
        {q.state_files().begin(),q.state_files().end()},{}};
      for(const auto& c:q.optimizer_contexts())out.optimizer_contexts.push_back({{},q.shard_id(),c.table_id(),
        c.optimizer_sequence(),c.learning_rate(),c.beta1_power(),c.beta2_power()});
      return out;
    });
}
}  // namespace eps
