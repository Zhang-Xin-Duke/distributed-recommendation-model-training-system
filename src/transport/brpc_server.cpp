#include "transport/brpc_server.hpp"

#include <chrono>
#include <functional>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <bthread/bthread.h>
#include "transport/rpc_dispatch.hpp"

namespace eps::transport {
void BrpcServer::Stats(google::protobuf::RpcController* c,const wire::StatsRequest* q,
                       wire::StatsReply* r,google::protobuf::Closure* done){
  Dispatch<RuntimeStats>(c,q,r,done,[&](const auto&){return service_.Stats();},
    [](const RuntimeStats& s,auto* out,auto&){
#define EPS_STAT(name) out->set_##name(s.name);
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
      EPS_STAT(dependency_prepare_nanoseconds) EPS_STAT(dependency_rebuilds) EPS_STAT(dependency_cache_hits)
      EPS_STAT(queued_observations) EPS_STAT(completed_observations) EPS_STAT(rejected_observations) EPS_STAT(dependency_budget_waits)
#undef EPS_STAT
    },{},CompletionExecutor::Observations());
}
using namespace eps::transport;

void BrpcServer::RestoredFrom(google::protobuf::RpcController* c,const wire::StatsRequest* q,
                            wire::RestoredSnapshotReply* r,google::protobuf::Closure* done) {
  Dispatch<RestoredSnapshot>(c,q,r,done,[&](const auto&){return service_.RestoredFrom();},
    [](RestoredSnapshot metadata,auto* out,auto&){
      out->set_loaded(metadata.loaded);out->set_pristine(metadata.pristine);
      out->set_run(metadata.run);out->set_source_run(metadata.source_run);
      out->set_file_bytes(metadata.bytes);out->set_checksum(metadata.checksum);
      out->set_shard(metadata.shard);out->set_shard_count(metadata.shard_count);
      out->set_partition_epoch(metadata.partition_epoch);
    });
}

void BrpcServer::ReadCheckpoint(google::protobuf::RpcController* c,const wire::SnapshotReadRequest* q,
                              wire::SnapshotChunkReply* r,google::protobuf::Closure* done) {
  Dispatch<SnapshotChunk>(c,q,r,done,[&](const auto&){
    return service_.ReadCheckpoint(Decode(q->identity()),q->offset(),q->max_bytes());
  },[](SnapshotChunk chunk,auto* out,auto&){
    out->set_offset(chunk.offset);out->set_total_bytes(chunk.total_bytes);
    out->set_data(std::move(chunk.data));out->set_eof(chunk.eof);
  });
}
void BrpcServer::ResumeCheckpoint(google::protobuf::RpcController* c,const wire::CheckpointIdentity* q,
                                wire::Ack* r,google::protobuf::Closure* done) {
  Dispatch<Ack>(c,q,r,done,[&](const auto&){return service_.ResumeCheckpoint(Decode(*q));},AckEncode);
}

void BrpcServer::CreateTable(google::protobuf::RpcController* c,const wire::CreateTableRequest* q,
                           wire::Ack* r,google::protobuf::Closure* done) {
  Dispatch<Ack>(c,q,r,done,[&](const auto&) {return service_.CreateTable(Decode(q->table()));},AckEncode);
}
void BrpcServer::Locations(google::protobuf::RpcController* c,const wire::RowsRequest* q,
                          wire::LocationsReply* r,google::protobuf::Closure* done){
  Dispatch<std::vector<LocatedRow>>(c,q,r,done,[&](const auto&){
    std::vector<RowKey> rows;for(const auto& row:q->rows())rows.push_back({row.table_id(),row.key()});
    return service_.Locations(std::move(rows));
  },[](auto rows,auto* out,auto&){for(const auto& row:rows)Encode(row,out->add_rows());});
}
void BrpcServer::MoveRows(google::protobuf::RpcController* c,const wire::MoveRequest* q,
                         wire::LocationsReply* r,google::protobuf::Closure* done){
  Dispatch<std::vector<LocatedRow>>(c,q,r,done,[&](const auto&){
    detail::Require(q->destination()==wire::CPU || q->destination()==wire::GPU,"invalid move destination");
    MoveRowsRequest request{q->run(),{},q->agent_id(),static_cast<Placement>(q->destination()-1),
      {q->network_bytes(),q->staging_bytes(),q->max_inflight_batches(),q->reserve_slots()}};
    for(const auto& row:q->rows())request.rows.push_back({row.table_id(),row.key()});
    return service_.MoveRows(std::move(request));
  },[](auto rows,auto* out,auto&){for(const auto& row:rows)Encode(row,out->add_rows());});
}
void BrpcServer::Lookup(google::protobuf::RpcController* c,const wire::LookupRequest* q,
                      wire::LookupReply* r,google::protobuf::Closure* done) {
  Dispatch<LookupReply>(c,q,r,done,[&](const auto& a) {
    detail::Require(q->features_size()<=4096,"too many features");
    LookupRequest request{Decode(q->identity()),q->partition_epoch(),q->read_only(),{}};
    for(const auto& f:q->features())request.features.push_back(Decode(f,a));
    return service_.Lookup(std::move(request));
  },[](LookupReply result,auto* out,auto& a) {
    out->set_ticket(result.ticket);
    for(const auto& f:result.cpu_contributions)Encode(f,out->add_cpu_contributions(),a);
    for(const auto& work:result.agent_work)Encode(work,out->add_agent_work(),a);
  });
}
void BrpcServer::Push(google::protobuf::RpcController* c,const wire::PushRequest* q,
                    wire::Ack* r,google::protobuf::Closure* done) {
  Dispatch<Ack>(c,q,r,done,[&](const auto& a) {
    detail::Require(q->feature_gradients_size()<=4096,"too many gradient features");
    PushRequest request{Decode(q->identity()),q->ticket(),{}};
    for(const auto& g:q->feature_gradients())request.feature_gradients.push_back(Decode(g,a));
    return service_.Push(std::move(request));
  },AckEncode);
}
void BrpcServer::Release(google::protobuf::RpcController* c,const wire::ReleaseRequest* q,
                       wire::Ack* r,google::protobuf::Closure* done) {
  Dispatch<Ack>(c,q,r,done,[&](const auto&) {
    return service_.Release({Decode(q->identity()),q->ticket()});
  },AckEncode);
}
void BrpcServer::ReadState(google::protobuf::RpcController* c,const wire::ReadStateRequest* q,
                         wire::ReadStateReply* r,google::protobuf::Closure* done) {
  Dispatch<std::vector<RowState>>(c,q,r,done,[&](const auto&) {
    std::vector<LocatedRow> rows;for(const auto& x:q->rows())rows.push_back(Decode(x));
    return service_.ReadState(std::move(rows));
  },[shard=shard_](auto result,auto* out,auto& a) {
    for(const auto& row:result)Encode(row,out->add_states(),a,shard);
  });
}
void BrpcServer::CompleteMigration(google::protobuf::RpcController* c,const wire::MigrationCompletion* q,
                                 wire::Ack* r,google::protobuf::Closure* done) {
  Dispatch<Ack>(c,q,r,done,[&](const auto&) {
    MigrationCompletion completion{q->plan_id(),{},q->completed()};
    for(const auto& row:q->rows())completion.rows.push_back(Decode(row));
    return service_.CompleteMigration(std::move(completion));
  },AckEncode);
}
void BrpcServer::DrainAndFlush(google::protobuf::RpcController* c,const wire::DrainRequest* q,
                             wire::SnapshotReply* r,google::protobuf::Closure* done) {
  Dispatch<SnapshotMetadata>(c,q,r,done,[&](const auto&) {return service_.DrainAndFlush(q->snapshot_id());},
    [](SnapshotMetadata result,auto* out,auto&) {
      out->set_snapshot_id(result.snapshot_id);out->set_shard_id(result.shard);
      out->set_partition_epoch(result.partition_epoch);
      for(const auto& file:result.state_files)out->add_state_files(file);
      for(const auto& context:result.optimizer_contexts) {
        auto* x=out->add_optimizer_contexts();x->set_table_id(context.table);
        x->set_optimizer_sequence(context.optimizer_sequence);x->set_learning_rate(context.learning_rate);
        x->set_beta1_power(context.beta1_power);x->set_beta2_power(context.beta2_power);
      }
    });
}
}  // namespace eps::transport
