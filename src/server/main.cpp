#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <nlohmann/json.hpp>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include "eps/runtime.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/cost_planner.hpp"
#include "transport/brpc_server.hpp"
#include "transport/brpc_synchronous_server.hpp"
#include "training/checkpoint_json.hpp"

using namespace eps;
namespace {
template <typename T>T Value(std::future<Result<T>> future) {
  auto r=future.get();if(auto error=std::get_if<Error>(&r))throw std::runtime_error(error->message);
  return std::get<T>(std::move(r));
}
}
int main(int argc,char** argv) {
  try {
    std::string config,restore,run,listen;
    for(int i=1;i<argc;++i) {
      const std::string arg=argv[i];
      if(i+1>=argc)throw std::runtime_error("expected --config, --restore, --run or --listen followed by a value");
      if(arg=="--config")config=argv[++i];else if(arg=="--restore")restore=argv[++i];
      else if(arg=="--run")run=argv[++i];else if(arg=="--listen")listen=argv[++i];
      else throw std::runtime_error("unknown command-line option");
    }
    if(config.empty())throw std::runtime_error("usage: eps_server --config FILE [--restore FILE] [--run ID] [--listen HOST:PORT]");
    std::ifstream stream(config);if(!stream)throw std::runtime_error("cannot open config");
    nlohmann::json j;stream>>j;
    if(j.value("schema_version",0)!=1)throw std::runtime_error("unsupported config version");
    const bool caching=j.contains("cache") && j["cache"].value("enabled",false);
    if(caching && (!j.contains("agents") || j["agents"].empty()))
      throw std::runtime_error("cache.enabled requires an agents list");
    const auto admission=j.value("cache",nlohmann::json::object()).value("admission",std::string("manual"));
    if(admission!="manual" && admission!="frequency" && admission!="reuse_cost")
      throw std::runtime_error("cache.admission must be manual, frequency or reuse_cost");
    if(admission!="manual" && !caching)
      throw std::runtime_error("automatic cache admission requires cache.enabled");
    const auto policy=j.value("access",nlohmann::json::object()).value("policy",std::string("cpu_only"));
    if(policy!="cpu_only" && policy!="transfer_unique" && policy!="cost")
      throw std::runtime_error("access.policy must be cpu_only, transfer_unique or cost");
    if(policy!="cpu_only" && !caching)throw std::runtime_error("GPU access routes require cache.enabled and an agent");
    RuntimeOptions options;
    const bool synchronous = j.contains("synchronous_training");
    if (synchronous) {
      if ((!restore.empty()) != j.at("synchronous_training").contains("restore"))
        throw std::runtime_error("parameter and synchronous group restore must be supplied together");
      options.exclusive_update_learner = 1023;
      options.start_paused = !restore.empty();
    }
    options.run_id=run.empty()?j.at("run_id").get<std::string>():run;
    options.shard_id=j.at("partition").at("shard_id");
    options.shard_count=j.at("partition").at("shard_count");
    options.partition_epoch=j.at("partition").at("epoch");
    options.snapshot_directory=j.value("snapshot_directory",std::string("snapshots"));
    if(j.contains("cache")){
      options.concurrent_agent_updates=j["cache"].value("concurrent_agent_updates",true);
      options.compact_gpu_gradients=j["cache"].value("compact_gradients",true);
    }
    if(j.contains("limits")) {
      const auto& l=j["limits"];
      options.max_request_bytes=l.value("max_request_bytes",options.max_request_bytes);
      options.max_live_tickets=l.value("max_live_tickets",options.max_live_tickets);
      options.state_bytes=l.value("cpu_state_bytes",options.state_bytes);
      options.ticket_payload_bytes=l.value("ticket_payload_bytes",options.ticket_payload_bytes);
      options.queue_payload_bytes=l.value("queue_payload_bytes",options.queue_payload_bytes);
      options.max_queued_operations=l.value("max_queued_operations",options.max_queued_operations);
      options.max_active_operations=l.value("max_active_operations",options.max_active_operations);
      options.completion_waiters=l.value("completion_waiters",options.completion_waiters);
      options.max_observation_operations=l.value("max_observation_operations",options.max_observation_operations);
      options.dependency_payload_bytes=l.value("dependency_payload_bytes",options.dependency_payload_bytes);
      options.cache_dependencies=l.value("cache_dependencies",options.cache_dependencies);
    }
    CpuParameterService service(options);
    if(!restore.empty()) {
      Value(service.Restore(restore));
      if (synchronous) for (const auto& t:j.at("tables"))
        Value(service.CreateTable(training::DecodeTable(t,options.shard_count,options.partition_epoch)));
    }
    else for(const auto& t:j.at("tables")) {
      if(t.value("dtype",std::string("float32"))!="float32" ||
         t.at("optimizer").value("kind",std::string())!="sparse_lazy_adam")
        throw std::runtime_error("only FP32 sparse_lazy_adam tables are supported");
      const auto& o=t.at("optimizer");const auto& init=t.at("initialization");
      TableDefinition table{t.at("id"),t.at("dimension"),options.shard_count,options.partition_epoch,
        {o.at("learning_rate"),o.at("beta1"),o.at("beta2"),o.at("epsilon"),
         o.at("initial_beta1_power"),o.at("initial_beta2_power")},
        init.at("min"),init.at("max"),init.at("seed")};
      if(o.contains("weight_bounds")) {
        const auto& b=o.at("weight_bounds");
        if(!b.is_array() || b.size()!=2)throw std::runtime_error("weight_bounds requires [minimum, maximum]");
        table.optimizer.weight_bounds=WeightBounds{b.at(0),b.at(1)};
      }
      Value(service.CreateTable(table));
    }
    if(caching)for(const auto& agent:j.at("agents")){
      const AgentId id=agent.at("id");
      Value(service.AttachAgent(id,std::make_shared<BrpcCacheBackend>(
        agent.at("endpoint"),options.run_id,options.shard_id,id)));
    }
    if(policy!="cpu_only"){
      const auto& access=j.at("access");
      const AgentId agent=access.value("agent_id",AgentId(0));
      std::shared_ptr<CostBasedPlanner> planner;
      if(policy=="cost"){
        PlannerOptions po;po.shard=options.shard_id;po.shard_count=options.shard_count;po.agent=agent;
        po.switch_margin=access.value("switch_margin",po.switch_margin);
        po.minimum_observations=access.value("minimum_observations",po.minimum_observations);
        planner=std::make_shared<CostBasedPlanner>(po);
        // Optional paired measurements from the actual workload. With no
        // observations, pooled features retain the conservative CPU baseline.
        for(const auto& sample:access.value("observations",nlohmann::json::array()))
          planner->Observe({sample.at("table"),sample.at("feature"),Pooling::sum,
            sample.at("occurrences"),sample.at("unique_rows"),sample.at("bags"),
            sample.at("cpu_microseconds"),sample.at("gpu_microseconds"),sample.value("queue_microseconds",0.0)});
      }
      Value(service.ConfigureAccess(agent,policy=="transfer_unique"?AccessRoute::transfer_unique:
        AccessRoute::cpu_pool,planner));
    }
    if(admission!="manual") {
      const auto& cache=j.at("cache");CachePolicyOptions co;
      co.mode=admission=="frequency"?CacheAdmissionMode::frequency:CacheAdmissionMode::reuse_cost;
      if(cache.contains("policy_agents"))co.agents=cache.at("policy_agents").get<std::vector<AgentId>>();
      else for(const auto& agent:j.at("agents"))co.agents.push_back(agent.at("id"));
      co.window_requests=cache.value("decision_window_requests",co.window_requests);
      co.max_tracked_rows=cache.value("max_tracked_rows",co.max_tracked_rows);
      co.window_decay=cache.value("window_decay",co.window_decay);
      co.minimum_accesses=cache.value("minimum_accesses",co.minimum_accesses);
      co.minimum_residence_windows=cache.value("minimum_residence_windows",co.minimum_residence_windows);
      co.replacement_margin=cache.value("replacement_margin",co.replacement_margin);
      co.max_rows_per_batch=cache.value("max_rows_per_batch",co.max_rows_per_batch);
      co.saved_microseconds_per_access=cache.value("saved_microseconds_per_access",co.saved_microseconds_per_access);
      co.load_microseconds_per_byte=cache.value("load_microseconds_per_byte",co.load_microseconds_per_byte);
      co.writeback_microseconds_per_byte=cache.value("writeback_microseconds_per_byte",co.writeback_microseconds_per_byte);
      co.cost_average_alpha=cache.value("cost_average_alpha",co.cost_average_alpha);
      co.learn_migration_costs=cache.value("learn_migration_costs",co.learn_migration_costs);
      co.budget.network_bytes=cache.value("network_bytes_per_window",co.budget.network_bytes);
      co.budget.staging_bytes=cache.value("staging_bytes_limit",co.budget.staging_bytes);
      co.budget.max_inflight_batches=cache.value("max_inflight_batches",co.budget.max_inflight_batches);
      co.budget.reserve_slots=cache.value("reserve_slots",co.budget.reserve_slots);
      Value(service.ConfigureCachePolicy(std::move(co)));
    }
    transport::BrpcServer rpc(service,options.shard_id);
    std::unique_ptr<transport::BrpcSynchronousServer> synchronous_rpc;
    if (synchronous) {
      const auto& config = j.at("synchronous_training");
      SynchronousGradientOptions group;
      group.run = options.run_id; group.epoch = options.partition_epoch;
      group.shard = options.shard_id; group.shards = options.shard_count;
      group.workers = config.at("workers");
      group.gradient_bytes = config.value("gradient_bytes", group.gradient_bytes);
      group.round_timeout = std::chrono::milliseconds(config.value("round_timeout_ms", 30000));
      const auto reduction = config.value("reduction", std::string("sample_mean"));
      if (reduction != "sample_mean" && reduction != "worker_mean" && reduction != "sum")
        throw std::runtime_error("invalid synchronous gradient reduction");
      group.reduction = reduction == "sample_mean" ? GradientReduction::sample_mean :
        reduction == "worker_mean" ? GradientReduction::worker_mean : GradientReduction::sum;
      for (const auto& table : j.at("tables"))
        group.tables.push_back({table.at("id"), table.at("dimension"), {}, {}});
      synchronous_rpc = std::make_unique<transport::BrpcSynchronousServer>(service, std::move(group));
      if (config.contains("restore")) {
        const auto& state=config.at("restore");
        const auto checkpoint=training::DecodeGroup(state.at("group"));
        const auto& fingerprint=state.at("parameter_snapshot");
        RestoredSnapshot expected;
        expected.source_run=fingerprint.at("source_run");expected.bytes=fingerprint.at("bytes");
        expected.checksum=fingerprint.at("checksum");expected.shard=fingerprint.at("shard");
        expected.shard_count=fingerprint.at("shards");expected.partition_epoch=fingerprint.at("epoch");
        auto result=synchronous_rpc->Restore(checkpoint,expected);
        if (auto* error=std::get_if<Error>(&result)) throw std::runtime_error(error->message);
      }
    }
    brpc::Server server;
    if(server.AddService(&rpc,brpc::SERVER_DOESNT_OWN_SERVICE)!=0)
      throw std::runtime_error("cannot register parameter service");
    if (synchronous_rpc && server.AddService(synchronous_rpc.get(),brpc::SERVER_DOESNT_OWN_SERVICE)!=0)
      throw std::runtime_error("cannot register synchronous gradient service");
    brpc::ServerOptions server_options;server_options.max_concurrency=synchronous ? 0 : 1024;
    const auto address=listen.empty()?j.at("listen").get<std::string>():listen;
    if(server.Start(address.c_str(),&server_options)!=0)throw std::runtime_error("cannot start parameter server");
    std::cout<<"READY "<<butil::endpoint2str(server.listen_address()).c_str()<<std::endl;
    if (!synchronous_rpc) server.RunUntilAskedToQuit();
    else {
      while (!brpc::IsAskedToQuit()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
      synchronous_rpc->Shutdown(); server.Stop(0); server.Join();
    }
    return 0;
  } catch(const std::exception& e) {std::cerr<<"eps_server: "<<e.what()<<'\n';return 1;}
}
