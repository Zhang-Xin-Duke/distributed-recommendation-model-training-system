#include <fstream>
#include <iostream>
#include <string>
#include <nlohmann/json.hpp>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include "eps/runtime.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/cost_planner.hpp"
#include "transport/brpc_server.hpp"

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
    const auto policy=j.value("access",nlohmann::json::object()).value("policy",std::string("cpu_only"));
    if(policy!="cpu_only" && policy!="transfer_unique" && policy!="cost")
      throw std::runtime_error("access.policy must be cpu_only, transfer_unique or cost");
    if(policy!="cpu_only" && !caching)throw std::runtime_error("GPU access routes require cache.enabled and an agent");
    RuntimeOptions options;
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
    }
    CpuParameterService service(options);
    if(!restore.empty())Value(service.Restore(restore));
    else for(const auto& t:j.at("tables")) {
      if(t.value("dtype",std::string("float32"))!="float32" ||
         t.at("optimizer").value("kind",std::string())!="sparse_lazy_adam")
        throw std::runtime_error("only FP32 sparse_lazy_adam tables are supported");
      const auto& o=t.at("optimizer");const auto& init=t.at("initialization");
      Value(service.CreateTable({t.at("id"),t.at("dimension"),options.shard_count,options.partition_epoch,
        {o.at("learning_rate"),o.at("beta1"),o.at("beta2"),o.at("epsilon"),
         o.at("initial_beta1_power"),o.at("initial_beta2_power")},
        init.at("min"),init.at("max"),init.at("seed")}));
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
    transport::BrpcServer rpc(service,options.shard_id);brpc::Server server;
    if(server.AddService(&rpc,brpc::SERVER_DOESNT_OWN_SERVICE)!=0)
      throw std::runtime_error("cannot register parameter service");
    brpc::ServerOptions server_options;server_options.max_concurrency=1024;
    const auto address=listen.empty()?j.at("listen").get<std::string>():listen;
    if(server.Start(address.c_str(),&server_options)!=0)throw std::runtime_error("cannot start parameter server");
    std::cout<<"READY "<<butil::endpoint2str(server.listen_address()).c_str()<<std::endl;
    server.RunUntilAskedToQuit();return 0;
  } catch(const std::exception& e) {std::cerr<<"eps_server: "<<e.what()<<'\n';return 1;}
}
