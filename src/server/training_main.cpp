#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <nlohmann/json.hpp>
#include "transport/brpc_training_server.hpp"

int main(int argc, char** argv) {
  try {
    if (argc != 3 || std::string(argv[1]) != "--config")
      throw std::runtime_error("usage: eps_training_coordinator --config FILE");
    std::ifstream file(argv[2]);
    if (!file) throw std::runtime_error("cannot open training config");
    nlohmann::json j; file >> j;
    if (j.at("schema_version") != 1) throw std::runtime_error("unsupported training config version");
    const auto policy = j.at("consistency").get<std::string>();
    if (policy != "sequential" && policy != "eventual" && policy != "bounded_delay")
      throw std::runtime_error("invalid training consistency policy");
    eps::TrainingProgressOptions options{j.at("run"), j.value("epoch", std::uint64_t(1)),
      j.at("workers"), policy == "sequential" ? eps::TrainingConsistency::sequential :
      policy == "eventual" ? eps::TrainingConsistency::eventual : eps::TrainingConsistency::bounded_delay,
      j.value("max_delay", std::uint64_t(0))};
    eps::transport::BrpcTrainingServer rpc(options, j.value("max_pending_rpcs", std::size_t(4096)));
    if(j.contains("replica_next_round")) {
      if(j.contains("restore"))throw std::runtime_error("choose replica recovery or checkpoint restore");
      auto result=rpc.RecoverSynchronousRound(j.at("replica_next_round"));
      if(const auto* error=std::get_if<eps::Error>(&result))throw std::runtime_error(error->message);
    }
    if (j.contains("restore")) {
      const auto& s = j.at("restore");
      const auto consistency = s.at("consistency").get<unsigned>();
      if (consistency > 2) throw std::runtime_error("invalid restored training policy");
      auto result = rpc.Restore({s.at("source_run"), s.at("epoch"),
        static_cast<eps::TrainingConsistency>(consistency), s.at("max_delay"),
        s.at("completed").get<std::vector<std::uint64_t>>()});
      if (const auto* error = std::get_if<eps::Error>(&result)) throw std::runtime_error(error->message);
    }
    brpc::Server server;
    if (server.AddService(&rpc, brpc::SERVER_DOESNT_OWN_SERVICE) != 0)
      throw std::runtime_error("cannot register training service");
    brpc::ServerOptions server_options;
    // Keep the global handler limit above the bounded waiting-RPC capacity so
    // Complete/Abort/Stats can still run while Begin calls are waiting.
    server_options.max_concurrency = 0;
    const auto address = j.value("listen", std::string("127.0.0.1:0"));
    if (server.Start(address.c_str(), &server_options) != 0)
      throw std::runtime_error("cannot start training coordinator");
    std::cout << "READY " << butil::endpoint2str(server.listen_address()).c_str() << std::endl;
    while (!brpc::IsAskedToQuit()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    rpc.Shutdown();
    server.Stop(0); server.Join();
  } catch (const std::exception& e) {
    std::cerr << "eps_training_coordinator: " << e.what() << '\n'; return 1;
  }
}
