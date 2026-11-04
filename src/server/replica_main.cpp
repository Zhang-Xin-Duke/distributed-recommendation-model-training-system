#include <fstream>
#include <iostream>
#include <thread>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include "transport/brpc_replica_server.hpp"
#include "replication/configuration.hpp"
int main(int argc,char** argv) {
  try {
    if((argc!=3 && argc!=7) || std::string(argv[1])!="--config")throw std::runtime_error("usage: eps_replica --config FILE [--test-publish-gate FILE --test-publish-round N]");
    std::ifstream file(argv[2]);if(!file)throw std::runtime_error("cannot open replica config");nlohmann::json j;file>>j;
    auto options=eps::replication::Configuration(j);
    if(argc==7) {
      if(std::string(argv[3])!="--test-publish-gate" || std::string(argv[5])!="--test-publish-round")throw std::runtime_error("invalid fault gate options");
      const auto path=std::filesystem::path(argv[4]);const auto target=std::stoull(argv[6]);
      options.before_publish=[path,target](std::uint64_t round) {
        if(round!=target)return;
        {std::ofstream waiting(path.string()+".waiting");waiting<<round;}
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(25);
        while(!std::filesystem::exists(path)) {
          if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("test publication gate timed out");
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      };
    }
    eps::transport::BrpcReplicaServer rpc(options);brpc::Server server;
    if(server.AddService(&rpc,brpc::SERVER_DOESNT_OWN_SERVICE)!=0 || server.Start(j.value("listen",std::string("127.0.0.1:0")).c_str(),nullptr)!=0)
      throw std::runtime_error("cannot start replica server");
    std::cout<<"READY "<<butil::endpoint2str(server.listen_address()).c_str()<<std::endl;
    while(!brpc::IsAskedToQuit())std::this_thread::sleep_for(std::chrono::milliseconds(50));server.Stop(0);server.Join();
  }catch(const std::exception& e){std::cerr<<"eps_replica: "<<e.what()<<'\n';return 1;}
}
