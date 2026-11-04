#include <fstream>
#include <iostream>
#include "eps/brpc_replica.hpp"
#include "replication/configuration.hpp"
int main(int argc,char** argv) {
  try {
    if(argc!=5 || std::string(argv[1])!="--config" || std::string(argv[3])!="--action")throw std::runtime_error("usage: eps_replica_control --config FILE --action head|fence");
    std::ifstream file(argv[2]);if(!file)throw std::runtime_error("cannot open replica control config");nlohmann::json j;file>>j;
    const auto scope=eps::replication::Configuration(j);
    eps::BrpcReplicaLog replica(j.at("replica_endpoint"),scope);
    const auto action=std::string(argv[4]);
    auto result=(action=="head" || action=="record")?replica.Head().get():action=="fence"?
      replica.Fence(j.at("expected_term"),j.at("primary")).get():throw std::runtime_error("unknown replica action");
    if(auto* e=std::get_if<eps::Error>(&result))throw std::runtime_error(e->message);
    const auto& h=std::get<eps::ReplicaHead>(result);
    if(action=="record") {
      const auto round=j.at("round").get<std::uint64_t>();
      if(round<h.base_round || round>=h.next_round)throw std::runtime_error("record is outside committed replica frontier");
      std::string data;
      for(std::uint64_t offset=0;;) {
        auto read=replica.Read(h.lease,false,round,offset).get();
        if(auto* e=std::get_if<eps::Error>(&read))throw std::runtime_error(e->message);
        const auto& chunk=std::get<eps::SnapshotChunk>(read);
        if(chunk.total_bytes>scope.record_bytes)throw std::runtime_error("replica record exceeds limit");
        data+=chunk.data;offset+=chunk.data.size();if(chunk.eof)break;
      }
      const auto image=eps::DecodeReplicatedUpdate(data,scope.record_bytes);
      auto clocks=nlohmann::json::array(),requests=nlohmann::json::array();
      for(const auto& c:image.contexts)clocks.push_back({{"table",c.table},{"next",c.optimizer_sequence}});
      for(const auto& p:image.requests)requests.push_back({{"learner",p.learner},{"next",p.next}});
      std::cout<<nlohmann::json{{"round",image.round},{"rows",image.rows.size()},{"optimizer_clocks",clocks},{"requests",requests},{"bytes",data.size()}}.dump()<<'\n';return 0;
    }
    std::cout<<nlohmann::json{{"term",h.lease.term},{"primary",h.lease.primary},{"baseline",h.baseline},
      {"base_round",h.base_round},{"next_round",h.next_round},{"bytes",h.bytes},{"baseline_bytes",h.baseline_bytes}}.dump()<<'\n';
  }catch(const std::exception& e){std::cerr<<"eps_replica_control: "<<e.what()<<'\n';return 1;}
}
