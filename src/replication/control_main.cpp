#include <fstream>
#include <iostream>
#include <charconv>
#include <sys/stat.h>
#include "store/snapshot.hpp"
#include "eps/brpc_replica.hpp"
#include "replication/configuration.hpp"
int main(int argc,char** argv) {
  try {
    if(argc!=5 || std::string(argv[1])!="--config" || std::string(argv[3])!="--action")throw std::runtime_error("usage: eps_replica_control --config FILE --action head|fence");
    std::ifstream file(argv[2]);if(!file)throw std::runtime_error("cannot open replica control config");nlohmann::json j;file>>j;
    const auto scope=eps::replication::Configuration(j);
    const auto action=std::string(argv[4]);
    if(action=="offline-head" || action=="offline-snapshots") {
      if(!std::filesystem::exists(scope.directory/"configuration"))throw std::runtime_error("offline inspection requires an existing replica log");
      eps::ReplicaLog log(scope);const auto h=log.Head();auto files=nlohmann::json::array();
      auto entry=[&](const std::filesystem::path& path) {
        struct stat s{};
        if(::lstat(path.c_str(),&s) || !S_ISREG(s.st_mode))throw std::runtime_error("retirement expects regular immutable files");
#ifdef __APPLE__
        const auto seconds=s.st_mtimespec.tv_sec,nanos=s.st_mtimespec.tv_nsec;
#else
        const auto seconds=s.st_mtim.tv_sec,nanos=s.st_mtim.tv_nsec;
#endif
        return nlohmann::json{{"path",std::filesystem::absolute(path).string()},{"device",s.st_dev},{"inode",s.st_ino},
          {"bytes",s.st_size},{"mtime_ns",std::int64_t(seconds)*1000000000+nanos}};
      };
      if(action=="offline-head") {
        for(const auto& file:std::filesystem::directory_iterator(scope.directory)) {
          const auto name=file.path().filename().string();
          if(name=="baseline.eps" || (name.starts_with("round-") && name.ends_with(".epsr")) ||
             (name.starts_with("upload-") && (name.ends_with(".part") || name.ends_with(".size"))))
            files.push_back(entry(file.path()));
          if(files.size()>16384)throw std::runtime_error("retirement file manifest exceeds 16384 entries");
        }
      } else {
        const auto root=std::filesystem::path(j.at("snapshot_directory").get<std::string>());
        if(std::filesystem::exists(root))for(const auto& directory:std::filesystem::directory_iterator(root)) {
          if(!directory.is_directory() || directory.is_symlink())throw std::runtime_error("unexpected snapshot directory entry");
          for(const auto& file:std::filesystem::directory_iterator(directory.path())) {
            const auto name=file.path().filename().string();
            if(name!="shard-"+std::to_string(scope.shard)+".bin") {
              if(!name.starts_with("shard-") || !name.ends_with(".bin"))throw std::runtime_error("unexpected snapshot payload");
              const auto digits=std::string_view(name).substr(6,name.size()-10);unsigned other=0;
              const auto parsed=std::from_chars(digits.data(),digits.data()+digits.size(),other);
              if(parsed.ec!=std::errc{} || parsed.ptr!=digits.data()+digits.size() || other>=scope.shards)
                throw std::runtime_error("unexpected snapshot shard name");
              continue;
            }
            const auto m=eps::detail::InspectSnapshot(file.path());
            if(m.run!=scope.run || m.shard!=scope.shard || m.shards!=scope.shards || m.epoch!=scope.epoch)
              throw std::runtime_error("snapshot retirement identity differs");
            for(const auto& c:m.optimizer_contexts)if(c.optimizer_sequence>j.at("retire_before").get<std::uint64_t>())
              throw std::runtime_error("snapshot contains updates beyond retirement frontier");
            files.push_back(entry(file.path()));
            if(files.size()>16384)throw std::runtime_error("snapshot retirement manifest exceeds 16384 entries");
          }
        }
      }
      std::cout<<nlohmann::json{{"base_round",h.base_round},{"next_round",h.next_round},{"baseline",h.baseline},
        {"term",h.lease.term},{"primary",h.lease.primary},{"files",files},{"retirement_protocol",1}}.dump()<<'\n';return 0;
    }
    eps::BrpcReplicaLog replica(j.at("replica_endpoint"),scope);
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
