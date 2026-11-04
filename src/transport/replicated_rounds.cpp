#include "transport/replicated_rounds.hpp"
#include "common/error.hpp"
#include <algorithm>
namespace eps::transport {
namespace {
template<class T>T Take(std::future<Result<T>> f){auto r=f.get();if(auto* e=std::get_if<Error>(&r))throw detail::Failure(*e);return std::get<T>(std::move(r));}
ReplicaLogOptions Local(ReplicaLogOptions options,const ReplicaLease& lease) {
  detail::Require(lease.term && !lease.primary.empty() && !options.directory.empty(),"replication requires a granted term and local directory");
  options.directory/= "term-"+std::to_string(lease.term);return options;
}
}
ReplicatedRoundWriter::ReplicatedRoundWriter(std::string endpoint,ReplicaLogOptions options,ReplicaLease lease)
    :options_(std::move(options)),lease_(std::move(lease)),replica_(std::move(endpoint),options_),local_(Local(options_,lease_)) {
  local_lease_=local_.Fence(0,lease_.primary);
}
ReplicaHead ReplicatedRoundWriter::Start(CpuParameterService& parameters,bool recover,bool leave_paused) {
  const auto head=Take(replica_.Head());
  detail::Require(head.lease==lease_ && head.base_round==options_.base_round,"replica grant/baseline round differs",Error::Code::stale_generation);
  if(recover) {
    detail::Require(head.baseline && local_.Head().next_round<=head.next_round,"replica recovery lacks baseline or local log contains unreplicated updates");
    auto copy=[&](bool baseline,std::uint64_t round) {
      std::string record;
      for(std::uint64_t offset=0;;) {
        const auto chunk=Take(replica_.Read(lease_,baseline,round,offset));
        local_.Write(local_lease_,baseline,round,chunk.total_bytes,offset,chunk.data);
        if(!baseline) {detail::Require(chunk.total_bytes<=options_.record_bytes,"replica record exceeds recovery budget");record+=chunk.data;}
        offset+=chunk.data.size();if(chunk.eof)break;
      }
      return record;
    };
    copy(true,head.base_round);
    Take(parameters.RestoreReplicaBaseline(options_.directory/("term-"+std::to_string(lease_.term))/"baseline.eps"));
    for(auto round=head.base_round;round<head.next_round;++round)
      Take(parameters.ImportReplicaUpdate(DecodeReplicatedUpdate(copy(false,round),options_.record_bytes)));
  } else {
    detail::Require(head.next_round==head.base_round,"fresh primary cannot overwrite a trained replica; recover it instead");
    const auto metadata=Take(parameters.TrainingMetadata());
    for(const auto& c:metadata.optimizer_contexts)detail::Require(c.optimizer_sequence==head.base_round,"replica baseline optimizer clock differs");
    Take(parameters.InitializeRequestFrontiers({{1023,head.base_round}}));
    const auto saved=Take(parameters.DrainAndFlush("replica-baseline"));
    // The snapshot is already immutable and fsynced. Share its inode locally
    // when possible; the remote replica still receives an independent copy.
    std::error_code link_error;
    std::filesystem::create_hard_link(saved.state_files.at(0),
      options_.directory/("term-"+std::to_string(lease_.term))/"baseline.eps",link_error);
    for(std::uint64_t offset=0;;) {
      const auto chunk=Take(parameters.ReadCheckpoint({options_.run,options_.epoch,saved.snapshot_id},offset,1<<20));
      local_.Write(local_lease_,true,head.base_round,chunk.total_bytes,offset,chunk.data);
      Take(replica_.Write(lease_,true,head.base_round,chunk.total_bytes,offset,chunk.data));
      offset+=chunk.data.size();if(chunk.eof)break;
    }
  }
  const auto result=Take(replica_.Head());detail::Require(result.lease==lease_ && result.next_round==head.next_round,"replica term/frontier changed during startup");
  const auto metadata=Take(parameters.TrainingMetadata());
  for(const auto& c:metadata.optimizer_contexts)detail::Require(c.optimizer_sequence==result.next_round,"replica replay optimizer clock differs");
  Take(parameters.InitializeRequestFrontiers({{1023,result.next_round}}));
  if(!leave_paused)Take(parameters.Resume());
  return result;
}
void ReplicatedRoundWriter::BeforeRound(std::uint64_t round) {
  const auto h=Take(replica_.Head());detail::Require(h.lease==lease_ && h.baseline && h.next_round==round,
    "replica primary term or round is fenced",Error::Code::stale_generation);
}
void ReplicatedRoundWriter::Commit(CpuParameterService& parameters,std::uint64_t round,const std::vector<RowKey>& keys) {
  const auto metadata=Take(parameters.TrainingMetadata());
  ReplicatedUpdate image{options_.run,options_.shard,options_.shards,options_.epoch,round,metadata.optimizer_contexts,{},metadata.requests};
  std::size_t at=0;
  while(at<keys.size()) {
    std::uint64_t bytes=0;std::vector<RowKey> group;
    while(at<keys.size()) {
      const auto id=keys[at].table;
      const auto definition=std::find_if(metadata.tables.begin(),metadata.tables.end(),[&](const auto& t){return t.id==id;});
      detail::Require(definition!=metadata.tables.end(),"replica image references unknown table");
      const auto cost=std::uint64_t(definition->dimension)*12+128;
      if(!group.empty() && bytes+cost>(8<<20))break;
      bytes+=cost;group.push_back(keys[at++]);
    }
    auto rows=Take(parameters.ReadState(Take(parameters.Locations(std::move(group)))));
    for(auto& row:rows)image.rows.push_back(std::move(row));
  }
  const auto data=EncodeReplicatedUpdate(image,options_.record_bytes);
  for(std::uint64_t offset=0;offset<data.size();) {
    const auto chunk=data.substr(offset,std::min<std::uint64_t>(1<<20,data.size()-offset));
    local_.Write(local_lease_,false,round,data.size(),offset,chunk);offset+=chunk.size();
  }
  for(std::uint64_t offset=0;offset<data.size();) {
    const auto chunk=data.substr(offset,std::min<std::uint64_t>(1<<20,data.size()-offset));
    Take(replica_.Write(lease_,false,round,data.size(),offset,chunk));offset+=chunk.size();
  }
}
}  // namespace eps::transport
