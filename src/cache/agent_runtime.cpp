#include "eps/agent_runtime.hpp"
#include "eps/cache_agent.hpp"

#include <map>
#include <typeindex>
#include "common/executor.hpp"

namespace eps {
struct AgentRuntime::Impl {
  Impl(std::string r,std::shared_ptr<CacheBackend> b,std::uint64_t limit)
    :run(std::move(r)),backend(std::move(b)),replay_limit(limit){
    detail::Require(!run.empty() && bool(backend),"invalid agent runtime");
  }
  struct Record {
    std::uint64_t next=0,bytes=0;std::string method;
    std::type_index type{typeid(void)};std::shared_ptr<void> result;
  };
  template<typename T,typename Size,typename Action>
  std::future<Result<T>> Command(AgentCommand id,std::string method,Size size,Action action){
    return executor.Submit<T>([this,id=std::move(id),method=std::move(method),size=std::move(size),action=std::move(action)]() mutable -> Result<T>{
      detail::Require(id.run==run && id.shard<1024,"invalid agent run/shard");
      auto& record=streams[id.shard];
      if(record.next && id.sequence==record.next-1){
        detail::Require(record.method==method && record.type==typeid(Result<T>),"control identity reused for another method");
        return *std::static_pointer_cast<Result<T>>(record.result);
      }
      detail::Require(id.sequence>=record.next,"control command was retired",Error::Code::retired_request);
      detail::Require(id.sequence!=UINT64_MAX,"control sequence exhausted");
      auto bytes=size();
      detail::Require(bytes<=replay_limit-(replay_bytes-record.bytes),"agent replay payload budget exhausted",Error::Code::capacity_exceeded);
      // Allocate the replay holder before executing a mutating command.
      auto retained=std::make_shared<Result<T>>(Error{Error::Code::run_failed,"command not completed"});
      Result<T> result;
      try{result=action();}
      catch(const detail::Failure& e){result=e.error();}
      *retained=std::move(result);
      record.result=retained;record.type=typeid(Result<T>);record.method.swap(method);
      replay_bytes=replay_bytes-record.bytes+bytes;record.bytes=bytes;record.next=id.sequence+1;
      return *retained;
    });
  }
  std::string run;std::shared_ptr<CacheBackend> backend;
  std::uint64_t replay_limit,replay_bytes=0;
  std::map<ShardId,Record> streams;
  std::map<TableId,TableDefinition> tables;
  detail::OrderedExecutor executor;
};
AgentRuntime::AgentRuntime(std::string run,std::shared_ptr<CacheBackend> backend,std::uint64_t bytes)
  :impl_(std::make_unique<Impl>(std::move(run),std::move(backend),bytes)){}
AgentRuntime::~AgentRuntime(){impl_->executor.Shutdown();}
std::future<Result<Ack>> AgentRuntime::Define(AgentCommand id,TableDefinition table){
  return impl_->Command<Ack>(std::move(id),"define",[]{return 0ULL;},[this,table]{
    detail::Await(impl_->backend->Define(table));impl_->tables.insert_or_assign(table.id,table);return Ack{};
  });
}
std::future<Result<std::uint64_t>> AgentRuntime::FreeSlots(std::string run,TableId table){
  return impl_->executor.Submit<std::uint64_t>([this,run=std::move(run),table]{
    detail::Require(run==impl_->run,"invalid agent run");return detail::Await(impl_->backend->FreeSlots(table));
  });
}
std::future<Result<std::vector<LocatedRow>>> AgentRuntime::Load(AgentCommand id,MigrationPlan plan,std::vector<RowState> rows){
  const auto bytes=plan.rows.size()*sizeof(LocatedRow);
  return impl_->Command<std::vector<LocatedRow>>(std::move(id),"load",[bytes]{return bytes;},
    [this,plan=std::move(plan),rows=std::move(rows)]() mutable {
      return detail::Await(impl_->backend->Load(std::move(plan),std::move(rows)));
    });
}
std::future<Result<std::vector<TensorBlock>>> AgentRuntime::Gather(std::string run,std::vector<AgentWork> work,std::vector<FeatureInput> layouts){
  return impl_->executor.Submit<std::vector<TensorBlock>>([this,run=std::move(run),work=std::move(work),layouts=std::move(layouts)]() mutable {
    detail::Require(run==impl_->run,"invalid agent run");
    return detail::Await(impl_->backend->GatherHost(std::move(work),std::move(layouts)));
  });
}
std::future<Result<std::vector<DeviceTensorBlock>>> AgentRuntime::GatherDevice(std::string run,std::vector<AgentWork> work,std::vector<FeatureInput> layouts){
  return impl_->executor.Submit<std::vector<DeviceTensorBlock>>([this,run=std::move(run),work=std::move(work),layouts=std::move(layouts)]() mutable {
    detail::Require(run==impl_->run,"invalid agent run");
    auto gpu=std::dynamic_pointer_cast<GpuCacheAgent>(impl_->backend);
    detail::Require(bool(gpu),"cache backend does not support device tensors");
    return detail::Await(gpu->GatherDevice(std::move(work),std::move(layouts)));
  });
}
std::future<Result<Ack>> AgentRuntime::Apply(AgentCommand id,UpdateContext context,RowGradients g){
  auto home=id.shard;auto run=id.run;
  return impl_->Command<Ack>(std::move(id),"apply",[]{return 0ULL;},[this,home,run=std::move(run),context,g=std::move(g)]() mutable {
    detail::Require(context.shard==home && context.original_request.run==run,"agent update context ownership mismatch");
    return detail::Await(impl_->backend->ApplyHost(context,std::move(g)));
  });
}
std::future<Result<std::vector<RowState>>> AgentRuntime::ReadBack(AgentCommand id,MigrationPlan plan){
  auto size=[this,rows=plan.rows]{
    std::uint64_t bytes=0;for(const auto& row:rows)bytes+=sizeof(RowState)+
      std::uint64_t(impl_->tables.at(row.row.table).dimension)*12;
    return bytes;
  };
  return impl_->Command<std::vector<RowState>>(std::move(id),"readback",std::move(size),
    [this,plan=std::move(plan)]() mutable {return detail::Await(impl_->backend->ReadBack(std::move(plan)));});
}
std::future<Result<Ack>> AgentRuntime::ApplyCompact(AgentCommand id,UpdateContext context,CompactGradients p){
  return impl_->Command<Ack>(id,"apply_compact",[]{return 0ULL;},[this,id,context,p=std::move(p)]()mutable{
    detail::Require(context.original_request.run==id.run && context.shard==id.shard,"compact update control owner mismatch");
    return detail::Await(impl_->backend->ApplyCompact(context,std::move(p)));
  });
}
std::future<Result<Ack>> AgentRuntime::CommitEviction(AgentCommand id,MigrationPlan plan){
  return impl_->Command<Ack>(std::move(id),"commit",[]{return 0ULL;},[this,plan=std::move(plan)]() mutable {
    return detail::Await(impl_->backend->CommitEviction(std::move(plan)));
  });
}
}  // namespace eps
