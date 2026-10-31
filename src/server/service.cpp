#include "eps/runtime.hpp"

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "access/batch_access.hpp"
#include "common/error.hpp"
#include "common/executor.hpp"
#include "eps/compact_gradients.hpp"
#include "server/request_tracker.hpp"
#include "store/snapshot.hpp"
#include "update/lazy_adam.hpp"

namespace eps {
using detail::Require;

class CpuParameterService::Impl {
 public:
  explicit Impl(RuntimeOptions options)
      : options_(std::move(options)), tracker_(options_.request_history),
        store_(std::make_unique<detail::ShardStore>(options_.shard_id,
          options_.shard_count, options_.partition_epoch, options_.state_bytes)) {
    Require(!options_.run_id.empty() && options_.run_id.size() <= 1024, "invalid run ID");
    Require(options_.request_history >= 4 && options_.max_live_tickets > 0 &&
            options_.max_queued_operations > 0, "invalid request limits");
    worker_ = std::thread([this] {
      while (true) {
        std::function<void()> task;
        {
          std::unique_lock lock(mutex_);
          changed_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
          if (queue_.empty() && stopping_) return;
          queued_payload_bytes_ -= queue_.front().payload_bytes;
          task = std::move(queue_.front().function); queue_.pop_front();
        }
        task();
      }
    });
  }
  ~Impl() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    changed_.notify_all(); worker_.join();
  }

  template <typename T, typename Fn>
  std::future<Result<T>> Submit(Fn fn, std::uint64_t payload_bytes = 0) {
    auto promise = std::make_shared<std::promise<Result<T>>>();
    auto future = promise->get_future();
    {
      std::lock_guard lock(mutex_);
      if (stopping_ || queue_.size() >= options_.max_queued_operations ||
          payload_bytes > options_.queue_payload_bytes - queued_payload_bytes_) {
        ++rejected_operations_;
        promise->set_value(Error{Error::Code::capacity_exceeded, "shard operation queue is full"});
        return future;
      }
      queue_.push_back({[this,promise, queued=std::chrono::steady_clock::now(), fn = std::move(fn)]() mutable {
        const auto begin=std::chrono::steady_clock::now();
        stats_.queue_wait_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(begin-queued).count();
        try { promise->set_value(fn()); }
        catch (const detail::Failure& e) {
          if(e.error().code==Error::Code::run_failed)fatal_=e.what();
          promise->set_value(e.error());
        }
        catch (const std::bad_alloc&) {
          promise->set_value(Error{Error::Code::capacity_exceeded, "allocation failed"});
        } catch (const std::exception& e) {
          fatal_=e.what();
          promise->set_value(Error{Error::Code::run_failed, e.what()});
        }
        ++stats_.executed_operations;
        stats_.executor_busy_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
      },payload_bytes});
      peak_queued_operations_=std::max<std::uint64_t>(peak_queued_operations_,queue_.size());
      queued_payload_bytes_ += payload_bytes;
    }
    changed_.notify_one(); return future;
  }

  void Identity(const RequestId& id, Phase phase) const {
    Require(id.run == options_.run_id && id.phase == phase, "invalid run or request phase");
  }
  void Ready() const {Require(fatal_.empty(),fatal_.c_str(),Error::Code::run_failed);}
  static auto RowId(RowKey key){return std::pair(key.table,key.key);}
  LocatedRow Location(RowKey key) const {
    if(auto it=locations_.find(RowId(key));it!=locations_.end())return it->second;
    return {key,options_.shard_id,Placement::cpu,0,0,0};
  }
  void Unpinned(RowKey row) const {
    for(const auto& [name,ticket]:tickets_)for(const auto& f:ticket.features)
      if(f.input.table==row.table)
        Require(std::find(f.input.keys.begin(),f.input.keys.end(),row.key)==f.input.keys.end(),
                "row is held by an active lookup",Error::Code::unavailable);
  }
  template<typename T>T AgentResult(std::future<Result<T>> future){
    auto result=future.get();if(auto error=std::get_if<Error>(&result)){
      if(error->code==Error::Code::unavailable || error->code==Error::Code::run_failed ||
         error->code==Error::Code::retired_request)fatal_=error->message;
      throw detail::Failure(*error);
    }return std::get<T>(std::move(result));
  }
  static std::string TicketName(const RequestId& id) {
    return std::to_string(id.learner) + "/" + std::to_string(id.sequence);
  }
  void Size(std::uint64_t& bytes, std::size_t count, std::size_t element_size) const {
    Require(count <= options_.max_request_bytes / element_size,
            "request array exceeds limit", Error::Code::capacity_exceeded);
    const auto add = count * element_size;
    Require(add <= options_.max_request_bytes - bytes, "request exceeds byte limit",
            Error::Code::capacity_exceeded);
    bytes += add;
  }

  LookupReply Lookup(LookupRequest request) {
    Ready();
    Identity(request.id, Phase::lookup);
    Require(request.partition_epoch == options_.partition_epoch, "partition epoch mismatch",
            Error::Code::stale_generation);
    const auto name = TicketName(request.id);
    const auto state = tracker_.Reserve(request.id);
    if (state != detail::RequestTracker::State::fresh) {
      ++stats_.duplicate_requests;
      auto it = tickets_.find(name);
      Require(it != tickets_.end(), "lookup is already released", Error::Code::retired_request);
      return it->second.reply;
    }
    try {
      Require(!draining_, "shard is draining", Error::Code::unavailable);
      Require(tickets_.size() < options_.max_live_tickets, "live ticket limit reached",
              Error::Code::capacity_exceeded);
      Require(!request.features.empty(), "lookup has no features");
      std::uint64_t bytes = 0;
      std::unordered_set<std::uint32_t> ids;
      for (const auto& f : request.features) {
        Require(ids.insert(f.feature).second, "duplicate feature ID");
        detail::ValidateFeature(f, *store_);
        Size(bytes, f.keys.size(), sizeof(Key));
        Size(bytes, f.offsets.size(), sizeof(std::uint64_t));
        Size(bytes, f.valid_lengths.size(), sizeof(std::uint64_t));
        const auto rows = f.pooling == Pooling::sequence ? f.keys.size() : f.valid_lengths.size();
        Size(bytes, rows, store_->Table(f.table).definition.dimension * sizeof(float));
      }
      Ticket ticket; ticket.id = request.id; ticket.read_only = request.read_only;
      ticket.reply.ticket = name;
      for (auto& f : request.features) {
        const auto route=planner_?planner_->SelectRoute(f):fallback_route_;
        const bool transfer=route==AccessRoute::transfer_unique;
        ticket.features.push_back(detail::PrepareFeature(std::move(f), *store_,
          [this,transfer](RowKey row){return !transfer && Location(row).placement==Placement::cpu;}));
        ticket.reply.cpu_contributions.push_back(ticket.features.back().contribution);
        const auto& input=ticket.features.back().input;
        using Group=std::pair<AgentId,AccessRoute>;
        std::map<Group,AgentWork> work;
        std::map<Group,std::map<Key,std::uint64_t>> indices;
        for(std::size_t i=0;i<input.keys.size();++i){
          auto location=Location({input.table,input.keys[i]});
          if(location.placement!=Placement::gpu && !transfer)continue;
          const Group group=location.placement==Placement::gpu?
            Group{location.agent,AccessRoute::resident_gpu}:Group{fallback_agent_,AccessRoute::transfer_unique};
          auto& task=work[group];task.feature=input.feature;
          task.route=group.second;task.agent=group.first;
          if(task.occurrence_to_unique.empty())task.occurrence_to_unique.assign(input.keys.size(),UINT64_MAX);
          auto [it,inserted]=indices[group].try_emplace(location.row.key,task.rows.size());
          if(inserted){
            task.rows.push_back(location);
            if(group.second==AccessRoute::transfer_unique){
              auto& table=store_->Table(location.row.table);
              const auto* values=table.At(table.index.at(location.row.key)).weights;
              Size(bytes,table.definition.dimension,sizeof(float));
              task.transferred_weights.insert(task.transferred_weights.end(),values,values+table.definition.dimension);
            }
          }
          task.occurrence_to_unique[i]=it->second;
        }
        for(auto& [group,task]:work)ticket.reply.agent_work.push_back(std::move(task));
        // Only layout is needed for backward; retain one cached feature payload.
        std::vector<float>().swap(ticket.features.back().contribution.values);
      }
      for (const auto& f : ticket.features)
        ticket.payload_bytes += (f.input.keys.capacity() + f.input.offsets.capacity() +
          f.input.valid_lengths.capacity() + f.unique_slots.capacity() +
          f.occurrence_to_unique.capacity()) * sizeof(std::uint64_t);
      for (const auto& f : ticket.reply.cpu_contributions)
        ticket.payload_bytes += f.values.capacity() * sizeof(float);
      for(const auto& work:ticket.reply.agent_work)
        ticket.payload_bytes+=work.rows.capacity()*sizeof(LocatedRow)+
          work.occurrence_to_unique.capacity()*sizeof(std::uint64_t)+
          work.transferred_weights.capacity()*sizeof(float);
      Require(ticket.payload_bytes <= options_.ticket_payload_bytes - ticket_payload_bytes_,
              "live ticket payload budget exhausted", Error::Code::capacity_exceeded);
      auto reply = ticket.reply;
      const auto ticket_bytes = ticket.payload_bytes;
      tickets_.emplace(name, std::move(ticket));
      ticket_payload_bytes_ += ticket_bytes;
      ++stats_.lookups;
      return reply;
    } catch (...) { tracker_.Abort(request.id); throw; }
  }

  Ack Push(PushRequest request) {
    Ready();
    Identity(request.id, Phase::push);
    const auto completed = tracker_.CompletedAs(request.id);
    if (completed) {
      Require(*completed == Phase::push, "lookup was released without an update");
      ++stats_.duplicate_requests; return {};
    }
      auto it = tickets_.find(request.ticket);
      Require(it != tickets_.end(), "unknown lookup ticket");
      auto& ticket = it->second;
      Require(!ticket.read_only && TicketName(request.id) == request.ticket &&
              request.id.learner == ticket.id.learner, "ticket does not belong to update");
      std::uint64_t bytes = 0;
      for (const auto& g : request.feature_gradients) Size(bytes, g.values.size(), sizeof(float));
      auto gradients = detail::ReduceGradients(ticket.features, request.feature_gradients, *store_);
      std::map<TableId,RowGradients> cpu;
      std::map<std::pair<AgentId,TableId>,RowGradients> device;
      for(const auto& [table,g]:gradients){
        auto& local=cpu[table];const auto dim=store_->Table(table).definition.dimension;
        for(std::size_t i=0;i<g.rows.size();++i){
          auto location=Location(g.rows[i]);
          auto& target=location.placement==Placement::gpu?device[{location.agent,table}]:local;
          target.rows.push_back(g.rows[i]);target.values.insert(target.values.end(),
            g.values.begin()+i*dim,g.values.begin()+(i+1)*dim);
        }
      }
      for(const auto& [table,g]:cpu)detail::ValidateLazyAdam(store_->Table(table),g);
      try{
        struct DeviceUpdate {UpdateContext context;std::variant<RowGradients,CompactGradients> payload;};
        std::map<AgentId,std::deque<DeviceUpdate>> groups;
        std::uint64_t reference_bytes=0,transferred_bytes=0,compact_commands=0;
        for(auto& [owner,g]:device){
          const auto& t=store_->Table(owner.second);
          UpdateContext context{request.id,options_.shard_id,owner.second,t.optimizer_sequence,
            t.definition.optimizer.learning_rate,t.beta1_power,t.beta2_power};
          const auto expanded=g.values.size()*sizeof(float);reference_bytes+=expanded;
          if(options_.compact_gpu_gradients && expanded>g.rows.size()*20+8){
            try{
              auto compact=detail::BuildCompactGradients(ticket.features,request.feature_gradients,g.rows,t.definition.dimension);
              const auto bytes=CompactGradientBytes(compact)-compact.rows.size()*sizeof(RowKey);
              if(bytes<expanded){
                groups[owner.first].push_back({context,std::move(compact)});
                transferred_bytes+=bytes;++compact_commands;continue;
              }
            }catch(const detail::Failure& e){
              // Highly repeated/sequence layouts can have larger reference
              // metadata than their already-reduced dense gradient.
              if(e.error().code!=Error::Code::capacity_exceeded)throw;
            }
          }
          transferred_bytes+=expanded;groups[owner.first].push_back({context,std::move(g)});
        }
        auto send=[this](AgentId agent,DeviceUpdate update){
          if(auto* compact=std::get_if<CompactGradients>(&update.payload))
            return agents_.at(agent)->ApplyCompact(update.context,std::move(*compact));
          return agents_.at(agent)->ApplyHost(update.context,std::get<RowGradients>(std::move(update.payload)));
        };
        // Independent GPU owners can execute one original Push concurrently.
        // Keep at most one control command per home/agent stream outstanding so
        // a later table cannot overtake that stream's earlier command.
        if(!options_.concurrent_agent_updates){
          for(auto& [agent,updates]:groups)for(auto& update:updates)
            AgentResult(send(agent,std::move(update)));
        }else for(;;){
          std::vector<std::future<Result<Ack>>> pending;pending.reserve(groups.size());
          for(auto& [agent,updates]:groups)if(!updates.empty()){
            auto update=std::move(updates.front());updates.pop_front();
            pending.push_back(send(agent,std::move(update)));
          }
          if(pending.empty())break;
          for(auto& completion:pending)AgentResult(std::move(completion));
        }
        stats_.expanded_gradient_bytes+=reference_bytes;stats_.transferred_gradient_bytes+=transferred_bytes;
        stats_.compact_update_commands+=compact_commands;
      }catch(const std::exception& e){fatal_=e.what();throw detail::Failure({Error::Code::run_failed,fatal_});}
      for(const auto& [table,g]:cpu)detail::ApplyLazyAdam(store_->Table(table),g);
      tracker_.Complete(request.id);
      ticket_payload_bytes_ -= ticket.payload_bytes;
      tickets_.erase(it); ++stats_.updates;
      return {};
  }

  Ack Release(ReleaseRequest request) {
    Identity(request.id, Phase::release);
    if (tracker_.CompletedAs(request.id)) {
      ++stats_.duplicate_requests; return {};
    }
      auto it = tickets_.find(request.ticket);
      Require(it != tickets_.end() && TicketName(request.id) == request.ticket,
              "release does not match a live ticket");
      tracker_.Complete(request.id);
      ticket_payload_bytes_ -= it->second.payload_bytes;
      tickets_.erase(it); return {};
  }

  std::vector<RowState> ReadState(const std::vector<LocatedRow>& rows) {
    Ready();
    std::uint64_t bytes = 0;
    for (const auto& located : rows) {
      const auto current=Location(located.row);
      Require(located.home == options_.shard_id && located.placement==current.placement &&
              located.generation==current.generation,"row placement changed",Error::Code::stale_generation);
      Size(bytes, store_->Table(located.row.table).definition.dimension, sizeof(float) * 3);
      Unpinned(located.row);
    }
    std::vector<RowState> result; result.reserve(rows.size());
    for(const auto& row:rows){
      if(row.placement==Placement::gpu){
        MigrationPlan plan{"read",Location(row.row).agent,{Location(row.row)},Placement::cpu,0};
        auto state=AgentResult(agents_.at(plan.agent)->ReadBack(std::move(plan)));
        Require(state.size()==1 && state[0].row==row.row,"invalid cache readback");
        result.push_back(std::move(state[0]));
      }else result.push_back(store_->Read(row.row.table,row.row.key));
    }
    return result;
  }

  std::vector<LocatedRow> Move(MoveRowsRequest request){
    Ready();Require(request.run==options_.run_id,"invalid migration run");
    Require(request.destination==Placement::cpu || request.destination==Placement::gpu,"invalid migration destination");
    Require(!request.rows.empty() && request.rows.size()<=4096 && request.budget.max_inflight_batches>0,
            "invalid migration batch");
    std::set<std::pair<TableId,Key>> unique;std::map<TableId,std::uint64_t> counts;
    std::uint64_t bytes=0;
    for(auto row:request.rows){
      Require(row.key%options_.shard_count==options_.shard_id && unique.insert(RowId(row)).second,
              "duplicate or wrongly routed migration row");
      Unpinned(row);auto& table=store_->Table(row.table);
      Require(table.index.contains(row.key),"migration requires an existing row");
      Size(bytes,table.definition.dimension,sizeof(float)*3);
      ++counts[row.table];
      auto location=Location(row);
      Require((request.destination==Placement::gpu && location.placement==Placement::cpu)||
        (request.destination==Placement::cpu && location.placement==Placement::gpu && location.agent==request.agent),
        "row is not at the expected migration source");
    }
    Require(agents_.contains(request.agent),"unknown cache agent");
    Require(bytes<=request.budget.network_bytes && bytes<=request.budget.staging_bytes &&
            counts.size()<=request.budget.max_inflight_batches,"migration budget exceeded",Error::Code::capacity_exceeded);
    auto backend=agents_.at(request.agent);
    if(request.destination==Placement::gpu)for(const auto& [table,n]:counts){
      const auto free=AgentResult(backend->FreeSlots(table));
      Require(free>=request.budget.reserve_slots && n<=free-request.budget.reserve_slots,
              "migration would exhaust reserved slots",Error::Code::capacity_exceeded);
    }
    MigrationPlan plan{options_.run_id+"/"+std::to_string(options_.shard_id)+"/"+
      std::to_string(next_move_++),request.agent,{},request.destination,bytes};
    std::vector<RowState> states;
    for(auto row:request.rows){
      auto location=Location(row);locations_.try_emplace(RowId(row),location);
      if(request.destination==Placement::gpu){
        ++location.generation;location.agent=request.agent;location.placement=Placement::loading;
        states.push_back(store_->Read(row.table,row.key));
      }plan.rows.push_back(location);
    }
    if(request.destination==Placement::gpu){
      auto loaded=AgentResult(backend->Load(plan,std::move(states)));
      Require(loaded.size()==plan.rows.size(),"cache load response count mismatch",Error::Code::run_failed);
      for(std::size_t i=0;i<loaded.size();++i){
        Require(loaded[i].row==plan.rows[i].row && loaded[i].generation==plan.rows[i].generation &&
          loaded[i].home==options_.shard_id && loaded[i].agent==request.agent && loaded[i].placement==Placement::gpu,
          "cache load response ownership mismatch",Error::Code::run_failed);
      }
      for(const auto& row:loaded)locations_.at(RowId(row.row))=row;
    }else{
      states=AgentResult(backend->ReadBack(plan));
      Require(states.size()==plan.rows.size(),"cache writeback count mismatch",Error::Code::run_failed);
      for(std::size_t i=0;i<states.size();++i){
        auto dim=store_->Table(states[i].row.table).definition.dimension;
        Require(states[i].row==plan.rows[i].row && states[i].weights.size()==dim &&
          states[i].first_moment.size()==dim && states[i].second_moment.size()==dim,
          "invalid cache writeback state",Error::Code::run_failed);
        for(auto* values:{&states[i].weights,&states[i].first_moment,&states[i].second_moment})
          for(auto x:*values)Require(std::isfinite(x),"non-finite cache writeback",Error::Code::run_failed);
        for(auto x:states[i].second_moment)Require(x>=0,"negative cache second moment",Error::Code::run_failed);
      }
      for(const auto& row:states)store_->Overwrite(row);
      AgentResult(backend->CommitEviction(plan));
      for(auto row:plan.rows){
        ++row.generation;row.placement=Placement::cpu;row.agent=0;row.slot=0;
        locations_.at(RowId(row.row))=row;
      }
    }
    stats_.migration_bytes+=bytes;
    std::vector<LocatedRow> result;result.reserve(request.rows.size());
    for(auto row:request.rows)result.push_back(Location(row));
    return result;
  }

  SnapshotMetadata Save(const std::string& id) {
    Ready();
    Require(!id.empty() && id.size() <= 128 && id != "." && id != ".." &&
      std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
      }), "invalid snapshot ID");
    draining_ = true;
    Require(tickets_.empty(), "drain in progress: complete or release outstanding tickets",
            Error::Code::unavailable);
    for(;;){
      std::vector<RowKey> rows;AgentId agent=0;std::uint64_t bytes=0;
      for(const auto& [key,loc]:locations_)if(loc.placement==Placement::gpu){
        if(rows.empty())agent=loc.agent;
        if(loc.agent!=agent)continue;
        auto row_bytes=std::uint64_t(store_->Table(loc.row.table).definition.dimension)*12;
        if(bytes+row_bytes>options_.max_request_bytes || rows.size()==64)break;
        rows.push_back(loc.row);bytes+=row_bytes;
      }
      if(rows.empty())break;
      Move({options_.run_id,std::move(rows),agent,Placement::cpu,
        {options_.max_request_bytes,options_.max_request_bytes,1024,0}});
    }
    const auto file = options_.snapshot_directory / id /
        ("shard-" + std::to_string(options_.shard_id) + ".bin");
    detail::WriteSnapshot(file, options_.run_id, *store_);
    SnapshotMetadata result{id, options_.shard_id, options_.partition_epoch, {file.string()}, {}};
    for (const auto& [table, t] : store_->tables())
      result.optimizer_contexts.push_back({{options_.run_id, 0, 0, Phase::maintenance},
        options_.shard_id, table, t.optimizer_sequence,
        t.definition.optimizer.learning_rate, t.beta1_power, t.beta2_power});
    return result;
  }

  struct Ticket {
    RequestId id;
    bool read_only;
    std::vector<detail::PreparedFeature> features;
    LookupReply reply;
    std::uint64_t payload_bytes = 0;
  };
  RuntimeOptions options_;
  detail::RequestTracker tracker_;
  std::unique_ptr<detail::ShardStore> store_;
  std::unordered_map<std::string, Ticket> tickets_;
  RuntimeStats stats_;
  std::uint64_t rejected_operations_=0,peak_queued_operations_=0;
  std::map<AgentId,std::shared_ptr<CacheBackend>> agents_;
  AgentId fallback_agent_=0;
  AccessRoute fallback_route_=AccessRoute::cpu_pool;
  std::shared_ptr<AccessPlanner> planner_;
  std::map<std::pair<TableId,Key>,LocatedRow> locations_;
  std::uint64_t next_move_=0;
  std::string fatal_;
  bool draining_ = false;
  std::uint64_t ticket_payload_bytes_ = 0;
  std::mutex mutex_;
  std::condition_variable changed_;
  struct Queued { std::function<void()> function; std::uint64_t payload_bytes; };
  std::deque<Queued> queue_;
  std::uint64_t queued_payload_bytes_ = 0;
  bool stopping_ = false;
  std::thread worker_;
};

CpuParameterService::CpuParameterService(RuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
CpuParameterService::~CpuParameterService() = default;
std::future<Result<Ack>> CpuParameterService::CreateTable(TableDefinition table) {
  return impl_->Submit<Ack>([this, table] {
    impl_->Ready();impl_->store_->Create(table);
    for(const auto& [id,backend]:impl_->agents_)impl_->AgentResult(backend->Define(table));
    return Ack{};
  });
}
std::future<Result<LookupReply>> CpuParameterService::Lookup(LookupRequest request) {
  std::uint64_t bytes = request.features.capacity() * sizeof(FeatureInput);
  for (const auto& f : request.features)
    bytes += (f.keys.capacity() + f.offsets.capacity() + f.valid_lengths.capacity()) * sizeof(std::uint64_t);
  return impl_->Submit<LookupReply>([this, request = std::move(request)]() mutable {
    return impl_->Lookup(std::move(request));
  }, bytes);
}
std::future<Result<Ack>> CpuParameterService::Push(PushRequest request) {
  std::uint64_t bytes = request.feature_gradients.capacity() * sizeof(TensorBlock);
  for (const auto& g : request.feature_gradients)
    bytes += g.values.capacity() * sizeof(float) + g.shape.capacity() * sizeof(std::uint64_t);
  return impl_->Submit<Ack>([this, request = std::move(request)]() mutable {
    return impl_->Push(std::move(request));
  }, bytes);
}
std::future<Result<Ack>> CpuParameterService::Release(ReleaseRequest request) {
  return impl_->Submit<Ack>([this, request = std::move(request)]() mutable {
    return impl_->Release(std::move(request));
  });
}
std::future<Result<std::vector<RowState>>> CpuParameterService::ReadState(std::vector<LocatedRow> rows) {
  const auto bytes = rows.size() * sizeof(LocatedRow);
  return impl_->Submit<std::vector<RowState>>([this, rows = std::move(rows)] {
    return impl_->ReadState(rows);
  }, bytes);
}
std::future<Result<Ack>> CpuParameterService::CompleteMigration(MigrationCompletion) {
  return impl_->Submit<Ack>([]() -> Result<Ack> {
    return Error{Error::Code::unavailable, "GPU cache agent is not attached"};
  });
}
std::future<Result<SnapshotMetadata>> CpuParameterService::DrainAndFlush(std::string id) {
  return impl_->Submit<SnapshotMetadata>([this, id = std::move(id)] { return impl_->Save(id); });
}
std::future<Result<Ack>> CpuParameterService::Restore(std::filesystem::path file) {
  return impl_->Submit<Ack>([this, file = std::move(file)] {
    Require(impl_->store_->empty() && impl_->tickets_.empty(), "restore requires a fresh service");
    auto store = detail::ReadSnapshot(file, impl_->options_.run_id,
      impl_->options_.shard_id, impl_->options_.shard_count,
      impl_->options_.partition_epoch, impl_->options_.state_bytes);
    impl_->store_ = std::move(store); return Ack{};
  });
}
std::future<Result<Ack>> CpuParameterService::Resume() {
  return impl_->Submit<Ack>([this] { impl_->Ready();impl_->draining_ = false; return Ack{}; });
}
std::future<Result<RuntimeStats>> CpuParameterService::Stats() {
  return impl_->Submit<RuntimeStats>([this] {
    auto s = impl_->stats_;
    s.rows = impl_->store_->rows(); s.state_bytes = impl_->store_->bytes();
    s.live_tickets = impl_->tickets_.size();
    s.ticket_payload_bytes = impl_->ticket_payload_bytes_;
    for(const auto& [key,loc]:impl_->locations_)if(loc.placement==Placement::gpu)++s.cached_rows;
    { std::lock_guard lock(impl_->mutex_); s.queued_payload_bytes = impl_->queued_payload_bytes_;
      s.rejected_operations=impl_->rejected_operations_;s.peak_queued_operations=impl_->peak_queued_operations_; }
    return s;
  });
}
std::future<Result<Ack>> CpuParameterService::AttachAgent(AgentId id,std::shared_ptr<CacheBackend> backend){
  return impl_->Submit<Ack>([this,id,backend=std::move(backend)]{
    impl_->Ready();Require(bool(backend) && !impl_->agents_.contains(id),"invalid/duplicate cache agent");
    for(const auto& [table,state]:impl_->store_->tables())impl_->AgentResult(backend->Define(state.definition));
    impl_->agents_.emplace(id,backend);return Ack{};
  });
}
std::future<Result<Ack>> CpuParameterService::ConfigureAccess(AgentId agent,AccessRoute route,
                                                           std::shared_ptr<AccessPlanner> planner){
  return impl_->Submit<Ack>([this,agent,route,planner=std::move(planner)]{
    impl_->Ready();Require(route==AccessRoute::cpu_pool || route==AccessRoute::transfer_unique,"invalid fallback route");
    Require((!planner && route==AccessRoute::cpu_pool) || impl_->agents_.contains(agent),"fallback GPU agent is not attached");
    impl_->fallback_agent_=agent;impl_->fallback_route_=route;impl_->planner_=planner;return Ack{};
  });
}
std::future<Result<std::vector<LocatedRow>>> CpuParameterService::Locations(std::vector<RowKey> rows){
  return impl_->Submit<std::vector<LocatedRow>>([this,rows=std::move(rows)]{
    impl_->Ready();std::vector<LocatedRow> out;out.reserve(rows.size());
    for(auto row:rows){
      Require(row.key%impl_->options_.shard_count==impl_->options_.shard_id &&
        impl_->store_->Table(row.table).index.contains(row.key),"unknown/wrongly routed row");
      out.push_back(impl_->Location(row));
    }return out;
  });
}
std::future<Result<std::vector<LocatedRow>>> CpuParameterService::MoveRows(MoveRowsRequest request){
  const auto bytes=request.rows.capacity()*sizeof(RowKey);
  return impl_->Submit<std::vector<LocatedRow>>([this,request=std::move(request)]() mutable {
    return impl_->Move(std::move(request));
  },bytes);
}
}  // namespace eps
