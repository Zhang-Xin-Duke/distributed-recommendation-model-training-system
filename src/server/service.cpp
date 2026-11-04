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
#include "cache/reuse_policy.hpp"
#include "common/error.hpp"
#include "common/executor.hpp"
#include "common/phase_timer.hpp"
#include "common/resumable_actor.hpp"
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
          options_.shard_count, options_.partition_epoch, options_.state_bytes)),
        actor_({options_.max_queued_operations,options_.max_active_operations,
                options_.completion_waiters,options_.queue_payload_bytes,options_.dependency_payload_bytes,
                options_.cache_dependencies,options_.max_observation_operations},
          [this](const Error& error){if(error.code==Error::Code::run_failed)fatal_=error.message;}) {
    Require(!options_.run_id.empty() && options_.run_id.size() <= 1024, "invalid run ID");
    Require(options_.request_history >= 4 && options_.max_live_tickets > 0 &&
            options_.max_queued_operations > 0, "invalid request limits");
    Require(!options_.exclusive_update_learner || *options_.exclusive_update_learner < 1024,
      "invalid exclusive update learner");
    draining_ = options_.start_paused;
  }

  template <typename T, typename Fn>
  std::future<Result<T>> Submit(Fn fn, std::uint64_t payload_bytes = 0) {
    return actor_.Submit<T>(std::move(fn),payload_bytes);
  }
  template<class T,class Fn,class Dependencies>
  std::future<Result<T>> SubmitIndependent(Fn fn,Dependencies dependencies,std::uint64_t bytes=0) {
    return actor_.Submit<T>(std::move(fn),std::move(dependencies),bytes);
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
  template<typename T>T CheckedAgentResult(Result<T> result){
    if(auto error=std::get_if<Error>(&result)){
      if(error->code==Error::Code::unavailable || error->code==Error::Code::run_failed ||
         error->code==Error::Code::retired_request)fatal_=error->message;
      throw detail::Failure(*error);
    }
    Ready();return std::get<T>(std::move(result));
  }
  template<class T,class Factory>detail::ActorTask<T> AgentCall(Factory factory) {
    detail::PhaseTimer timer(stats_.agent_wait_nanoseconds);
    auto results=co_await actor_.WaitAll<T>(1,[&](auto& pending){pending.push_back(factory());});
    co_return CheckedAgentResult(std::move(results.front()));
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
  void Dependency(detail::ActorResources& resources,detail::ActorResource resource) const {
    Require(resources.keys.size()<options_.dependency_payload_bytes/sizeof(detail::ActorResource),
      "shard dependency budget exceeded",Error::Code::capacity_exceeded);
    resources.keys.push_back(resource);
  }
  detail::ActorResources Lifecycle(const RequestId& id) const {
    detail::ActorResources r;Dependency(r,{detail::ActorResource::Kind::lifecycle,id.learner,id.sequence});return r;
  }
  void RowDependency(detail::ActorResources& r,RowKey row,bool agent=false) const {
    Dependency(r,{detail::ActorResource::Kind::row,row.table,row.key});
    auto location=Location(row);
    if(agent && location.placement==Placement::gpu)
      Dependency(r,{detail::ActorResource::Kind::agent,location.agent,0});
  }
  detail::ActorResources LookupDependencies(const LookupRequest& request) const {
    auto r=Lifecycle(request.id);
    for(const auto& f:request.features)for(auto key:f.keys)RowDependency(r,{f.table,key});return r;
  }
  detail::ActorResources PushDependencies(const PushRequest& request) const {
    auto r=Lifecycle(request.id);
    if(cache_policy_)Dependency(r,{detail::ActorResource::Kind::maintenance,0,0});
    if(auto it=tickets_.find(request.ticket);it!=tickets_.end()) {
      for(const auto& f:it->second.features) {
        Dependency(r,{detail::ActorResource::Kind::optimizer,f.input.table,0});
        for(auto key:f.input.keys)RowDependency(r,{f.input.table,key},true);
      }
    }else r.cacheable=false;
    return r;
  }
  detail::ActorResources ReleaseDependencies(const ReleaseRequest& request) const {
    auto r=Lifecycle(request.id);
    if(cache_policy_)Dependency(r,{detail::ActorResource::Kind::maintenance,0,0});return r;
  }
  detail::ActorResources StateDependencies(const std::vector<LocatedRow>& rows) const {
    detail::ActorResources r;for(const auto& row:rows)RowDependency(r,row.row,true);return r;
  }
  detail::ActorResources MoveDependencies(const MoveRowsRequest& request) const {
    detail::ActorResources r;Dependency(r,{detail::ActorResource::Kind::agent,request.agent,0});
    for(auto row:request.rows)RowDependency(r,row);return r;
  }

  LookupReply Lookup(LookupRequest request) {
    Ready();
    Identity(request.id, Phase::lookup);
    Require(!options_.exclusive_update_learner || request.read_only ||
      request.id.learner == *options_.exclusive_update_learner,
      "this training mode requires read-only worker Pull and synchronous gradient Submit");
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
    detail::PhaseTimer timer(stats_.lookup_prepare_nanoseconds);
    try {
      Require(!draining_, "shard is draining", Error::Code::unavailable);
      restored_.pristine=false;
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
      if(cache_policy_) {
        try {
          std::vector<RowKey> rows;
          for(const auto& f:tickets_.at(name).features)
            for(auto key:f.input.keys)rows.push_back({f.input.table,key});
          cache_policy_->Observe(rows);
        } catch(const std::bad_alloc&) { ++stats_.policy_failures; }
      }
      return reply;
    } catch (...) { tracker_.Abort(request.id); throw; }
  }

  detail::ActorTask<Ack> Push(PushRequest request) {
    Ready();
    Identity(request.id, Phase::push);
    Require(!options_.exclusive_update_learner || request.id.learner == *options_.exclusive_update_learner,
      "ordinary worker Push is disabled for synchronous training");
    const auto completed = tracker_.CompletedAs(request.id);
    if (completed) {
      Require(*completed == Phase::push, "lookup was released without an update");
      ++stats_.duplicate_requests; co_return Ack{};
    }
      auto it = tickets_.find(request.ticket);
      Require(it != tickets_.end(), "unknown lookup ticket");
      auto& ticket = it->second;
      Require(!ticket.read_only && TicketName(request.id) == request.ticket &&
              request.id.learner == ticket.id.learner, "ticket does not belong to update");
      std::map<TableId,RowGradients> cpu;
      std::map<std::pair<AgentId,TableId>,RowGradients> device;
      {
        detail::PhaseTimer timer(stats_.gradient_prepare_nanoseconds);
      std::uint64_t bytes = 0;
      for (const auto& g : request.feature_gradients) Size(bytes, g.values.size(), sizeof(float));
      auto gradients = detail::ReduceGradients(ticket.features, request.feature_gradients, *store_);
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
      }
      try{
        struct DeviceUpdate {UpdateContext context;std::variant<RowGradients,CompactGradients> payload;};
        std::map<AgentId,std::deque<DeviceUpdate>> groups;
        std::uint64_t reference_bytes=0,transferred_bytes=0,compact_commands=0;
        {
        detail::PhaseTimer timer(stats_.gpu_packet_prepare_nanoseconds,!device.empty());
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
        }
        auto send=[this](AgentId agent,DeviceUpdate update){
          detail::PhaseTimer timer(stats_.update_dispatch_nanoseconds);
          if(auto* compact=std::get_if<CompactGradients>(&update.payload))
            return agents_.at(agent)->ApplyCompact(update.context,std::move(*compact));
          return agents_.at(agent)->ApplyHost(update.context,std::get<RowGradients>(std::move(update.payload)));
        };
        // Independent GPU owners can execute one original Push concurrently.
        // Keep at most one control command per home/agent stream outstanding so
        // a later table cannot overtake that stream's earlier command.
        if(!options_.concurrent_agent_updates){
          for(auto& [agent,updates]:groups)for(auto& update:updates)
            co_await AgentCall<Ack>([&]{return send(agent,std::move(update));});
        }else for(;;){
          if(std::all_of(groups.begin(),groups.end(),[](const auto& g){return g.second.empty();}))break;
          std::vector<Result<Ack>> completed;
          {detail::PhaseTimer timer(stats_.agent_wait_nanoseconds);
            completed=co_await actor_.WaitAll<Ack>(groups.size(),[&](auto& pending){
              for(auto& [agent,updates]:groups)if(!updates.empty()){
                auto update=std::move(updates.front());updates.pop_front();
                pending.push_back(send(agent,std::move(update)));
              }
            });}
          for(auto& result:completed)CheckedAgentResult(std::move(result));
        }
        stats_.expanded_gradient_bytes+=reference_bytes;stats_.transferred_gradient_bytes+=transferred_bytes;
        stats_.compact_update_commands+=compact_commands;
      }catch(const std::exception& e){fatal_=e.what();throw detail::Failure({Error::Code::run_failed,fatal_});}
      {detail::PhaseTimer timer(stats_.cpu_optimizer_nanoseconds,!cpu.empty());
        for(const auto& [table,g]:cpu)detail::ApplyLazyAdam(store_->Table(table),g);}
      tracker_.Complete(request.id);
      ticket_payload_bytes_ -= ticket.payload_bytes;
      tickets_.erase(request.ticket); ++stats_.updates;
      co_await MaybeMaintainCache();
      co_return Ack{};
  }

  detail::ActorTask<Ack> Release(ReleaseRequest request) {
    Identity(request.id, Phase::release);
    if(request.cancel_unseen) {
      Require(!options_.exclusive_update_learner || request.id.learner!=*options_.exclusive_update_learner,
        "read cancellation cannot retire the optimizer actor");
      Require(request.partition_epoch==options_.partition_epoch,"read cancellation partition epoch differs",Error::Code::stale_generation);
      const auto name=TicketName(request.id);
      Require(request.ticket.empty() || request.ticket==name,"read cancellation ticket identity differs");
      detail::RequestTracker::State state;
      try {state=tracker_.Reserve(request.id);}
      catch(const detail::Failure& e) {
        if(e.error().code==Error::Code::retired_request){++stats_.duplicate_requests;co_return Ack{};}
        throw;
      }
      if(state==detail::RequestTracker::State::completed) {
        Require(tracker_.CompletedAs(request.id)==Phase::release,"read cancellation cannot undo an applied update");
        ++stats_.duplicate_requests;co_return Ack{};
      }
      auto it=tickets_.find(name);
      Require(it!=tickets_.end() || state==detail::RequestTracker::State::fresh,"pending read cancellation lacks its ticket");
      tracker_.Complete(request.id);
      if(it!=tickets_.end()){ticket_payload_bytes_-=it->second.payload_bytes;tickets_.erase(it);}
      co_await MaybeMaintainCache();co_return Ack{};
    }
    if (tracker_.CompletedAs(request.id)) {
      ++stats_.duplicate_requests; co_return Ack{};
    }
      auto it = tickets_.find(request.ticket);
      Require(it != tickets_.end() && TicketName(request.id) == request.ticket,
              "release does not match a live ticket");
      tracker_.Complete(request.id);
      ticket_payload_bytes_ -= it->second.payload_bytes;
      tickets_.erase(it); co_await MaybeMaintainCache();co_return Ack{};
  }

  detail::ActorTask<std::vector<RowState>> ReadState(const std::vector<LocatedRow>& rows) {
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
        auto state=co_await AgentCall<std::vector<RowState>>([&]{return agents_.at(plan.agent)->ReadBack(std::move(plan));});
        Require(state.size()==1 && state[0].row==row.row,"invalid cache readback");
        result.push_back(std::move(state[0]));
      }else result.push_back(store_->Read(row.row.table,row.row.key));
    }
    co_return result;
  }

  detail::ActorTask<std::vector<LocatedRow>> Move(MoveRowsRequest request){
    detail::PhaseTimer timer(stats_.migration_nanoseconds);
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
      const auto free=co_await AgentCall<std::uint64_t>([&]{return backend->FreeSlots(table);});
      Require(free>=request.budget.reserve_slots && n<=free-request.budget.reserve_slots,
              "migration would exhaust reserved slots",Error::Code::capacity_exceeded);
    }
    MigrationPlan plan{options_.run_id+"/"+std::to_string(options_.shard_id)+"/"+
      std::to_string(next_move_++),request.agent,{},request.destination,bytes};
    plan.reserve_slots=request.destination==Placement::gpu?request.budget.reserve_slots:0;
    std::vector<RowState> states;
    for(auto row:request.rows){
      auto location=Location(row);locations_.try_emplace(RowId(row),location);
      if(request.destination==Placement::gpu){
        ++location.generation;location.agent=request.agent;location.placement=Placement::loading;
        states.push_back(store_->Read(row.table,row.key));
      }plan.rows.push_back(location);
    }
    if(request.destination==Placement::gpu){
      auto loaded=co_await AgentCall<std::vector<LocatedRow>>([&]{return backend->Load(plan,std::move(states));});
      Require(loaded.size()==plan.rows.size(),"cache load response count mismatch",Error::Code::run_failed);
      for(std::size_t i=0;i<loaded.size();++i){
        Require(loaded[i].row==plan.rows[i].row && loaded[i].generation==plan.rows[i].generation &&
          loaded[i].home==options_.shard_id && loaded[i].agent==request.agent && loaded[i].placement==Placement::gpu,
          "cache load response ownership mismatch",Error::Code::run_failed);
      }
      for(const auto& row:loaded)locations_.at(RowId(row.row))=row;
    }else{
      states=co_await AgentCall<std::vector<RowState>>([&]{return backend->ReadBack(plan);});
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
      co_await AgentCall<Ack>([&]{return backend->CommitEviction(plan);});
      for(auto row:plan.rows){
        ++row.generation;row.placement=Placement::cpu;row.agent=0;row.slot=0;
        locations_.at(RowId(row.row))=row;
      }
    }
    stats_.migration_bytes+=bytes;
    actor_.InvalidateDependencies();
    std::vector<LocatedRow> result;result.reserve(request.rows.size());
    for(auto row:request.rows)result.push_back(Location(row));
    co_return result;
  }

  detail::ActorTask<Ack> MaintainCache() {
    const auto& o=cache_policy_->options();
    using Id=std::pair<TableId,Key>;
    std::set<Id> pinned;
    for(const auto& [name,ticket]:tickets_)for(const auto& feature:ticket.features)
      for(auto key:feature.input.keys)pinned.emplace(feature.input.table,key);
    std::vector<LocatedRow> resident;
    for(const auto& [id,location]:locations_)
      if(location.placement==Placement::gpu)resident.push_back(location);
    cache_policy_->SynchronizeResidents(resident);
    using Group=std::pair<AgentId,TableId>;
    std::map<Group,std::vector<RowKey>> candidates;
    for(auto row:cache_policy_->TrackedRows()) {
      if(Location(row).placement!=Placement::cpu)continue;
      if(pinned.contains(RowId(row))) { ++stats_.policy_pinned_skips;continue; }
      const auto bytes=std::uint64_t(store_->Table(row.table).definition.dimension)*12+8;
      if(cache_policy_->AdmissionValue(row,bytes)<=0)continue;
      const auto agent=o.agents[(row.key/options_.shard_count)%o.agents.size()];
      candidates[{agent,row.table}].push_back(row);
    }
    auto remaining=std::min(o.budget.network_bytes,o.budget.staging_bytes);
    auto batches=o.budget.max_inflight_batches;
    const auto migrate=[&](AgentId agent,std::vector<RowKey> rows,Placement destination,
                           std::uint64_t row_bytes)->detail::ActorTask<Ack> {
      const auto bytes=rows.size()*row_bytes;
      Require(bytes<=remaining && batches>0,"cache policy exhausted migration budget");
      const auto begin=std::chrono::steady_clock::now();
      // Keep the owning aggregate explicit across the suspension expression.
      // GCC 11 mishandles a braced temporary with string/vector members here.
      MoveRowsRequest migration{options_.run_id,std::move(rows),agent,destination,
        {remaining,remaining,1,destination==Placement::gpu?o.budget.reserve_slots:0}};
      auto moved=co_await Move(std::move(migration));
      remaining-=bytes;--batches;
      ++stats_.policy_migration_batches;stats_.policy_migration_bytes+=bytes;
      if(destination==Placement::gpu)stats_.policy_admitted_rows+=moved.size();
      else stats_.policy_evicted_rows+=moved.size();
      cache_policy_->ObserveMigration(destination,bytes,
        std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count());
      if(destination==Placement::gpu)for(const auto& row:moved)cache_policy_->Admitted(row);
      co_return Ack{};
    };
    for(auto& [group,rows]:candidates) {
      const auto [agent,table]=group;
      const auto row_bytes=std::uint64_t(store_->Table(table).definition.dimension)*12+8;
      const auto max_rows=std::min<std::uint64_t>(o.max_rows_per_batch,options_.max_request_bytes/row_bytes);
      std::sort(rows.begin(),rows.end(),[&](auto a,auto b) {
        const auto av=cache_policy_->AdmissionValue(a,row_bytes),bv=cache_policy_->AdmissionValue(b,row_bytes);
        return av==bv?RowId(a)<RowId(b):av>bv;
      });
      std::size_t next=0;
      while(next<rows.size()) {
        if(!batches || remaining<row_bytes || !max_rows) {
          stats_.policy_budget_skips+=rows.size()-next;break;
        }
        if(cache_policy_->AdmissionValue(rows[next],row_bytes)<=0)break;
        const auto free=co_await AgentCall<std::uint64_t>([&]{return agents_.at(agent)->FreeSlots(table);});
        const auto usable=free>o.budget.reserve_slots?free-o.budget.reserve_slots:0;
        if(usable) {
          const auto count=std::min<std::uint64_t>({usable,remaining/row_bytes,
            max_rows,rows.size()-next});
          std::vector<RowKey> admit;
          for(std::size_t n=0;n<count && cache_policy_->AdmissionValue(rows[next+n],row_bytes)>0;++n)
            admit.push_back(rows[next+n]);
          if(admit.empty())break;
          const auto moved=admit.size();co_await migrate(agent,std::move(admit),Placement::gpu,row_bytes);
          next+=moved;continue;
        }
        if(batches<2) { stats_.policy_budget_skips+=rows.size()-next;break; }
        std::vector<LocatedRow> victims;
        for(const auto& row:resident) {
          const auto current=Location(row.row);
          if(current.placement!=Placement::gpu || current.generation!=row.generation ||
            row.agent!=agent || row.row.table!=table)continue;
          if(pinned.contains(RowId(row.row))) { ++stats_.policy_pinned_skips;continue; }
          if(cache_policy_->CanReplace(row))victims.push_back(row);
        }
        std::sort(victims.begin(),victims.end(),[&](const auto& a,const auto& b) {
          const auto av=cache_policy_->RetainedValue(a.row),bv=cache_policy_->RetainedValue(b.row);
          return av==bv?RowId(a.row)<RowId(b.row):av<bv;
        });
        const auto deficit=o.budget.reserve_slots>free?o.budget.reserve_slots-free:0;
        if(deficit>=victims.size() || deficit>=max_rows)break;
        const auto limit=std::min<std::uint64_t>({rows.size()-next,
          max_rows-deficit,victims.size()-deficit});
        if(deficit+2>remaining/row_bytes) {
          stats_.policy_budget_skips+=rows.size()-next;break;
        }
        std::size_t replace=0;double saved=0,cost=0;
        for(std::size_t n=0;n<deficit;++n)
          cost+=cache_policy_->RetainedValue(victims[n].row)+cache_policy_->WritebackCost(row_bytes);
        for(std::size_t n=1;n<=limit;++n) {
          if(deficit+2*n>remaining/row_bytes)break;
          const auto value=cache_policy_->AdmissionValue(rows[next+n-1],row_bytes);
          if(value<=0)break;
          saved+=value;
          cost+=cache_policy_->RetainedValue(victims[deficit+n-1].row)+cache_policy_->WritebackCost(row_bytes);
          if(saved>cost*(1+o.replacement_margin))replace=n;
        }
        if(!replace)break;
        std::vector<RowKey> evict;
        for(std::size_t n=0;n<deficit+replace;++n)evict.push_back(victims[n].row);
        co_await migrate(agent,std::move(evict),Placement::cpu,row_bytes);
        // Requery free slots and revalidate the reserve in Move. Another home
        // shard may have claimed capacity; these rows safely remain CPU-owned.
      }
    }
    co_return Ack{};
  }
  detail::ActorTask<Ack> MaybeMaintainCache() {
    if(!cache_policy_ || draining_ || !fatal_.empty() || !cache_policy_->CompleteRequest())co_return Ack{};
    co_await actor_.Exclusive();
    Ready();
    ++stats_.policy_windows;
    detail::PhaseTimer timer(stats_.cache_maintenance_nanoseconds);
    try { co_await MaintainCache(); }
    catch(const detail::Failure& e) {
      ++stats_.policy_failures;
      if(!fatal_.empty() || e.error().code==Error::Code::run_failed)throw;
    }
    catch(const std::bad_alloc&) { ++stats_.policy_failures; }
    co_return Ack{};
  }

  detail::ActorTask<SnapshotMetadata> Save(const std::string& id) {
    Ready();
    Require(!id.empty() && id.size() <= 128 && id != "." && id != ".." &&
      std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
      }), "invalid snapshot ID");
    const bool already_saved = draining_ && saved_checkpoint_ == id;
    draining_ = true;
    if (!already_saved) saved_checkpoint_.clear();
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
      MoveRowsRequest migration{options_.run_id,std::move(rows),agent,Placement::cpu,
        {options_.max_request_bytes,options_.max_request_bytes,1024,0}};
      co_await Move(std::move(migration));
    }
    const auto file = options_.snapshot_directory / id /
        ("shard-" + std::to_string(options_.shard_id) + ".bin");
    if (!already_saved) detail::WriteSnapshot(file, options_.run_id, *store_);
    SnapshotMetadata result{id, options_.shard_id, options_.partition_epoch, {file.string()}, {}};
    for (const auto& [table, t] : store_->tables())
      result.optimizer_contexts.push_back({{options_.run_id, 0, 0, Phase::maintenance},
        options_.shard_id, table, t.optimizer_sequence,
        t.definition.optimizer.learning_rate, t.beta1_power, t.beta2_power});
    saved_checkpoint_=id;
    co_return result;
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
  std::map<AgentId,std::shared_ptr<CacheBackend>> agents_;
  AgentId fallback_agent_=0;
  AccessRoute fallback_route_=AccessRoute::cpu_pool;
  std::shared_ptr<AccessPlanner> planner_;
  std::unique_ptr<detail::ReusePolicy> cache_policy_;
  std::map<std::pair<TableId,Key>,LocatedRow> locations_;
  std::uint64_t next_move_=0;
  std::string fatal_;
  bool draining_ = false;
  std::string saved_checkpoint_;
  RestoredSnapshot restored_;
  std::uint64_t ticket_payload_bytes_ = 0;
  // Last member: accepted continuations and completion workers drain before
  // the store, tickets, backends and error observer's state are destroyed.
  detail::ResumableActor actor_;
};

CpuParameterService::CpuParameterService(RuntimeOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
CpuParameterService::~CpuParameterService() {
  // Accepted jobs capture this and access impl_; keep the pointer intact until
  // their actor continuations and completion workers have all finished.
  impl_->actor_.Shutdown();
}
std::future<Result<Ack>> CpuParameterService::CreateTable(TableDefinition table) {
  return impl_->Submit<Ack>([this,table]()->detail::ActorTask<Ack>{
    impl_->Ready();const bool existed=impl_->store_->tables().contains(table.id);
    Require(existed || !impl_->draining_, "cannot add a table while parameter admission is paused",
      Error::Code::unavailable);
    impl_->store_->Create(table);
    if(!existed)impl_->restored_.pristine=false;
    for(const auto& [id,backend]:impl_->agents_)
      co_await impl_->AgentCall<Ack>([&]{return backend->Define(table);});
    co_return Ack{};
  });
}
std::future<Result<LookupReply>> CpuParameterService::Lookup(LookupRequest request) {
  std::uint64_t bytes=request.features.capacity()*sizeof(FeatureInput);
  for(const auto& f:request.features)
    bytes+=(f.keys.capacity()+f.offsets.capacity()+f.valid_lengths.capacity())*sizeof(std::uint64_t);
  auto input=std::make_shared<LookupRequest>(std::move(request));
  return impl_->SubmitIndependent<LookupReply>([this,input]{return impl_->Lookup(std::move(*input));},
    [this,input]{return impl_->LookupDependencies(*input);},bytes);
}
std::future<Result<Ack>> CpuParameterService::Push(PushRequest request) {
  std::uint64_t bytes=request.feature_gradients.capacity()*sizeof(TensorBlock);
  for(const auto& g:request.feature_gradients)
    bytes+=g.values.capacity()*sizeof(float)+g.shape.capacity()*sizeof(std::uint64_t);
  auto input=std::make_shared<PushRequest>(std::move(request));
  return impl_->SubmitIndependent<Ack>([this,input]{return impl_->Push(std::move(*input));},
    [this,input]{return impl_->PushDependencies(*input);},bytes);
}
std::future<Result<ParameterTrainingMetadata>> CpuParameterService::TrainingMetadata() {
  return impl_->actor_.Observe<ParameterTrainingMetadata>([this] {
    impl_->Ready(); ParameterTrainingMetadata result;
    result.provenance = impl_->restored_;
    result.provenance.run = impl_->options_.run_id;
    result.provenance.shard = impl_->options_.shard_id;
    result.provenance.shard_count = impl_->options_.shard_count;
    result.provenance.partition_epoch = impl_->options_.partition_epoch;
    result.requests = impl_->tracker_.Frontiers();
    for (const auto& [id, table] : impl_->store_->tables()) {
      result.tables.push_back(table.definition);
      result.optimizer_contexts.push_back({{impl_->options_.run_id, 1023, 0, Phase::maintenance},
        impl_->options_.shard_id, id, table.optimizer_sequence, table.definition.optimizer.learning_rate,
        table.beta1_power, table.beta2_power});
    }
    return result;
  });
}
std::future<Result<Ack>> CpuParameterService::Release(ReleaseRequest request) {
  auto input=std::make_shared<ReleaseRequest>(std::move(request));
  return impl_->SubmitIndependent<Ack>([this,input]{return impl_->Release(std::move(*input));},
    [this,input]{return impl_->ReleaseDependencies(*input);});
}
std::future<Result<std::vector<RowState>>> CpuParameterService::ReadState(std::vector<LocatedRow> rows) {
  auto bytes=rows.capacity()*sizeof(LocatedRow);auto input=std::make_shared<std::vector<LocatedRow>>(std::move(rows));
  return impl_->SubmitIndependent<std::vector<RowState>>([this,input]{return impl_->ReadState(*input);},
    [this,input]{return impl_->StateDependencies(*input);},bytes);
}
std::future<Result<Ack>> CpuParameterService::CompleteMigration(MigrationCompletion) {
  return impl_->Submit<Ack>([]()->Result<Ack>{
    return Error{Error::Code::unavailable,"GPU cache agent is not attached"};
  });
}
std::future<Result<SnapshotMetadata>> CpuParameterService::DrainAndFlush(std::string id) {
  return impl_->Submit<SnapshotMetadata>([this,id=std::move(id)]{return impl_->Save(id);});
}
std::future<Result<Ack>> CpuParameterService::Restore(std::filesystem::path file) {
  return impl_->Submit<Ack>([this,file=std::move(file)]{
    impl_->Ready();
    Require(!impl_->restored_.loaded && impl_->store_->empty() && impl_->tickets_.empty(),"restore requires a fresh service");
    RestoredSnapshot provenance;
    auto store=detail::ReadSnapshot(file,impl_->options_.run_id,
      impl_->options_.shard_id,impl_->options_.shard_count,
      impl_->options_.partition_epoch,impl_->options_.state_bytes,&provenance);
    impl_->store_=std::move(store);impl_->restored_=std::move(provenance);return Ack{};
  });
}
std::future<Result<RestoredSnapshot>> CpuParameterService::RestoredFrom() {
  return impl_->Submit<RestoredSnapshot>([this]{
    impl_->Ready();auto metadata=impl_->restored_;
    metadata.run=impl_->options_.run_id;metadata.shard=impl_->options_.shard_id;
    metadata.shard_count=impl_->options_.shard_count;metadata.partition_epoch=impl_->options_.partition_epoch;
    return metadata;
  });
}
std::future<Result<Ack>> CpuParameterService::RestoreReplicaBaseline(std::filesystem::path file) {
  return impl_->Submit<Ack>([this,file=std::move(file)] {
    impl_->Ready();Require(impl_->draining_ && !impl_->restored_.loaded && impl_->store_->empty() && impl_->tickets_.empty(),
      "replica restoration requires a fresh paused service");
    impl_->store_=detail::ReadSnapshot(file,impl_->options_.run_id,impl_->options_.shard_id,
      impl_->options_.shard_count,impl_->options_.partition_epoch,impl_->options_.state_bytes,&impl_->restored_,true);
    return Ack{};
  });
}
std::future<Result<Ack>> CpuParameterService::ImportReplicaUpdate(ReplicatedUpdate image) {
  return impl_->Submit<Ack>([this,image=std::move(image)] {
    impl_->Ready();Require(impl_->draining_ && impl_->restored_.loaded && impl_->tickets_.empty() && impl_->agents_.empty() &&
      image.run==impl_->options_.run_id && image.shard==impl_->options_.shard_id && image.shards==impl_->options_.shard_count &&
      image.epoch==impl_->options_.partition_epoch && image.round!=UINT64_MAX && image.contexts.size()==impl_->store_->tables().size(),
      "replica image restoration identity/state mismatch");
    std::set<TableId> tables;
    for(const auto& c:image.contexts) {
      auto& t=impl_->store_->Table(c.table);
      Require(tables.insert(c.table).second && c.shard==image.shard && t.optimizer_sequence==image.round &&
        c.optimizer_sequence==image.round+1 && c.learning_rate==t.definition.optimizer.learning_rate &&
        c.beta1_power==t.beta1_power*t.definition.optimizer.beta1 && c.beta2_power==t.beta2_power*t.definition.optimizer.beta2,
        "replica optimizer image is not the next update");
    }
    std::set<std::pair<TableId,Key>> seen;
    for(const auto& r:image.rows) {
      auto& t=impl_->store_->Table(r.row.table);
      Require(tables.contains(r.row.table) && r.row.key%image.shards==image.shard &&
        seen.emplace(r.row.table,r.row.key).second && r.weights.size()==t.definition.dimension &&
        r.first_moment.size()==r.weights.size() && r.second_moment.size()==r.weights.size(),"invalid replica row layout");
      const auto slot=impl_->store_->Ensure(r.row.table,r.row.key);
      Require(*t.At(slot).updates!=UINT64_MAX && r.row_updates==*t.At(slot).updates+1,"replica row update count is not contiguous");
      for(const auto* a:{&r.weights,&r.first_moment,&r.second_moment})for(auto f:*a)Require(std::isfinite(f),"non-finite replica image");
      for(auto f:r.second_moment)Require(f>=0,"negative replicated second moment");
    }
    for(const auto& r:image.rows)impl_->store_->Overwrite(r);
    for(const auto& c:image.contexts){auto& t=impl_->store_->Table(c.table);t.optimizer_sequence=c.optimizer_sequence;t.beta1_power=c.beta1_power;t.beta2_power=c.beta2_power;}
    impl_->tracker_.RestoreFrontiers(image.requests);impl_->restored_.pristine=false;
    return Ack{};
  });
}
std::future<Result<Ack>> CpuParameterService::InitializeRequestFrontiers(std::vector<RequestFrontier> progress) {
  return impl_->Submit<Ack>([this,progress=std::move(progress)] {
    impl_->Ready();Require(impl_->tickets_.empty() && impl_->stats_.lookups==0 && impl_->stats_.updates==0,
      "request frontiers must be restored before serving parameter traffic");
    impl_->tracker_.RestoreFrontiers(progress);return Ack{};
  });
}
std::future<Result<Ack>> CpuParameterService::CertifyReadFrontiers(std::vector<RequestFrontier> progress) {
  return impl_->Submit<Ack>([this,progress=std::move(progress)] {
    impl_->Ready();Require(impl_->options_.exclusive_update_learner==1023 && impl_->tickets_.empty(),
      "read progress certification requires a quiescent synchronous group");
    impl_->tracker_.CertifyReads(progress);return Ack{};
  });
}
std::future<Result<Ack>> CpuParameterService::InitializeReplicaRows(std::vector<std::pair<TableId,std::uint64_t>> counts) {
  return impl_->Submit<Ack>([this,counts=std::move(counts)] {
    impl_->Ready();Require(impl_->tickets_.empty() && impl_->agents_.empty() && impl_->stats_.lookups==0 && impl_->stats_.updates==0,
      "replica baseline rows must be initialized before serving");
    for(const auto& [table,total]:counts) {
      Require(total>0 && total<=(1ULL<<34),"invalid replica table cardinality");
      const auto previous=impl_->store_->rows();
      for(std::uint64_t key=impl_->options_.shard_id;key<total;key+=impl_->options_.shard_count)impl_->store_->Ensure(table,key);
      if(impl_->store_->rows()!=previous)impl_->restored_.pristine=false;
    }
    return Ack{};
  });
}
std::future<Result<Ack>> CpuParameterService::Resume() {
  return impl_->Submit<Ack>([this]{impl_->Ready();impl_->draining_=false;return Ack{};});
}
std::future<Result<SnapshotChunk>> CpuParameterService::ReadCheckpoint(
    CheckpointIdentity identity,std::uint64_t offset,std::uint32_t max_bytes) {
  return impl_->Submit<SnapshotChunk>([this,identity=std::move(identity),offset,max_bytes]{
    impl_->Ready();
    Require(identity.run==impl_->options_.run_id &&
      identity.partition_epoch==impl_->options_.partition_epoch,
      "checkpoint run or partition epoch mismatch");
    Require(impl_->draining_ && !impl_->saved_checkpoint_.empty() &&
      identity.snapshot_id==impl_->saved_checkpoint_,"checkpoint is not the current frozen save",
      Error::Code::unavailable);
    return detail::ReadSnapshotChunk(impl_->options_.snapshot_directory/identity.snapshot_id/
      ("shard-"+std::to_string(impl_->options_.shard_id)+".bin"),offset,max_bytes);
  });
}
std::future<Result<Ack>> CpuParameterService::ResumeCheckpoint(CheckpointIdentity identity) {
  return impl_->Submit<Ack>([this,identity=std::move(identity)]{
    impl_->Ready();
    Require(identity.run==impl_->options_.run_id &&
      identity.partition_epoch==impl_->options_.partition_epoch,
      "checkpoint run or partition epoch mismatch");
    Require(!impl_->saved_checkpoint_.empty() && identity.snapshot_id==impl_->saved_checkpoint_,
      "checkpoint resume requires the current successful save",Error::Code::unavailable);
    impl_->draining_=false;return Ack{};
  });
}
std::future<Result<RuntimeStats>> CpuParameterService::Stats() {
  return impl_->actor_.Observe<RuntimeStats>([this]{
    auto s=impl_->stats_;auto scheduling=impl_->actor_.Snapshot();
    s.rows=impl_->store_->rows();s.state_bytes=impl_->store_->bytes();
    s.live_tickets=impl_->tickets_.size();s.ticket_payload_bytes=impl_->ticket_payload_bytes_;
    s.policy_tracked_rows=impl_->cache_policy_?impl_->cache_policy_->tracked_rows():0;
    for(const auto& [key,loc]:impl_->locations_)if(loc.placement==Placement::gpu)++s.cached_rows;
    s.queued_payload_bytes=scheduling.queued_payload;s.rejected_operations=scheduling.rejected;
    s.peak_queued_operations=scheduling.peak_queued;s.executed_operations=scheduling.completed;
    s.queue_wait_nanoseconds=scheduling.queue_wait_ns;s.executor_busy_nanoseconds=scheduling.busy_ns;
    s.active_operations=scheduling.active;s.waiting_operations=scheduling.waiting;
    s.active_payload_bytes=scheduling.active_payload;s.dependency_payload_bytes=scheduling.dependency_bytes;
    s.dependency_prepare_nanoseconds=scheduling.dependency_prepare_ns;
    s.dependency_rebuilds=scheduling.dependency_rebuilds;s.dependency_cache_hits=scheduling.dependency_cache_hits;
    s.queued_observations=scheduling.queued_observations;s.completed_observations=scheduling.completed_observations;
    s.rejected_observations=scheduling.rejected_observations;s.dependency_budget_waits=scheduling.dependency_budget_waits;
    return s;
  });
}
std::future<Result<Ack>> CpuParameterService::AttachAgent(AgentId id,std::shared_ptr<CacheBackend> backend) {
  return impl_->Submit<Ack>([this,id,backend=std::move(backend)]()->detail::ActorTask<Ack>{
    impl_->Ready();Require(bool(backend) && !impl_->agents_.contains(id),"invalid/duplicate cache agent");
    for(const auto& [table,state]:impl_->store_->tables())
      co_await impl_->AgentCall<Ack>([&]{return backend->Define(state.definition);});
    impl_->agents_.emplace(id,backend);co_return Ack{};
  });
}
std::future<Result<Ack>> CpuParameterService::ConfigureAccess(AgentId agent,AccessRoute route,
                                                           std::shared_ptr<AccessPlanner> planner) {
  return impl_->Submit<Ack>([this,agent,route,planner=std::move(planner)]{
    impl_->Ready();Require(route==AccessRoute::cpu_pool || route==AccessRoute::transfer_unique,"invalid fallback route");
    Require((!planner && route==AccessRoute::cpu_pool) || impl_->agents_.contains(agent),"fallback GPU agent is not attached");
    impl_->fallback_agent_=agent;impl_->fallback_route_=route;impl_->planner_=planner;return Ack{};
  });
}
std::future<Result<Ack>> CpuParameterService::ConfigureCachePolicy(CachePolicyOptions options) {
  return impl_->Submit<Ack>([this,options=std::move(options)]() mutable {
    impl_->Ready();auto policy=std::make_unique<detail::ReusePolicy>(std::move(options));
    for(auto agent:policy->options().agents)
      Require(impl_->agents_.contains(agent),"cache policy agent is not attached");
    if(policy->options().mode==CacheAdmissionMode::manual)policy.reset();
    impl_->cache_policy_=std::move(policy);impl_->actor_.InvalidateDependencies();return Ack{};
  });
}
std::future<Result<std::vector<LocatedRow>>> CpuParameterService::Locations(std::vector<RowKey> rows) {
  auto bytes=rows.capacity()*sizeof(RowKey);auto input=std::make_shared<std::vector<RowKey>>(std::move(rows));
  return impl_->SubmitIndependent<std::vector<LocatedRow>>([this,input]{
    impl_->Ready();std::vector<LocatedRow> out;out.reserve(input->size());
    for(auto row:*input){
      Require(row.key%impl_->options_.shard_count==impl_->options_.shard_id &&
        impl_->store_->Table(row.table).index.contains(row.key),"unknown/wrongly routed row");
      out.push_back(impl_->Location(row));
    }return out;
  },[this,input]{detail::ActorResources resources;for(auto row:*input)impl_->RowDependency(resources,row);return resources;},bytes);
}
std::future<Result<std::vector<LocatedRow>>> CpuParameterService::MoveRows(MoveRowsRequest request) {
  auto bytes=request.rows.capacity()*sizeof(RowKey);auto input=std::make_shared<MoveRowsRequest>(std::move(request));
  return impl_->SubmitIndependent<std::vector<LocatedRow>>([this,input]{return impl_->Move(std::move(*input));},
    [this,input]{return impl_->MoveDependencies(*input);},bytes);
}
}  // namespace eps
