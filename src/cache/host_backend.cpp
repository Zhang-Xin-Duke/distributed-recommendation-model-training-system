#include "eps/cache_backend.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include "common/executor.hpp"

namespace eps {
namespace {
using RowId=std::pair<TableId,Key>;
RowId Id(RowKey row){return {row.table,row.key};}
}
struct HostCacheBackend::Impl {
  struct Entry {LocatedRow location;RowState state;};
  Impl(AgentId a,std::uint64_t b):agent(a),budget(b){}
  Entry& Find(const LocatedRow& loc){
    auto it=rows.find(Id(loc.row));
    detail::Require(it!=rows.end() && it->second.location.generation==loc.generation &&
      it->second.location.slot==loc.slot && loc.agent==agent,"cache row generation mismatch",Error::Code::stale_generation);
    return it->second;
  }
  AgentId agent;std::uint64_t budget,bytes=0,next_slot=0;
  std::map<TableId,TableDefinition> tables;
  std::map<RowId,Entry> rows;
  // Last member: executor stops before its referenced state is destroyed.
  detail::OrderedExecutor executor;
};
HostCacheBackend::HostCacheBackend(AgentId a,std::uint64_t b):impl_(std::make_unique<Impl>(a,b)){}
HostCacheBackend::~HostCacheBackend()=default;
std::future<Result<std::uint64_t>> HostCacheBackend::FreeSlots(TableId table){
  return impl_->executor.Submit<std::uint64_t>([this,table]{
    auto bytes=std::uint64_t(impl_->tables.at(table).dimension)*12+8;
    return (impl_->budget-impl_->bytes)/bytes;
  });
}
std::future<Result<Ack>> HostCacheBackend::Define(TableDefinition t){
  return impl_->executor.Submit<Ack>([this,t]{
    detail::Require(t.dimension>0 && t.dimension<=65536,"invalid cache table dimension");
    auto [it,inserted]=impl_->tables.emplace(t.id,t);
    detail::Require(inserted||it->second==t,"cache table configuration mismatch");return Ack{};
  });
}
std::future<Result<std::vector<LocatedRow>>> HostCacheBackend::Load(MigrationPlan plan,std::vector<RowState> states){
  return impl_->executor.Submit<std::vector<LocatedRow>>([this,plan=std::move(plan),states=std::move(states)]{
    detail::Require(plan.agent==impl_->agent && plan.rows.size()==states.size() &&
      plan.destination==Placement::gpu,"invalid cache load plan");
    std::uint64_t required=0;std::set<RowId> unique;
    for(std::size_t i=0;i<states.size();++i){
      const auto& state=states[i];const auto& loc=plan.rows[i];
      detail::Require(state.row==loc.row && unique.insert(Id(state.row)).second &&
        !impl_->rows.contains(Id(state.row)),"cache load contains an existing/duplicate row");
      const auto dim=impl_->tables.at(state.row.table).dimension;
      detail::Require(state.weights.size()==dim && state.first_moment.size()==dim && state.second_moment.size()==dim,
                      "cache state dimension mismatch");
      for(auto* values:{&state.weights,&state.first_moment,&state.second_moment})
        for(auto v:*values)detail::Require(std::isfinite(v),"non-finite cache state");
      for(auto v:state.second_moment)detail::Require(v>=0,"negative second moment");
      required+=std::uint64_t(dim)*12+8;
    }
    detail::Require(required<=impl_->budget-impl_->bytes,"cache state budget exhausted",Error::Code::capacity_exceeded);
    auto next=impl_->rows; // Reference backend provides atomic publication on allocation failure.
    std::vector<LocatedRow> locations;locations.reserve(states.size());
    for(std::size_t i=0;i<states.size();++i){
      auto location=plan.rows[i];location.agent=impl_->agent;location.slot=impl_->next_slot+i;
      location.placement=Placement::gpu;
      next.emplace(Id(states[i].row),Impl::Entry{location,states[i]});locations.push_back(location);
    }
    impl_->rows.swap(next);impl_->next_slot+=states.size();impl_->bytes+=required;return locations;
  });
}
std::future<Result<std::vector<TensorBlock>>> HostCacheBackend::GatherHost(
    std::vector<AgentWork> work,std::vector<FeatureInput> layouts){
  return impl_->executor.Submit<std::vector<TensorBlock>>([this,work=std::move(work),layouts=std::move(layouts)]{
    std::vector<TensorBlock> output;
    for(const auto& task:work){
      auto it=std::find_if(layouts.begin(),layouts.end(),[&](const auto& f){return f.feature==task.feature;});
      detail::Require(it!=layouts.end() && task.agent==impl_->agent,"invalid cache feature work");
      const auto& f=*it;const auto dim=impl_->tables.at(f.table).dimension;
      const bool transfer=task.route==AccessRoute::transfer_unique;
      detail::Require(transfer || task.route==AccessRoute::resident_gpu,"invalid gather route");
      detail::Require(transfer?task.transferred_weights.size()==task.rows.size()*dim:
        task.transferred_weights.empty(),"invalid transferred weight layout");
      detail::Require(task.occurrence_to_unique.size()==f.keys.size() &&
        !f.offsets.empty() && f.offsets.front()==0 && f.offsets.back()==f.keys.size() &&
        std::is_sorted(f.offsets.begin(),f.offsets.end()),"invalid cached feature layout");
      TensorBlock tensor{f.feature,{f.pooling==Pooling::sequence?f.keys.size():f.offsets.size()-1,dim},{}};
      tensor.values.resize(tensor.shape[0]*dim,0);
      for(std::size_t b=0;b+1<f.offsets.size();++b)for(auto i=f.offsets[b];i<f.offsets[b+1];++i){
        const auto index=task.occurrence_to_unique[i];if(index==UINT64_MAX)continue;
        detail::Require(index<task.rows.size() && task.rows[index].row==RowKey{f.table,f.keys[i]},"invalid cached occurrence map");
        const float* weights;
        if(transfer){
          detail::Require(task.rows[index].placement==Placement::cpu,"transferred row is not CPU-owned");
          weights=task.transferred_weights.data()+index*dim;
        }else weights=impl_->Find(task.rows[index]).state.weights.data();
        const auto row=f.pooling==Pooling::sequence?i:b;
        for(std::uint32_t d=0;d<dim;++d)tensor.values[row*dim+d]+=weights[d];
      }
      output.push_back(std::move(tensor));
    }
    return output;
  });
}
std::future<Result<Ack>> HostCacheBackend::ApplyHost(UpdateContext c,RowGradients gradients){
  return impl_->executor.Submit<Ack>([this,c,gradients=std::move(gradients)]{
    const auto& t=impl_->tables.at(c.table);const auto dim=t.dimension;const auto& o=t.optimizer;
    detail::Require(gradients.values.size()==gradients.rows.size()*dim &&
      c.beta1_power>=0 && c.beta1_power<1 && c.beta2_power>=0 && c.beta2_power<1 &&
      std::isfinite(c.learning_rate) && c.learning_rate>0,"invalid cache optimizer context");
    std::set<RowId> unique;std::vector<RowState> updated;updated.reserve(gradients.rows.size());
    const auto lr=c.learning_rate*std::sqrt(1-c.beta2_power)/(1-c.beta1_power);
    for(std::size_t i=0;i<gradients.rows.size();++i){
      auto key=gradients.rows[i];detail::Require(key.table==c.table && unique.insert(Id(key)).second,"duplicate cached gradient row");
      auto it=impl_->rows.find(Id(key));
      detail::Require(it!=impl_->rows.end() && it->second.location.home==c.shard,"cached gradient routed to wrong owner");
      auto row=it->second.state;
      for(std::uint32_t d=0;d<dim;++d){
        auto g=gradients.values[i*dim+d];
        row.first_moment[d]=o.beta1*row.first_moment[d]+(1-o.beta1)*g;
        row.second_moment[d]=o.beta2*row.second_moment[d]+(1-o.beta2)*g*g;
        row.weights[d]-=lr*row.first_moment[d]/(std::sqrt(row.second_moment[d])+o.epsilon);
        detail::Require(std::isfinite(g)&&std::isfinite(row.weights[d])&&std::isfinite(row.first_moment[d])&&
          std::isfinite(row.second_moment[d]),"cached optimizer overflow");
      }
      ++row.row_updates;updated.push_back(std::move(row));
    }
    for(auto& row:updated)impl_->rows.at(Id(row.row)).state=std::move(row);
    return Ack{};
  });
}
std::future<Result<std::vector<RowState>>> HostCacheBackend::ReadBack(MigrationPlan plan){
  return impl_->executor.Submit<std::vector<RowState>>([this,plan=std::move(plan)]{
    std::vector<RowState> rows;rows.reserve(plan.rows.size());
    for(const auto& loc:plan.rows)rows.push_back(impl_->Find(loc).state);
    return rows;
  });
}
std::future<Result<Ack>> HostCacheBackend::CommitEviction(MigrationPlan plan){
  return impl_->executor.Submit<Ack>([this,plan=std::move(plan)]{
    std::set<RowId> unique;
    for(const auto& loc:plan.rows){impl_->Find(loc);detail::Require(unique.insert(Id(loc.row)).second,"duplicate eviction row");}
    for(const auto& loc:plan.rows){
      impl_->bytes-=std::uint64_t(impl_->tables.at(loc.row.table).dimension)*12+8;
      impl_->rows.erase(Id(loc.row));
    }return Ack{};
  });
}
}  // namespace eps
