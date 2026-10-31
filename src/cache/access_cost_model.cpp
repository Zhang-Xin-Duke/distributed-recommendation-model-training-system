#include "eps/cost_planner.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include "common/error.hpp"

namespace eps {
struct CostBasedPlanner::Impl {
  explicit Impl(PlannerOptions o):options(o) {
    detail::Require(o.moving_average_alpha>0 && o.moving_average_alpha<=1 &&
      o.switch_margin>=0 && o.switch_margin<1 && o.minimum_observations>0 &&
      o.shard_count>0 && o.shard<o.shard_count,"invalid cost planner options");
  }
  struct Estimate {
    double cpu=0,gpu=0;
    std::uint64_t samples=0;
    AccessRoute selected=AccessRoute::cpu_pool;
  };
  PlannerOptions options;
  mutable std::mutex mutex;
  mutable std::map<std::pair<TableId,std::uint32_t>,Estimate> estimates;
  std::map<TableId,std::uint64_t> free_slots;
  mutable std::uint64_t next_plan=0;
};
CostBasedPlanner::CostBasedPlanner(PlannerOptions o):impl_(std::make_unique<Impl>(o)) {}
CostBasedPlanner::~CostBasedPlanner()=default;
void CostBasedPlanner::Observe(AccessObservation o) {
  detail::Require(std::isfinite(o.cpu_route_microseconds) && o.cpu_route_microseconds>=0 &&
    std::isfinite(o.gpu_route_microseconds) && o.gpu_route_microseconds>=0 &&
    std::isfinite(o.queue_microseconds) && o.queue_microseconds>=0 &&
    o.unique_rows<=o.occurrences,"invalid access observation");
  std::lock_guard lock(impl_->mutex);
  auto& e=impl_->estimates[{o.table,o.feature}];
  const auto alpha=e.samples?impl_->options.moving_average_alpha:1;
  e.cpu=(1-alpha)*e.cpu+alpha*o.cpu_route_microseconds;
  e.gpu=(1-alpha)*e.gpu+alpha*o.gpu_route_microseconds;
  ++e.samples;
}
AccessRoute CostBasedPlanner::SelectRoute(const FeatureInput& feature) const {
  if(feature.pooling==Pooling::sequence)return AccessRoute::transfer_unique;
  std::lock_guard lock(impl_->mutex);
  auto& e=impl_->estimates[{feature.table,feature.feature}];
  if(e.samples<impl_->options.minimum_observations)return AccessRoute::cpu_pool;
  const auto margin=1-impl_->options.switch_margin;
  if(e.selected==AccessRoute::cpu_pool && e.gpu<e.cpu*margin)e.selected=AccessRoute::transfer_unique;
  else if(e.selected==AccessRoute::transfer_unique && e.cpu<e.gpu*margin)e.selected=AccessRoute::cpu_pool;
  return e.selected;
}
void CostBasedPlanner::SetFreeSlots(TableId table,std::uint64_t slots) {
  std::lock_guard lock(impl_->mutex);impl_->free_slots[table]=slots;
}
std::vector<MigrationPlan> CostBasedPlanner::PlanMigration(
    std::vector<AdmissionCandidate> candidates,MigrationBudget budget) const {
  std::lock_guard lock(impl_->mutex);
  auto score=[](const AdmissionCandidate& c) {
    return (c.expected_accesses*c.saved_microseconds_per_access-
            c.loading_microseconds-c.writeback_microseconds)/c.state_bytes;
  };
  std::erase_if(candidates,[&](const auto& c) {
    return !c.state_bytes || c.row.key%impl_->options.shard_count!=impl_->options.shard ||
      !std::isfinite(c.expected_accesses) || c.expected_accesses<0 ||
      !std::isfinite(c.saved_microseconds_per_access) || c.saved_microseconds_per_access<0 ||
      !std::isfinite(c.loading_microseconds) || c.loading_microseconds<0 ||
      !std::isfinite(c.writeback_microseconds) || c.writeback_microseconds<0 ||
      !std::isfinite(score(c)) || score(c)<=0;
  });
  std::sort(candidates.begin(),candidates.end(),[&](const auto& a,const auto& b) {
    if(score(a)!=score(b))return score(a)>score(b);
    return std::pair(a.row.table,a.row.key)<std::pair(b.row.table,b.row.key);
  });
  auto capacity=impl_->free_slots;
  std::map<TableId,std::size_t> group;
  std::set<std::pair<TableId,Key>> selected;
  std::vector<MigrationPlan> plans;
  auto bytes=std::min(budget.network_bytes,budget.staging_bytes);
  for(const auto& c:candidates) {
    if(c.state_bytes>bytes || capacity[c.row.table]<=budget.reserve_slots ||
       selected.contains({c.row.table,c.row.key}))continue;
    if(!group.contains(c.row.table)) {
      if(plans.size()>=budget.max_inflight_batches)continue;
      group[c.row.table]=plans.size();
      plans.push_back({"proposal-"+std::to_string(impl_->options.shard)+"-"+
        std::to_string(impl_->next_plan++),impl_->options.agent,{},Placement::gpu,0});
    }
    auto& plan=plans[group.at(c.row.table)];
    plan.rows.push_back({c.row,impl_->options.shard,Placement::cpu,impl_->options.agent,0,0});
    plan.estimated_bytes+=c.state_bytes;bytes-=c.state_bytes;
    --capacity[c.row.table];selected.emplace(c.row.table,c.row.key);
  }
  return plans;
}
}  // namespace eps
