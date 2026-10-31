#include "cache/reuse_policy.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include "common/error.hpp"

namespace eps::detail {
ReusePolicy::ReusePolicy(CachePolicyOptions options):options_(std::move(options)) {
  const auto& o=options_;
  Require(o.mode==CacheAdmissionMode::manual || o.mode==CacheAdmissionMode::frequency ||
    o.mode==CacheAdmissionMode::reuse_cost,"invalid cache admission mode");
  Require(o.window_requests>0 && o.max_tracked_rows>0 && o.max_rows_per_batch>0 &&
    o.max_rows_per_batch<=4096,"invalid cache policy bounds");
  const auto finite=[](double x){return std::isfinite(x) && x>=0;};
  Require(finite(o.window_decay) && o.window_decay>0 && o.window_decay<=1 &&
    finite(o.minimum_accesses) && o.minimum_accesses>0 &&
    finite(o.replacement_margin) && finite(o.saved_microseconds_per_access) &&
    finite(o.load_microseconds_per_byte) && finite(o.writeback_microseconds_per_byte) &&
    finite(o.cost_average_alpha) && o.cost_average_alpha>0 && o.cost_average_alpha<=1,
    "invalid cache policy costs or decay");
  std::set<AgentId> agents(o.agents.begin(),o.agents.end());
  Require(agents.size()==o.agents.size() &&
    (o.mode==CacheAdmissionMode::manual || !agents.empty()),"invalid cache policy agents");
  Require(o.mode==CacheAdmissionMode::manual || (o.budget.max_inflight_batches>0 &&
    o.budget.network_bytes>0 && o.budget.staging_bytes>0),"invalid cache policy budget");
}
void ReusePolicy::Observe(const std::vector<RowKey>& rows) {
  std::set<Id> unique;
  for(auto row:rows)unique.emplace(KeyOf(row));
  for(auto id:unique) {
    if(auto it=observations_.find(id);it!=observations_.end()) {
      auto& v=it->second;
      v.accesses=v.accesses*std::pow(options_.window_decay,double(window_-v.window))+1;
      v.window=window_;lru_.splice(lru_.end(),lru_,v.lru);
    } else {
      if(observations_.size()==options_.max_tracked_rows) {
        observations_.erase(lru_.front());lru_.pop_front();
      }
      lru_.push_back(id);
      try { observations_.emplace(id,Observation{1,window_,std::prev(lru_.end())}); }
      catch(...) { lru_.pop_back();throw; }
    }
  }
}
bool ReusePolicy::CompleteRequest() {
  if(++completed_<options_.window_requests)return false;
  completed_=0;++window_;return true;
}
std::vector<RowKey> ReusePolicy::TrackedRows() const {
  std::vector<RowKey> out;out.reserve(observations_.size());
  for(const auto& [id,v]:observations_)out.push_back({id.first,id.second});
  return out;
}
double ReusePolicy::ExpectedAccesses(RowKey row) const {
  const auto it=observations_.find(KeyOf(row));
  if(it==observations_.end())return 0;
  const auto& v=it->second;
  // CompleteRequest advances the window before its maintenance decisions.
  const auto observed_window=window_?window_-1:0;
  return v.accesses*std::pow(options_.window_decay,
    double(observed_window>v.window?observed_window-v.window:0));
}
double ReusePolicy::AdmissionValue(RowKey row,std::uint64_t bytes) const {
  const auto accesses=ExpectedAccesses(row);
  if(accesses<options_.minimum_accesses)return 0;
  if(options_.mode==CacheAdmissionMode::frequency)return accesses;
  const auto value=accesses*options_.saved_microseconds_per_access-
    bytes*(options_.load_microseconds_per_byte+options_.writeback_microseconds_per_byte);
  return std::isfinite(value)?std::max(0.,value):0;
}
double ReusePolicy::RetainedValue(RowKey row) const {
  const auto value=ExpectedAccesses(row)*(options_.mode==CacheAdmissionMode::frequency?
    1:options_.saved_microseconds_per_access);
  return std::isfinite(value)?value:0;
}
double ReusePolicy::WritebackCost(std::uint64_t bytes) const {
  return options_.mode==CacheAdmissionMode::frequency?0:bytes*options_.writeback_microseconds_per_byte;
}
void ReusePolicy::ObserveMigration(Placement destination,std::uint64_t bytes,double us) {
  if(!options_.learn_migration_costs || !bytes || !std::isfinite(us) || us<0)return;
  auto& cost=destination==Placement::gpu?options_.load_microseconds_per_byte:
    options_.writeback_microseconds_per_byte;
  const auto measured=us/bytes;
  cost=(1-options_.cost_average_alpha)*cost+options_.cost_average_alpha*measured;
}
void ReusePolicy::SynchronizeResidents(const std::vector<LocatedRow>& rows) {
  std::set<Id> live;
  for(const auto& row:rows) {
    const auto id=KeyOf(row.row);live.insert(id);
    auto [it,inserted]=residents_.try_emplace(id,Residence{row.generation,window_});
    if(!inserted && it->second.generation!=row.generation)
      it->second={row.generation,window_};
  }
  std::erase_if(residents_,[&](const auto& v){return !live.contains(v.first);});
}
bool ReusePolicy::CanReplace(const LocatedRow& row) const {
  const auto it=residents_.find(KeyOf(row.row));
  return it!=residents_.end() && it->second.generation==row.generation &&
    window_-it->second.window>=options_.minimum_residence_windows;
}
void ReusePolicy::Admitted(const LocatedRow& row) {
  residents_.insert_or_assign(KeyOf(row.row),Residence{row.generation,window_});
}
}  // namespace eps::detail
