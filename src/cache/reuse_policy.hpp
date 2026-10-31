#pragma once

#include <list>
#include <map>
#include <utility>
#include "eps/cache_policy.hpp"

namespace eps::detail {
// Called only by the home shard's sequencer; owns no parameter state.
class ReusePolicy {
 public:
  explicit ReusePolicy(CachePolicyOptions options);
  const CachePolicyOptions& options() const { return options_; }
  void Observe(const std::vector<RowKey>& rows);
  bool CompleteRequest();
  std::uint64_t window() const { return window_; }
  std::size_t tracked_rows() const { return observations_.size(); }
  std::vector<RowKey> TrackedRows() const;
  double ExpectedAccesses(RowKey row) const;
  double AdmissionValue(RowKey row, std::uint64_t bytes) const;
  double RetainedValue(RowKey row) const;
  double WritebackCost(std::uint64_t bytes) const;
  void ObserveMigration(Placement destination, std::uint64_t bytes, double microseconds);
  void SynchronizeResidents(const std::vector<LocatedRow>& rows);
  bool CanReplace(const LocatedRow& row) const;
  void Admitted(const LocatedRow& row);
 private:
  using Id = std::pair<TableId,Key>;
  static Id KeyOf(RowKey row) { return {row.table,row.key}; }
  struct Observation {
    double accesses;
    std::uint64_t window;
    std::list<Id>::iterator lru;
  };
  struct Residence { std::uint64_t generation, window; };
  CachePolicyOptions options_;
  std::uint64_t completed_ = 0, window_ = 0;
  std::list<Id> lru_;
  std::map<Id,Observation> observations_;
  // Proportional to current resident rows, independent of access-history size.
  std::map<Id,Residence> residents_;
};
}  // namespace eps::detail
