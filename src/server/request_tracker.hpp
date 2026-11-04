#pragma once

#include <map>
#include <optional>
#include <tuple>
#include <set>
#include <algorithm>

#include "common/error.hpp"
#include "eps/types.hpp"

namespace eps::detail {

// One bounded lifecycle sequence window per learner. Completed holes remain until
// their predecessors complete; an old packet can never become a new operation.
class RequestTracker {
 public:
  explicit RequestTracker(std::uint32_t window) : window_(window) {}
  std::vector<RequestFrontier> Frontiers() const {
    std::vector<RequestFrontier> result;
    for (const auto& [learner,s]:streams_) {
      auto next=s.retired_next;
      while(next!=UINT64_MAX) {auto it=s.entries.find(next);if(it==s.entries.end() || !it->second.completed)break;++next;}
      result.push_back({learner,next});
    }
    return result;
  }
  void RestoreFrontiers(const std::vector<RequestFrontier>& progress) {
    std::set<std::uint32_t> seen;
    for (const auto& p:progress) {
      Require(p.learner<1024 && seen.insert(p.learner).second,"invalid restored request stream");
      auto& s=streams_[p.learner];Require(s.entries.empty() && p.next>=s.retired_next,"request recovery needs empty monotonic streams");
      s.retired_next=p.next;
    }
  }
  void CertifyReads(const std::vector<RequestFrontier>& progress) {
    std::set<std::uint32_t> seen;
    for(const auto& p:progress) {
      Require(p.learner<1023 && seen.insert(p.learner).second,"invalid read-completion certificate");
      const auto& s=streams_[p.learner];
      for(const auto& [sequence,record]:s.entries)if(sequence<p.next)
        Require(record.completed,"read certificate overlaps an unfinished request");
    }
    for(const auto& p:progress) {
      auto& s=streams_[p.learner];s.retired_next=std::max(s.retired_next,p.next);
      s.entries.erase(s.entries.begin(),s.entries.lower_bound(s.retired_next));
    }
  }
  enum class State { fresh, pending, completed };
  State Reserve(const RequestId& id) {
    Require(id.learner < 1024, "learner ID exceeds supported run limit");
    auto& s = streams_[id.learner];
    if (auto it = s.entries.find(id.sequence); it != s.entries.end())
      return it->second.completed ? State::completed : State::pending;
    if (s.entries.size() >= window_) {
      while (s.entries.size() >= window_ / 2) {
        auto it = s.entries.find(s.retired_next);
        if (it == s.entries.end() || !it->second.completed) break;
        s.entries.erase(it);
        ++s.retired_next;
      }
    }
    Require(id.sequence >= s.retired_next, "request is outside retained history",
            Error::Code::retired_request);
    Require(id.sequence - s.retired_next < window_, "request sequence window is full",
            Error::Code::capacity_exceeded);
    s.entries.emplace(id.sequence, Record{});
    return State::fresh;
  }
  void Complete(const RequestId& id) {
    streams_.at(id.learner).entries.at(id.sequence) = {true, id.phase};
  }
  std::optional<Phase> CompletedAs(const RequestId& id) const {
    auto stream = streams_.find(id.learner);
    Require(stream != streams_.end(), "unknown request lifecycle");
    Require(id.sequence >= stream->second.retired_next,
            "request is outside retained history", Error::Code::retired_request);
    auto it = stream->second.entries.find(id.sequence);
    Require(it != stream->second.entries.end(), "unknown request lifecycle");
    return it->second.completed ? std::optional(it->second.final_phase) : std::nullopt;
  }
  void Abort(const RequestId& id) {
    streams_.at(id.learner).entries.erase(id.sequence);
  }
 private:
  struct Record { bool completed = false; Phase final_phase = Phase::lookup; };
  struct Stream {
    std::uint64_t retired_next = 0;
    std::map<std::uint64_t, Record> entries;
  };
  std::uint32_t window_;
  std::map<std::uint32_t, Stream> streams_;
};

}  // namespace eps::detail
