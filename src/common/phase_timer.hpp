#pragma once

#include <chrono>
#include <cstdint>

namespace eps::detail {
// Sequencer-owned wall time. Inclusive scopes such as cache maintenance,
// migration and agent waiting overlap; they are not disjoint CPU-time buckets.
class PhaseTimer {
 public:
  explicit PhaseTimer(std::uint64_t& elapsed,bool enabled=true)
      :elapsed_(elapsed),enabled_(enabled),begin_(enabled?Clock::now():Clock::time_point{}) {}
  ~PhaseTimer() {
    if(enabled_)elapsed_+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin_).count();
  }
  PhaseTimer(const PhaseTimer&)=delete;
  PhaseTimer& operator=(const PhaseTimer&)=delete;
 private:
  using Clock=std::chrono::steady_clock;
  std::uint64_t& elapsed_;
  bool enabled_;
  Clock::time_point begin_;
};
}  // namespace eps::detail
