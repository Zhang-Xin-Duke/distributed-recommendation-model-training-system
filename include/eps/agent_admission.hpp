#pragma once
#include <cstdint>

namespace eps {
struct AgentAdmissionOptions {
  std::uint64_t max_active=32;
  std::uint64_t max_pending=4096;  // Active plus waiting requests.
  std::uint64_t request_bytes=256ULL<<20;
};
struct AgentAdmissionStats {
  std::uint64_t active_operations=0,queued_operations=0,pending_request_bytes=0;
  std::uint64_t peak_active_operations=0,peak_pending_operations=0,peak_request_bytes=0;
  std::uint64_t accepted_requests=0,completed_requests=0,rejected_requests=0;
  std::uint64_t queue_wait_nanoseconds=0,dispatch_failures=0;
  std::uint64_t active_limit=0,pending_limit=0,request_bytes_limit=0;
};
}  // namespace eps
