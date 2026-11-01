#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include "eps/agent_admission.hpp"
#include "common/error.hpp"

namespace eps::detail {
// A permit follows the asynchronous completion, rather than the dispatch call.
// One dispatcher preserves FIFO start order without blocking BRPC worker threads.
class AgentAdmissionQueue {
  struct State {
    std::mutex mutex;
    std::condition_variable changed;
    AgentAdmissionOptions options;
    AgentAdmissionStats stats;
    std::uint64_t pending=0;
    bool closing=false;
  };
  struct Permit {
    std::shared_ptr<State> owner;
    std::uint64_t bytes;
    bool registered=false,started=false;  // Protected by owner's mutex.
    ~Permit(){
      {
        std::lock_guard lock(owner->mutex);
        if(!registered)return;
        if(started)--owner->stats.active_operations;
        --owner->pending;
        owner->stats.pending_request_bytes-=bytes;
        ++owner->stats.completed_requests;
      }
      owner->changed.notify_all();
    }
  };
 public:
  using Job=std::function<void(std::shared_ptr<void>)>;
  explicit AgentAdmissionQueue(AgentAdmissionOptions options={});
  ~AgentAdmissionQueue();
  AgentAdmissionQueue(const AgentAdmissionQueue&)=delete;
  void Submit(std::uint64_t bytes,Job job);
  AgentAdmissionStats Stats() const;
  void Shutdown();
 private:
  struct Entry {
    Job job;
    std::shared_ptr<Permit> permit;
    std::chrono::steady_clock::time_point queued;
  };
  std::shared_ptr<State> state_;
  std::deque<Entry> queue_;
  std::thread worker_;
  std::once_flag shutdown_;
};

inline AgentAdmissionQueue::AgentAdmissionQueue(AgentAdmissionOptions options)
  :state_(std::make_shared<State>()){
  Require(options.max_active>0 && options.max_active<=128 &&
    options.max_pending>=options.max_active && options.max_pending<=65536 &&
    options.request_bytes>0,"invalid Agent admission limits");
  state_->options=options;
  state_->stats.active_limit=options.max_active;
  state_->stats.pending_limit=options.max_pending;
  state_->stats.request_bytes_limit=options.request_bytes;
  worker_=std::thread([this]{
    for(;;){
      Entry entry;
      {
        std::unique_lock lock(state_->mutex);
        state_->changed.wait(lock,[&]{return
          (state_->closing && state_->pending==0) ||
          (!queue_.empty() && state_->stats.active_operations<state_->options.max_active);});
        if(state_->closing && state_->pending==0)return;
        entry=std::move(queue_.front());queue_.pop_front();
        entry.permit->started=true;
        auto& s=state_->stats;++s.active_operations;
        s.peak_active_operations=std::max(s.peak_active_operations,s.active_operations);
        s.queue_wait_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now()-entry.queued).count();
      }
      try{entry.job(std::move(entry.permit));}
      catch(...){std::lock_guard lock(state_->mutex);++state_->stats.dispatch_failures;}
    }
  });
}
inline AgentAdmissionQueue::~AgentAdmissionQueue(){Shutdown();}
inline void AgentAdmissionQueue::Submit(std::uint64_t bytes,Job job){
  Require(bool(job),"empty Agent admission job");
  auto permit=std::make_shared<Permit>();permit->owner=state_;permit->bytes=bytes;
  {
    std::lock_guard lock(state_->mutex);auto& s=state_->stats;
    if(state_->closing){++s.rejected_requests;
      throw Failure({Error::Code::unavailable,"Agent admission is closing"});}
    if(state_->pending>=state_->options.max_pending ||
        bytes>state_->options.request_bytes-s.pending_request_bytes){
      ++s.rejected_requests;
      throw Failure({Error::Code::capacity_exceeded,"Agent admission budget exhausted"});
    }
    queue_.push_back({std::move(job),permit,std::chrono::steady_clock::now()});
    permit->registered=true;
    ++state_->pending;++s.accepted_requests;s.pending_request_bytes+=bytes;
    s.peak_pending_operations=std::max(s.peak_pending_operations,state_->pending);
    s.peak_request_bytes=std::max(s.peak_request_bytes,s.pending_request_bytes);
  }
  state_->changed.notify_all();
}
inline AgentAdmissionStats AgentAdmissionQueue::Stats() const{
  std::lock_guard lock(state_->mutex);auto result=state_->stats;
  result.queued_operations=queue_.size();return result;
}
inline void AgentAdmissionQueue::Shutdown(){
  std::call_once(shutdown_,[this]{
    {std::lock_guard lock(state_->mutex);state_->closing=true;}
    state_->changed.notify_all();worker_.join();
  });
}
}  // namespace eps::detail
