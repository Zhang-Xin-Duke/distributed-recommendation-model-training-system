#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>
#include "common/error.hpp"

namespace eps::transport {
// Futures are produced by ordered service executors. A small native waiter pool
// blocks on their notifications without pinning BRPC workers or polling every
// outstanding request. Intrusive queue nodes are allocated before admission.
class CompletionExecutor {
 public:
  struct Task {
    virtual ~Task()=default;
    virtual void Run()=0;
    std::unique_ptr<Task> next;
  };
  class Reservation {
   public:
    explicit Reservation(CompletionExecutor& owner):owner_(&owner){
      std::lock_guard lock(owner.mutex_);
      detail::Require(!owner.stopping_ && owner.outstanding_<owner.limit_,
        "RPC completion queue is full",Error::Code::capacity_exceeded);
      ++owner.outstanding_;
    }
    Reservation(const Reservation&)=delete;
    ~Reservation(){if(owner_)owner_->Release();}
    void Publish(std::unique_ptr<Task> task) noexcept {
      auto* owner=std::exchange(owner_,nullptr);owner->Post(std::move(task));
    }
   private:CompletionExecutor* owner_;
  };
  static CompletionExecutor& Instance(){static CompletionExecutor executor;return executor;}
  ~CompletionExecutor(){
    {std::lock_guard lock(mutex_);stopping_=true;}
    changed_.notify_all();for(auto& worker:workers_)worker.join();
  }
 private:
  CompletionExecutor(){
    workers_.reserve(4);
    try{for(int i=0;i<4;++i)workers_.emplace_back([this]{
      for(;;){
        std::unique_ptr<Task> task;
        {std::unique_lock lock(mutex_);changed_.wait(lock,[&]{return stopping_ || bool(head_);});
          if(!head_ && stopping_)return;
          task=std::move(head_);head_=std::move(task->next);if(!head_)tail_=nullptr;
        }
        // Pending owns a closure guard, so its response is completed even if
        // encoding throws. Keep the executor available for other requests.
        try{task->Run();}catch(...){}
        Release();
      }
    });}catch(...){
      {std::lock_guard lock(mutex_);stopping_=true;}
      changed_.notify_all();for(auto& worker:workers_)worker.join();throw;
    }
  }
  void Release(){std::lock_guard lock(mutex_);--outstanding_;}
  void Post(std::unique_ptr<Task> task) noexcept {
    {std::lock_guard lock(mutex_);auto* next=task.get();
      if(tail_)tail_->next=std::move(task);else head_=std::move(task);tail_=next;
    }changed_.notify_one();
  }
  const std::size_t limit_=4096;
  std::mutex mutex_;std::condition_variable changed_;
  std::unique_ptr<Task> head_;Task* tail_=nullptr;
  std::size_t outstanding_=0;bool stopping_=false;std::vector<std::thread> workers_;
};
}  // namespace eps::transport
