#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include "common/error.hpp"

namespace eps::detail {
class OrderedExecutor {
 public:
  explicit OrderedExecutor(std::size_t max_operations=256):limit_(max_operations) {
    worker_=std::thread([this] {
      for(;;){
        std::function<void()> job;
        {std::unique_lock lock(mutex_);changed_.wait(lock,[&]{return stop_||!jobs_.empty();});
          if(stop_&&jobs_.empty())return;
          job=std::move(jobs_.front());jobs_.pop_front();}
        job();
      }
    });
  }
  ~OrderedExecutor(){
    {std::lock_guard lock(mutex_);stop_=true;}changed_.notify_all();worker_.join();
  }
  template<typename T,typename F>std::future<Result<T>> Submit(F fn){
    auto p=std::make_shared<std::promise<Result<T>>>();auto f=p->get_future();
    {std::lock_guard lock(mutex_);
      if(stop_||jobs_.size()>=limit_){p->set_value(Error{Error::Code::capacity_exceeded,"agent queue is full"});return f;}
      jobs_.emplace_back([p,fn=std::move(fn)]() mutable {
        try{p->set_value(fn());}
        catch(const Failure& e){p->set_value(e.error());}
        catch(const std::bad_alloc&){p->set_value(Error{Error::Code::capacity_exceeded,"agent allocation failed"});}
        catch(const std::exception& e){p->set_value(Error{Error::Code::run_failed,e.what()});}
      });
    }changed_.notify_one();return f;
  }
 private:
  std::size_t limit_;
  std::mutex mutex_;std::condition_variable changed_;
  std::deque<std::function<void()>> jobs_;bool stop_=false;std::thread worker_;
};
template<typename T>T Await(std::future<Result<T>> future){
  auto result=future.get();if(auto error=std::get_if<Error>(&result))throw Failure(*error);
  return std::get<T>(std::move(result));
}
}  // namespace eps::detail
