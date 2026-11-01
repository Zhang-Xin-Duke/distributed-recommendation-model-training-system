#pragma once

#include <algorithm>
#include <chrono>
#include <compare>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>
#include "common/actor_task.hpp"
#include "common/error.hpp"

namespace eps::detail {

struct ActorResource {
  enum class Kind { lifecycle, optimizer, agent, maintenance, row };
  Kind kind;
  std::uint64_t first=0,second=0;
  auto operator<=>(const ActorResource&) const=default;
};
struct ActorResources {
  bool barrier=false;
  std::vector<ActorResource> keys;
  bool cacheable=true;
  void Normalize(){std::sort(keys.begin(),keys.end());keys.erase(std::unique(keys.begin(),keys.end()),keys.end());}
  std::uint64_t bytes() const{return keys.capacity()*sizeof(ActorResource);}
  bool Conflicts(const ActorResources& b) const {
    if(barrier || b.barrier)return true;
    auto a_it=keys.begin(),b_it=b.keys.begin();
    while(a_it!=keys.end() && b_it!=b.keys.end()) {
      if(*a_it==*b_it)return true;
      if(*a_it<*b_it)++a_it;else ++b_it;
    }return false;
  }
};

class ResumableActor {
 public:
  struct Options {
    std::size_t max_queued=1024,max_active=16,waiters=4;
    std::uint64_t payload_bytes=64ULL<<20,dependency_bytes=64ULL<<20;
    bool cache_dependencies=true;
  };
  struct Stats {
    std::uint64_t queued=0,active=0,waiting=0,queued_payload=0,active_payload=0,
      dependency_bytes=0,rejected=0,peak_queued=0,completed=0,queue_wait_ns=0,busy_ns=0,
      dependency_prepare_ns=0,dependency_rebuilds=0,dependency_cache_hits=0;
  };
 private:
  struct Job;
  struct External {
    virtual ~External()=default;
    virtual void Observe() noexcept=0;
    std::coroutine_handle<> continuation;
    Job* owner=nullptr;
    External* next=nullptr;
  };
  struct Job {
    virtual ~Job()=default;
    virtual ActorResources Dependencies()=0;
    virtual std::coroutine_handle<> Start()=0;
    virtual void Finish() noexcept=0;
    virtual void Reject(Error) noexcept=0;
    std::unique_ptr<Job> next;
    ActorResources resources;
    bool finished=false;
    bool dependencies_valid=false;
    std::uint64_t dependencies_epoch=0;
    std::coroutine_handle<> exclusive_continuation{};
    std::uint64_t payload=0;
    std::chrono::steady_clock::time_point queued;
  };
  static Error CurrentError() {
    try{throw;}
    catch(const Failure& e){return e.error();}
    catch(const std::bad_alloc&){return {Error::Code::capacity_exceeded,"actor allocation failed"};}
    catch(const std::exception& e){return {Error::Code::run_failed,e.what()};}
    catch(...){return {Error::Code::run_failed,"unknown actor failure"};}
  }
  template<class T,class Fn> static ActorTask<T> Invoke(Fn* fn) {
    if constexpr(IsActorTask<std::invoke_result_t<Fn>>::value)co_return co_await (*fn)();
    else if constexpr(std::is_same_v<std::invoke_result_t<Fn>,Result<T>>) {
      auto result=(*fn)();if(auto error=std::get_if<Error>(&result))throw Failure(*error);
      co_return std::get<T>(std::move(result));
    }
    else co_return (*fn)();
  }
  template<class T,class Fn,class DependenciesFn> struct TypedJob final:Job {
    TypedJob(Fn f,DependenciesFn d,std::function<void(const Error&)>* observer)
      :fn(std::move(f)),dependencies(std::move(d)),on_error(observer){}
    ActorResources Dependencies() override{return dependencies();}
    std::coroutine_handle<> Start() override {
      task.emplace(Invoke<T>(&fn));task->handle().promise().completed=&this->finished;
      return task->handle();
    }
    void Finish() noexcept override {
      try{promise.set_value(task->Take());}
      catch(...){Reject(CurrentError());}
    }
    void Reject(Error error) noexcept override {
      try{if(*on_error)(*on_error)(error);promise.set_value(std::move(error));}catch(...){std::terminate();}
    }
    Fn fn;DependenciesFn dependencies;
    std::function<void(const Error&)>* on_error;
    std::promise<Result<T>> promise;
    std::optional<ActorTask<T>> task;
  };
  template<class T> struct BatchWait final:External {
    explicit BatchWait(std::size_t n){futures.reserve(n);results.reserve(n);}
    bool Ready() {
      bool ready=true;
      for(auto& f:futures) {
        try{if(f.wait_for(std::chrono::seconds(0))!=std::future_status::ready)ready=false;}
        catch(...){if(!error)error=std::current_exception();}
      }return ready;
    }
    void Observe() noexcept override {
      for(auto& future:futures) {
        try{results.push_back(future.get());}
        catch(...){if(!error)error=std::current_exception();}
      }
    }
    std::vector<std::future<Result<T>>> futures;
    std::vector<Result<T>> results;
    std::exception_ptr error;
  };
 public:
  explicit ResumableActor(Options options,std::function<void(const Error&)> observer={})
    :options_(options),on_error_(std::move(observer)) {
    Require(options_.max_queued>0 && options_.max_queued<=1048576 &&
      options_.max_active>0 && options_.max_active<=1024 && options_.waiters>0 && options_.waiters<=64 &&
      options_.payload_bytes>0 && options_.dependency_bytes>0,"invalid actor bounds");
    active_.reserve(options_.max_active);scratch_.reserve(options_.max_queued);
    waiters_.reserve(options_.waiters);
    try {
      for(std::size_t i=0;i<options_.waiters;++i)waiters_.emplace_back([this]{WaiterLoop();});
      worker_=std::thread([this]{Run();});
    }catch(...) {
      {std::lock_guard lock(mutex_);waiters_stopping_=true;}
      waiter_changed_.notify_all();for(auto& worker:waiters_)worker.join();throw;
    }
  }
  ResumableActor(const ResumableActor&)=delete;
  ~ResumableActor(){Shutdown();}
  void Shutdown(){
    std::call_once(shutdown_,[this]{
      {std::lock_guard lock(mutex_);closed_=true;}changed_.notify_all();worker_.join();
      for(auto& waiter:waiters_)waiter.join();
    });
  }
  template<class T,class Fn,class DependenciesFn>
  std::future<Result<T>> Submit(Fn fn,DependenciesFn dependencies,std::uint64_t payload=0) {
    auto job=std::make_unique<TypedJob<T,Fn,DependenciesFn>>(std::move(fn),std::move(dependencies),&on_error_);
    auto future=job->promise.get_future();
    job->payload=payload;job->queued=std::chrono::steady_clock::now();
    {
      std::lock_guard lock(mutex_);
      if(closed_ || stats_.queued>=options_.max_queued ||
         payload>options_.payload_bytes-stats_.queued_payload-stats_.active_payload) {
        ++stats_.rejected;
        // Admission rejection runs on the submitting thread; never call the
        // actor-state error observer here.
        job->promise.set_value(Error{Error::Code::capacity_exceeded,"shard operation queue is full"});return future;
      }
      auto* tail=job.get();
      if(tail_)tail_->next=std::move(job);else head_=std::move(job);
      tail_=tail;++stats_.queued;stats_.queued_payload+=payload;
      stats_.peak_queued=std::max(stats_.peak_queued,stats_.queued);
    }
    changed_.notify_one();return future;
  }
  template<class T,class Fn> std::future<Result<T>> Submit(Fn fn,std::uint64_t payload=0) {
    return Submit<T>(std::move(fn),[]{return ActorResources{true,{}};},payload);
  }
  template<class T,class Producer> class AwaitBatch {
   public:
    AwaitBatch(ResumableActor& a,std::size_t count,Producer producer)
      :actor_(a),producer_(std::move(producer)),state_(std::make_unique<BatchWait<T>>(count)),limit_(count){}
    bool await_ready() noexcept{return false;}
    bool await_suspend(std::coroutine_handle<> continuation) {
      // All waiter/result/queue storage exists before Producer dispatches work.
      try{producer_(state_->futures);}
      catch(...){state_->error=std::current_exception();}
      if(state_->futures.size()>limit_ && !state_->error) {
        try{throw Failure({Error::Code::run_failed,"external producer exceeded reserved completion slots"});}
        catch(...){state_->error=std::current_exception();}
      }
      if(state_->Ready()){state_->Observe();return false;}
      state_->continuation=continuation;actor_.Park(state_.get());return true;
    }
    std::vector<Result<T>> await_resume() {
      if(state_->error)std::rethrow_exception(state_->error);return std::move(state_->results);
    }
   private:
    ResumableActor& actor_;Producer producer_;std::unique_ptr<BatchWait<T>> state_;std::size_t limit_;
  };
  template<class T,class Producer> auto WaitAll(std::size_t count,Producer producer) {
    return AwaitBatch<T,Producer>(*this,count,std::move(producer));
  }
  class ExclusiveAwaiter {
   public:
    explicit ExclusiveAwaiter(ResumableActor& actor):actor_(actor){}
    bool await_ready() noexcept{return false;}
    bool await_suspend(std::coroutine_handle<> continuation) {
      std::lock_guard lock(actor_.mutex_);
      auto* owner=actor_.executing_;
      Require(owner!=nullptr,"exclusive upgrade outside an actor operation",Error::Code::run_failed);
      // Callers sharing an upgrade path must hold the same maintenance resource
      // from admission. That allows at most one upgrade waiter at a time.
      for(const auto& job:actor_.active_)
        Require(!job->exclusive_continuation,"concurrent exclusive upgrades",Error::Code::run_failed);
      owner->resources.barrier=true;
      if(actor_.active_.size()==1)return false;
      owner->exclusive_continuation=continuation;++actor_.stats_.waiting;return true;
    }
    void await_resume() noexcept{}
   private:ResumableActor& actor_;
  };
  ExclusiveAwaiter Exclusive(){return ExclusiveAwaiter(*this);}
  // Actor-thread only. Active operations retain their acquired dependencies;
  // queued plans are rebuilt after ownership or policy interpretation changes.
  void InvalidateDependencies() noexcept {++dependency_epoch_;}
  Stats Snapshot() const {std::lock_guard lock(mutex_);return stats_;}
 private:
  void Park(External* wait) noexcept {
    {std::lock_guard lock(mutex_);
      wait->owner=executing_;
      if(wait_tail_)wait_tail_->next=wait;else wait_head_=wait;
      wait_tail_=wait;++stats_.waiting;
    }waiter_changed_.notify_one();
  }
  void WaiterLoop() {
    for(;;) {
      External* task;
      {std::unique_lock lock(mutex_);waiter_changed_.wait(lock,[this]{return waiters_stopping_ || wait_head_;});
        if(!wait_head_ && waiters_stopping_)return;
        task=wait_head_;wait_head_=task->next;if(!wait_head_)wait_tail_=nullptr;task->next=nullptr;
      }
      task->Observe();
      {std::lock_guard lock(mutex_);
        if(wake_tail_)wake_tail_->next=task;else wake_head_=task;wake_tail_=task;
      }changed_.notify_one();
    }
  }
  // Called under mutex, solely by the actor. Dependency resolvers inspect actor
  // state; they must be pure and must not submit another operation.
  std::unique_ptr<Job> TakeRunnable() {
    if(active_.size()>=options_.max_active)return {};
    scratch_.clear();
    auto* link=&head_;Job* previous=nullptr;
    while(*link) {
      auto& candidate=**link;
      if(!options_.cache_dependencies || !candidate.dependencies_valid || candidate.dependencies_epoch!=dependency_epoch_) {
        stats_.dependency_bytes-=candidate.resources.bytes();
        candidate.resources=ActorResources{};candidate.dependencies_valid=false;
        ActorResources resources;
        const auto begin=std::chrono::steady_clock::now();
        try{resources=candidate.Dependencies();resources.Normalize();}
        catch(...){
          stats_.dependency_prepare_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
          ++stats_.dependency_rebuilds;
          auto failed=std::move(*link);*link=std::move(failed->next);if(!*link)tail_=previous;
          --stats_.queued;stats_.queued_payload-=failed->payload;
          auto error=CurrentError();if(error.code==Error::Code::capacity_exceeded)++stats_.rejected;
          ++stats_.completed;failed->Reject(std::move(error));continue;
        }
        stats_.dependency_prepare_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
        ++stats_.dependency_rebuilds;
        const auto bytes=resources.bytes();
        if(bytes>options_.dependency_bytes) {
          auto failed=std::move(*link);*link=std::move(failed->next);if(!*link)tail_=previous;
          --stats_.queued;stats_.queued_payload-=failed->payload;
          ++stats_.rejected;++stats_.completed;
          failed->Reject({Error::Code::capacity_exceeded,"shard dependency budget exceeded"});continue;
        }
        if(bytes>options_.dependency_bytes-stats_.dependency_bytes) {
          // Reclaim optional queued plans instead of rejecting accepted work.
          // With active work, its next completion wakes this bounded scan.
          for(auto* job=head_.get();job;job=job->next.get()) {
            stats_.dependency_bytes-=job->resources.bytes();job->resources=ActorResources{};
            job->dependencies_valid=false;
          }
          scratch_.clear();return {};
        }
        candidate.resources=std::move(resources);stats_.dependency_bytes+=bytes;
        candidate.dependencies_valid=candidate.resources.cacheable;candidate.dependencies_epoch=dependency_epoch_;
      }else ++stats_.dependency_cache_hits;
      const auto& resources=candidate.resources;
      bool blocked=false;
      for(const auto& job:active_)if(resources.Conflicts(job->resources)){blocked=true;break;}
      for(const auto* older:scratch_)if(resources.Conflicts(older->resources)){blocked=true;break;}
      if(!blocked) {
        auto next=std::move(*link);*link=std::move(next->next);if(!*link)tail_=previous;
        --stats_.queued;stats_.queued_payload-=next->payload;
        ++stats_.active;stats_.active_payload+=next->payload;
        stats_.queue_wait_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-next->queued).count();
        scratch_.clear();return next;
      }
      if(resources.barrier)break;
      scratch_.push_back(&candidate);
      previous=&candidate;link=&candidate.next;
    }
    scratch_.clear();return {};
  }
  void Reap() {
    for(auto it=active_.begin();it!=active_.end();) {
      if(!(*it)->finished){++it;continue;}
      auto done=std::move(*it);it=active_.erase(it);
      {std::lock_guard lock(mutex_);--stats_.active;stats_.active_payload-=done->payload;
        stats_.dependency_bytes-=done->resources.bytes();++stats_.completed;}
      done->Finish();
    }
  }
  void Run() {
    for(;;) {
      std::coroutine_handle<> continuation{};std::unique_ptr<Job> next;Job* owner=nullptr;
      {
        std::unique_lock lock(mutex_);
        for(;;) {
          if(wake_head_) {
            auto* ready=wake_head_;wake_head_=ready->next;if(!wake_head_)wake_tail_=nullptr;
            ready->next=nullptr;continuation=ready->continuation;owner=ready->owner;--stats_.waiting;break;
          }
          if(active_.size()==1 && active_.front()->exclusive_continuation) {
            owner=active_.front().get();continuation=std::exchange(owner->exclusive_continuation,{});
            --stats_.waiting;break;
          }
          const auto planning_begin=std::chrono::steady_clock::now();
          next=TakeRunnable();
          stats_.busy_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-planning_begin).count();
          if(next)break;
          if(!active_.size() && head_)continue;
          if(closed_ && !head_ && active_.empty()) {
            waiters_stopping_=true;lock.unlock();waiter_changed_.notify_all();return;
          }
          changed_.wait(lock);
        }
      }
      auto begin=std::chrono::steady_clock::now();
      if(next) {
        auto* raw=next.get();active_.push_back(std::move(next));
        try{continuation=raw->Start();owner=raw;}
        catch(...){raw->Reject(CurrentError());raw->finished=true;
          // A task that could not allocate its frame has no result to Finish.
          {std::lock_guard lock(mutex_);--stats_.active;stats_.active_payload-=raw->payload;
            stats_.dependency_bytes-=raw->resources.bytes();++stats_.completed;}
          active_.pop_back();continuation={};}
      }
      executing_=owner;if(continuation)continuation.resume();executing_=nullptr;
      Reap();
      {std::lock_guard lock(mutex_);stats_.busy_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();}
    }
  }
  Options options_;std::function<void(const Error&)> on_error_;
  mutable std::mutex mutex_;std::condition_variable changed_,waiter_changed_;
  std::unique_ptr<Job> head_;Job* tail_=nullptr;
  std::vector<std::unique_ptr<Job>> active_;
  Job* executing_=nullptr;
  std::vector<Job*> scratch_;
  std::uint64_t dependency_epoch_=0;
  External* wait_head_=nullptr;External* wait_tail_=nullptr;
  External* wake_head_=nullptr;External* wake_tail_=nullptr;
  Stats stats_;bool closed_=false,waiters_stopping_=false;
  std::vector<std::thread> waiters_;std::thread worker_;
  std::once_flag shutdown_;
};
}  // namespace eps::detail
