#pragma once

#include <coroutine>
#include <exception>
#include <optional>
#include <utility>

namespace eps::detail {

// Lazy, move-only task. A nested task transfers control on the same actor;
// only the external-future awaiter may resume it from the actor's event queue.
template<class T> class ActorTask {
 public:
  struct promise_type;
  using Handle=std::coroutine_handle<promise_type>;
  struct promise_type {
    std::optional<T> value;
    std::exception_ptr error;
    std::coroutine_handle<> continuation{};
    bool* completed=nullptr;
    ActorTask get_return_object(){return ActorTask{Handle::from_promise(*this)};}
    std::suspend_always initial_suspend() noexcept{return {};}
    struct Final {
      bool await_ready() noexcept{return false;}
      std::coroutine_handle<> await_suspend(Handle h) noexcept {
        auto& p=h.promise();if(p.completed)*p.completed=true;
        return p.continuation?p.continuation:std::noop_coroutine();
      }
      void await_resume() noexcept{}
    };
    Final final_suspend() noexcept{return {};}
    void return_value(T result){value.emplace(std::move(result));}
    void unhandled_exception() noexcept{error=std::current_exception();}
  };
  explicit ActorTask(Handle h):handle_(h){}
  ActorTask(ActorTask&& other) noexcept:handle_(std::exchange(other.handle_,{})){}
  ActorTask& operator=(ActorTask&& other) noexcept {
    if(this!=&other){if(handle_)handle_.destroy();handle_=std::exchange(other.handle_,{});}return *this;
  }
  ActorTask(const ActorTask&)=delete;
  ~ActorTask(){if(handle_)handle_.destroy();}
  Handle handle() const{return handle_;}
  T Take(){auto& p=handle_.promise();if(p.error)std::rethrow_exception(p.error);return std::move(*p.value);}
  struct Awaiter {
    Handle child;
    Awaiter(Awaiter&& other) noexcept:child(std::exchange(other.child,{})){}
    explicit Awaiter(Handle h):child(h){}
    ~Awaiter(){if(child)child.destroy();}
    bool await_ready() noexcept{return false;}
    std::coroutine_handle<> await_suspend(std::coroutine_handle<> parent) noexcept {
      child.promise().continuation=parent;return child;
    }
    T await_resume(){auto& p=child.promise();if(p.error)std::rethrow_exception(p.error);return std::move(*p.value);}
  };
  Awaiter operator co_await() && noexcept{return Awaiter{std::exchange(handle_,{})};}
 private:
  Handle handle_;
};

template<class T> struct IsActorTask {static constexpr bool value=false;};
template<class T> struct IsActorTask<ActorTask<T>> {static constexpr bool value=true;};

}  // namespace eps::detail
