#include "common/resumable_actor.hpp"
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace eps;
using namespace eps::detail;
using namespace std::chrono_literals;
namespace {
void Check(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class T>T Get(std::future<Result<T>> value) {
  Check(value.wait_for(3s)==std::future_status::ready,"actor future did not complete");
  auto result=value.get();if(auto error=std::get_if<Error>(&result))throw std::runtime_error(error->message);
  return std::get<T>(std::move(result));
}
template<class T>Error ErrorOf(std::future<Result<T>> value) {
  Check(value.wait_for(3s)==std::future_status::ready,"actor error did not complete");
  auto result=value.get();Check(std::holds_alternative<Error>(result),"expected actor error");return std::get<Error>(result);
}
ActorResources Rows(std::initializer_list<unsigned> rows) {
  ActorResources r;for(auto row:rows)r.keys.push_back({ActorResource::Kind::row,1,row});return r;
}
struct Gate {
  std::promise<Result<int>> result;
  std::promise<void> entered;
  std::future<void> started=entered.get_future();
};
ActorTask<int> Waiting(ResumableActor& actor,Gate& gate,std::thread::id* resumed=nullptr) {
  auto batch=co_await actor.WaitAll<int>(1,[&](auto& pending) {
    pending.push_back(gate.result.get_future());gate.entered.set_value();
  });
  if(resumed)*resumed=std::this_thread::get_id();
  if(auto error=std::get_if<Error>(&batch.at(0)))throw Failure(*error);
  co_return std::get<int>(batch.at(0));
}
void WaitStarted(Gate& gate){Check(gate.started.wait_for(3s)==std::future_status::ready,"wait producer did not start");}

void IndependentAndFair() {
  ResumableActor actor({});Gate first,second;std::thread::id owner,resumed;
  Get(actor.Submit<int>([&]{owner=std::this_thread::get_id();return 0;}));
  auto a=actor.Submit<int>([&]{return Waiting(actor,first,&resumed);},[]{return Rows({1});});WaitStarted(first);
  auto b=actor.Submit<int>([&]{return Waiting(actor,second);},[]{return Rows({1,2});});
  auto c=actor.Submit<int>([]{return 3;},[]{return Rows({2});});
  auto d=actor.Submit<int>([]{return 4;},[]{return Rows({3});});
  Check(Get(std::move(d))==4,"independent request was blocked by an external wait");
  Check(b.wait_for(0s)!=std::future_status::ready && c.wait_for(0s)!=std::future_status::ready,
    "a conflicting request overtook its predecessor");
  auto during=actor.Snapshot();Check(during.active==1 && during.waiting==1 && during.queued==2,"suspended actor accounting changed");
  first.result.set_value(1);Check(Get(std::move(a))==1,"first continuation changed value");WaitStarted(second);
  Check(c.wait_for(0s)!=std::future_status::ready,"younger conflicting work overtook an active predecessor");
  second.result.set_value(2);Check(Get(std::move(b))==2 && Get(std::move(c))==3,"dependency release lost a request");
  Check(owner==resumed,"external completion resumed outside the actor thread");
  auto after=actor.Snapshot();Check(after.active==0 && after.waiting==0 && after.dependency_bytes==0,"completed actor resources leaked");
}

void BarrierAndPayload() {
  ResumableActor::Options options;options.max_queued=1;options.max_active=1;options.payload_bytes=10;
  ResumableActor actor(options);Gate gate;
  auto a=actor.Submit<int>([&]{return Waiting(actor,gate);},[]{return Rows({1});},6);WaitStarted(gate);
  Check(ErrorOf(actor.Submit<int>([]{return 0;},[]{return Rows({2});},5)).code==Error::Code::capacity_exceeded,
    "active request payload escaped admission accounting");
  auto b=actor.Submit<int>([]{return 2;},[]{return Rows({2});},4);
  Check(ErrorOf(actor.Submit<int>([]{return 3;},[]{return Rows({3});})).code==Error::Code::capacity_exceeded,"queued job count was unbounded");
  auto during=actor.Snapshot();Check(during.active_payload==6 && during.queued_payload==4 && during.rejected==2,"payload/rejection counters changed");
  gate.result.set_value(1);Check(Get(std::move(a))==1 && Get(std::move(b))==2,"bounded requests failed after release");

  ResumableActor barriers({});Gate held;std::atomic<unsigned> order{0};
  auto first=barriers.Submit<int>([&]{return Waiting(barriers,held);},[]{return Rows({1});});WaitStarted(held);
  auto barrier=barriers.Submit<int>([&]{Check(order.fetch_add(1)==0,"barrier ran after a younger request");return 2;});
  auto later=barriers.Submit<int>([&]{Check(order.fetch_add(1)==1,"request crossed the queued barrier");return 3;},[]{return Rows({7});});
  Check(later.wait_for(30ms)!=std::future_status::ready,"barrier allowed younger independent admission");
  held.result.set_value(1);Get(std::move(first));Get(std::move(barrier));Get(std::move(later));

  options={};options.dependency_bytes=1;ResumableActor tiny(options);
  Check(ErrorOf(tiny.Submit<int>([]{return 0;},[]{return Rows({1});})).code==Error::Code::capacity_exceeded,"oversized dependency metadata was admitted");
  Check(Get(tiny.Submit<int>([]{return 7;},[]{return ActorResources{};}))==7,"dependency rejection stalled later work");
}

ActorTask<int> Nested(ResumableActor& actor,Gate& gate,unsigned depth,std::thread::id owner) {
  Check(std::this_thread::get_id()==owner,"nested task started off actor");
  int result=depth?co_await Nested(actor,gate,depth-1,owner):co_await Waiting(actor,gate);
  Check(std::this_thread::get_id()==owner,"nested task resumed off actor");co_return result+1;
}
void NestedAndFailureCleanup() {
  std::atomic<unsigned> failures{0};std::thread::id owner;
  ResumableActor actor({},[&](const Error&){Check(owner==std::this_thread::get_id(),"failure observer ran off actor");++failures;});
  Get(actor.Submit<int>([&]{owner=std::this_thread::get_id();return 0;}));
  Gate nested;auto task=actor.Submit<int>([&]{return Nested(actor,nested,12,owner);});WaitStarted(nested);
  nested.result.set_value(10);Check(Get(std::move(task))==23,"nested task lost a result or continuation");
  Check(ErrorOf(actor.Submit<int>([]()->int{throw Failure({Error::Code::invalid_argument,"invalid input"});})).code==Error::Code::invalid_argument,"typed failure mapping changed");
  Check(ErrorOf(actor.Submit<int>([]()->int{throw std::bad_alloc();})).code==Error::Code::capacity_exceeded,"allocation error mapping changed");

  std::promise<Result<int>> held,failed;std::promise<void> dispatched;auto started=dispatched.get_future();
  auto fanout=actor.Submit<int>([&]()->ActorTask<int>{
    auto values=co_await actor.WaitAll<int>(2,[&](auto& pending) {
      pending.push_back(failed.get_future());pending.push_back(held.get_future());dispatched.set_value();
      throw std::runtime_error("second dispatch failed after submission");
    });co_return values.size();
  });
  Check(started.wait_for(3s)==std::future_status::ready,"fan-out dispatch did not run");
  failed.set_value(Error{Error::Code::unavailable,"owner failed"});
  Check(fanout.wait_for(30ms)!=std::future_status::ready,"a dispatch failure abandoned another submitted future");
  held.set_value(2);Check(ErrorOf(std::move(fanout)).code==Error::Code::run_failed,"dispatch exception was lost");
  Check(failures==3,"completed failure notifications were lost");
  Check(Get(actor.Submit<int>([]{return 42;}))==42,"failure poisoned generic actor scheduling");
}

void ShutdownDrains() {
  auto actor=std::make_unique<ResumableActor>(ResumableActor::Options{});auto* raw=actor.get();Gate gate;
  auto first=actor->Submit<int>([&]{return Waiting(*raw,gate);},[]{return Rows({1});});WaitStarted(gate);
  auto later=actor->Submit<int>([]{return 2;},[]{return Rows({1});});
  std::promise<void> started,finished;auto closing=started.get_future(),closed=finished.get_future();
  std::thread shutdown([&]{started.set_value();actor.reset();finished.set_value();});closing.wait();
  Check(closed.wait_for(30ms)!=std::future_status::ready,"actor destroyed a live continuation");
  gate.result.set_value(1);Check(Get(std::move(first))==1 && Get(std::move(later))==2,"shutdown dropped accepted work");
  Check(closed.wait_for(3s)==std::future_status::ready,"actor shutdown did not join completion workers");shutdown.join();
}
void ExclusiveUpgrade() {
  ResumableActor actor({});Gate held,upgrading;std::promise<void> entering;auto entering_future=entering.get_future();
  auto first=actor.Submit<int>([&]{return Waiting(actor,held);},[]{return Rows({1});});WaitStarted(held);
  auto exclusive=actor.Submit<int>([&]()->ActorTask<int>{
    co_await Waiting(actor,upgrading);entering.set_value();
    co_await actor.Exclusive();co_return 2;
  },[]{auto r=Rows({2});r.keys.push_back({ActorResource::Kind::maintenance,0,0});return r;});
  WaitStarted(upgrading);upgrading.result.set_value(2);
  Check(entering_future.wait_for(3s)==std::future_status::ready,"exclusive upgrade did not start");
  auto later=actor.Submit<int>([]{return 3;},[]{return Rows({3});});
  Check(later.wait_for(30ms)!=std::future_status::ready && exclusive.wait_for(0s)!=std::future_status::ready,
    "exclusive upgrade crossed an active operation or admitted younger work");
  held.result.set_value(1);Get(std::move(first));Check(Get(std::move(exclusive))==2 && Get(std::move(later))==3,"exclusive upgrade failed to resume after drain");
}
}
int main() {
  try{IndependentAndFair();BarrierAndPayload();NestedAndFailureCleanup();ExclusiveUpgrade();ShutdownDrains();
    std::cout<<"PASS resumable actor: independent progress, FIFO conflicts, barriers, bounded admission, nested continuations and drain\n";return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL actor: "<<e.what()<<'\n';return 1;}
}
