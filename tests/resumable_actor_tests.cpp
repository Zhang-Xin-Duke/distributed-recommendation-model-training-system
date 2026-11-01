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
void DependencyPlanCache() {
  for(bool cached:{false,true}) {
    ResumableActor::Options options;options.cache_dependencies=cached;ResumableActor actor(options);
    Gate gate;std::atomic<unsigned> builds{0};
    auto a=actor.Submit<int>([&]{return Waiting(actor,gate);},[]{return Rows({1});});WaitStarted(gate);
    auto b=actor.Submit<int>([]{return 2;},[&]{++builds;return Rows({1});});
    for(unsigned n=0;n<12;++n)Get(actor.Submit<int>([]{return 3;},[]{return Rows({3});}));
    Check(cached?builds==1:builds>1,"queued dependency plans were not cached/selectably rebuilt");
    if(cached) {
      Check(actor.Snapshot().dependency_cache_hits>0,"dependency cache hits were not recorded");
      Get(actor.Submit<int>([&]{actor.InvalidateDependencies();return 0;},[]{return Rows({4});}));
      Get(actor.Submit<int>([]{return 0;},[]{return Rows({4});}));
      Check(builds==2,"ownership/policy epoch did not invalidate a queued plan");
    }
    gate.result.set_value(1);Get(std::move(a));Get(std::move(b));
    auto stats=actor.Snapshot();Check(stats.dependency_bytes==0 && stats.dependency_rebuilds>0 &&
      stats.dependency_prepare_ns>0,"dependency cache accounting or timing was lost");
  }
  ResumableActor::Options options;options.dependency_bytes=3*sizeof(ActorResource);ResumableActor bounded(options);Gate gate;
  auto a=bounded.Submit<int>([&]{return Waiting(bounded,gate);},[]{return Rows({1});});WaitStarted(gate);
  auto b=bounded.Submit<int>([]{return 2;},[]{return Rows({1});});
  auto c=bounded.Submit<int>([]{return 3;},[]{return Rows({1,2});});
  auto d=bounded.Submit<int>([]{return 4;},[]{return Rows({3});});
  Check(bounded.Snapshot().dependency_bytes<=options.dependency_bytes,"queued plans exceeded their shared byte budget");
  gate.result.set_value(1);Get(std::move(a));Get(std::move(b));Get(std::move(c));Get(std::move(d));
  Check(bounded.Snapshot().dependency_bytes==0,"evicted queued plans leaked their reservation");

  ResumableActor unknown({});Gate held;bool barrier=false;std::atomic<unsigned> rebuilt{0};
  auto first=unknown.Submit<int>([&]{return Waiting(unknown,held);},[]{return Rows({1});});WaitStarted(held);
  auto late=unknown.Submit<int>([]{return 2;},[&]{++rebuilt;auto r=Rows({1});r.cacheable=false;r.barrier=barrier;return r;});
  Get(unknown.Submit<int>([&]{barrier=true;return 0;},[]{return Rows({3});}));
  auto younger=unknown.Submit<int>([]{return 3;},[]{return Rows({4});});
  Check(younger.wait_for(30ms)!=std::future_status::ready && rebuilt>=2,"an incomplete plan was cached as a stable dependency set");
  held.result.set_value(1);Get(std::move(first));Get(std::move(late));Get(std::move(younger));
}
void OverloadObservationsAndPlanRetention() {
  ResumableActor::Options options;options.max_active=2;options.max_queued=128;
  options.dependency_bytes=9*sizeof(ActorResource);
  ResumableActor actor(options);Gate held;
  auto active=actor.Submit<int>([&]{return Waiting(actor,held);},[]{return Rows({1});});WaitStarted(held);
  std::atomic<unsigned> builds{0};std::vector<std::future<Result<int>>> queued;
  for(unsigned i=0;i<32;++i)queued.push_back(actor.Submit<int>([i]{return int(i);},[&,i]{++builds;return Rows({1,100+i});}));
  // Empty-dependency observations must pass the blocked, budget-limited prefix.
  for(unsigned i=0;i<16;++i) {
    auto snapshot=Get(actor.Observe<ResumableActor::Stats>([&]{return actor.Snapshot();}));
    Check(snapshot.active==1 && snapshot.waiting==1 && snapshot.queued==32 &&
      snapshot.dependency_bytes<=options.dependency_bytes,"observation changed held data work or escaped its budget");
  }
  auto before=Get(actor.Observe<unsigned>([&]{return builds.load();}));
  held.result.set_value(9);Get(std::move(active));
  for(unsigned i=0;i<32;++i)Check(Get(std::move(queued[i]))==int(i),"budget-limited FIFO request changed");
  Check(before<=5,"blocked prefix plans were repeatedly discarded/rebuilt");
  Check(actor.Snapshot().dependency_bytes==0,"plan retention leaked after drain");

  options.max_active=1;ResumableActor full(options);Gate gate;
  auto first=full.Submit<int>([&]{return Waiting(full,gate);});WaitStarted(gate);
  auto stats=Get(full.Observe<ResumableActor::Stats>([&]{return full.Snapshot();}));
  gate.result.set_value(1);Get(std::move(first));
  Check(stats.active==1 && stats.waiting==1,"full active set prevented state observation");
}
void BoundedObservationFairness() {
  ResumableActor::Options options;options.max_observations=2;
  ResumableActor actor(options);std::promise<void> entered,release;
  auto started=entered.get_future(),open=release.get_future();std::vector<int> order;
  auto held=actor.Submit<int>([&]{entered.set_value();open.wait();return 0;});started.wait();
  auto a=actor.Observe<int>([&]{order.push_back(1);return 1;});
  auto data=actor.Submit<int>([&]{order.push_back(2);return 2;});
  auto b=actor.Observe<int>([&]{order.push_back(3);return 3;});
  auto rejected=actor.Observe<int>([]{return 4;});
  const auto snapshot=actor.Snapshot();release.set_value();
  Get(std::move(held));Get(std::move(a));Get(std::move(data));Get(std::move(b));
  Check(ErrorOf(std::move(rejected)).code==Error::Code::capacity_exceeded &&
    snapshot.queued_observations==2 && snapshot.rejected_observations==1,"observation admission was unbounded");
  Check(order==std::vector<int>({1,2,3}),"observations starved runnable data work");
  Check(ErrorOf(actor.Observe<int>([]()->int{throw std::bad_alloc();})).code==Error::Code::capacity_exceeded,
    "observation allocation failure was lost");
  Check(Get(actor.Submit<int>([]{return 7;}))==7,"failed observation poisoned data work");
  actor.Shutdown();
  Check(ErrorOf(actor.Observe<int>([]{return 0;})).code==Error::Code::unavailable,"observation accepted after shutdown");
}
void GrowingOldestPlanReclaimsYoungerCache() {
  ResumableActor::Options options;options.dependency_bytes=4*sizeof(ActorResource);
  ResumableActor actor(options);Gate held;bool grew=false;std::promise<void> planned;
  auto seen=planned.get_future();bool notified=false;
  auto active=actor.Submit<int>([&]()->ActorTask<int>{
    co_await Waiting(actor,held);grew=true;actor.InvalidateDependencies();co_return 1;
  },[]{return Rows({1});});WaitStarted(held);
  auto oldest=actor.Submit<int>([]{return 2;},[&]{return grew?Rows({1,2,3}):Rows({1});});
  auto younger=actor.Submit<int>([]{return 3;},[&]{if(!notified){notified=true;planned.set_value();}return Rows({1,4});});
  Check(seen.wait_for(3s)==std::future_status::ready,"younger cached plan was not prepared");
  held.result.set_value(1);Get(std::move(active));
  Check(Get(std::move(oldest))==2 && Get(std::move(younger))==3,
    "growing oldest plan stalled behind younger cached reservations");
  Check(actor.Snapshot().dependency_bytes==0,"reclaimed younger cache leaked bytes");
}
}
int main() {
  try{IndependentAndFair();BarrierAndPayload();NestedAndFailureCleanup();ExclusiveUpgrade();DependencyPlanCache();OverloadObservationsAndPlanRetention();BoundedObservationFairness();GrowingOldestPlanReclaimsYoungerCache();ShutdownDrains();
    std::cout<<"PASS resumable actor: independent progress, FIFO conflicts, barriers, bounded admission, nested continuations and drain\n";return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL actor: "<<e.what()<<'\n';return 1;}
}
