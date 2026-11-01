#include "common/resumable_actor.hpp"
#include <atomic>
#include <iostream>
using namespace eps;using namespace eps::detail;using namespace std::chrono_literals;
ActorResources Row(unsigned key=1){ActorResources r;r.keys.push_back({ActorResource::Kind::row,1,1});if(key!=1)r.keys.push_back({ActorResource::Kind::row,1,key});return r;}
ActorTask<int> Hold(ResumableActor& a,std::promise<Result<int>>& p,std::promise<void>& entered){auto r=co_await a.WaitAll<int>(1,[&](auto& f){f.push_back(p.get_future());entered.set_value();});co_return std::get<int>(r[0]);}
int main(){ResumableActor::Options o;o.max_active=2;o.dependency_bytes=9*sizeof(ActorResource);ResumableActor a(o);
 std::promise<Result<int>> p;std::promise<void> entered;auto ready=entered.get_future();std::atomic<unsigned> builds{0};
 auto held=a.Submit<int>([&]{return Hold(a,p,entered);},[]{return Row();});ready.wait();
 std::vector<std::future<Result<int>>> work;
 for(unsigned i=0;i<5;++i)work.push_back(a.Submit<int>([]{return 1;},[&,i]{++builds;return Row(100+i);}));
 auto deadline=std::chrono::steady_clock::now()+2s;while(builds<5 && std::chrono::steady_clock::now()<deadline)std::this_thread::sleep_for(1ms);
 for(int i=0;i<10;++i){work.push_back(a.Submit<int>([]{return 2;},[]{return ActorResources{};}));std::this_thread::sleep_for(10ms);}
 auto before=builds.load();p.set_value(1);held.get();for(auto& f:work)f.get();
 std::cout<<"{\"blocked_prefix_dependency_builds_before_release\":"<<before<<",\"input_prefix_jobs\":5,\"wake_notifications\":10,\"final_dependency_bytes\":"<<a.Snapshot().dependency_bytes<<"}\n";}
