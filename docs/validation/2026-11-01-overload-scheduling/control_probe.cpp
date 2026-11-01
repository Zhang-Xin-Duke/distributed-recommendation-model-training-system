#include "common/resumable_actor.hpp"
#include <iostream>
using namespace eps;using namespace eps::detail;using namespace std::chrono_literals;
ActorTask<int> Hold(ResumableActor& a,std::promise<Result<int>>& p,std::promise<void>& entered){
 auto values=co_await a.WaitAll<int>(1,[&](auto& f){f.push_back(p.get_future());entered.set_value();});co_return std::get<int>(values[0]);}
int main(){ResumableActor::Options o;o.max_active=1;ResumableActor a(o);std::promise<Result<int>> p;std::promise<void> entered;auto ready=entered.get_future();
 auto held=a.Submit<int>([&]{return Hold(a,p,entered);});ready.wait();
#ifdef EPS_OBSERVATION_LANE
 auto read=a.Observe<int>([]{return 7;});
#else
 auto read=a.Submit<int>([]{return 7;},[]{return ActorResources{};});
#endif
 bool observed=read.wait_for(100ms)==std::future_status::ready;p.set_value(1);auto h=held.get();auto r=read.get();
 std::cout<<"{\"observation_completed_while_data_slot_held\":"<<(observed?"true":"false")<<",\"read_result\":"<<std::get<int>(r)<<"}\n";}
