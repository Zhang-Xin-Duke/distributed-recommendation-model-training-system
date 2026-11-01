#include "common/agent_admission.hpp"
#include <atomic>
#include <future>
#include <iostream>
#include <vector>
using namespace eps;
using eps::detail::AgentAdmissionQueue;
void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<typename F>void Capacity(F fn){
  try{fn();throw std::runtime_error("expected admission rejection");}
  catch(const eps::detail::Failure& e){Check(e.error().code==Error::Code::capacity_exceeded,"wrong admission error");}
}
int main(){try{
  {
    AgentAdmissionQueue q({1,3,100});
    std::promise<std::shared_ptr<void>> acquired;auto held=acquired.get_future();
    std::vector<int> order;
    q.Submit(40,[&](auto permit){order.push_back(1);acquired.set_value(std::move(permit));});
    auto permit=held.get();
    q.Submit(20,[&](auto){order.push_back(2);});
    q.Submit(20,[&](auto){order.push_back(3);});
    Capacity([&]{q.Submit(0,[](auto){});});
    auto s=q.Stats();Check(s.active_operations==1 && s.queued_operations==2 &&
      s.pending_request_bytes==80,"permit did not follow asynchronous completion");
    permit.reset();q.Shutdown();s=q.Stats();
    Check(order==std::vector<int>({1,2,3}),"admission start order changed");
    Check(s.active_operations==0 && s.queued_operations==0 && s.pending_request_bytes==0 &&
      s.peak_active_operations==1 && s.peak_pending_operations==3 && s.peak_request_bytes==80 &&
      s.accepted_requests==3 && s.completed_requests==3 && s.rejected_requests==1,"count budget accounting");
    try{q.Submit(1,[](auto){});throw std::runtime_error("accepted after shutdown");}
    catch(const eps::detail::Failure& e){Check(e.error().code==Error::Code::unavailable,"closing status");}
  }
  {
    AgentAdmissionQueue q({1,3,50});
    std::promise<std::shared_ptr<void>> acquired;auto held=acquired.get_future();
    q.Submit(40,[&](auto permit){acquired.set_value(std::move(permit));});
    auto permit=held.get();Capacity([&]{q.Submit(11,[](auto){});});
    q.Submit(10,[](auto){});
    auto stopped=std::async(std::launch::async,[&]{q.Shutdown();});
    Check(stopped.wait_for(std::chrono::milliseconds(20))==std::future_status::timeout,"shutdown discarded active request");
    permit.reset();stopped.get();auto s=q.Stats();
    Check(s.accepted_requests==2 && s.completed_requests==2 && s.peak_request_bytes==50 &&
      s.pending_request_bytes==0,"byte budget or shutdown drain");
  }
  {
    AgentAdmissionQueue q({1,4,100});
    q.Submit(1,[](auto){throw std::runtime_error("dispatch test exception");});
    std::atomic<int> ran=0;q.Submit(1,[&](auto){++ran;});q.Shutdown();auto s=q.Stats();
    Check(ran==1 && s.dispatch_failures==1 && s.completed_requests==2 &&
      s.pending_request_bytes==0,"exception leaked admission permit");
  }
  {
    AgentAdmissionQueue q({4,64,512});std::atomic<std::uint64_t> ran=0,rejected=0;
    std::vector<std::thread> producers;
    for(int i=0;i<8;++i)producers.emplace_back([&]{for(int j=0;j<512;++j){
      try{q.Submit(8,[&](auto){++ran;});}
      catch(const eps::detail::Failure& e){Check(e.error().code==Error::Code::capacity_exceeded,"unexpected concurrent status");++rejected;}
    }});
    for(auto& t:producers)t.join();q.Shutdown();auto s=q.Stats();
    Check(s.accepted_requests+s.rejected_requests==4096 && s.accepted_requests==ran &&
      s.completed_requests==ran && s.rejected_requests==rejected && s.peak_active_operations<=4 &&
      s.peak_pending_operations<=64 && s.peak_request_bytes<=512 &&
      s.active_operations==0 && s.queued_operations==0 && s.pending_request_bytes==0,"concurrent admission accounting");
  }
  std::cout<<"Agent admission: FIFO, completion lifetime, count/byte limits, exception cleanup and shutdown passed\n";
  return 0;
}catch(const std::exception& e){std::cerr<<"FAIL admission: "<<e.what()<<'\n';return 1;}}
