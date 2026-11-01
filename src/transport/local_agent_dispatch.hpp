#pragma once
#include "common/agent_admission.hpp"
#include "transport/completion_executor.hpp"

namespace eps::transport {
// Native input accounting includes reserved vector capacity retained by the job.
inline std::uint64_t GatherInputBytes(const std::vector<AgentWork>& work,
    const std::vector<FeatureInput>& layouts,const std::string& run){
  std::uint64_t bytes=sizeof(work)+sizeof(layouts)+sizeof(run);
  auto add=[&](std::uint64_t count,std::uint64_t width){
    detail::Require(count<=(UINT64_MAX-bytes)/width,"native gather size overflow");
    bytes+=count*width;
  };
  add(run.capacity(),1);add(work.capacity(),sizeof(AgentWork));add(layouts.capacity(),sizeof(FeatureInput));
  for(const auto& task:work){add(task.rows.capacity(),sizeof(LocatedRow));
    add(task.occurrence_to_unique.capacity(),sizeof(std::uint64_t));
    add(task.transferred_weights.capacity(),sizeof(float));}
  for(const auto& f:layouts){add(f.keys.capacity(),sizeof(Key));add(f.offsets.capacity(),sizeof(std::uint64_t));
    add(f.valid_lengths.capacity(),sizeof(std::uint64_t));}
  return bytes;
}
template<typename T>struct LocalAgentCompletion final:CompletionExecutor::Task {
  std::shared_ptr<void> permit;
  std::shared_ptr<std::promise<Result<T>>> output;
  std::future<Result<T>> input;
  void Run() override {
    try{output->set_value(input.get());}
    catch(const std::exception& e){output->set_value(Error{Error::Code::run_failed,e.what()});}
  }
};
template<typename T,typename Start>
std::future<Result<T>> LocalAgentDispatch(detail::AgentAdmissionQueue& admission,
    const std::string& run,std::vector<AgentWork> work,std::vector<FeatureInput> layouts,Start start){
  auto output=std::make_shared<std::promise<Result<T>>>();auto future=output->get_future();
  try{
    const auto bytes=GatherInputBytes(work,layouts,run);
    admission.Submit(bytes,[output,start=std::move(start),run,work=std::move(work),layouts=std::move(layouts)]
        (std::shared_ptr<void> permit) mutable {
      try{
        CompletionExecutor::Reservation reservation(CompletionExecutor::Instance());
        auto task=std::make_unique<LocalAgentCompletion<T>>();
        task->permit=std::move(permit);task->output=output;
        task->input=start(run,std::move(work),std::move(layouts));
        reservation.Publish(std::move(task));
      }catch(const detail::Failure& e){output->set_value(e.error());}
       catch(const std::bad_alloc&){output->set_value(Error{Error::Code::capacity_exceeded,"native gather allocation failed"});}
       catch(const std::exception& e){output->set_value(Error{Error::Code::run_failed,e.what()});}
    });
  }catch(const detail::Failure& e){output->set_value(e.error());}
   catch(const std::bad_alloc&){output->set_value(Error{Error::Code::capacity_exceeded,"native gather admission allocation failed"});}
   catch(const std::exception& e){output->set_value(Error{Error::Code::run_failed,e.what()});}
  return future;
}
}  // namespace eps::transport
