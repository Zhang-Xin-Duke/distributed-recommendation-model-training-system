#include "access/prepare_queue.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <set>
#include "common/error.hpp"

namespace eps::detail {
namespace {
template<typename T>T Get(Result<T> value) {
  if(auto e=std::get_if<Error>(&value))throw Failure(*e);
  return std::get<T>(std::move(value));
}
void Add(std::uint64_t& total,std::uint64_t n,std::uint64_t size,std::uint64_t limit) {
  Require(size && n<=limit/size && total<=limit-n*size,
      "adapter preparation byte budget exceeded",Error::Code::capacity_exceeded);
  total+=n*size;
}
std::uint64_t Estimate(const std::vector<entropy::Feature>& features,std::uint32_t shards,PreparationLimits limits) {
  Require(shards && shards<=1024 && !features.empty() && features.size()<=limits.max_features,
    "invalid adapter preparation dimensions");
  std::uint64_t input=0,prepared=0;
  Add(input,features.capacity(),sizeof(entropy::Feature),limits.max_input_bytes);
  std::set<std::uint32_t> ids;
  for(const auto& f:features) {
    Require(ids.insert(f.feature).second,"duplicate adapter feature");
    Require(f.pooling==Pooling::sum || f.pooling==Pooling::mean || f.pooling==Pooling::sequence,"invalid adapter pooling");
    Require(!f.offsets.empty() && f.offsets.front()==0 && f.offsets.back()==f.keys.size() &&
      std::is_sorted(f.offsets.begin(),f.offsets.end()),"invalid adapter row splits");
    Add(input,f.keys.capacity(),8,limits.max_input_bytes);
    Add(input,f.offsets.capacity(),8,limits.max_input_bytes);
  }
  Add(prepared,1,sizeof(PreparedLookup),limits.max_prepared_bytes);
  Add(prepared,features.size(),sizeof(FeatureInput),limits.max_prepared_bytes);
  Add(prepared,shards,sizeof(std::vector<FeatureInput>)+sizeof(std::vector<std::vector<std::uint64_t>>),limits.max_prepared_bytes);
  Add(prepared,std::uint64_t(shards)*features.size(),sizeof(FeatureInput)+sizeof(std::vector<std::uint64_t>),limits.max_prepared_bytes);
  for(const auto& f:features) {
    Add(prepared,f.keys.size(),24,limits.max_prepared_bytes);
    Add(prepared,std::uint64_t(shards+1)*(2*f.offsets.size()-1),8,limits.max_prepared_bytes);
  }
  // Payload/container reservation: original input, prepared buffers, padding
  // maps and temporary scratch. Two copies of derived metadata cover conversion
  // scratch and capacity rounding; job count also bounds allocation headers.
  std::uint64_t derived=prepared;
  for(const auto& f:features)Add(derived,f.keys.size(),8,UINT64_MAX);
  Add(derived,features.size(),sizeof(std::vector<std::uint64_t>)+sizeof(std::uint64_t)+sizeof(Pooling)+64,UINT64_MAX);
  Add(derived,shards,8,UINT64_MAX);
  Add(input,derived,2,UINT64_MAX);
  return input;
}
}
Result<std::uint64_t> EstimateAdapterPreparation(const std::vector<entropy::Feature>& f,std::uint32_t shards,PreparationLimits limits) {
  try{return Estimate(f,shards,limits);}catch(const Failure& e){return e.error();}
  catch(const std::bad_alloc&){return Error{Error::Code::capacity_exceeded,"preparation estimate allocation failed"};}
}
Result<PreparedAdapterInput> PrepareAdapterInput(std::vector<entropy::Feature> features,std::uint32_t shards,PreparationLimits limits) {
  try {
    (void)Estimate(features,shards,limits);
    PreparedAdapterInput out;std::vector<FeatureInput> compact;
    compact.reserve(features.size());out.positions.resize(features.size());
    out.original_sizes.reserve(features.size());out.pooling.reserve(features.size());
    for(std::size_t fi=0;fi<features.size();++fi) {
      const auto& f=features[fi];FeatureInput input{f.table,f.feature,f.pooling,{},{},{}};
      input.keys.reserve(f.keys.size());input.offsets.reserve(f.offsets.size());
      input.valid_lengths.reserve(f.offsets.size()-1);input.offsets.push_back(0);
      auto& positions=out.positions[fi];
      if(f.pooling==Pooling::sequence)positions.reserve(f.keys.size());
      for(std::size_t b=0;b+1<f.offsets.size();++b) {
        const auto begin=input.keys.size();
        for(auto i=f.offsets[b];i<f.offsets[b+1];++i)
          if(!f.padding_key || f.keys[i]!=*f.padding_key) {
            input.keys.push_back(f.keys[i]);
            if(f.pooling==Pooling::sequence)positions.push_back(i);
          }
        input.valid_lengths.push_back(input.keys.size()-begin);input.offsets.push_back(input.keys.size());
      }
      out.original_sizes.push_back(f.keys.size());out.pooling.push_back(f.pooling);
      compact.push_back(std::move(input));
    }
    out.lookup=Get(PrepareLookupInputs(std::move(compact),shards,limits));return out;
  }catch(const Failure& e){return e.error();}
  catch(const std::bad_alloc&){return Error{Error::Code::capacity_exceeded,"adapter preparation allocation failed"};}
}

struct PrepareQueue::Job {
  enum class State {queued,running,ready};
  std::uint64_t token,bytes;
  std::vector<entropy::Feature> input;
  std::optional<Result<PreparedAdapterInput>> result;
  State state=State::queued;
  bool claimed=false,cancelled=false,reserved=true;
};
PrepareQueue::PrepareQueue(std::uint32_t shards,entropy::PreparationOptions options,Processor processor)
    :shards_(shards),options_(options),processor_(std::move(processor)) {
  Require(shards && shards<=1024 && options.max_pending_batches && options.payload_bytes && options.lookup_limits.max_input_bytes &&
    options.lookup_limits.max_prepared_bytes && options.lookup_limits.max_features,
    "invalid preparation queue options");
  if(!processor_)processor_=[=](auto input){return PrepareAdapterInput(std::move(input),shards,options.lookup_limits);};
  worker_=std::thread([this]{Run();});
}
PrepareQueue::~PrepareQueue() {
  {std::lock_guard lock(mutex_);closed_=true;stopping_=true;CancelAll();}
  changed_.notify_all();worker_.join();
}
void PrepareQueue::Reclaim(Job& job) {
  if(!job.reserved)return;
  job.reserved=false;--stats_.reserved_jobs;stats_.reserved_bytes-=job.bytes;
}
void PrepareQueue::CancelAll() {
  for(auto& [token,job]:pending_) {
    if(!job->cancelled){job->cancelled=true;++stats_.cancelled;}
    if(job->state!=Job::State::running) {
      job->input={};job->result.reset();Reclaim(*job);
    }
  }
  pending_.clear();jobs_.clear();
}
Result<std::uint64_t> PrepareQueue::Submit(std::vector<entropy::Feature> features) {
  auto estimate=EstimateAdapterPreparation(features,shards_,options_.lookup_limits);
  std::lock_guard lock(mutex_);
  try {
    Require(!closed_,"preparation admission is closed",Error::Code::unavailable);
    const auto bytes=Get(std::move(estimate));
    Require(next_!=UINT64_MAX && stats_.reserved_jobs<options_.max_pending_batches &&
      bytes<=options_.payload_bytes && stats_.reserved_bytes<=options_.payload_bytes-bytes,
      "preparation queue is full",Error::Code::capacity_exceeded);
    auto job=std::make_shared<Job>();job->token=next_;job->bytes=bytes;job->input=std::move(features);
    pending_.emplace(next_,job);
    try {jobs_.push_back(job);}catch(...){pending_.erase(next_);throw;}
    ++next_;++stats_.accepted;++stats_.reserved_jobs;stats_.reserved_bytes+=bytes;
    stats_.peak_reserved_bytes=std::max(stats_.peak_reserved_bytes,stats_.reserved_bytes);
    changed_.notify_one();return job->token;
  }catch(const Failure& e){++stats_.rejected;return e.error();}
  catch(const std::bad_alloc&){++stats_.rejected;return Error{Error::Code::capacity_exceeded,"preparation admission allocation failed"};}
}
void PrepareQueue::Run() {
  for(;;) {
    std::shared_ptr<Job> job;
    {std::unique_lock lock(mutex_);changed_.wait(lock,[&]{return stopping_||!jobs_.empty();});
      if(stopping_)return;
      job=jobs_.front();jobs_.pop_front();job->state=Job::State::running;}
    const auto begin=std::chrono::steady_clock::now();
    Result<PreparedAdapterInput> result=Error{Error::Code::run_failed,"preparation did not complete"};
    try {result=processor_(std::move(job->input));}
    catch(const Failure& e){result=e.error();}
    catch(const std::bad_alloc&){result=Error{Error::Code::capacity_exceeded,"preparation worker allocation failed"};}
    catch(const std::exception& e){result=Error{Error::Code::run_failed,e.what()};}
    {std::lock_guard lock(mutex_);
      stats_.prepare_nanoseconds+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
      if(job->cancelled) {
        result=Error{Error::Code::unavailable,"preparation cancelled"};job->input={};Reclaim(*job);
      }else {
        if(std::holds_alternative<Error>(result))++stats_.failed;else ++stats_.completed;
        job->result.emplace(std::move(result));job->state=Job::State::ready;
      }
    }changed_.notify_all();
  }
}
Result<PreparedAdapterInput> PrepareQueue::Take(std::uint64_t token) {
  std::unique_lock lock(mutex_);
  auto it=pending_.find(token);
  if(it==pending_.end())return Error{Error::Code::retired_request,"unknown or cancelled preparation token"};
  if(it!=pending_.begin() || it->second->claimed)return Error{Error::Code::unavailable,"consume prepared inputs in FIFO order"};
  const auto job=it->second;job->claimed=true;
  changed_.wait(lock,[&]{return job->cancelled || job->state==Job::State::ready;});
  if(job->cancelled)return Error{Error::Code::unavailable,"preparation admission was closed"};
  auto result=std::move(*job->result);job->result.reset();pending_.erase(token);Reclaim(*job);++stats_.consumed;
  return result;
}
Result<Ack> PrepareQueue::Cancel(std::uint64_t token) {
  std::lock_guard lock(mutex_);auto it=pending_.find(token);
  if(it==pending_.end())return Error{Error::Code::retired_request,"unknown or consumed preparation token"};
  auto job=it->second;
  if(job->claimed)return Error{Error::Code::unavailable,"prepared input is already claimed"};
  job->cancelled=true;++stats_.cancelled;pending_.erase(it);
  if(job->state!=Job::State::running) {
    std::erase(jobs_,job);job->input={};job->result.reset();Reclaim(*job);
  }
  changed_.notify_all();return Ack{};
}
bool PrepareQueue::Empty() const {std::lock_guard lock(mutex_);return pending_.empty();}
Result<Ack> PrepareQueue::CloseIfEmpty() {
  std::lock_guard lock(mutex_);
  if(!pending_.empty())return Error{Error::Code::unavailable,"consume or cancel prepared inputs before saving"};
  closed_=true;return Ack{};
}
void PrepareQueue::Close() {
  {std::lock_guard lock(mutex_);closed_=true;CancelAll();}changed_.notify_all();
}
entropy::PreparationStats PrepareQueue::Stats() const {
  std::lock_guard lock(mutex_);auto out=stats_;out.pending=pending_.size();
  for(const auto& [token,job]:pending_)out.claimed+=job->claimed;
  return out;
}
}  // namespace eps::detail
