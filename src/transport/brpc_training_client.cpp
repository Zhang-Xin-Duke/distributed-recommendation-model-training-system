#include "eps/brpc_training.hpp"

#include "transport/rpc_call.hpp"

namespace eps {
namespace t = transport;
namespace wire = v1;
struct BrpcTrainingProgress::Impl {
  brpc::Channel channel;
  int timeout_ms;
  std::string run;
  std::uint64_t epoch;
  Impl(const std::string& endpoint, std::string id, std::uint64_t generation, int timeout)
      : timeout_ms(timeout), run(std::move(id)), epoch(generation) {
    detail::Require(!run.empty() && run.size() <= 256 && epoch > 0 && timeout > 0,
                    "invalid training RPC options");
    brpc::ChannelOptions options;
    options.protocol = "baidu_std";
    options.timeout_ms = timeout;
    options.max_retry = 2;
    detail::Require(channel.Init(endpoint.c_str(), &options) == 0,
                    "cannot initialize training channel");
  }
  void Control(wire::TrainingControl& request) const {
    request.set_run(run); request.set_epoch(epoch);
  }
  void Task(wire::TrainingTask& request, std::uint32_t worker, std::uint64_t iteration) const {
    request.set_run(run); request.set_epoch(epoch);
    request.set_worker(worker); request.set_iteration(iteration);
  }
};

BrpcTrainingProgress::BrpcTrainingProgress(std::string endpoint, std::string run,
                                         std::uint64_t epoch, int timeout)
    : impl_(std::make_shared<Impl>(endpoint, std::move(run), epoch, timeout)) {}
BrpcTrainingProgress::~BrpcTrainingProgress() = default;

std::future<Result<TrainingPermit>> BrpcTrainingProgress::Begin(std::uint32_t w, std::uint64_t i) {
  return t::Call<wire::TrainingTask, wire::TrainingPermitReply, TrainingPermit>(impl_,
      &wire::TrainingService_Stub::Begin,
      [p=impl_, w, i](auto& q, auto&) { p->Task(q, w, i); },
      [p=impl_, w, i](const auto& r, const auto&) {
        const auto& q = r.task();
        detail::Require(q.run() == p->run && q.epoch() == p->epoch &&
                         q.worker() == w && q.iteration() == i,
                         "training permit identity mismatch");
        return TrainingPermit{{q.run(), q.epoch(), w, i}, r.observed_completion_floor()};
      });
}
std::future<Result<Ack>> BrpcTrainingProgress::Complete(std::uint32_t w, std::uint64_t i) {
  return t::Call<wire::TrainingTask, wire::Ack, Ack>(impl_, &wire::TrainingService_Stub::Complete,
      [p=impl_, w, i](auto& q, auto&) { p->Task(q, w, i); }, t::AckDecoder);
}
std::future<Result<TrainingProgressStats>> BrpcTrainingProgress::Stats() {
  return t::Call<wire::TrainingControl, wire::TrainingStatsReply, TrainingProgressStats>(impl_,
      &wire::TrainingService_Stub::Stats,
      [p=impl_](auto& q, auto&) { p->Control(q); }, [](const auto& r, const auto&) {
        return TrainingProgressStats{{r.completed().begin(), r.completed().end()},
          r.completion_floor(), r.granted_tasks(), r.completed_tasks(), r.duplicate_completions(),
          r.active(), r.waiting(), r.paused(), r.failed()};
      });
}
std::future<Result<TrainingProgressCheckpoint>> BrpcTrainingProgress::Checkpoint() {
  return t::Call<wire::TrainingControl, wire::TrainingCheckpointReply, TrainingProgressCheckpoint>(impl_,
      &wire::TrainingService_Stub::Checkpoint,
      [p=impl_](auto& q, auto&) { p->Control(q); }, [](const auto& r, const auto&) {
        detail::Require(r.consistency() <= 2 && r.completed_size() <= 1024,
                        "invalid training checkpoint reply");
        return TrainingProgressCheckpoint{r.source_run(), r.epoch(),
          static_cast<TrainingConsistency>(r.consistency()), r.max_delay(),
          {r.completed().begin(), r.completed().end()}};
      });
}
#define EPS_CONTROL(Name) \
std::future<Result<Ack>> BrpcTrainingProgress::Name() { \
  return t::Call<wire::TrainingControl, wire::Ack, Ack>(impl_, &wire::TrainingService_Stub::Name, \
      [p=impl_](auto& q, auto&) { p->Control(q); }, t::AckDecoder); \
}
EPS_CONTROL(Pause)
EPS_CONTROL(Resume)
#undef EPS_CONTROL
std::future<Result<Ack>> BrpcTrainingProgress::Abort(std::string reason) {
  return t::Call<wire::TrainingControl, wire::Ack, Ack>(impl_, &wire::TrainingService_Stub::Abort,
      [p=impl_, reason=std::move(reason)](auto& q, auto&) { p->Control(q); q.set_reason(reason); },
      t::AckDecoder);
}
}  // namespace eps
