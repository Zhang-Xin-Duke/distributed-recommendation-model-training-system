#include "eps/brpc_synchronous_gradients.hpp"
#include "transport/rpc_call.hpp"

namespace eps {
namespace t = transport;
struct BrpcSynchronousGradients::Impl {
  brpc::Channel channel;
  int timeout_ms;
  std::string run;
  std::uint64_t epoch;
  Impl(const std::string& endpoint, std::string r, std::uint64_t e, int timeout)
      : timeout_ms(timeout), run(std::move(r)), epoch(e) {
    detail::Require(!run.empty() && run.size() <= 256 && epoch && timeout > 0,
                    "invalid synchronous gradient RPC options");
    brpc::ChannelOptions options; options.protocol = "baidu_std";
    options.timeout_ms = timeout; options.max_retry = 2;
    detail::Require(channel.Init(endpoint.c_str(), &options) == 0, "cannot open gradient RPC channel");
  }
  void Control(v1::TrainingControl& q) const { q.set_run(run); q.set_epoch(epoch); }
};
BrpcSynchronousGradients::BrpcSynchronousGradients(std::string endpoint, std::string run,
                                                 std::uint64_t epoch, int timeout)
    : impl_(std::make_shared<Impl>(endpoint, std::move(run), epoch, timeout)) {}
BrpcSynchronousGradients::~BrpcSynchronousGradients() = default;
std::future<Result<Ack>> BrpcSynchronousGradients::Submit(GradientContribution c) {
  return t::Call<v1::GradientContribution, v1::Ack, Ack>(impl_, &v1::SynchronousGradientService_Stub::Submit,
    [c=std::move(c), p=impl_](auto& q, auto& a) {
      detail::Require(c.task.run == p->run && c.task.epoch == p->epoch && c.tables.size() <= 256,
                      "invalid gradient RPC identity/table count");
      auto* task = q.mutable_task(); task->set_run(c.task.run); task->set_epoch(c.task.epoch);
      task->set_worker(c.task.worker); task->set_iteration(c.task.iteration);
      q.set_sample_count(c.sample_count);
      for (const auto& table : c.tables) {
        auto* out = q.add_tables(); out->set_table_id(table.table); out->set_dimension(table.dimension);
        t::Pack(table.keys, out->mutable_keys(), a); t::Pack(table.values, out->mutable_values(), a);
      }
    }, t::AckDecoder);
}
std::future<Result<SynchronousGradientStats>> BrpcSynchronousGradients::Stats() {
  return t::Call<v1::TrainingControl, v1::SynchronousGradientStatsReply, SynchronousGradientStats>(impl_,
    &v1::SynchronousGradientService_Stub::Stats,
    [p=impl_](auto& q, auto&) { p->Control(q); }, [](const auto& r, const auto&) {
      return SynchronousGradientStats{r.next_round(), r.received_workers(), r.retained_bytes(),
        r.duplicate_submissions(), r.applying(), r.failed()};
    });
}
std::future<Result<Ack>> BrpcSynchronousGradients::Abort(std::string reason) {
  return t::Call<v1::TrainingControl, v1::Ack, Ack>(impl_, &v1::SynchronousGradientService_Stub::Abort,
    [p=impl_, reason=std::move(reason)](auto& q, auto&) { p->Control(q); q.set_reason(reason); }, t::AckDecoder);
}
}  // namespace eps
