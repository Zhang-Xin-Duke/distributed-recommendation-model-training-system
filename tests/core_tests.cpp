#include "eps/runtime.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using namespace eps;
namespace {
void Check(bool b, const char* why) { if (!b) throw std::runtime_error(why); }
void Near(float a, float b, float tolerance = 1e-5f) {
  if (!std::isfinite(a) || std::abs(a - b) > tolerance)
    throw std::runtime_error("numeric mismatch: " + std::to_string(a) + " vs " + std::to_string(b));
}
template <typename T> T Value(std::future<Result<T>> future) {
  auto result = future.get();
  if (auto e = std::get_if<Error>(&result)) throw std::runtime_error(e->message);
  return std::get<T>(std::move(result));
}
template <typename T> Error Failure(std::future<Result<T>> future) {
  auto r = future.get(); Check(std::holds_alternative<Error>(r), "expected rejection");
  return std::get<Error>(r);
}
struct Temporary {
  std::filesystem::path path;
  Temporary() {
    char name[] = "/tmp/eps-tests-XXXXXX";
    auto* p = ::mkdtemp(name); Check(p != nullptr, "temporary directory failed"); path = p;
  }
  ~Temporary() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
RuntimeOptions Options(std::string run = "test") {
  RuntimeOptions o; o.run_id = std::move(run); return o;
}
TableDefinition Table(std::uint32_t count = 1) {
  return {1, 2, count, 1, {0.1f, 0.9f, 0.999f, 1e-8f, 0.9f, 0.999f}, 0.25f, 0.25f, 42};
}
RequestId Id(std::uint64_t seq, Phase phase, std::string run = "test", std::uint32_t learner = 0) {
  return {std::move(run), learner, seq, phase};
}
FeatureInput Feature(std::uint32_t id, Pooling mode, std::vector<Key> keys,
                     std::vector<std::uint64_t> offsets,
                     std::vector<std::uint64_t> lengths) {
  return {1, id, mode, std::move(keys), std::move(offsets), std::move(lengths)};
}
LookupReply Lookup(CpuParameterService& s, std::uint64_t seq,
                   std::vector<FeatureInput> features, bool read_only = false,
                   std::string run = "test", std::uint32_t learner = 0) {
  return Value(s.Lookup({Id(seq, Phase::lookup, std::move(run), learner), 1,
                         read_only, std::move(features)}));
}
RowState Read(CpuParameterService& s, Key key, ShardId home = 0) {
  return Value(s.ReadState({{{1, key}, home, Placement::cpu, 0, 0, 0}})).at(0);
}
void Release(CpuParameterService& s, std::uint64_t seq, const std::string& ticket,
             std::string run = "test", std::uint32_t learner = 0) {
  Value(s.Release({Id(seq, Phase::release, std::move(run), learner), ticket}));
}

void PoolingAndUpdate() {
  CpuParameterService s(Options()); Value(s.CreateTable(Table()));
  auto out = Lookup(s, 0, {Feature(0, Pooling::sum, {1,1,2}, {0,3,3}, {3,0}),
                            Feature(1, Pooling::mean, {1,3}, {0,2}, {2})});
  Check(out.cpu_contributions[0].shape == std::vector<std::uint64_t>({2,2}), "sum layout");
  Near(out.cpu_contributions[0].values[0], 0.75f);
  Near(out.cpu_contributions[0].values[2], 0);
  Near(out.cpu_contributions[1].values[0], 0.5f); // unnormalized partial sum
  Value(s.Push({Id(0, Phase::push), out.ticket,
    {{0, {2,2}, {1,2,0,0}}, {1, {1,2}, {4,6}}}}));
  const auto a = Read(s,1), b = Read(s,2), c = Read(s,3);
  Near(a.first_moment[0],0.4f); Near(a.first_moment[1],0.7f);
  Near(a.second_moment[0],0.016f); Near(a.second_moment[1],0.049f);
  Near(b.first_moment[0],0.1f); Near(c.first_moment[0],0.2f);
  Near(a.weights[0],0.15f); Check(a.row_updates == 1, "repeated ID applied twice");
  Value(s.Push({Id(0, Phase::push), out.ticket, {{0,{1,1},{999}}}}));
  Check(Read(s,1).row_updates == 1, "duplicate Push applied twice");
}

void SequenceAndPinnedRows() {
  CpuParameterService s(Options()); auto t=Table();t.initialization_min=-1;t.initialization_max=1;
  Value(s.CreateTable(t));
  auto out=Lookup(s,0,{Feature(0,Pooling::sequence,{8,2,8},{0,2,3},{2,1})},true);
  const auto& v=out.cpu_contributions[0].values;
  Check(v.size()==6 && v[0]==v[4] && v[1]==v[5],"sequence positions lost");
  Check(Failure(s.ReadState({{{1,8},0,Placement::cpu,0,0,0}})).code==Error::Code::unavailable,
        "active row was not pinned");
  Release(s,0,out.ticket); Read(s,8);
}

void InvalidUpdateIsRetryable() {
  CpuParameterService s(Options());Value(s.CreateTable(Table()));
  auto out=Lookup(s,0,{Feature(0,Pooling::sum,{1},{0,1},{1}),
                       Feature(1,Pooling::sum,{2},{0,1},{1})});
  Failure(s.Push({Id(0,Phase::push),out.ticket,{{0,{1,2},{1,2}},{1,{1,1},{4}}}}));
  Value(s.Push({Id(0,Phase::push),out.ticket,{{0,{1,2},{1,2}},{1,{1,2},{3,4}}}}));
  Check(Read(s,1).row_updates==1 && Read(s,2).row_updates==1,"update was not atomic");
}

void DrainAndRestore() {
  Temporary dir; auto options=Options();options.snapshot_directory=dir.path;
  CpuParameterService original(options);Value(original.CreateTable(Table()));
  auto out=Lookup(original,0,{Feature(0,Pooling::sum,{1},{0,1},{1})});
  Failure(original.DrainAndFlush("step-1"));
  Failure(original.Lookup({Id(1,Phase::lookup),1,false,{Feature(0,Pooling::sum,{1},{0,1},{1})}}));
  Value(original.Push({Id(0,Phase::push),out.ticket,{{0,{1,2},{1,2}}}}));
  auto saved=Value(original.DrainAndFlush("step-1"));
  CpuParameterService restored(Options("resumed"));Value(restored.Restore(saved.state_files[0]));
  auto a=Read(original,1),b=Read(restored,1);
  Check(a.weights==b.weights && a.first_moment==b.first_moment &&
        a.second_moment==b.second_moment && a.row_updates==b.row_updates,"restore changed state");
  Value(original.Resume());
  auto x=Lookup(original,1,{Feature(0,Pooling::sum,{1},{0,1},{1})});
  auto y=Lookup(restored,0,{Feature(0,Pooling::sum,{1},{0,1},{1})},false,"resumed");
  Value(original.Push({Id(1,Phase::push),x.ticket,{{0,{1,2},{3,4}}}}));
  Value(restored.Push({Id(0,Phase::push,"resumed"),y.ticket,{{0,{1,2},{3,4}}}}));
  a=Read(original,1);b=Read(restored,1);
  Check(a.weights==b.weights && a.first_moment==b.first_moment && a.second_moment==b.second_moment,
        "resume did not preserve optimizer trajectory");
  CpuParameterService same_run(Options());Failure(same_run.Restore(saved.state_files[0]));
  Failure(original.DrainAndFlush("step-1")); // immutable snapshot
}

void CorruptSnapshotIsRejected() {
  Temporary dir;auto options=Options();options.snapshot_directory=dir.path;
  CpuParameterService s(options);Value(s.CreateTable(Table()));
  auto x=Lookup(s,0,{Feature(0,Pooling::sum,{1},{0,1},{1})},true);Release(s,0,x.ticket);
  auto saved=Value(s.DrainAndFlush("one"));
  std::fstream file(saved.state_files[0],std::ios::binary|std::ios::in|std::ios::out);
  file.seekg(-9,std::ios::end);char c;file.read(&c,1);c^=1;file.seekp(-9,std::ios::end);file.write(&c,1);file.close();
  CpuParameterService fresh(Options("new"));Failure(fresh.Restore(saved.state_files[0]));
  Check(Value(fresh.Stats()).rows==0,"failed restore published partial state");
}

void ShardsAndInitializer() {
  auto a=Options(),b=Options();a.shard_count=b.shard_count=2;b.shard_id=1;
  CpuParameterService even(a),odd(b);auto t=Table(2);t.initialization_min=-1;t.initialization_max=1;
  Value(even.CreateTable(t));Value(odd.CreateTable(t));
  Failure(even.Lookup({Id(0,Phase::lookup),1,true,{Feature(0,Pooling::sum,{1},{0,1},{1})}}));
  auto x=Lookup(even,0,{Feature(0,Pooling::sum,{2},{0,1},{1})},true);
  auto y=Lookup(odd,0,{Feature(0,Pooling::sum,{1},{0,1},{1})},true);
  Release(even,0,x.ticket);Release(odd,0,y.ticket);
  CpuParameterService all(Options());t.shard_count=1;Value(all.CreateTable(t));
  auto z=Lookup(all,0,{Feature(0,Pooling::sequence,{1,2},{0,2},{2})},true);Release(all,0,z.ticket);
  Check(Read(all,1).weights==Read(odd,1,1).weights &&
        Read(all,2).weights==Read(even,2).weights,"initialization depends on shard topology");
}

void BoundedHistoryWithMixedRequests() {
  auto options=Options();options.request_history=8;
  CpuParameterService s(options);Value(s.CreateTable(Table()));
  for(std::uint64_t seq=0;seq<100;++seq) {
    auto x=Lookup(s,seq,{Feature(0,Pooling::sum,{1},{0,1},{1})},seq%2==0);
    if(seq%2==0) Release(s,seq,x.ticket);
    else Value(s.Push({Id(seq,Phase::push),x.ticket,{{0,{1,2},{1,1}}}}));
  }
  Check(Failure(s.Push({Id(1,Phase::push),"0/1",{}})).code==Error::Code::retired_request,
        "retired update was admitted again");
  Check(Read(s,1).row_updates==50,"mixed request history lost updates");
}

void BudgetAndEpoch() {
  auto options=Options();options.state_bytes=32;options.max_live_tickets=1;
  CpuParameterService s(options);Value(s.CreateTable(Table()));
  Failure(s.Lookup({Id(0,Phase::lookup),2,true,{Feature(0,Pooling::sum,{1},{0,1},{1})}}));
  auto x=Lookup(s,0,{Feature(0,Pooling::sum,{1},{0,1},{1})},true);
  Failure(s.Lookup({Id(1,Phase::lookup),1,true,{Feature(0,Pooling::sum,{1},{0,1},{1})}}));
  Release(s,0,x.ticket);
  Failure(s.Lookup({Id(1,Phase::lookup),1,true,{Feature(0,Pooling::sum,{2},{0,1},{1})}}));
  Check(Value(s.Stats()).state_bytes<=32,"state memory budget exceeded");
}
void PayloadBudgets() {
  auto o=Options();o.ticket_payload_bytes=1;
  CpuParameterService tickets(o);Value(tickets.CreateTable(Table()));
  Check(Failure(tickets.Lookup({Id(0,Phase::lookup),1,true,
    {Feature(0,Pooling::sum,{1},{0,1},{1})}})).code==Error::Code::capacity_exceeded,"ticket budget ignored");
  Check(Value(tickets.Stats()).live_tickets==0,"rejected lookup leaked ticket");
  o=Options();o.queue_payload_bytes=1;
  CpuParameterService queue(o);Value(queue.CreateTable(Table()));
  Check(Failure(queue.Lookup({Id(0,Phase::lookup),1,true,
    {Feature(0,Pooling::sum,{1},{0,1},{1})}})).code==Error::Code::capacity_exceeded,"queue byte budget ignored");
}
}

int main() {
  const std::vector<std::pair<const char*,std::function<void()>>> checks={
    {"pooling and sparse update",PoolingAndUpdate},
    {"sequence and pinned rows",SequenceAndPinnedRows},
    {"invalid update is retryable",InvalidUpdateIsRetryable},
    {"drain and exact optimizer restore",DrainAndRestore},
    {"corrupt snapshot",CorruptSnapshotIsRejected},
    {"partitioning and initializer",ShardsAndInitializer},
    {"mixed request history",BoundedHistoryWithMixedRequests},
    {"resource bounds and epoch",BudgetAndEpoch},
    {"ticket and queue payload limits",PayloadBudgets}};
  int failed=0;
  for(const auto& [name,fn]:checks) {
    try { fn();std::cout<<"PASS "<<name<<'\n'; }
    catch(const std::exception& e) { ++failed;std::cerr<<"FAIL "<<name<<": "<<e.what()<<'\n'; }
  }
  return failed?1:0;
}
