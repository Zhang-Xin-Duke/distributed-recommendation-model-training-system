#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include <set>
#include "eps/entropy_adapter.hpp"
#include "eps/brpc_client.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/cache_resolver.hpp"

namespace py=pybind11;
using namespace eps;
namespace {
template<typename T>T Take(Result<T> r) {
  if(auto e=std::get_if<Error>(&r)) throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
class Backend {
 public:
  Backend(std::vector<std::string> endpoints,std::string run,std::uint32_t learner,
          std::uint64_t epoch,std::size_t max_live,const py::dict& preparation)
      : run_(std::move(run)),epoch_(epoch),count_(endpoints.size()) {
    std::vector<std::shared_ptr<ParameterService>> shards;
    for(auto& e:endpoints){
      auto connection=std::make_shared<BrpcParameterService>(e);
      connections_.push_back(connection);shards.push_back(std::move(connection));
    }
    entropy::AdapterOptions options{run_,learner,epoch,max_live,4096};
    const std::set<std::string> known{"max_pending_batches","payload_bytes","max_input_bytes","max_prepared_bytes","max_features"};
    for(auto item:preparation)if(!known.contains(py::cast<std::string>(item.first)))
      throw std::invalid_argument("unknown preparation option");
    auto u64=[&](const char* key,std::uint64_t fallback){return preparation.contains(key)?py::cast<std::uint64_t>(preparation[key]):fallback;};
    auto& q=options.preparation;q.max_pending_batches=u64("max_pending_batches",q.max_pending_batches);
    q.payload_bytes=u64("payload_bytes",q.payload_bytes);
    q.lookup_limits.max_input_bytes=u64("max_input_bytes",q.lookup_limits.max_input_bytes);
    q.lookup_limits.max_prepared_bytes=u64("max_prepared_bytes",q.lookup_limits.max_prepared_bytes);
    if(preparation.contains("max_features"))q.lookup_limits.max_features=py::cast<std::uint32_t>(preparation["max_features"]);
    adapter_=std::make_unique<entropy::TableAdapter>(std::move(options),std::move(shards));
  }
  void Create(const std::string& definition) {
    const auto t=nlohmann::json::parse(definition);
    if(t.value("dtype",std::string("float32"))!="float32" ||
       t.at("optimizer").at("kind")!="sparse_lazy_adam")
      throw std::invalid_argument("supported table format: FP32 sparse_lazy_adam");
    const auto& o=t.at("optimizer");const auto& init=t.at("initialization");
    TableDefinition table{t.at("id"),t.at("dimension"),count_,epoch_,
      {o.at("learning_rate"),o.at("beta1"),o.at("beta2"),o.at("epsilon"),
       o.at("initial_beta1_power"),o.at("initial_beta2_power")},
      init.at("min"),init.at("max"),init.at("seed")};
    if(o.contains("weight_bounds")) {
      const auto& b=o.at("weight_bounds");
      if(!b.is_array() || b.size()!=2)throw std::invalid_argument("weight_bounds requires [minimum, maximum]");
      table.optimizer.weight_bounds=WeightBounds{b.at(0),b.at(1)};
    }
    py::gil_scoped_release release;Take(adapter_->CreateTable(table));
  }
  std::vector<entropy::Feature> Parse(const py::list& descriptors) {
    std::vector<entropy::Feature> features;
    for(auto item:descriptors) {
      auto d=py::cast<py::dict>(item);
      auto keys=py::cast<py::array_t<std::uint64_t,py::array::c_style|py::array::forcecast>>(d["keys"]);
      auto splits=py::cast<py::array_t<std::uint64_t,py::array::c_style|py::array::forcecast>>(d["offsets"]);
      if(keys.ndim()!=1 || splits.ndim()!=1)throw std::invalid_argument("IDs and offsets must be vectors");
      auto pooling=py::cast<std::string>(d["pooling"]);
      if(pooling!="sum" && pooling!="mean" && pooling!="sequence")throw std::invalid_argument("invalid pooling");
      entropy::Feature f{py::cast<TableId>(d["table"]),py::cast<std::uint32_t>(d["feature"]),
        pooling=="sum"?Pooling::sum:pooling=="mean"?Pooling::mean:Pooling::sequence,
        {keys.data(),keys.data()+keys.size()},{splits.data(),splits.data()+splits.size()},std::nullopt};
      if(d.contains("padding_key") && !d["padding_key"].is_none())f.padding_key=py::cast<Key>(d["padding_key"]);
      features.push_back(std::move(f));
    }
    return features;
  }
  py::tuple Convert(entropy::PullResult result) {
    try {
      py::list arrays;
      for(const auto& block:result.features) {
        py::array_t<float> array({py::ssize_t(block.shape[0]),py::ssize_t(block.shape[1])});
        std::copy(block.values.begin(),block.values.end(),array.mutable_data());arrays.append(std::move(array));
      }
      return py::make_tuple(result.handle,arrays);
    }catch(...) {
      {py::gil_scoped_release release;try{(void)adapter_->Release(result.handle);}catch(...){} }
      throw;
    }
  }
  py::tuple Pull(const py::list& descriptors,bool read_only) {
    auto features=Parse(descriptors);entropy::PullResult result;
    {py::gil_scoped_release release;result=Take(adapter_->Pull(std::move(features),read_only));}
    return Convert(std::move(result));
  }
  std::uint64_t Prepare(const py::list& descriptors) {
    auto features=Parse(descriptors);py::gil_scoped_release release;
    return Take(adapter_->PrepareAsync(std::move(features)));
  }
  py::tuple PullPrepared(std::uint64_t token,bool read_only) {
    entropy::PullResult result;
    {py::gil_scoped_release release;result=Take(adapter_->PullPrepared(token,read_only));}
    return Convert(std::move(result));
  }
  void CancelPrepared(std::uint64_t token) {py::gil_scoped_release release;Take(adapter_->CancelPrepared(token));}
  std::map<std::string,std::uint64_t> PreparationStats() {
    py::gil_scoped_release release;const auto s=adapter_->PreparationStatus();
    return {{"accepted",s.accepted},{"completed",s.completed},{"failed",s.failed},{"consumed",s.consumed},
      {"cancelled",s.cancelled},{"rejected",s.rejected},{"pending",s.pending},{"claimed",s.claimed},
      {"reserved_jobs",s.reserved_jobs},{"reserved_bytes",s.reserved_bytes},{"peak_reserved_bytes",s.peak_reserved_bytes},
      {"prepare_nanoseconds",s.prepare_nanoseconds}};
  }
  void Push(std::uint64_t handle,const std::vector<std::uint32_t>& ids,const py::list& arrays) {
    if(ids.size()!=arrays.size())throw std::invalid_argument("gradient count mismatch");
    std::vector<TensorBlock> gradients;
    for(std::size_t i=0;i<ids.size();++i) {
      auto array=py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(arrays[i]);
      if(array.ndim()!=2)throw std::invalid_argument("feature gradient must have rank two");
      gradients.push_back({ids[i],{std::uint64_t(array.shape(0)),std::uint64_t(array.shape(1))},
        {array.data(),array.data()+array.size()}});
    }
    py::gil_scoped_release release;Take(adapter_->Push(handle,std::move(gradients)));
  }
  void Release(std::uint64_t h) {py::gil_scoped_release release;Take(adapter_->Release(h));}
  void Join() {py::gil_scoped_release release;Take(adapter_->Join());}
  std::vector<std::string> Save(std::string id) {
    py::gil_scoped_release release;auto metadata=Take(adapter_->Save(std::move(id)));
    std::vector<std::string> files;
    for(auto& m:metadata)files.insert(files.end(),m.state_files.begin(),m.state_files.end());return files;
  }
  void Cache(std::vector<std::pair<AgentId,std::string>> endpoints,bool coalesce) {
    std::map<AgentId,std::shared_ptr<CacheBackend>> agents;
    for(auto& [id,endpoint]:endpoints)agents[id]=std::make_shared<BrpcCacheBackend>(endpoint,run_,0,id);
    adapter_->SetCacheResolver(MakeCacheResolver(std::move(agents)),coalesce);
  }
  std::vector<std::map<std::string,std::uint64_t>> Stats() {
    py::gil_scoped_release release;
    std::vector<std::future<Result<RuntimeStats>>> pending;
    for(const auto& connection:connections_)pending.push_back(connection->Stats());
    std::vector<std::map<std::string,std::uint64_t>> result;
    for(auto& future:pending) {
      const auto s=Take(future.get());std::map<std::string,std::uint64_t> row;
#define EPS_STAT(name) row[#name]=s.name;
      EPS_STAT(lookups) EPS_STAT(updates) EPS_STAT(duplicate_requests) EPS_STAT(rows)
      EPS_STAT(state_bytes) EPS_STAT(live_tickets) EPS_STAT(ticket_payload_bytes)
      EPS_STAT(queued_payload_bytes) EPS_STAT(cached_rows) EPS_STAT(migration_bytes)
      EPS_STAT(executed_operations) EPS_STAT(queue_wait_nanoseconds) EPS_STAT(executor_busy_nanoseconds)
      EPS_STAT(rejected_operations) EPS_STAT(peak_queued_operations)
      EPS_STAT(expanded_gradient_bytes) EPS_STAT(transferred_gradient_bytes) EPS_STAT(compact_update_commands)
      EPS_STAT(policy_windows) EPS_STAT(policy_tracked_rows) EPS_STAT(policy_admitted_rows)
      EPS_STAT(policy_evicted_rows) EPS_STAT(policy_pinned_skips) EPS_STAT(policy_budget_skips)
      EPS_STAT(policy_migration_batches) EPS_STAT(policy_migration_bytes) EPS_STAT(policy_failures)
      EPS_STAT(lookup_prepare_nanoseconds) EPS_STAT(gradient_prepare_nanoseconds) EPS_STAT(gpu_packet_prepare_nanoseconds) EPS_STAT(update_dispatch_nanoseconds) EPS_STAT(cpu_optimizer_nanoseconds) EPS_STAT(agent_wait_nanoseconds) EPS_STAT(migration_nanoseconds) EPS_STAT(cache_maintenance_nanoseconds)
      EPS_STAT(active_operations) EPS_STAT(waiting_operations) EPS_STAT(active_payload_bytes) EPS_STAT(dependency_payload_bytes)
      EPS_STAT(dependency_prepare_nanoseconds) EPS_STAT(dependency_rebuilds) EPS_STAT(dependency_cache_hits)
#undef EPS_STAT
      result.push_back(std::move(row));
    }
    return result;
  }
 private:
  std::string run_;std::uint64_t epoch_;std::uint32_t count_;
  std::vector<std::shared_ptr<BrpcParameterService>> connections_;
  std::unique_ptr<entropy::TableAdapter> adapter_;
};
}
PYBIND11_MODULE(_eps_native,m) {
  py::class_<Backend>(m,"Backend")
    .def(py::init<std::vector<std::string>,std::string,std::uint32_t,std::uint64_t,std::size_t,const py::dict&>(),
      py::arg("endpoints"),py::arg("run"),py::arg("learner")=0,py::arg("epoch")=1,py::arg("max_live")=128,py::arg("preparation")=py::dict())
    .def("create_table",&Backend::Create).def("pull",&Backend::Pull,py::arg("features"),py::arg("read_only")=false)
    .def("prepare",&Backend::Prepare)
    .def("pull_prepared",&Backend::PullPrepared,py::arg("token"),py::arg("read_only")=false)
    .def("cancel_prepared",&Backend::CancelPrepared).def("preparation_stats",&Backend::PreparationStats)
    .def("push",&Backend::Push).def("release",&Backend::Release).def("join",&Backend::Join)
    .def("save",&Backend::Save).def("stats",&Backend::Stats)
    .def("cache_endpoints",&Backend::Cache,py::arg("agents"),py::arg("coalesce")=false);
}
