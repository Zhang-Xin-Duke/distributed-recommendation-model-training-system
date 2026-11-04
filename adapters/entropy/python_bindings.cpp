#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include <set>
#include <atomic>
#include "eps/entropy_adapter.hpp"
#include "eps/brpc_client.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/cache_resolver.hpp"
#include "eps/brpc_training.hpp"
#include "eps/brpc_synchronous_gradients.hpp"

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
      : run_(std::move(run)),epoch_(epoch),count_(endpoints.size()),learner_(learner),endpoints_(endpoints) {
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
  bool ReadHealthy() {py::gil_scoped_release release;return adapter_->ReadLifecycleHealthy();}
  void Resume(std::string id) {py::gil_scoped_release release;Take(adapter_->Resume(std::move(id)));}
  py::tuple ReadCheckpoint(std::string id,std::uint32_t shard,std::uint64_t offset,
      std::uint32_t max_bytes,std::uint64_t checksum,py::bytes previous_tail) {
    const auto tail=py::cast<std::string>(previous_tail);
    if(tail.size()!=std::min<std::uint64_t>(offset,8) ||
       (offset==0 && checksum!=14695981039346656037ULL))
      throw std::invalid_argument("invalid checkpoint checksum continuation");
    SnapshotChunk chunk;std::string next_tail;
    {py::gil_scoped_release release;
      chunk=Take(adapter_->ReadCheckpoint(std::move(id),shard,offset,max_bytes));
      const auto combined=tail+chunk.data;const auto body=combined.size()>8?combined.size()-8:0;
      for(std::size_t i=0;i<body;++i)checksum=(checksum^static_cast<unsigned char>(combined[i]))*1099511628211ULL;
      next_tail=combined.substr(body);
      if(chunk.eof) {
        if(next_tail.size()!=8)throw std::runtime_error("checkpoint checksum trailer missing");
        std::uint64_t expected=0;for(unsigned i=0;i<8;++i)expected|=std::uint64_t(static_cast<unsigned char>(next_tail[i]))<<(8*i);
        if(expected!=checksum)throw std::runtime_error("exported checkpoint checksum mismatch");
      }
    }
    return py::make_tuple(chunk.offset,chunk.total_bytes,py::bytes(chunk.data),chunk.eof,checksum,py::bytes(next_tail));
  }
  std::vector<std::string> Save(std::string id) {
    py::gil_scoped_release release;auto metadata=Take(adapter_->Save(std::move(id)));
    std::vector<std::string> files;
    for(auto& m:metadata)files.insert(files.end(),m.state_files.begin(),m.state_files.end());return files;
  }
  void Cache(py::sequence endpoints,bool coalesce) {
    std::map<AgentId,std::shared_ptr<CacheBackend>> agents;
    for(auto endpoint:endpoints) {
      auto fields=py::cast<py::sequence>(endpoint);
      if(fields.size()!=2 && fields.size()!=3)throw std::invalid_argument("cache endpoint requires (id, endpoint[, owner_run])");
      const auto id=py::cast<AgentId>(fields[0]);
      const auto scope=fields.size()==3?py::cast<std::string>(fields[2]):run_;
      if(scope.empty() || agents.contains(id))throw std::invalid_argument("empty owner run or repeated cache agent");
      agents[id]=std::make_shared<BrpcCacheBackend>(py::cast<std::string>(fields[1]),scope,0,id);
    }
    adapter_->SetCacheResolver(MakeCacheResolver(std::move(agents)),coalesce);
  }
  py::list RestoredFrom() {
    std::vector<RestoredSnapshot> metadata;
    {py::gil_scoped_release release;
      std::vector<std::future<Result<RestoredSnapshot>>> pending;
      for(const auto& connection:connections_)pending.push_back(connection->RestoredFrom());
      for(auto& future:pending)metadata.push_back(Take(future.get()));
    }
    py::list result;
    for(const auto& m:metadata) {
      py::dict row;row["loaded"]=m.loaded;row["pristine"]=m.pristine;
      row["run"]=m.run;row["source_run"]=m.source_run;row["bytes"]=m.bytes;row["checksum"]=m.checksum;
      row["shard"]=m.shard;row["shard_count"]=m.shard_count;row["partition_epoch"]=m.partition_epoch;
      result.append(std::move(row));
    }return result;
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
      EPS_STAT(queued_observations) EPS_STAT(completed_observations) EPS_STAT(rejected_observations) EPS_STAT(dependency_budget_waits)
#undef EPS_STAT
      result.push_back(std::move(row));
    }
    return result;
  }
  void LookupCache(bool enabled) {
    py::gil_scoped_release release;
    for (const auto& connection : connections_) connection->SetLookupLayoutCache(enabled);
  }
  std::vector<std::map<std::string,std::uint64_t>> LookupTransport() {
    py::gil_scoped_release release;
    std::vector<std::map<std::string,std::uint64_t>> result;
    for (const auto& connection : connections_) {
      const auto s = connection->LookupTransport();
      result.push_back({{"full_requests",s.full_requests},{"token_requests",s.token_requests},
        {"fallback_requests",s.fallback_requests},{"request_bytes",s.request_bytes},
        {"avoided_request_bytes",s.avoided_request_bytes},{"cache_entries",s.cache.entries},
        {"cache_bytes",s.cache.bytes},{"cache_hits",s.cache.hits},{"cache_misses",s.cache.misses},
        {"cache_evictions",s.cache.evictions},{"cache_expirations",s.cache.expirations},
        {"cache_rejected",s.cache.rejected}});
    }
    return result;
  }
  void ConfigureTraining(std::string coordinator,unsigned retry_ms) {
    if (progress_) throw std::invalid_argument("training already configured");
    if (learner_ >= 1023) throw std::invalid_argument("synchronous training worker ID must be below 1023");
    if(retry_ms>20000)throw std::invalid_argument("gradient retry window exceeds 20 seconds");
    auto progress = std::make_unique<BrpcTrainingProgress>(std::move(coordinator), run_, epoch_, 20000);
    std::vector<std::unique_ptr<BrpcSynchronousGradients>> collectors;
    for (const auto& endpoint : endpoints_)
      collectors.push_back(std::make_unique<BrpcSynchronousGradients>(endpoint, run_, epoch_,retry_ms?4000:25000));
    retry_ms_=retry_ms;
    progress_ = std::move(progress); collectors_ = std::move(collectors);
  }
  std::uint64_t BeginTraining(std::uint64_t iteration) {
    if (!progress_) throw std::invalid_argument("training is not configured");
    py::gil_scoped_release release;
    return Take(progress_->Begin(learner_, iteration).get()).observed_completion_floor;
  }
  void CompleteTraining(std::uint64_t iteration) {
    if (!progress_) throw std::invalid_argument("training is not configured");
    py::gil_scoped_release release; Take(progress_->Complete(learner_, iteration).get());
  }
  void SubmitSynchronous(std::uint64_t iteration, std::uint64_t samples, const py::list& tables) {
    if (!progress_ || tables.empty() || tables.size() > 256 || !samples)
      throw std::invalid_argument("invalid synchronous training submission");
    std::vector<GradientContribution> packets(count_);
    std::uint64_t completed_reads=0;
    const auto progress=adapter_->CompletedReadFrontier();
    if(auto* n=std::get_if<std::uint64_t>(&progress))completed_reads=*n;
    else if(retry_ms_)throw std::runtime_error(std::get<Error>(progress).message);
    for (auto& packet : packets) { packet.task = {run_, epoch_, learner_, iteration}; packet.sample_count = samples;packet.completed_reads=completed_reads; }
    TableId previous = 0; bool first = true;
    std::uint64_t retained = 0;
    for (auto item : tables) {
      const auto d = py::cast<py::dict>(item);
      const auto id = py::cast<TableId>(d["table"]);
      if (!first && id <= previous) throw std::invalid_argument("synchronous tables must be unique and sorted");
      first = false; previous = id;
      const auto keys = py::cast<py::array_t<std::uint64_t,py::array::c_style|py::array::forcecast>>(d["keys"]);
      const auto values = py::cast<py::array_t<float,py::array::c_style|py::array::forcecast>>(d["values"]);
      if (keys.ndim() != 1 || values.ndim() != 2 || values.shape(0) != keys.size() ||
          values.shape(1) <= 0 || values.shape(1) > 65536)
        throw std::invalid_argument("invalid synchronous gradient array layout");
      const auto bytes = keys.nbytes() + values.nbytes();
      if (bytes > (16ULL<<20) || retained > (16ULL<<20) - bytes)
        throw std::invalid_argument("synchronous gradient arrays exceed byte limit");
      retained += bytes;
      const auto dim = static_cast<std::uint32_t>(values.shape(1));
      for (auto& packet : packets) packet.tables.push_back({id, dim, {}, {}});
      for (py::ssize_t row = 0; row < keys.size(); ++row) {
        const auto key = keys.data()[row];
        if (row && keys.data()[row-1] >= key) throw std::invalid_argument("gradient Keys must be unique and sorted");
        auto& table = packets[key % count_].tables.back();
        table.keys.push_back(key);
        table.values.insert(table.values.end(), values.data()+row*dim, values.data()+(row+1)*dim);
      }
    }
    py::gil_scoped_release release;
    std::vector<BrpcSynchronousGradients*> group;
    for(const auto& collector:collectors_)group.push_back(collector.get());
    retry_count_+=Take(SubmitGradientGroup(group,packets,std::chrono::milliseconds(retry_ms_))).retries;
  }
  std::uint64_t TrainingRetryCount()const{return retry_count_.load();}
  std::uint64_t ReadFrontier(){return Take(adapter_->CompletedReadFrontier());}
  void RestoreReadFrontier(std::uint64_t next){Take(adapter_->RestoreReadFrontier(next));}
  void AbortTraining(std::string reason) {
    py::gil_scoped_release release;
    std::vector<std::future<Result<Ack>>> pending;
    if (progress_) pending.push_back(progress_->Abort(reason));
    for (auto& collector : collectors_) pending.push_back(collector->Abort(reason));
    for (auto& f : pending) (void)f.get();
  }
 private:
  std::string run_;std::uint64_t epoch_;std::uint32_t count_;
  std::uint32_t learner_;
  unsigned retry_ms_=0;
  std::atomic<std::uint64_t> retry_count_{0};
  std::vector<std::string> endpoints_;
  std::unique_ptr<BrpcTrainingProgress> progress_;
  std::vector<std::unique_ptr<BrpcSynchronousGradients>> collectors_;
  std::vector<std::shared_ptr<BrpcParameterService>> connections_;
  std::unique_ptr<entropy::TableAdapter> adapter_;
};
}
PYBIND11_MODULE(_eps_native,m) {
  m.attr("owner_incarnations")=true;
  m.attr("read_lifecycle_cleanup")=true;
  py::class_<Backend>(m,"Backend")
    .def(py::init<std::vector<std::string>,std::string,std::uint32_t,std::uint64_t,std::size_t,const py::dict&>(),
      py::arg("endpoints"),py::arg("run"),py::arg("learner")=0,py::arg("epoch")=1,py::arg("max_live")=128,py::arg("preparation")=py::dict())
    .def("create_table",&Backend::Create).def("pull",&Backend::Pull,py::arg("features"),py::arg("read_only")=false)
    .def("prepare",&Backend::Prepare)
    .def("pull_prepared",&Backend::PullPrepared,py::arg("token"),py::arg("read_only")=false)
    .def("cancel_prepared",&Backend::CancelPrepared).def("preparation_stats",&Backend::PreparationStats)
    .def("push",&Backend::Push).def("release",&Backend::Release).def("join",&Backend::Join)
    .def("read_lifecycle_healthy",&Backend::ReadHealthy)
    .def("restored_from",&Backend::RestoredFrom)
    .def("save",&Backend::Save).def("stats",&Backend::Stats)
    .def("lookup_layout_cache",&Backend::LookupCache).def("lookup_transport_stats",&Backend::LookupTransport)
    .def("resume",&Backend::Resume).def("read_checkpoint",&Backend::ReadCheckpoint)
    .def("configure_training",&Backend::ConfigureTraining,py::arg("coordinator"),py::arg("retry_ms")=0)
    .def("training_retry_count",&Backend::TrainingRetryCount)
    .def("completed_read_frontier",&Backend::ReadFrontier).def("restore_read_frontier",&Backend::RestoreReadFrontier)
    .def("begin_training",&Backend::BeginTraining).def("complete_training",&Backend::CompleteTraining)
    .def("submit_synchronous",&Backend::SubmitSynchronous).def("abort_training",&Backend::AbortTraining)
    .def("cache_endpoints",&Backend::Cache,py::arg("agents"),py::arg("coalesce")=false);
}
