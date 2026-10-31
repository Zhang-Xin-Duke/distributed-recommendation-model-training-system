#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
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
          std::uint64_t epoch,std::size_t max_live)
      : run_(std::move(run)),epoch_(epoch),count_(endpoints.size()) {
    std::vector<std::shared_ptr<ParameterService>> shards;
    for(auto& e:endpoints)shards.push_back(std::make_shared<BrpcParameterService>(e));
    adapter_=std::make_unique<entropy::TableAdapter>(entropy::AdapterOptions{
      run_,learner,epoch,max_live,4096},std::move(shards));
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
    py::gil_scoped_release release;Take(adapter_->CreateTable(table));
  }
  py::tuple Pull(const py::list& descriptors,bool read_only) {
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
    entropy::PullResult result;
    {py::gil_scoped_release release;result=Take(adapter_->Pull(std::move(features),read_only));}
    py::list arrays;
    for(const auto& block:result.features) {
      py::array_t<float> array({py::ssize_t(block.shape[0]),py::ssize_t(block.shape[1])});
      std::copy(block.values.begin(),block.values.end(),array.mutable_data());arrays.append(std::move(array));
    }
    return py::make_tuple(result.handle,arrays);
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
  void Cache(std::vector<std::pair<AgentId,std::string>> endpoints) {
    std::map<AgentId,std::shared_ptr<CacheBackend>> agents;
    for(auto& [id,endpoint]:endpoints)agents[id]=std::make_shared<BrpcCacheBackend>(endpoint,run_,0,id);
    adapter_->SetCacheResolver(MakeCacheResolver(std::move(agents)));
  }
 private:
  std::string run_;std::uint64_t epoch_;std::uint32_t count_;
  std::unique_ptr<entropy::TableAdapter> adapter_;
};
}
PYBIND11_MODULE(_eps_native,m) {
  py::class_<Backend>(m,"Backend")
    .def(py::init<std::vector<std::string>,std::string,std::uint32_t,std::uint64_t,std::size_t>(),
      py::arg("endpoints"),py::arg("run"),py::arg("learner")=0,py::arg("epoch")=1,py::arg("max_live")=128)
    .def("create_table",&Backend::Create).def("pull",&Backend::Pull,py::arg("features"),py::arg("read_only")=false)
    .def("push",&Backend::Push).def("release",&Backend::Release).def("join",&Backend::Join)
    .def("save",&Backend::Save).def("cache_endpoints",&Backend::Cache);
}
