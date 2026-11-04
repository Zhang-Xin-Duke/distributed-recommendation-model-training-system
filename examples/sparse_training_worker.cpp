// A real sparse-logistic training worker used to exercise the distributed
// execution path. Model/data choices here are separate from the PS protocol.
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>
#include <nlohmann/json.hpp>
#include "eps/brpc_client.hpp"
#include "eps/brpc_training.hpp"
#include "eps/brpc_synchronous_gradients.hpp"
#include "eps/entropy_adapter.hpp"

using namespace eps;
namespace {
template<class T> T Take(Result<T> result) {
  if (const auto* e = std::get_if<Error>(&result)) throw std::runtime_error(e->message);
  return std::get<T>(std::move(result));
}
template<class T> T Take(std::future<Result<T>> future) { return Take(future.get()); }
double Loss(double score, double label) {
  return std::max(score, 0.) - label * score + std::log1p(std::exp(-std::abs(score)));
}
}  // namespace

int main(int argc, char** argv) {
  std::unique_ptr<BrpcTrainingProgress> progress;
  std::vector<std::unique_ptr<BrpcSynchronousGradients>> collectives;
  try {
    if (argc != 3 || std::string(argv[1]) != "--config")
      throw std::runtime_error("usage: eps_training_worker --config FILE");
    std::ifstream file(argv[2]);
    if (!file) throw std::runtime_error("cannot open worker config");
    nlohmann::json j; file >> j;
    if (j.at("schema_version") != 1) throw std::runtime_error("unsupported worker config version");
    const std::string run = j.at("run");
    const std::uint64_t epoch = j.value("epoch", std::uint64_t(1));
    const std::uint32_t worker = j.at("worker");
    const auto iterations = j.at("iterations").get<std::uint64_t>();
    if (iterations == 0 || iterations > 1000000000)
      throw std::runtime_error("iterations must be in [1, 1000000000]");
    std::vector<std::shared_ptr<ParameterService>> shards;
    const auto mode = j.value("update_mode", std::string("per_worker"));
    if (mode != "per_worker" && mode != "synchronous") throw std::runtime_error("invalid update mode");
    const bool synchronous = mode == "synchronous";
    for (const auto& address : j.at("shards")) {
      shards.push_back(std::make_shared<BrpcParameterService>(address.get<std::string>()));
      if (synchronous) collectives.push_back(std::make_unique<BrpcSynchronousGradients>(address, run, epoch));
    }
    if (shards.empty() || shards.size() > 1024) throw std::runtime_error("invalid shard group");
    entropy::TableAdapter adapter({run, worker, epoch}, shards);
    TableDefinition table{1, 1, static_cast<std::uint32_t>(shards.size()), epoch,
      {j.value("learning_rate", .02f), .9f, .999f, 1e-8f, .9f, .999f}, 0, 0, 42};
    Take(adapter.CreateTable(table));
    std::vector<double> labels;
    entropy::Feature feature{1, 0, Pooling::sum, {}, {0}, {}};
    const auto& samples = j.at("samples");
    if (samples.empty() || samples.size() > 65536) throw std::runtime_error("invalid training batch size");
    for (const auto& sample : samples) {
      const double label = sample.at("label");
      if (!std::isfinite(label) || label < 0 || label > 1)
        throw std::runtime_error("binary labels must be in [0,1]");
      labels.push_back(label);
      auto keys = sample.at("keys").get<std::vector<Key>>();
      if (keys.empty() || keys.size() > 65536 || feature.keys.size() + keys.size() > 1048576)
        throw std::runtime_error("invalid sparse training input size");
      feature.keys.insert(feature.keys.end(), keys.begin(), keys.end());
      feature.offsets.push_back(feature.keys.size());
    }
    progress = std::make_unique<BrpcTrainingProgress>(j.at("coordinator"), run, epoch);
    const auto slow_ms = j.value("delay_ms", 0u);
    const auto slow_every = j.value("delay_every", 1u);
    if (slow_ms > 60000 || slow_every == 0) throw std::runtime_error("invalid straggler configuration");
    std::uint64_t completed = 0, max_lead = 0, wait_ns = 0;
    double first_loss = 0, last_loss = 0;
    for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      auto permit = Take(progress->Begin(worker, iteration));
      wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count();
      if (permit.observed_completion_floor > iteration)
        throw std::runtime_error("training floor advanced past this worker");
      max_lead = std::max(max_lead, iteration - permit.observed_completion_floor);
      if (slow_ms && iteration % slow_every == 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(slow_ms));
      auto batch = Take(adapter.Pull({feature}, synchronous));
      auto gradients = batch.features;
      auto& values = gradients.at(0).values;
      if (values.size() != labels.size()) throw std::runtime_error("unexpected model output shape");
      double loss = 0;
      for (std::size_t sample = 0; sample < labels.size(); ++sample) {
        const double score = values[sample];
        if (!std::isfinite(score)) throw std::runtime_error("non-finite model score");
        loss += Loss(score, labels[sample]) / labels.size();
        // Each worker update is the gradient of its local mean batch loss.
        // LazyAdam advances once per acknowledged worker update on each shard.
        values[sample] = static_cast<float>((1. / (1. + std::exp(-score)) - labels[sample]) / labels.size());
      }
      if (!synchronous) Take(adapter.Push(batch.handle, std::move(gradients)));
      else {
        std::map<Key, double> rows;
        for (std::size_t sample = 0; sample < labels.size(); ++sample)
          for (auto at = feature.offsets[sample]; at < feature.offsets[sample+1]; ++at)
            rows[feature.keys[at]] += values[sample];
        Take(adapter.Release(batch.handle));
        std::vector<GradientContribution> packets;
        for (std::size_t home = 0; home < shards.size(); ++home)
          packets.push_back({{run, epoch, worker, iteration}, labels.size(), {{1, 1, {}, {}}}});
        for (const auto& [key, gradient] : rows) {
          auto& table = packets[key % shards.size()].tables[0];
          table.keys.push_back(key); table.values.push_back(static_cast<float>(gradient));
        }
        std::vector<std::future<Result<Ack>>> acknowledgements;
        for (std::size_t home = 0; home < shards.size(); ++home)
          acknowledgements.push_back(collectives[home]->Submit(packets[home]));
        // Submit to every shard before waiting; even an empty shard contribution
        // is required for that shard's fixed-group round to finish.
        for (auto& ack : acknowledgements) Take(std::move(ack));
        if (j.value("retry_completed_submit", false))
          for (std::size_t home = 0; home < shards.size(); ++home)
            Take(collectives[home]->Submit(std::move(packets[home])));
      }
      // All owning shards have acknowledged the actual update before the
      // coordinator advances this worker's contiguous completed-task count.
      Take(progress->Complete(worker, iteration));
      ++completed;
      if (!iteration) first_loss = loss;
      last_loss = loss;
    }
    Take(adapter.Join());
    std::cout << nlohmann::json{{"worker", worker}, {"completed", completed},
      {"input_batches_committed", completed}, {"max_admission_lead", max_lead},
      {"admission_wait_ns", wait_ns}, {"first_local_loss", first_loss},
      {"last_local_loss", last_loss}, {"gradient_reduction", "local_batch_mean"},
      {"optimizer_step", synchronous ? "per_group_round" : "per_worker_update"}}.dump() << '\n';
  } catch (const std::exception& e) {
    if (progress) { try { (void)progress->Abort(e.what()).get(); } catch (...) {} }
    for (auto& collective : collectives) { try { (void)collective->Abort(e.what()).get(); } catch (...) {} }
    std::cerr << "eps_training_worker: " << e.what() << '\n'; return 1;
  }
}
