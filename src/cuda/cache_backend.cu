#include "eps/cuda_cache_backend.hpp"
#include "eps/compact_gradients.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include "common/executor.hpp"
#include "common/optimizer_bounds.hpp"

namespace eps {
namespace {
void Cuda(cudaError_t status) {
  if (status != cudaSuccess)
    throw detail::Failure({status == cudaErrorMemoryAllocation
        ? Error::Code::capacity_exceeded : Error::Code::run_failed,
        cudaGetErrorString(status)});
}
struct Buffer {
  void* pointer = nullptr;
  int device = 0;
  bool pinned = false;
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { Reset(); }
  void Reset() noexcept {
    if (!pointer) return;
    cudaSetDevice(device);
    if (pinned) cudaFreeHost(pointer); else cudaFree(pointer);
    pointer = nullptr;
  }
  void Allocate(std::uint64_t bytes, int ordinal, bool host = false) {
    device = ordinal; pinned = host; Cuda(cudaSetDevice(device));
    if (bytes) {
      if (host) Cuda(cudaMallocHost(&pointer, bytes));
      else Cuda(cudaMalloc(&pointer, bytes));
    }
  }
  template<typename T> T* As() const { return static_cast<T*>(pointer); }
};
struct Stream {
  cudaStream_t value = nullptr;
  int device;
  explicit Stream(int d) : device(d) {
    Cuda(cudaSetDevice(d));
    Cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
  }
  ~Stream() { cudaSetDevice(device); if (value) cudaStreamDestroy(value); }
};
struct Event {
  cudaEvent_t value = nullptr;
  int device;
  explicit Event(int d) : device(d) {
    Cuda(cudaSetDevice(d));
    Cuda(cudaEventCreateWithFlags(&value, cudaEventDisableTiming));
  }
  ~Event() { cudaSetDevice(device); if (value) cudaEventDestroy(value); }
};
unsigned Blocks(std::uint64_t n) {
  return static_cast<unsigned>(std::min<std::uint64_t>((n + 255) / 256, 4096));
}
__global__ void ScatterState(float* target, const float* packed,
    const std::uint64_t* slots, std::uint64_t count, std::uint64_t width) {
  for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count * width; i += std::uint64_t(blockDim.x) * gridDim.x)
    target[slots[i / width] * width + i % width] = packed[i];
}
__global__ void ReadStateKernel(const float* source, float* packed,
    const std::uint64_t* slots, std::uint64_t count, std::uint64_t width) {
  for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count * width; i += std::uint64_t(blockDim.x) * gridDim.x)
    packed[i] = source[slots[i / width] * width + i % width];
}
__global__ void GatherKernel(const float* state, float* output,
    const std::uint64_t* slots, const std::uint64_t* offsets,
    std::uint64_t rows, std::uint32_t dim, std::uint64_t stride, bool sequence) {
  for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < rows * dim; i += std::uint64_t(blockDim.x) * gridDim.x) {
    const auto row = i / dim, d = i % dim;
    float sum = 0;
    const auto begin = sequence ? row : offsets[row];
    const auto end = sequence ? row + 1 : offsets[row + 1];
    for (auto pos = begin; pos < end; ++pos)
      if (slots[pos] != UINT64_MAX) sum += state[slots[pos] * stride + d];
    output[i] = sum;
  }
}
__global__ void AdamKernel(float* state, const float* gradients,
    const std::uint64_t* slots, std::uint64_t count, std::uint32_t dim,
    float beta1, float beta2, float lr, float epsilon, float minimum, float maximum,
    bool validate, int* invalid) {
  for (std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count * dim; i += std::uint64_t(blockDim.x) * gridDim.x) {
    const auto base = slots[i / dim] * dim * 3 + i % dim;
    const auto g = gradients[i];
    const auto m = beta1 * state[base + dim] + (1 - beta1) * g;
    const auto v = beta2 * state[base + dim * 2] + (1 - beta2) * g * g;
    const auto w = state[base] - lr * m / (sqrtf(v) + epsilon);
    if (validate) {
      if (!isfinite(g) || !isfinite(m) || !isfinite(v) || !isfinite(w))
        atomicExch(invalid, 1);
    } else {
      state[base] = fminf(maximum,fmaxf(minimum,w));
      state[base + dim] = m; state[base + dim * 2] = v;
    }
  }
}
__global__ void ReduceFeatureGradients(const float* sources,const float* scales,
    const std::uint64_t* offsets,const std::uint64_t* references,float* gradients,
    std::uint64_t rows,std::uint32_t dim){
  for(std::uint64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<rows*dim;
      i+=std::uint64_t(blockDim.x)*gridDim.x){
    const auto row=i/dim,d=i%dim;float sum=0;
    for(auto ref=offsets[row];ref<offsets[row+1];++ref)
      sum+=sources[references[ref]*dim+d]*scales[ref];
    gradients[i]=sum;
  }
}
struct OutputBudget {
  std::atomic<std::uint64_t> used{0};
  std::atomic<std::uint64_t> allocations{0};
  std::uint64_t limit = 0;
};
struct Allocation {
  Buffer buffer;
  Event ready;
  std::shared_ptr<OutputBudget> budget;
  std::uint64_t bytes;
  Allocation(int device, std::uint64_t n, std::shared_ptr<OutputBudget> b)
      : ready(device), budget(std::move(b)), bytes(n) {
    detail::Require(n <= budget->limit, "device output is too large",
                    Error::Code::capacity_exceeded);
    const auto previous = budget->used.fetch_add(n);
    if (previous > budget->limit - n) {
      budget->used.fetch_sub(n);
      throw detail::Failure({Error::Code::capacity_exceeded,
                             "outstanding device output budget exhausted"});
    }
    try { buffer.Allocate(n, device); ++budget->allocations; }
    catch (...) { budget->used.fetch_sub(n); throw; }
  }
  ~Allocation() { buffer.Reset(); budget->used.fetch_sub(bytes); }
};
class TensorView final : public DeviceTensor {
 public:
  TensorView(std::shared_ptr<Allocation> owner, std::uint64_t offset,
             std::uint64_t count)
      : owner_(std::move(owner)), offset_(offset), count_(count) {}
  void* data() const override {
    return owner_->buffer.pointer ? owner_->buffer.As<float>() + offset_ : nullptr;
  }
  std::uint64_t elements() const override { return count_; }
  int device() const override { return owner_->buffer.device; }
  void WaitOnStream(void* stream) const override {
    Cuda(cudaSetDevice(device()));
    Cuda(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), owner_->ready.value, 0));
  }
  std::vector<float> CopyToHost() const override {
    Cuda(cudaSetDevice(device())); Cuda(cudaEventSynchronize(owner_->ready.value));
    std::vector<float> values(count_);
    if (count_) Cuda(cudaMemcpy(values.data(), data(), count_ * 4, cudaMemcpyDeviceToHost));
    return values;
  }
 private:
  std::shared_ptr<Allocation> owner_;
  std::uint64_t offset_, count_;
};
}  // namespace

struct CudaCacheBackend::Impl {
  struct Entry { LocatedRow location; std::uint64_t updates; };
  struct Table {
    TableDefinition definition;
    Buffer state;
    std::vector<std::uint64_t> free;
    std::unordered_map<Key, Entry> rows;
  };
  struct GatherPart {
    std::uint32_t feature;
    Table* table;
    Pooling pooling;
    std::uint64_t slot_offset, offsets_offset, output_offset, rows;
    const float* source;
    std::uint64_t stride;
  };
  explicit Impl(CudaCacheOptions o)
      : options(o), stream(o.device), done(o.device),
        output_budget(std::make_shared<OutputBudget>()) {
    detail::Require(o.rows_per_table > 0 && o.rows_per_table <= (1ULL << 24) &&
                    o.transfer_bytes >= 1024 && o.transfer_bytes <= (64ULL << 20),
                    "invalid CUDA cache limits");
    output_budget->limit = o.outstanding_output_bytes;
    host_values.Allocate(o.transfer_bytes, o.device, true);
    host_indices.Allocate(o.transfer_bytes, o.device, true);
    device_values.Allocate(o.transfer_bytes, o.device);
    device_indices.Allocate(o.transfer_bytes, o.device);
    device_transient.Allocate(o.transfer_bytes, o.device);
    invalid.Allocate(sizeof(int), o.device);
    executor = std::make_unique<detail::OrderedExecutor>(128);
  }
  ~Impl() {
    executor.reset();
    cudaSetDevice(options.device); cudaStreamSynchronize(stream.value);
  }
  void Activate() { Cuda(cudaSetDevice(options.device)); }
  void Finish() {
    Cuda(cudaEventRecord(done.value, stream.value));
    Cuda(cudaEventSynchronize(done.value));
  }
  Table& Get(TableId id) {
    auto it = tables.find(id);
    detail::Require(it != tables.end(), "unknown CUDA cache table");
    return *it->second;
  }
  Entry& Find(const LocatedRow& loc) {
    auto& t = Get(loc.row.table);
    auto it = t.rows.find(loc.row.key);
    detail::Require(it != t.rows.end() && loc.agent == options.agent &&
        it->second.location.generation == loc.generation &&
        it->second.location.slot == loc.slot && it->second.location.home == loc.home,
        "CUDA cache generation mismatch", Error::Code::stale_generation);
    return it->second;
  }
  void Limit(std::uint64_t bytes) const {
    detail::Require(bytes <= options.transfer_bytes, "CUDA staging budget exceeded",
                    Error::Code::capacity_exceeded);
  }
  std::pair<std::vector<GatherPart>, std::uint64_t> PrepareGather(
      const std::vector<AgentWork>& work, const std::vector<FeatureInput>& layouts) {
    std::vector<GatherPart> parts; parts.reserve(work.size());
    std::uint64_t indices = 0, values = 0, transferred = 0;
    auto* index = host_indices.As<std::uint64_t>();
    for (const auto& task : work) {
      auto it = std::find_if(layouts.begin(), layouts.end(),
                           [&](const auto& f) { return f.feature == task.feature; });
      detail::Require(it != layouts.end() && task.agent == options.agent &&
          (task.route == AccessRoute::resident_gpu || task.route==AccessRoute::transfer_unique), "invalid CUDA gather work");
      const auto& f = *it;
      auto& table = Get(f.table); const auto dim = table.definition.dimension;
      const bool transfer=task.route==AccessRoute::transfer_unique;
      detail::Require(transfer?task.transferred_weights.size()==task.rows.size()*dim:
        task.transferred_weights.empty(),"invalid CUDA transferred weights");
      const auto transfer_offset=transferred;
      if(transfer){
        Limit((transferred+task.transferred_weights.size())*4);
        for(float value:task.transferred_weights)detail::Require(std::isfinite(value),"non-finite transferred weight");
        std::copy(task.transferred_weights.begin(),task.transferred_weights.end(),host_values.As<float>()+transferred);
        transferred+=task.transferred_weights.size();
      }
      detail::Require(task.occurrence_to_unique.size() == f.keys.size() &&
          !f.offsets.empty() && f.offsets.front() == 0 &&
          f.offsets.back() == f.keys.size() &&
          std::is_sorted(f.offsets.begin(), f.offsets.end()), "invalid CUDA gather layout");
      Limit((indices + f.keys.size() + f.offsets.size()) * 8);
      const auto slot_offset = indices;
      for (std::size_t i = 0; i < f.keys.size(); ++i) {
        const auto unique = task.occurrence_to_unique[i];
        if (unique == UINT64_MAX) index[indices++] = UINT64_MAX;
        else {
          detail::Require(unique < task.rows.size() &&
              task.rows[unique].row == RowKey{f.table, f.keys[i]},
              "invalid CUDA occurrence map");
          if(transfer){
            detail::Require(task.rows[unique].placement==Placement::cpu,"transferred row is not CPU-owned");
            index[indices++]=unique;
          }else index[indices++] = Find(task.rows[unique]).location.slot;
        }
      }
      const auto offsets_offset = indices;
      std::copy(f.offsets.begin(), f.offsets.end(), index + indices);
      indices += f.offsets.size();
      const auto rows = f.pooling == Pooling::sequence
          ? f.keys.size() : f.offsets.size() - 1;
      Limit((values + rows * dim) * 4);
      parts.push_back({f.feature, &table, f.pooling, slot_offset, offsets_offset,
                       values, rows,transfer?device_transient.As<float>()+transfer_offset:table.state.As<float>(),
                       std::uint64_t(dim)*(transfer?1:3)});
      values += rows * dim;
    }
    if (indices) Cuda(cudaMemcpyAsync(device_indices.pointer, host_indices.pointer,
                       indices * 8, cudaMemcpyHostToDevice, stream.value));
    if(transferred)Cuda(cudaMemcpyAsync(device_transient.pointer,host_values.pointer,
      transferred*4,cudaMemcpyHostToDevice,stream.value));
    return {std::move(parts), values};
  }
  void LaunchGather(const std::vector<GatherPart>& parts, float* output) {
    for (const auto& p : parts) {
      const auto dim = p.table->definition.dimension;
      if (!p.rows) continue;
      GatherKernel<<<Blocks(p.rows * dim), 256, 0, stream.value>>>(
          p.source, output + p.output_offset,
          device_indices.As<std::uint64_t>() + p.slot_offset,
          device_indices.As<std::uint64_t>() + p.offsets_offset,
          p.rows, dim, p.stride, p.pooling == Pooling::sequence);
      Cuda(cudaGetLastError());
    }
  }
  Table& ValidateUpdate(const UpdateContext& context,const std::vector<RowKey>& rows){
    auto& t=Get(context.table);
    detail::Require(std::isfinite(context.learning_rate) && context.learning_rate>0 &&
      context.beta1_power>=0 && context.beta1_power<1 && context.beta2_power>=0 &&
      context.beta2_power<1,"invalid CUDA update context");
    Limit(rows.size()*std::uint64_t(t.definition.dimension)*4);
    std::set<Key> unique;
    for(auto row:rows){auto it=t.rows.find(row.key);
      detail::Require(row.table==context.table && unique.insert(row.key).second &&
        it!=t.rows.end() && it->second.location.home==context.shard &&
        it->second.updates!=UINT64_MAX,"invalid CUDA gradient owner/counter");}
    return t;
  }
  Ack ApplyPrepared(UpdateContext context,const std::vector<RowKey>& rows,std::uint64_t slots_offset){
    auto& t=Get(context.table);const auto dim=t.definition.dimension;
    if(rows.empty())return {};
    Cuda(cudaMemsetAsync(invalid.pointer,0,sizeof(int),stream.value));
    const auto& o=t.definition.optimizer;
    const auto minimum=o.weight_bounds?o.weight_bounds->minimum:-std::numeric_limits<float>::max();
    const auto maximum=o.weight_bounds?o.weight_bounds->maximum:std::numeric_limits<float>::max();
    const auto lr=context.learning_rate*std::sqrt(1-context.beta2_power)/(1-context.beta1_power);
    AdamKernel<<<Blocks(rows.size()*dim),256,0,stream.value>>>(t.state.As<float>(),device_values.As<float>(),
      device_indices.As<std::uint64_t>()+slots_offset,rows.size(),dim,o.beta1,o.beta2,lr,o.epsilon,minimum,maximum,true,invalid.As<int>());
    Cuda(cudaGetLastError());int rejected=0;
    Cuda(cudaMemcpyAsync(&rejected,invalid.pointer,sizeof(int),cudaMemcpyDeviceToHost,stream.value));Finish();
    detail::Require(!rejected,"CUDA optimizer state would overflow");
    AdamKernel<<<Blocks(rows.size()*dim),256,0,stream.value>>>(t.state.As<float>(),device_values.As<float>(),
      device_indices.As<std::uint64_t>()+slots_offset,rows.size(),dim,o.beta1,o.beta2,lr,o.epsilon,minimum,maximum,false,invalid.As<int>());
    Cuda(cudaGetLastError());Finish();for(auto row:rows)++t.rows.at(row.key).updates;return {};
  }
  CudaCacheOptions options;
  Stream stream;
  Event done;
  Buffer host_values, host_indices, device_values, device_indices, device_transient, invalid;
  std::map<TableId, std::unique_ptr<Table>> tables;
  std::uint64_t allocated_state = 0;
  std::shared_ptr<OutputBudget> output_budget;
  std::unique_ptr<detail::OrderedExecutor> executor;
};

CudaCacheBackend::CudaCacheBackend(CudaCacheOptions o) : impl_(std::make_unique<Impl>(o)) {}
CudaCacheBackend::~CudaCacheBackend() = default;

std::future<Result<Ack>> CudaCacheBackend::Define(TableDefinition d) {
  return impl_->executor->Submit<Ack>([this, d] {
    detail::ValidateWeightBounds(d.optimizer);
    impl_->Activate();
    detail::Require(d.dimension > 0 && d.dimension <= 65536, "invalid CUDA dimension");
    if (auto it = impl_->tables.find(d.id); it != impl_->tables.end()) {
      detail::Require(it->second->definition == d, "CUDA table configuration mismatch");
      return Ack{};
    }
    const auto bytes = impl_->options.rows_per_table * (std::uint64_t(d.dimension) * 12 + 8);
    detail::Require(bytes <= impl_->options.state_bytes - impl_->allocated_state,
                    "CUDA table state budget exhausted", Error::Code::capacity_exceeded);
    auto t = std::make_unique<Impl::Table>(); t->definition = d;
    t->state.Allocate(impl_->options.rows_per_table * std::uint64_t(d.dimension) * 12,
                      impl_->options.device);
    t->free.resize(impl_->options.rows_per_table);
    std::iota(t->free.rbegin(), t->free.rend(), 0);
    impl_->tables.emplace(d.id, std::move(t)); impl_->allocated_state += bytes;
    return Ack{};
  });
}
std::future<Result<std::uint64_t>> CudaCacheBackend::FreeSlots(TableId id) {
  return impl_->executor->Submit<std::uint64_t>([this, id] {
    return std::uint64_t(impl_->Get(id).free.size());
  });
}
std::future<Result<std::vector<LocatedRow>>> CudaCacheBackend::Load(
    MigrationPlan plan, std::vector<RowState> states) {
  return impl_->executor->Submit<std::vector<LocatedRow>>(
      [this, plan = std::move(plan), states = std::move(states)] {
    impl_->Activate();
    detail::Require(plan.agent == impl_->options.agent && plan.rows.size() == states.size() &&
                    plan.destination == Placement::gpu, "invalid CUDA load plan");
    std::map<TableId, std::vector<std::size_t>> groups;
    std::set<std::pair<TableId, Key>> unique; std::uint64_t bytes = 0;
    for (std::size_t i = 0; i < states.size(); ++i) {
      const auto& row = states[i]; auto& t = impl_->Get(row.row.table);
      const auto dim = t.definition.dimension;
      detail::Require(row.row == plan.rows[i].row && !t.rows.contains(row.row.key) &&
          unique.emplace(row.row.table, row.row.key).second, "duplicate/existing CUDA load row");
      detail::Require(row.weights.size() == dim && row.first_moment.size() == dim &&
                      row.second_moment.size() == dim, "CUDA load shape mismatch");
      for (auto* v : {&row.weights, &row.first_moment, &row.second_moment})
        for (auto x : *v) detail::Require(std::isfinite(x), "non-finite CUDA load state");
      for (auto x : row.second_moment) detail::Require(x >= 0, "negative CUDA second moment");
      bytes += std::uint64_t(dim) * 12; groups[row.row.table].push_back(i);
    }
    impl_->Limit(bytes); impl_->Limit(states.size() * 8);
    for (const auto& [id, rows] : groups) {
      auto& t = impl_->Get(id);
      detail::Require(plan.reserve_slots<=t.free.size() && rows.size()<=t.free.size()-plan.reserve_slots,
                      "CUDA cache slots or reserve exhausted",
                      Error::Code::capacity_exceeded);
      t.rows.reserve(t.rows.size() + rows.size());
    }
    std::vector<LocatedRow> result(states.size());
    std::vector<std::pair<TableId, Key>> inserted; inserted.reserve(states.size());
    struct Group { Impl::Table* table; std::uint64_t values, slots, count; };
    std::vector<Group> copies; copies.reserve(groups.size());
    std::uint64_t value_offset = 0, slot_offset = 0;
    try {
      for (const auto& [id, rows] : groups) {
        auto& t = impl_->Get(id); const auto dim = t.definition.dimension;
        copies.push_back({&t, value_offset, slot_offset, rows.size()});
        for (auto i : rows) {
          const auto slot = t.free.back();
          auto loc = plan.rows[i]; loc.slot = slot; loc.agent = impl_->options.agent;
          loc.placement = Placement::gpu;
          t.rows.emplace(loc.row.key, Impl::Entry{loc, states[i].row_updates});
          t.free.pop_back(); inserted.emplace_back(id, loc.row.key); result[i] = loc;
          impl_->host_indices.As<std::uint64_t>()[slot_offset++] = slot;
          for (auto* v : {&states[i].weights, &states[i].first_moment, &states[i].second_moment}) {
            std::copy(v->begin(), v->end(), impl_->host_values.As<float>() + value_offset);
            value_offset += dim;
          }
        }
      }
      if (bytes) Cuda(cudaMemcpyAsync(impl_->device_values.pointer, impl_->host_values.pointer,
                                     bytes, cudaMemcpyHostToDevice, impl_->stream.value));
      if (slot_offset) Cuda(cudaMemcpyAsync(impl_->device_indices.pointer,
          impl_->host_indices.pointer, slot_offset * 8, cudaMemcpyHostToDevice, impl_->stream.value));
      for (const auto& c : copies) if (c.count) {
        const auto width = std::uint64_t(c.table->definition.dimension) * 3;
        ScatterState<<<Blocks(c.count * width), 256, 0, impl_->stream.value>>>(
            c.table->state.As<float>(), impl_->device_values.As<float>() + c.values,
            impl_->device_indices.As<std::uint64_t>() + c.slots, c.count, width);
        Cuda(cudaGetLastError());
      }
      impl_->Finish(); return result;
    } catch (...) {
      cudaStreamSynchronize(impl_->stream.value);
      for (auto [id, key] : inserted) {
        auto& t = impl_->Get(id);
        t.free.push_back(t.rows.at(key).location.slot); t.rows.erase(key);
      }
      throw;
    }
  });
}
std::future<Result<std::vector<TensorBlock>>> CudaCacheBackend::GatherHost(
    std::vector<AgentWork> work, std::vector<FeatureInput> layouts) {
  return impl_->executor->Submit<std::vector<TensorBlock>>(
      [this, work = std::move(work), layouts = std::move(layouts)] {
    impl_->Activate();
    auto [parts, values] = impl_->PrepareGather(work, layouts);
    impl_->LaunchGather(parts, impl_->device_values.As<float>());
    if (values) Cuda(cudaMemcpyAsync(impl_->host_values.pointer, impl_->device_values.pointer,
                                    values * 4, cudaMemcpyDeviceToHost, impl_->stream.value));
    impl_->Finish();
    std::vector<TensorBlock> out; out.reserve(parts.size());
    for (const auto& p : parts) {
      const auto count = p.rows * p.table->definition.dimension;
      auto* begin = impl_->host_values.As<float>() + p.output_offset;
      out.push_back({p.feature, {p.rows, p.table->definition.dimension}, {begin, begin + count}});
    }
    return out;
  });
}
std::future<Result<std::vector<DeviceTensorBlock>>> CudaCacheBackend::GatherDevice(
    std::vector<AgentWork> work, std::vector<FeatureInput> layouts) {
  return impl_->executor->Submit<std::vector<DeviceTensorBlock>>(
      [this, work = std::move(work), layouts = std::move(layouts)] {
    impl_->Activate();
    auto [parts, values] = impl_->PrepareGather(work, layouts);
    auto allocation = std::make_shared<Allocation>(impl_->options.device, values * 4,
                                                  impl_->output_budget);
    std::vector<DeviceTensorBlock> out; out.reserve(parts.size());
    for (const auto& p : parts)
      out.push_back({p.feature, {p.rows, p.table->definition.dimension},
          std::make_shared<TensorView>(allocation, p.output_offset,
                                      p.rows * p.table->definition.dimension)});
    impl_->LaunchGather(parts, allocation->buffer.As<float>());
    Cuda(cudaEventRecord(allocation->ready.value, impl_->stream.value));
    impl_->Finish(); // Host index staging is reusable after the submitted work completes.
    return out;
  });
}
std::future<Result<Ack>> CudaCacheBackend::ApplyHost(UpdateContext context, RowGradients g) {
  return impl_->executor->Submit<Ack>([this,context,g=std::move(g)] {
    impl_->Activate();auto& t=impl_->ValidateUpdate(context,g.rows);const auto dim=t.definition.dimension;
    detail::Require(g.values.size()==g.rows.size()*dim,"invalid CUDA gradient shape");
    impl_->Limit(g.rows.size()*8);
    for(std::size_t i=0;i<g.rows.size();++i)impl_->host_indices.As<std::uint64_t>()[i]=t.rows.at(g.rows[i].key).location.slot;
    if(g.rows.empty())return Ack{};
    std::copy(g.values.begin(),g.values.end(),impl_->host_values.As<float>());
    Cuda(cudaMemcpyAsync(impl_->device_values.pointer,impl_->host_values.pointer,g.values.size()*4,cudaMemcpyHostToDevice,impl_->stream.value));
    Cuda(cudaMemcpyAsync(impl_->device_indices.pointer,impl_->host_indices.pointer,g.rows.size()*8,cudaMemcpyHostToDevice,impl_->stream.value));
    return impl_->ApplyPrepared(context,g.rows,0);
  });
}
std::future<Result<Ack>> CudaCacheBackend::ApplyCompact(UpdateContext context,CompactGradients p){
  return impl_->executor->Submit<Ack>([this,context,p=std::move(p)]{
    impl_->Activate();ValidateCompactGradients(p);
    auto& t=impl_->ValidateUpdate(context,p.rows);const auto dim=t.definition.dimension;
    detail::Require(p.dimension==dim,"compact CUDA table dimension mismatch");
    const auto slots_offset=p.row_offsets.size()+p.source_rows.size();
    const auto index_count=slots_offset+p.rows.size();
    const auto value_count=p.source_values.size()+p.scales.size();
    impl_->Limit(index_count*8);impl_->Limit(value_count*4);
    if(p.rows.empty())return Ack{};
    auto* indices=impl_->host_indices.As<std::uint64_t>();
    std::copy(p.row_offsets.begin(),p.row_offsets.end(),indices);
    std::copy(p.source_rows.begin(),p.source_rows.end(),indices+p.row_offsets.size());
    for(std::size_t i=0;i<p.rows.size();++i)indices[slots_offset+i]=t.rows.at(p.rows[i].key).location.slot;
    auto* values=impl_->host_values.As<float>();
    std::copy(p.source_values.begin(),p.source_values.end(),values);
    std::copy(p.scales.begin(),p.scales.end(),values+p.source_values.size());
    Cuda(cudaMemcpyAsync(impl_->device_transient.pointer,values,value_count*4,cudaMemcpyHostToDevice,impl_->stream.value));
    Cuda(cudaMemcpyAsync(impl_->device_indices.pointer,indices,index_count*8,cudaMemcpyHostToDevice,impl_->stream.value));
    ReduceFeatureGradients<<<Blocks(p.rows.size()*dim),256,0,impl_->stream.value>>>(
      impl_->device_transient.As<float>(),impl_->device_transient.As<float>()+p.source_values.size(),
      impl_->device_indices.As<std::uint64_t>(),impl_->device_indices.As<std::uint64_t>()+p.row_offsets.size(),
      impl_->device_values.As<float>(),p.rows.size(),dim);
    Cuda(cudaGetLastError());return impl_->ApplyPrepared(context,p.rows,slots_offset);
  });
}
std::future<Result<std::vector<RowState>>> CudaCacheBackend::ReadBack(MigrationPlan plan) {
  return impl_->executor->Submit<std::vector<RowState>>([this, plan = std::move(plan)] {
    impl_->Activate();
    std::map<TableId, std::vector<std::size_t>> groups;
    std::uint64_t bytes = 0;
    std::vector<RowState> out(plan.rows.size());
    for (std::size_t i = 0; i < plan.rows.size(); ++i) {
      const auto& loc = plan.rows[i]; auto& entry = impl_->Find(loc);
      const auto dim = impl_->Get(loc.row.table).definition.dimension;
      out[i] = {loc.row, std::vector<float>(dim), std::vector<float>(dim),
                std::vector<float>(dim), entry.updates};
      groups[loc.row.table].push_back(i); bytes += std::uint64_t(dim) * 12;
    }
    impl_->Limit(bytes); impl_->Limit(plan.rows.size() * 8);
    struct Part { Impl::Table* table; std::uint64_t values, slots, count; };
    std::vector<Part> parts; std::uint64_t values = 0, slots = 0;
    for (const auto& [id, indices] : groups) {
      auto& t = impl_->Get(id); parts.push_back({&t, values, slots, indices.size()});
      for (auto i : indices) {
        impl_->host_indices.As<std::uint64_t>()[slots++] =
            impl_->Find(plan.rows[i]).location.slot;
        values += std::uint64_t(t.definition.dimension) * 3;
      }
    }
    if (slots) Cuda(cudaMemcpyAsync(impl_->device_indices.pointer,
        impl_->host_indices.pointer, slots * 8, cudaMemcpyHostToDevice, impl_->stream.value));
    for (const auto& p : parts) if (p.count) {
      const auto width = std::uint64_t(p.table->definition.dimension) * 3;
      ReadStateKernel<<<Blocks(p.count * width), 256, 0, impl_->stream.value>>>(
          p.table->state.As<float>(), impl_->device_values.As<float>() + p.values,
          impl_->device_indices.As<std::uint64_t>() + p.slots, p.count, width);
      Cuda(cudaGetLastError());
    }
    if (bytes) Cuda(cudaMemcpyAsync(impl_->host_values.pointer,
        impl_->device_values.pointer, bytes, cudaMemcpyDeviceToHost, impl_->stream.value));
    impl_->Finish();
    std::uint64_t offset = 0;
    for (const auto& [id, indices] : groups) for (auto i : indices) {
      const auto dim = impl_->Get(id).definition.dimension;
      for (auto* v : {&out[i].weights, &out[i].first_moment, &out[i].second_moment}) {
        std::copy_n(impl_->host_values.As<float>() + offset, dim, v->begin()); offset += dim;
      }
    }
    return out;
  });
}
std::future<Result<Ack>> CudaCacheBackend::CommitEviction(MigrationPlan plan) {
  return impl_->executor->Submit<Ack>([this, plan = std::move(plan)] {
    std::set<std::pair<TableId, Key>> unique;
    for (const auto& loc : plan.rows) {
      impl_->Find(loc);
      detail::Require(unique.emplace(loc.row.table, loc.row.key).second,
                      "duplicate CUDA eviction row");
    }
    for (const auto& loc : plan.rows) {
      auto& t = impl_->Get(loc.row.table); t.free.push_back(loc.slot); t.rows.erase(loc.row.key);
    }
    return Ack{};
  });
}
std::future<Result<CudaMemoryStats>> CudaCacheBackend::MemoryStats() {
  return impl_->executor->Submit<CudaMemoryStats>([this] {
    std::uint64_t state = 0;
    for (const auto& [id, table] : impl_->tables)
      state += impl_->options.rows_per_table * std::uint64_t(table->definition.dimension) * 12;
    return CudaMemoryStats{state, impl_->options.transfer_bytes * 3 + sizeof(int),
      impl_->output_budget->used.load(), impl_->output_budget->allocations.load()};
  });
}
}  // namespace eps

