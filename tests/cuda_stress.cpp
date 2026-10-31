#include "eps/cuda_cache_backend.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>

using namespace eps;
namespace {
template<typename T>T Take(std::future<Result<T>> f){auto r=f.get();
  if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
void Check(bool ok,const char* m){if(!ok)throw std::runtime_error(m);}
float Weight(Key key,std::uint32_t d){return 0.001f*(float(key%17)+float(d%13));}
struct Query {std::vector<AgentWork> work;std::vector<FeatureInput> layouts;};
Query MakeQuery(const std::vector<LocatedRow>& rows,std::uint32_t keys,std::mt19937_64& random){
  constexpr std::uint64_t length=16;
  FeatureInput layout{1,0,Pooling::sum,{},{0},{}};
  AgentWork work{0,AccessRoute::resident_gpu,0,{},{},{}};
  std::unordered_map<Key,std::uint64_t> unique;unique.reserve(keys);
  for(std::uint32_t i=0;i<keys;++i){
    const auto limit=random()%100<95?std::min<std::size_t>(rows.size(),32768):rows.size();
    const auto& row=rows[random()%limit];layout.keys.push_back(row.row.key);
    auto [it,added]=unique.try_emplace(row.row.key,work.rows.size());
    if(added)work.rows.push_back(row);work.occurrence_to_unique.push_back(it->second);
    if((i+1)%length==0){layout.offsets.push_back(i+1);layout.valid_lengths.push_back(length);}
  }
  return {{std::move(work)},{std::move(layout)}};
}
}
int main(int argc,char** argv){
  try{
    const std::uint64_t rows=argc>1?std::stoull(argv[1]):262144;
    const std::uint32_t dimension=argc>2?std::stoul(argv[2]):128;
    const int workers=argc>3?std::stoi(argv[3]):8,batches=argc>4?std::stoi(argv[4]):64;
    constexpr std::uint32_t keys_per_batch=4096;
    Check(rows>0&&rows<=1048576&&dimension>0&&dimension<=512&&workers>0&&workers<=32&&batches>0&&batches<=1000,
      "invalid CUDA stress dimensions");
    CudaCacheOptions options;options.rows_per_table=rows;
    options.state_bytes=rows*(std::uint64_t(dimension)*12+8);
    CudaCacheBackend backend(options);
    Take(backend.Define({1,dimension,1,1,{0.01f,0.9f,0.999f,1e-8f,0.9f,0.999f},0,0,42}));
    std::vector<LocatedRow> locations;locations.reserve(rows);
    const auto chunk=std::min<std::uint64_t>(4096,options.transfer_bytes/(std::uint64_t(dimension)*12));
    for(std::uint64_t begin=0;begin<rows;begin+=chunk){
      MigrationPlan plan{"stress-load",0,{},Placement::gpu,0};std::vector<RowState> state;
      for(auto i=begin;i<std::min(rows,begin+chunk);++i){
        RowState r{{1,i+1},std::vector<float>(dimension),std::vector<float>(dimension,0),std::vector<float>(dimension,0),0};
        for(std::uint32_t d=0;d<dimension;++d)r.weights[d]=Weight(i+1,d);
        plan.rows.push_back({r.row,0,Placement::loading,0,1,0});state.push_back(std::move(r));
      }
      auto loaded=Take(backend.Load(std::move(plan),std::move(state)));
      locations.insert(locations.end(),loaded.begin(),loaded.end());
    }
    std::mt19937_64 warm_random(1);auto warm=MakeQuery(locations,keys_per_batch,warm_random);
    Take(backend.GatherHost(warm.work,warm.layouts));
    const auto memory_before=Take(backend.MemoryStats());
    std::size_t free_before=0,total=0,free_after=0;
    Check(cudaMemGetInfo(&free_before,&total)==cudaSuccess,"cannot read CUDA memory state");
    std::atomic<bool> start{false};std::atomic<int> ready{0};
    std::mutex mutex;std::string failure;float maximum_error=0;
    std::vector<std::thread> callers;
    for(int worker=0;worker<workers;++worker)callers.emplace_back([&,worker]{
      ++ready;while(!start.load())std::this_thread::yield();
      try{
        std::mt19937_64 random(42+worker);float local_error=0;
        for(int batch=0;batch<batches;++batch){
          auto query=MakeQuery(locations,keys_per_batch,random);
          auto result=Take(backend.GatherHost(query.work,query.layouts));
          Check(result.size()==1&&result[0].values.size()==std::uint64_t(keys_per_batch/16)*dimension,"stress output shape mismatch");
          const auto& input=query.layouts[0];
          for(std::size_t bag=0;bag<input.valid_lengths.size();++bag)for(std::uint32_t d=0;d<dimension;++d){
            float expected=0;for(auto i=input.offsets[bag];i<input.offsets[bag+1];++i)expected+=Weight(input.keys[i],d);
            const auto actual=result[0].values[bag*dimension+d];
            Check(std::isfinite(actual),"non-finite stress result");local_error=std::max(local_error,std::abs(expected-actual));
          }
        }
        std::lock_guard lock(mutex);maximum_error=std::max(maximum_error,local_error);
      }catch(const std::exception& e){std::lock_guard lock(mutex);failure=e.what();}
    });
    while(ready.load()!=workers)std::this_thread::yield();
    const auto begin=std::chrono::steady_clock::now();start=true;
    for(auto& caller:callers)caller.join();
    const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    Check(failure.empty(),failure.c_str());Check(maximum_error<1e-5f,"CUDA stress comparison exceeded tolerance");
    Check(cudaMemGetInfo(&free_after,&total)==cudaSuccess,"cannot read final CUDA memory state");
    const auto growth=free_before>free_after?free_before-free_after:0;
    // Driver free-memory readings can change independently on a shared GPU.
    // Assert the allocator state owned by this backend; report the device-wide
    // reading only as an observation.
    const auto memory_after=Take(backend.MemoryStats());
    Check(memory_before==memory_after,"pooled reads changed backend device allocations");
    std::cout<<"{\"test\":\"cuda_checked_read_stress\",\"rows\":"<<rows
      <<",\"dimension\":"<<dimension<<",\"workers\":"<<workers
      <<",\"checked_batches\":"<<workers*batches<<",\"key_occurrences\":"<<std::uint64_t(workers)*batches*keys_per_batch
      <<",\"weight_working_set_bytes\":"<<rows*dimension*4
      <<",\"maximum_absolute_error\":"<<maximum_error<<",\"device_memory_growth_bytes\":"<<growth
      <<",\"backend_state_bytes\":"<<memory_after.resident_state_bytes
      <<",\"backend_staging_bytes\":"<<memory_after.staging_device_bytes
      <<",\"read_phase_output_allocations\":"<<memory_after.output_allocations-memory_before.output_allocations
      <<",\"wall_seconds_including_reference_checks\":"<<seconds<<"}\n";
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL CUDA stress: "<<e.what()<<'\n';return 1;}
}
