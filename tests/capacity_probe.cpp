#include "eps/brpc_client.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/distributed_client.hpp"
#include "eps/cache_resolver.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <latch>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <thread>

using namespace eps;
using Clock=std::chrono::steady_clock;
namespace {
template<typename T>T Take(Result<T> r){if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<typename T>T Take(std::future<Result<T>> f){return Take(f.get());}
std::vector<std::string> Split(std::string s){std::vector<std::string> out;std::size_t begin=0;
  for(;;){auto end=s.find(',',begin);out.push_back(s.substr(begin,end-begin));if(end==s.npos)return out;begin=end+1;}}
double Ms(Clock::time_point begin){return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
double Percentile(std::vector<double> v,double q){if(v.empty())return 0;std::sort(v.begin(),v.end());return v[std::size_t((v.size()-1)*q)];}
}
int main(int argc,char** argv){
  try {
    if(argc!=11)throw std::runtime_error("usage: eps_capacity_probe SHARDS AGENTS RUN ROWS DIM KEYS BAG LEARNER_LIST SECONDS REPEATS");
    const auto endpoints=Split(argv[1]),agent_endpoints=Split(argv[2]);const std::string run=argv[3];
    const std::uint64_t rows=std::stoull(argv[4]);const std::uint32_t dim=std::stoul(argv[5]),keys=std::stoul(argv[6]),bag=std::stoul(argv[7]);
    const char* gather_mode=std::getenv("EPS_COALESCE_GATHERS");
    if(gather_mode && std::string(gather_mode)!="0" && std::string(gather_mode)!="1")throw std::runtime_error("EPS_COALESCE_GATHERS must be 0 or 1");
    const bool coalesce=gather_mode && std::string(gather_mode)=="1";
    const char* pattern_env=std::getenv("EPS_CAPACITY_ACCESS_PATTERN");
    const std::string access_pattern=pattern_env?pattern_env:"resident_mixture";
    if(access_pattern!="resident_mixture" && access_pattern!="hot_shard")
      throw std::runtime_error("EPS_CAPACITY_ACCESS_PATTERN must be resident_mixture or hot_shard");
    const auto levels=Split(argv[8]);const double duration=std::stod(argv[9]);const int repeats=std::stoi(argv[10]);
    if(rows<1024 || rows>1048576 || !dim || dim>512 || !keys || keys>8192 || !bag || keys%bag || duration<=0 || duration>60 || repeats<1 || repeats>10)
      throw std::runtime_error("invalid capacity workload dimensions");
    std::vector<std::shared_ptr<ParameterService>> shards;
    std::vector<std::shared_ptr<BrpcParameterService>> admin;
    for(const auto& endpoint:endpoints){auto s=std::make_shared<BrpcParameterService>(endpoint,30000);shards.push_back(s);admin.push_back(s);}
    std::map<AgentId,std::shared_ptr<CacheBackend>> agents;
    for(AgentId id=0;id<agent_endpoints.size();++id)
      agents[id]=std::make_shared<BrpcCacheBackend>(agent_endpoints[id],run,0,id,30000);
    auto admission_stats=[&](bool drained){
      nlohmann::json records=nlohmann::json::array();
      for(const auto& [id,backend]:agents){
        try{
          auto agent=std::dynamic_pointer_cast<BrpcCacheBackend>(backend);
          auto deadline=Clock::now()+std::chrono::seconds(3);
          auto s=Take(agent->AdmissionStats());
          while(drained && (s.active_operations || s.queued_operations) && Clock::now()<deadline){
            std::this_thread::sleep_for(std::chrono::milliseconds(1));s=Take(agent->AdmissionStats());
          }
          nlohmann::json out{{"agent",id}};
#define EPS_STAT(field) out[#field]=s.field;
          EPS_STAT(active_operations) EPS_STAT(queued_operations) EPS_STAT(pending_request_bytes)
          EPS_STAT(peak_active_operations) EPS_STAT(peak_pending_operations) EPS_STAT(peak_request_bytes)
          EPS_STAT(accepted_requests) EPS_STAT(completed_requests) EPS_STAT(rejected_requests)
          EPS_STAT(queue_wait_nanoseconds) EPS_STAT(dispatch_failures)
          EPS_STAT(active_limit) EPS_STAT(pending_limit) EPS_STAT(request_bytes_limit)
#undef EPS_STAT
          records.push_back(std::move(out));
        }catch(const std::exception& e){records.push_back({{"agent",id},{"error",e.what()}});}
      }
      return records;
    };
    auto resolve=MakeCacheResolver(agents);
    DistributedClient setup(shards);setup.SetCacheResolver(resolve,coalesce);
    Take(setup.CreateTable({1,dim,std::uint32_t(shards.size()),1,{0.001f,0.9f,0.999f,1e-8f,0.9f,0.999f},-0.01f,0.01f,42}));
    std::uint64_t sequence=0;
    for(std::uint64_t begin=0;begin<rows;begin+=1024){
      FeatureInput f{1,0,Pooling::sequence,{},{0},{}};
      for(auto key=begin;key<std::min(rows,begin+1024);++key)f.keys.push_back(key+1);
      f.offsets.push_back(f.keys.size());f.valid_lengths.push_back(f.keys.size());
      auto batch=Take(setup.Lookup({{run,999,sequence++,Phase::lookup},1,true,{std::move(f)}}));Take(setup.Release(batch));
    }
    const auto cached=rows/2;
    // Capture independent CPU-owned weights before migration. The later read
    // check spans resident/cold rows, repeats, empty bags and sequence positions.
    std::vector<Key> reference_keys;
    for(Key i=1;i<=32;++i)for(Key key:{i,cached+i,rows-32+i})reference_keys.push_back(key);
    std::map<Key,std::vector<float>> reference_weights;
    for(std::size_t home=0;home<shards.size();++home) {
      std::vector<RowKey> sample;
      for(auto key:reference_keys)if(key%shards.size()==home)sample.push_back({1,key});
      if(sample.empty())continue;
      auto states=Take(admin[home]->ReadState(Take(admin[home]->Locations(std::move(sample)))));
      for(auto& state:states)reference_weights.emplace(state.row.key,std::move(state.weights));
    }
    for(std::size_t home=0;home<shards.size();++home)for(AgentId agent=0;agent<agents.size();++agent){
      std::vector<RowKey> move;
      auto flush=[&]{if(move.empty())return;Take(admin[home]->MoveRows({run,std::move(move),agent,Placement::gpu,{16ULL<<20,16ULL<<20,1,0}}));move.clear();};
      for(Key key=1;key<=cached;++key)if(key%shards.size()==home && (key/shards.size())%agents.size()==agent){
        move.push_back({1,key});if(move.size()==1024)flush();}
      flush();
    }
    std::vector<Key> check_keys;
    for(std::size_t i=0;i<128;++i)check_keys.push_back(reference_keys[i%reference_keys.size()]);
    std::vector<FeatureInput> check_features{
      {1,701,Pooling::mean,check_keys,{0,32,32,128},{32,0,96}},
      {1,702,Pooling::sequence,check_keys,{0,0,64,128},{0,64,64}},
      {1,703,Pooling::sum,check_keys,{0,64,128},{64,64}}};
    auto checked_batch=Take(setup.Lookup({{run,999,sequence++,Phase::lookup},1,true,check_features}));
    std::uint64_t preflight_values=0;double preflight_error=0;
    for(std::size_t fi=0;fi<check_features.size();++fi) {
      const auto& input=check_features[fi];const auto& output=checked_batch.features().at(fi);
      std::vector<float> expected_values(output.values.size(),0);
      for(std::size_t b=0;b<input.valid_lengths.size();++b) {
        for(auto i=input.offsets[b];i<input.offsets[b+1];++i) {
          const auto row=input.pooling==Pooling::sequence?i:b;
          for(std::size_t d=0;d<dim;++d)expected_values[row*dim+d]+=reference_weights.at(input.keys[i])[d];
        }
        if(input.pooling==Pooling::mean && input.valid_lengths[b])
          for(std::size_t d=0;d<dim;++d)expected_values[b*dim+d]/=input.valid_lengths[b];
      }
      for(std::size_t i=0;i<expected_values.size();++i) {
        const auto error=std::abs(double(output.values[i])-expected_values[i]);
        if(!std::isfinite(output.values[i]) || error>5e-6)
          throw std::runtime_error("preflight distributed feature differs from captured CPU weights");
        preflight_error=std::max(preflight_error,error);++preflight_values;
      }
    }
    Take(setup.Release(checked_batch));
    std::vector<std::atomic<std::uint64_t>> expected(rows+1);
    constexpr int max_learners=2048;
    std::vector<std::uint64_t> next_sequences(max_learners,0);
    for(int warm=0;warm<3;++warm){
      FeatureInput f{1,0,Pooling::mean,{},{0},{}};
      for(std::uint32_t i=0;i<keys;++i){f.keys.push_back(1+i%rows);
        if((i+1)%bag==0){f.offsets.push_back(i+1);f.valid_lengths.push_back(bag);}}
      std::set<Key> unique(f.keys.begin(),f.keys.end());
      auto batch=Take(setup.Lookup({{run,999,sequence++,Phase::lookup},1,false,{std::move(f)}}));
      auto gradients=batch.features();for(auto& g:gradients)std::fill(g.values.begin(),g.values.end(),0.01f);
      Take(setup.Push(batch,std::move(gradients)));for(auto key:unique)++expected[key];
    }
    std::uint32_t group=0;
    for(const auto& level:levels)for(int repeat=0;repeat<repeats;++repeat){
      std::vector<RuntimeStats> stats_before;
      const int learners=std::stoi(level);if(learners<1 || learners>max_learners)throw std::runtime_error("invalid Learner count");
      std::latch ready(learners),start(1);Clock::time_point deadline;
      std::mutex mutex;std::vector<double> latencies,lookups,pushes;std::uint64_t completed=0,failures=0;std::string error;
      std::vector<std::thread> threads;
      try{for(int learner=0;learner<learners;++learner)threads.emplace_back([&,learner]{
        DistributedClient client(shards);client.SetCacheResolver(resolve,coalesce);
        try{Take(client.CreateTable({1,dim,std::uint32_t(shards.size()),1,{0.001f,0.9f,0.999f,1e-8f,0.9f,0.999f},-0.01f,0.01f,42}));}
        catch(const std::exception& e){std::lock_guard lock(mutex);error=e.what();++failures;ready.count_down();return;}
        ready.count_down();start.wait();
        std::mt19937_64 random(42+learner+group*1000);std::uint64_t seq=next_sequences[learner];
        std::vector<double> local_latency,local_lookup,local_push;
        try{
          while(Clock::now()<deadline){
            FeatureInput f{1,0,Pooling::mean,{},{0},{}};f.keys.reserve(keys);
            for(std::uint32_t i=0;i<keys;++i){
              const auto bound=random()%100<95?cached:rows;
              const Key key=access_pattern=="hot_shard"
                ? (1+random()%(bound/shards.size()))*shards.size()
                : 1+random()%bound;
              f.keys.push_back(key);
              if((i+1)%bag==0){f.offsets.push_back(i+1);f.valid_lengths.push_back(bag);}}
            std::set<Key> unique(f.keys.begin(),f.keys.end());
            const auto begin=Clock::now();
            auto batch=Take(client.Lookup({{run,std::uint32_t(learner),seq++,Phase::lookup},1,false,{std::move(f)}}));
            const auto lookup_ms=Ms(begin);auto gradients=batch.features();
            for(auto& g:gradients)for(auto& value:g.values){if(!std::isfinite(value))throw std::runtime_error("non-finite feature");value=0.01f;}
            const auto push_begin=Clock::now();Take(client.Push(batch,std::move(gradients)));
            local_latency.push_back(Ms(begin));local_lookup.push_back(lookup_ms);local_push.push_back(Ms(push_begin));
            for(auto key:unique)++expected[key];
          }
        }catch(const std::exception& e){std::lock_guard lock(mutex);error=e.what();++failures;}
        next_sequences[learner]=seq;
        std::lock_guard lock(mutex);completed+=local_latency.size();
        latencies.insert(latencies.end(),local_latency.begin(),local_latency.end());
        lookups.insert(lookups.end(),local_lookup.begin(),local_lookup.end());pushes.insert(pushes.end(),local_push.begin(),local_push.end());
      });}catch(...){
        // A load-generator thread limit must not destroy joinable threads.
        // The default deadline prevents mutation when startup is incomplete.
        start.count_down();for(auto& thread:threads)thread.join();throw;
      }
      ready.wait();
      for(auto& shard:admin)stats_before.push_back(Take(shard->Stats()));
      const auto begin=Clock::now();deadline=begin+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(duration));start.count_down();
      for(auto& thread:threads)thread.join();const auto elapsed=Ms(begin)/1000;
      if(failures){
        // Preserve the failed point without presenting partially acknowledged
        // cross-shard updates as a verified throughput or optimizer trajectory.
        nlohmann::json diagnosis{{"test","distributed_sparse_training_overload"},
          {"gpu_agents",agents.size()},{"ps_shards",shards.size()},
          {"rows",rows},{"dimension",dim},{"keys_per_step",keys},
          {"learners",learners},{"repeat",repeat},{"errors",failures},
          {"acknowledged_steps_before_failure",completed},{"elapsed_seconds",elapsed},
          {"error",error},{"optimizer_trajectory_verified",false}};
        auto sample_stats=[&]{
          nlohmann::json samples=nlohmann::json::array();
          for(std::size_t home=0;home<admin.size();++home){
            try{auto s=Take(admin[home]->Stats());
              samples.push_back({{"shard",home},{"responsive",true},
                {"live_tickets",s.live_tickets},{"active_operations",s.active_operations},
                {"waiting_operations",s.waiting_operations},
                {"queued_payload_bytes",s.queued_payload_bytes},
                {"updates",s.updates},{"rejected_operations",s.rejected_operations}});
            }catch(const std::exception& e){samples.push_back({{"shard",home},
              {"responsive",false},{"error",e.what()}});}
          }
          return samples;
        };
        diagnosis["post_failure_shards"]=sample_stats();
        try{
          DistributedClient health(shards);health.SetCacheResolver(resolve,coalesce);
          Take(health.CreateTable({1,dim,std::uint32_t(shards.size()),1,
            {0.001f,0.9f,0.999f,1e-8f,0.9f,0.999f},-0.01f,0.01f,42}));
          FeatureInput f{1,991,Pooling::sum,{},{0},{}};
          for(Key key=1;key<=16;++key)f.keys.push_back(key);
          f.offsets.push_back(f.keys.size());f.valid_lengths.push_back(f.keys.size());
          auto batch=Take(health.Lookup({{run,1001,0,Phase::lookup},1,true,{std::move(f)}}));
          bool finite=true;
          for(const auto& output:batch.features())for(auto value:output.values)finite&=std::isfinite(value);
          Take(health.Release(batch));
          diagnosis["fresh_read_only_probe"]={{"completed",true},{"finite",finite}};
        }catch(const std::exception& e){diagnosis["fresh_read_only_probe"]=
          {{"completed",false},{"error",e.what()}};}
        std::uint64_t checked=0;bool finite=true;
        try{
          for(Key key=1;key<=16;++key){
            auto home=key%shards.size();
            auto state=Take(admin[home]->ReadState(Take(admin[home]->Locations({{1,key}})))).at(0);
            for(const auto* values:{&state.weights,&state.first_moment,&state.second_moment})
              for(auto value:*values)finite&=std::isfinite(value);
            ++checked;
          }
          diagnosis["sampled_state"]={{"completed",true},{"checked_rows",checked},{"finite",finite}};
        }catch(const std::exception& e){diagnosis["sampled_state"]=
          {{"completed",false},{"checked_rows",checked},
           {"finite",checked?nlohmann::json(finite):nlohmann::json(nullptr)},{"error",e.what()}};}
        diagnosis["after_health_probe_shards"]=sample_stats();
        diagnosis["agent_admission"]=admission_stats(false);
        std::cout<<diagnosis.dump()<<std::endl;
        throw std::runtime_error("capacity run failed: "+error);
      }
      std::map<std::string,std::uint64_t> phases;
      std::uint64_t busy_ns=0,wait_ns=0,operations=0,admission_failures=0;
      std::uint64_t dependency_ns=0,dependency_rebuilds=0,dependency_cache_hits=0;
      std::uint64_t expanded_bytes=0,transferred_bytes=0,compact_commands=0;
      for(std::size_t i=0;i<admin.size();++i){auto s=Take(admin[i]->Stats());
        phases["lookup_prepare_nanoseconds"]+=s.lookup_prepare_nanoseconds-stats_before[i].lookup_prepare_nanoseconds;
        phases["gradient_prepare_nanoseconds"]+=s.gradient_prepare_nanoseconds-stats_before[i].gradient_prepare_nanoseconds;
        phases["gpu_packet_prepare_nanoseconds"]+=s.gpu_packet_prepare_nanoseconds-stats_before[i].gpu_packet_prepare_nanoseconds;
        phases["update_dispatch_nanoseconds"]+=s.update_dispatch_nanoseconds-stats_before[i].update_dispatch_nanoseconds;
        phases["cpu_optimizer_nanoseconds"]+=s.cpu_optimizer_nanoseconds-stats_before[i].cpu_optimizer_nanoseconds;
        phases["agent_wait_nanoseconds"]+=s.agent_wait_nanoseconds-stats_before[i].agent_wait_nanoseconds;
        phases["migration_nanoseconds"]+=s.migration_nanoseconds-stats_before[i].migration_nanoseconds;
        phases["cache_maintenance_nanoseconds"]+=s.cache_maintenance_nanoseconds-stats_before[i].cache_maintenance_nanoseconds;
        busy_ns+=s.executor_busy_nanoseconds-stats_before[i].executor_busy_nanoseconds;
        wait_ns+=s.queue_wait_nanoseconds-stats_before[i].queue_wait_nanoseconds;
        operations+=s.executed_operations-stats_before[i].executed_operations;
        admission_failures+=s.rejected_operations-stats_before[i].rejected_operations;
        dependency_ns+=s.dependency_prepare_nanoseconds-stats_before[i].dependency_prepare_nanoseconds;
        dependency_rebuilds+=s.dependency_rebuilds-stats_before[i].dependency_rebuilds;
        dependency_cache_hits+=s.dependency_cache_hits-stats_before[i].dependency_cache_hits;
        expanded_bytes+=s.expanded_gradient_bytes-stats_before[i].expanded_gradient_bytes;
        transferred_bytes+=s.transferred_gradient_bytes-stats_before[i].transferred_gradient_bytes;
        compact_commands+=s.compact_update_commands-stats_before[i].compact_update_commands;
        if(s.live_tickets)throw std::runtime_error("load left an outstanding training ticket");}
      std::uint64_t checked=0;
      for(std::uint64_t sample=0;sample<32;++sample){
        // Offset the sample stride so power-of-two shard counts are covered,
        // rather than repeatedly selecting only one key-modulo partition.
        const Key key=1+(sample*(rows/32)+sample)%rows;
        const auto home=key%shards.size();auto loc=Take(admin[home]->Locations({{1,key}}));
        auto state=Take(admin[home]->ReadState(loc)).at(0);
        if(state.row_updates!=expected[key])throw std::runtime_error("accepted training update count mismatch");
        for(auto value:state.weights)if(!std::isfinite(value))throw std::runtime_error("non-finite trained parameter");++checked;
      }
      nlohmann::json result{{"test","distributed_sparse_training_capacity"},{"gpu_agents",agents.size()},{"ps_shards",shards.size()},
        {"rows",rows},{"dimension",dim},{"cached_rows",cached},{"keys_per_step",keys},{"bag_length",bag},{"learners",learners},{"repeat",repeat},
        {"completed_steps",completed},{"errors",failures},{"elapsed_seconds",elapsed},{"steps_per_second",completed/elapsed},
        {"id_occurrences_per_second",completed*keys/elapsed},{"step_p50_ms",Percentile(latencies,0.5)},{"step_p95_ms",Percentile(latencies,0.95)},
        {"step_p99_ms",Percentile(latencies,0.99)},{"lookup_p99_ms",Percentile(lookups,0.99)},{"push_p99_ms",Percentile(pushes,0.99)},
        {"checked_parameter_counters",checked}};
      result["phase_nanoseconds"]=phases;
      result["sequencer_busy_fraction"]=busy_ns/1e9/(elapsed*admin.size());
      result["executor_busy_scope"]="actor execution and dependency planning; suspended external waits excluded";
      result["queue_wait_per_operation_ms"]=operations?wait_ns/1e6/operations:0;
      result["rejected_admissions"]=admission_failures;
      result["dependency_prepare_nanoseconds"]=dependency_ns;
      result["dependency_rebuilds"]=dependency_rebuilds;
      result["dependency_cache_hits"]=dependency_cache_hits;
      result["expanded_gradient_bytes"]=expanded_bytes;result["transferred_gradient_bytes"]=transferred_bytes;
      result["compact_update_commands"]=compact_commands;
      result["coalesced_gathers"]=coalesce;
      result["access_pattern"]=access_pattern;
      result["preflight_values_checked"]=preflight_values;
      result["preflight_max_absolute_error"]=preflight_error;
      result["agent_admission"]=admission_stats(true);
      result["agent_admission_scope"]="setup, preflight, warmup and all trials in this probe";
      result["start_monotonic_seconds"]=std::chrono::duration<double>(begin.time_since_epoch()).count();
      result["end_monotonic_seconds"]=std::chrono::duration<double>(begin.time_since_epoch()).count()+elapsed;
      std::cout<<result.dump()<<std::endl;++group;
    }
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL capacity probe: "<<e.what()<<'\n';return 1;}
}
