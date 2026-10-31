#include "eps/brpc_client.hpp"
#include "eps/brpc_cache_backend.hpp"
#include "eps/cache_resolver.hpp"
#include "eps/distributed_client.hpp"
#include "eps/runtime.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>

using namespace eps;
using Clock=std::chrono::steady_clock;
namespace {
template<typename T>T Take(Result<T> r){if(auto e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<typename T>T Take(std::future<Result<T>> f){return Take(f.get());}
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
double Milliseconds(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
double Quantile(std::vector<double> v,double q){std::sort(v.begin(),v.end());return v.at(std::size_t((v.size()-1)*q));}
TableDefinition Table(std::uint32_t shards,std::uint32_t dimension){
  return {1,dimension,shards,1,{0.001f,0.9f,0.999f,1e-8f,0.9f,0.999f},-0.01f,0.01f,42};
}
struct Counters {
  std::uint64_t windows=0,admitted=0,evicted=0,bytes=0,batches=0,failures=0,pinned=0,budget=0,cached=0,tracked=0,updates=0;
};
Counters Metrics(const std::vector<std::shared_ptr<BrpcParameterService>>& admin){
  Counters out;
  for(const auto& shard:admin){const auto s=Take(shard->Stats());Check(!s.live_tickets,"outstanding trace ticket");
    out.windows+=s.policy_windows;out.admitted+=s.policy_admitted_rows;out.evicted+=s.policy_evicted_rows;
    out.bytes+=s.policy_migration_bytes;out.batches+=s.policy_migration_batches;out.failures+=s.policy_failures;
    out.pinned+=s.policy_pinned_skips;out.budget+=s.policy_budget_skips;out.cached+=s.cached_rows;
    out.tracked+=s.policy_tracked_rows;out.updates+=s.updates;}
  return out;
}
void WarmAgent(const std::string& endpoint,const std::string& run,std::uint32_t dim){
  // A separate, valid home stream/table warms CUDA kernels without changing
  // the benchmark table or its optimizer. Its only row is evicted before timing.
  BrpcCacheBackend warm(endpoint,run,1023,0,30000);
  auto definition=Table(1024,dim);definition.id=999;Take(warm.Define(definition));
  RowState state{{999,1023},std::vector<float>(dim,0),std::vector<float>(dim,0),std::vector<float>(dim,0),0};
  MigrationPlan plan{"warmup",0,{{state.row,1023,Placement::loading,0,0,1}},Placement::gpu,dim*12ULL+8};
  auto rows=Take(warm.Load(plan,{state}));
  FeatureInput input{999,0,Pooling::mean,{1023,1023},{0,2},{2}};
  Take(warm.GatherHost({{0,AccessRoute::resident_gpu,0,rows,{0,0},{}}},{input}));
  Take(warm.ApplyHost({{run,0,0,Phase::push},1023,999,0,0.001f,0.9f,0.999f},
    {{state.row},std::vector<float>(dim,0.01f)}));
  plan.rows=rows;plan.destination=Placement::cpu;
  auto checked=Take(warm.ReadBack(plan));Check(checked.size()==1 && checked[0].row_updates==1,"warmup update failed");
  Take(warm.CommitEviction(plan));
}
}
int main(int argc,char** argv){
  try{
    Check(argc==2,"usage: eps_cache_policy_probe CONFIG.json");std::ifstream input(argv[1]);nlohmann::json j;input>>j;
    const auto mode=j.at("mode").get<std::string>(),run=j.at("run").get<std::string>();
    Check(mode=="cpu_only" || mode=="static" || mode=="frequency" || mode=="reuse_cost","invalid trace mode");
    const std::uint32_t rows=j.at("rows"),dim=j.at("dimension"),keys=j.at("keys"),bag=j.at("bag"),hot=j.at("hot_rows"),steps=j.at("phase_steps");
    const std::uint32_t slots=j.at("cache_rows"),reserve=j.at("reserve_slots");
    const bool calibration=j.value("calibration",false),coalesce=j.value("coalesce",false);
    Check(rows>=1024 && rows<=262144 && dim>0 && dim<=512 && keys>0 && keys<=8192 && keys%4==0 &&
      bag>0 && keys%bag==0 && hot>0 && steps>0 && steps<=512 && reserve<slots && hot<=slots-reserve &&
      std::uint64_t(hot)*2+std::uint64_t(steps)*(keys/4)<rows,"invalid trace dimensions");
    std::vector<std::shared_ptr<ParameterService>> services;
    std::vector<std::shared_ptr<BrpcParameterService>> admin;
    for(const auto& endpoint:j.at("endpoints")){
      auto s=std::make_shared<BrpcParameterService>(endpoint.get<std::string>(),30000);admin.push_back(s);services.push_back(s);
    }
    const auto agent_endpoint=j.at("agent").get<std::string>();
    auto reader=std::make_shared<BrpcCacheBackend>(agent_endpoint,run,0,0,30000);
    auto resolver=MakeCacheResolver({{0,reader}});
    std::uint64_t resident=0,transferred=0,gather_calls=0;
    DistributedClient actual(services);
    actual.SetCacheResolver([&](auto work,auto layouts){
      ++gather_calls;
      for(const auto& task:work){const auto count=std::count_if(task.occurrence_to_unique.begin(),task.occurrence_to_unique.end(),[](auto x){return x!=UINT64_MAX;});
        if(task.route==AccessRoute::resident_gpu)resident+=count;else transferred+=count;}
      return resolver(std::move(work),std::move(layouts));
    },coalesce);
    Take(actual.CreateTable(Table(services.size(),dim)));
    RuntimeOptions options;options.run_id="trace-reference";options.state_bytes=4ULL<<30;
    auto reference_service=std::make_shared<CpuParameterService>(options);DistributedClient reference({reference_service});
    Take(reference.CreateTable(Table(1,dim)));WarmAgent(agent_endpoint,run,dim);
    std::uint64_t setup_sequence=0,sequence=0;
    std::set<Key> touched;std::vector<std::uint64_t> expected_updates(rows+1,0);
    for(std::uint64_t begin=1;begin<=hot;begin+=1024){
      FeatureInput f{1,0,Pooling::sequence,{},{0},{}};
      for(auto key=begin;key<=std::min<std::uint64_t>(hot,begin+1023);++key){f.keys.push_back(key);touched.insert(key);}
      f.offsets.push_back(f.keys.size());f.valid_lengths.push_back(f.keys.size());
      auto a=Take(actual.Lookup({{run,999,setup_sequence,Phase::lookup},1,true,{f}}));Take(actual.Release(a));
      auto b=Take(reference.Lookup({{options.run_id,999,setup_sequence++,Phase::lookup},1,true,{f}}));Take(reference.Release(b));
    }
    if(mode!="cpu_only")for(std::size_t home=0;home<admin.size();++home){
      std::vector<RowKey> batch;
      const auto flush=[&]{if(!batch.empty()){Take(admin[home]->MoveRows({run,std::move(batch),0,Placement::gpu,{16ULL<<20,16ULL<<20,1,reserve}}));batch.clear();}};
      for(Key key=1;key<=hot;++key)if(key%admin.size()==home){batch.push_back({1,key});if(batch.size()==256)flush();}flush();
    }
    const auto initial=Metrics(admin);Check(initial.cached==(mode=="cpu_only"?0:hot),"initial cache differs from specification");
    std::vector<nlohmann::json> records;
    std::uint64_t fingerprint=14695981039346656037ULL;
    const std::vector<std::string> names=calibration?std::vector<std::string>{"stable"}:
      std::vector<std::string>{"stable","shift","scan_mixed"};
    for(std::size_t phase=0;phase<names.size();++phase){
      std::mt19937_64 random(42+phase*10007);std::vector<FeatureInput> trace;trace.reserve(steps);
      std::uint64_t initially_hot_unique=0,total_unique=0;
      for(std::uint32_t step=0;step<steps;++step){
        FeatureInput f{1,0,Pooling::mean,{},{0},{}};f.keys.reserve(keys);
        const auto cold_begin=std::uint64_t(hot)*2+std::uint64_t(steps)*(keys/4)+1;
        for(std::uint32_t k=0;k<keys;++k){
          Key key;
          if(phase==2 && k%4==0)key=std::uint64_t(hot)*2+1+step*(keys/4)+k/4;
          else if(phase<2 && random()%100>=95)key=cold_begin+random()%(rows-cold_begin+1);
          else key=1+(phase==1?hot:0)+random()%hot;
          f.keys.push_back(key);touched.insert(key);
          for(int byte=0;byte<8;++byte){fingerprint^=(key>>(byte*8))&255;fingerprint*=1099511628211ULL;}
          if((k+1)%bag==0){f.offsets.push_back(k+1);f.valid_lengths.push_back(bag);}
        }
        std::set<Key> unique(f.keys.begin(),f.keys.end());total_unique+=unique.size();
        for(auto key:unique){++expected_updates[key];if(key<=hot)++initially_hot_unique;}
        trace.push_back(std::move(f));
      }
      const auto values=std::uint64_t(keys/bag)*dim;
      std::vector<float> captured(std::uint64_t(steps)*values);
      std::vector<TensorBlock> gradients{{0,{keys/bag,dim},std::vector<float>(values,0.01f)}};
      std::vector<double> latency;latency.reserve(steps);
      const auto before=Metrics(admin);const auto resident_begin=resident,transfer_begin=transferred,gather_begin=gather_calls;
      const auto first_sequence=sequence;const auto begin=Clock::now();
      for(std::uint32_t step=0;step<steps;++step){
        const auto tick=Clock::now();auto b=Take(actual.Lookup({{run,0,sequence++,Phase::lookup},1,false,{trace[step]}}));
        Check(b.features().size()==1 && b.features()[0].values.size()==values,"trace feature shape changed");
        std::copy(b.features()[0].values.begin(),b.features()[0].values.end(),captured.begin()+step*values);
        Take(actual.Push(b,gradients));latency.push_back(Milliseconds(tick));
      }
      const auto elapsed=Milliseconds(begin)/1000;const auto after=Metrics(admin);
      Check(after.updates-before.updates==std::uint64_t(steps)*admin.size(),"trace update acknowledgement count mismatch");
      Check(after.cached<=slots-reserve,"cache consumed reserved capacity");
      Check(after.bytes<=after.windows*j.at("network_bytes_per_window").get<std::uint64_t>() &&
        after.batches<=after.windows*j.at("max_inflight_batches").get<std::uint64_t>(),"policy window bounds exceeded");
      double feature_error=0;
      for(std::uint32_t step=0;step<steps;++step){
        auto b=Take(reference.Lookup({{options.run_id,0,first_sequence+step,Phase::lookup},1,false,{trace[step]}}));
        for(std::uint64_t v=0;v<values;++v){const auto actual_value=captured[step*values+v];
          const auto error=std::abs(double(actual_value)-b.features()[0].values[v]);
          Check(std::isfinite(actual_value) && error<=5e-6,"trace feature differs from CPU replay");feature_error=std::max(feature_error,error);}
        Take(reference.Push(b,gradients));
      }
      const auto resident_count=resident-resident_begin,transfer_count=transferred-transfer_begin,total=std::uint64_t(steps)*keys;
      Check(resident_count+transfer_count<=total,"trace access accounting mismatch");
      nlohmann::json record{{"test","sparse_cache_policy_trace"},{"mode",mode},{"phase",names[phase]},
        {"calibration",calibration},{"rows",rows},{"dimension",dim},{"keys_per_step",keys},{"bag_length",bag},
        {"hot_rows",hot},{"cache_rows",slots},{"initial_cached_rows",initial.cached},{"ps_shards",admin.size()},{"gpu_agents",1},
        {"learners",1},{"completed_steps",steps},{"errors",0},{"elapsed_seconds",elapsed},{"steps_per_second",steps/elapsed},
        {"step_mean_ms",elapsed*1000/steps},{"step_p50_ms",Quantile(latency,.5)},{"step_p99_ms",Quantile(latency,.99)},
        {"input_id_occurrences",total},{"resident_id_occurrences",resident_count},{"transferred_id_occurrences",transfer_count},
        {"cpu_pooled_id_occurrences",total-resident_count-transfer_count},{"resident_hit_fraction",double(resident_count)/total},
        {"resolver_calls",gather_calls-gather_begin},{"cached_rows_end",after.cached},{"policy_windows",after.windows-before.windows},
        {"policy_admitted_rows",after.admitted-before.admitted},{"policy_evicted_rows",after.evicted-before.evicted},
        {"policy_migration_bytes",after.bytes-before.bytes},{"policy_migration_batches",after.batches-before.batches},
        {"policy_failures",after.failures-before.failures},{"policy_pinned_skips",after.pinned-before.pinned},
        {"policy_budget_skips",after.budget-before.budget},{"checked_feature_values",steps*values},{"feature_max_absolute_error",feature_error},
        {"mean_unique_rows_per_step",double(total_unique)/steps},{"mean_initial_hot_unique_per_step",double(initially_hot_unique)/steps},
        {"saved_microseconds_per_access",j.at("saved_microseconds_per_access")},{"coalesced_gathers",coalesce}};
      record["start_monotonic_seconds"]=std::chrono::duration<double>(begin.time_since_epoch()).count();
      record["end_monotonic_seconds"]=record["start_monotonic_seconds"].get<double>()+elapsed;
      records.push_back(std::move(record));std::cerr<<"PHASE "<<names[phase]<<" completed and replayed\n";
    }
    for(const auto& shard:admin)Take(shard->DrainAndFlush("trace-final"));
    Check(Take(reader->FreeSlots(1))==slots,"trace drain did not restore agent capacity");
    double state_error=0;std::uint64_t checked_rows=0,checked_values=0;
    for(std::size_t home=0;home<admin.size();++home){
      std::vector<RowKey> batch;
      const auto flush=[&]{
        if(batch.empty())return;
        auto states=Take(admin[home]->ReadState(Take(admin[home]->Locations(batch))));
        std::vector<LocatedRow> locations;for(auto key:batch)locations.push_back({key,0,Placement::cpu,0,0,0});
        auto expected=Take(reference_service->ReadState(std::move(locations)));
        Check(states.size()==expected.size(),"state replay count mismatch");
        for(std::size_t i=0;i<states.size();++i){
          const auto& a=states[i];const auto& b=expected[i];
          Check(a.row==b.row && a.row_updates==b.row_updates && a.row_updates==expected_updates.at(a.row.key),"trace row update counter mismatch");
          for(auto vectors:{std::pair(&a.weights,&b.weights),std::pair(&a.first_moment,&b.first_moment),std::pair(&a.second_moment,&b.second_moment)}){
            Check(vectors.first->size()==dim && vectors.second->size()==dim,"state shape mismatch");
            for(std::size_t d=0;d<dim;++d){const auto value=(*vectors.first)[d];const auto error=std::abs(double(value)-(*vectors.second)[d]);
              Check(std::isfinite(value) && error<=5e-6,"trace final state differs from CPU replay");state_error=std::max(state_error,error);++checked_values;}
          }++checked_rows;
        }batch.clear();
      };
      for(auto key:touched)if(key%admin.size()==home){batch.push_back({1,key});if(batch.size()==512)flush();}flush();
    }
    std::ostringstream identity;identity<<std::hex<<std::setfill('0')<<std::setw(16)<<fingerprint;
    for(auto& record:records){record["trace_fnv1a64"]=identity.str();record["checked_final_rows"]=checked_rows;
      record["checked_final_state_values"]=checked_values;record["state_max_absolute_error"]=state_error;
      record["final_free_slots"]=slots;std::cout<<record.dump()<<std::endl;}
    return 0;
  }catch(const std::exception& e){std::cerr<<"FAIL cache policy trace: "<<e.what()<<'\n';return 1;}
}
