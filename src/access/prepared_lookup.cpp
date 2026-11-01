#include "eps/prepared_lookup.hpp"

#include <algorithm>
#include <set>
#include "common/error.hpp"

namespace eps {
namespace {
void Add(std::uint64_t& total,std::uint64_t count,std::uint64_t size,std::uint64_t limit) {
  detail::Require(size && count<=limit/size && total<=limit-count*size,
    "prepared lookup byte budget exceeded",Error::Code::capacity_exceeded);
  total+=count*size;
}
std::uint64_t Estimate(const std::vector<FeatureInput>& features,std::uint32_t shards,PreparationLimits limits) {
  detail::Require(shards>0 && shards<=1024 && limits.max_features>0 &&
    !features.empty() && features.size()<=limits.max_features,"invalid prepared lookup dimensions");
  std::set<std::uint32_t> ids;std::uint64_t input=0;
  Add(input,features.capacity(),sizeof(FeatureInput),limits.max_input_bytes);
  for(const auto& f:features) {
    detail::Require(ids.insert(f.feature).second,"duplicate prepared feature");
    detail::Require(f.pooling==Pooling::sum || f.pooling==Pooling::mean || f.pooling==Pooling::sequence,"invalid prepared pooling");
    detail::Require(!f.offsets.empty() && f.offsets.front()==0 && f.offsets.back()==f.keys.size() &&
      std::is_sorted(f.offsets.begin(),f.offsets.end()) && f.offsets.size()==f.valid_lengths.size()+1,"invalid prepared offsets");
    for(std::size_t b=0;b<f.valid_lengths.size();++b)
      detail::Require(f.valid_lengths[b]==f.offsets[b+1]-f.offsets[b],"prepared lookup needs original bag lengths");
    for(auto capacity:{f.keys.capacity(),f.offsets.capacity(),f.valid_lengths.capacity()})
      Add(input,capacity,sizeof(std::uint64_t),limits.max_input_bytes);
  }
  // Vector objects, original buffers, routed keys/positions and replicated bag
  // metadata. Allocation headers are bounded by feature/shard/container counts.
  std::uint64_t bytes=input;
  Add(bytes,1,sizeof(PreparedLookup),limits.max_prepared_bytes);
  Add(bytes,shards,sizeof(std::vector<FeatureInput>)+sizeof(std::vector<std::vector<std::uint64_t>>),limits.max_prepared_bytes);
  Add(bytes,std::uint64_t(shards)*features.size(),sizeof(FeatureInput)+sizeof(std::vector<std::uint64_t>),limits.max_prepared_bytes);
  for(const auto& f:features) {
    Add(bytes,f.keys.size(),2*sizeof(std::uint64_t),limits.max_prepared_bytes);
    Add(bytes,std::uint64_t(shards)*f.offsets.size(),sizeof(std::uint64_t),limits.max_prepared_bytes);
    Add(bytes,std::uint64_t(shards)*f.valid_lengths.size(),sizeof(std::uint64_t),limits.max_prepared_bytes);
  }
  return bytes;
}
}
Result<std::uint64_t> EstimatePreparedLookupBytes(const std::vector<FeatureInput>& f,std::uint32_t shards,PreparationLimits limits) {
  try{return Estimate(f,shards,limits);}catch(const detail::Failure& e){return e.error();}
  catch(const std::bad_alloc&){return Error{Error::Code::capacity_exceeded,"prepared lookup estimate allocation failed"};}
}
Result<PreparedLookup> PrepareLookupInputs(std::vector<FeatureInput> features,std::uint32_t shards,PreparationLimits limits) {
  try {
    PreparedLookup out;out.bytes_=Estimate(features,shards,limits);out.shard_count_=shards;
    out.original_=std::move(features);out.parts_.resize(shards);
    for(auto& part:out.parts_) {part.features.resize(out.original_.size());part.positions.resize(out.original_.size());}
    for(std::size_t fi=0;fi<out.original_.size();++fi) {
      const auto& f=out.original_[fi];std::vector<std::size_t> counts(shards,0);
      for(auto key:f.keys)++counts[key%shards];
      for(std::uint32_t s=0;s<shards;++s) {
        auto& dst=out.parts_[s].features[fi];dst.table=f.table;dst.feature=f.feature;dst.pooling=f.pooling;
        dst.keys.reserve(counts[s]);dst.offsets.reserve(f.offsets.size());dst.offsets.push_back(0);
        dst.valid_lengths=f.valid_lengths;out.parts_[s].positions[fi].reserve(counts[s]);
      }
      for(std::size_t b=0;b<f.valid_lengths.size();++b) {
        for(auto i=f.offsets[b];i<f.offsets[b+1];++i) {
          auto& part=out.parts_[f.keys[i]%shards];part.features[fi].keys.push_back(f.keys[i]);part.positions[fi].push_back(i);
        }
        for(auto& part:out.parts_)part.features[fi].offsets.push_back(part.features[fi].keys.size());
      }
    }
    std::uint64_t actual=sizeof(PreparedLookup);
    Add(actual,out.original_.capacity(),sizeof(FeatureInput),limits.max_prepared_bytes);
    for(const auto& f:out.original_)for(auto n:{f.keys.capacity(),f.offsets.capacity(),f.valid_lengths.capacity()})
      Add(actual,n,8,limits.max_prepared_bytes);
    Add(actual,out.parts_.capacity(),sizeof(PreparedLookup::Part),limits.max_prepared_bytes);
    for(const auto& part:out.parts_) {
      Add(actual,part.features.capacity(),sizeof(FeatureInput),limits.max_prepared_bytes);
      Add(actual,part.positions.capacity(),sizeof(std::vector<std::uint64_t>),limits.max_prepared_bytes);
      for(const auto& f:part.features)for(auto n:{f.keys.capacity(),f.offsets.capacity(),f.valid_lengths.capacity()})
        Add(actual,n,8,limits.max_prepared_bytes);
      for(const auto& positions:part.positions)Add(actual,positions.capacity(),8,limits.max_prepared_bytes);
    }
    out.bytes_=actual;
    return out;
  }catch(const detail::Failure& e){return e.error();}
  catch(const std::bad_alloc&){return Error{Error::Code::capacity_exceeded,"prepared lookup allocation failed"};}
}
}  // namespace eps
