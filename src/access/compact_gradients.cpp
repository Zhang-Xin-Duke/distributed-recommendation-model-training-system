#include "eps/compact_gradients.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include "common/error.hpp"
#include "access/batch_access.hpp"

namespace eps {
void ValidateCompactGradients(const CompactGradients& p){
  detail::Require(p.dimension>0 && p.dimension<=4096 && p.rows.size()<=(1ULL<<20),"invalid compact gradient dimensions");
  detail::Require(p.source_values.size()%p.dimension==0 && p.row_offsets.size()==p.rows.size()+1 &&
    p.source_rows.size()==p.scales.size() && !p.row_offsets.empty() && p.row_offsets.front()==0 &&
    p.row_offsets.back()==p.source_rows.size() && std::is_sorted(p.row_offsets.begin(),p.row_offsets.end()),"invalid compact gradient offsets");
  detail::Require(CompactGradientBytes(p)<=(16ULL<<20),"compact gradient exceeds byte limit",Error::Code::capacity_exceeded);
  std::set<std::pair<TableId,Key>> unique;
  for(auto row:p.rows)detail::Require(unique.emplace(row.table,row.key).second,"duplicate compact parameter row");
  const auto source_count=p.source_values.size()/p.dimension;
  for(auto source:p.source_rows)detail::Require(source<source_count,"compact gradient reference is out of bounds");
  for(auto value:p.source_values)detail::Require(std::isfinite(value),"non-finite compact gradient value");
  for(auto scale:p.scales)detail::Require(std::isfinite(scale) && scale>=0,"invalid compact gradient scale");
}
std::uint64_t CompactGradientBytes(const CompactGradients& p){
  return p.rows.size()*sizeof(RowKey)+(p.source_values.size()+p.scales.size())*sizeof(float)+
    (p.row_offsets.size()+p.source_rows.size())*sizeof(std::uint64_t);
}
RowGradients ExpandCompactGradients(const CompactGradients& p){
  ValidateCompactGradients(p);
  detail::Require(p.rows.size()<=(16ULL<<20)/(p.dimension*4ULL),"expanded gradient exceeds byte limit",Error::Code::capacity_exceeded);
  RowGradients out{p.rows,std::vector<float>(p.rows.size()*p.dimension,0)};
  for(std::size_t row=0;row<p.rows.size();++row)for(std::uint32_t d=0;d<p.dimension;++d){
    float sum=0;
    for(auto ref=p.row_offsets[row];ref<p.row_offsets[row+1];++ref)
      sum+=p.source_values[p.source_rows[ref]*p.dimension+d]*p.scales[ref];
    detail::Require(std::isfinite(sum),"compact gradient reduction overflow");
    out.values[row*p.dimension+d]=sum;
  }
  return out;
}
}  // namespace eps

namespace eps::detail {
CompactGradients BuildCompactGradients(const std::vector<PreparedFeature>& features,
    const std::vector<TensorBlock>& gradients,const std::vector<RowKey>& rows,std::uint32_t dimension){
  Require(!rows.empty(),"empty compact gradient target");
  CompactGradients out{rows,dimension,{},{0},{},{}};
  std::map<Key,std::size_t> target;
  for(std::size_t i=0;i<rows.size();++i){Require(rows[i].table==rows.front().table,"mixed compact gradient tables");target.emplace(rows[i].key,i);}
  std::map<std::uint32_t,const TensorBlock*> by_id;
  for(const auto& g:gradients)by_id.emplace(g.feature,&g);
  struct Reference{std::uint64_t source;float scale;};
  std::vector<std::vector<Reference>> references(rows.size());
  for(const auto& f:features)if(f.input.table==rows.front().table){
    const auto& gradient=*by_id.at(f.input.feature);
    std::map<std::uint64_t,std::uint64_t> source_index;
    for(std::size_t bag=0;bag<f.input.valid_lengths.size();++bag){
      const float scale=f.input.pooling==Pooling::mean && f.input.valid_lengths[bag]?1.0f/f.input.valid_lengths[bag]:1.0f;
      for(auto i=f.input.offsets[bag];i<f.input.offsets[bag+1];++i){
        auto selected=target.find(f.input.keys[i]);if(selected==target.end())continue;
        const auto source=f.input.pooling==Pooling::sequence?i:bag;
        auto [it,added]=source_index.try_emplace(source,out.source_values.size()/dimension);
        if(added)out.source_values.insert(out.source_values.end(),gradient.values.begin()+source*dimension,gradient.values.begin()+(source+1)*dimension);
        references[selected->second].push_back({it->second,scale});
      }
    }
  }
  for(const auto& refs:references){
    for(auto ref:refs){out.source_rows.push_back(ref.source);out.scales.push_back(ref.scale);}
    out.row_offsets.push_back(out.source_rows.size());
  }
  ValidateCompactGradients(out);return out;
}
}  // namespace eps::detail
