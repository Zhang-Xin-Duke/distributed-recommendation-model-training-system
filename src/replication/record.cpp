#include "eps/replica_log.hpp"
#include "common/error.hpp"
#include <bit>
#include <cmath>
#include <set>

namespace eps {
namespace {
using detail::Require;
struct Binary {
  std::string bytes;
  std::size_t position = 0;
  std::uint64_t limit;
  void U(std::uint64_t n) {
    Require(bytes.size() <= limit && limit-bytes.size() >= 8,"replica record exceeds limit",Error::Code::capacity_exceeded);
    for(unsigned i=0;i<8;++i)bytes.push_back(static_cast<char>(n>>(8*i)));
  }
  std::uint64_t U() {
    Require(position <= bytes.size() && bytes.size()-position>=8,"truncated replica record");
    std::uint64_t n=0;for(unsigned i=0;i<8;++i)n|=std::uint64_t(static_cast<unsigned char>(bytes[position++]))<<(8*i);
    return n;
  }
  void F(float value) {
    Require(std::isfinite(value) && bytes.size()<=limit && limit-bytes.size()>=4,"invalid replica float/size");
    const auto n=std::bit_cast<std::uint32_t>(value);for(unsigned i=0;i<4;++i)bytes.push_back(char(n>>(8*i)));
  }
  float F() {
    Require(position<=bytes.size() && bytes.size()-position>=4,"truncated replica float");
    std::uint32_t n=0;for(unsigned i=0;i<4;++i)n|=std::uint32_t(static_cast<unsigned char>(bytes[position++]))<<(8*i);
    const auto f=std::bit_cast<float>(n);Require(std::isfinite(f),"non-finite replica state");return f;
  }
};
std::uint64_t Hash(const std::string& s,std::size_t n) {
  std::uint64_t hash=14695981039346656037ULL;
  for(std::size_t i=0;i<n;++i)hash=(hash^static_cast<unsigned char>(s[i]))*1099511628211ULL;
  return hash;
}
constexpr std::uint64_t magic=0x315445535045ULL;
void Validate(const ReplicatedUpdate& u) {
  Require(!u.run.empty() && u.run.size()<=256 && u.shards && u.shards<=1024 && u.shard<u.shards &&
    u.epoch && u.round!=UINT64_MAX && !u.contexts.empty() && u.contexts.size()<=256 && u.requests.size()<=1024,
    "invalid replica image identity");
  std::set<TableId> tables;
  for(const auto& c:u.contexts) Require(tables.insert(c.table).second && c.shard==u.shard &&
    c.optimizer_sequence==u.round+1 && std::isfinite(c.learning_rate) && c.learning_rate>0 &&
    std::isfinite(c.beta1_power) && c.beta1_power>=0 && c.beta1_power<1 &&
    std::isfinite(c.beta2_power) && c.beta2_power>=0 && c.beta2_power<1,"invalid replica optimizer clocks");
  std::set<std::pair<TableId,Key>> rows;
  for(const auto& r:u.rows) {
    Require(tables.contains(r.row.table) && r.row.key%u.shards==u.shard &&
      rows.emplace(r.row.table,r.row.key).second && !r.weights.empty() && r.weights.size()<=65536 &&
      r.weights.size()==r.first_moment.size() && r.weights.size()==r.second_moment.size() && r.row_updates>0,
      "invalid replica row image");
    for(const auto* a:{&r.weights,&r.first_moment,&r.second_moment})for(auto f:*a)Require(std::isfinite(f),"non-finite replica row");
    for(auto f:r.second_moment)Require(f>=0,"negative replicated second moment");
  }
  std::set<std::uint32_t> learners;
  for(const auto& p:u.requests)Require(p.learner<1024 && learners.insert(p.learner).second,"invalid replica request frontier");
}
}
std::string EncodeReplicatedUpdate(const ReplicatedUpdate& u,std::uint64_t limit) {
  Validate(u);Binary b{{},0,limit};b.U(magic);b.U(u.run.size());
  Require(u.run.size()<=limit-b.bytes.size(),"replica identity exceeds record");b.bytes+=u.run;
  b.U(u.shard);b.U(u.shards);b.U(u.epoch);b.U(u.round);b.U(u.contexts.size());
  for(const auto& c:u.contexts) {b.U(c.table);b.U(c.shard);b.U(c.optimizer_sequence);b.F(c.learning_rate);b.F(c.beta1_power);b.F(c.beta2_power);}
  b.U(u.rows.size());
  for(const auto& r:u.rows) {b.U(r.row.table);b.U(r.row.key);b.U(r.row_updates);b.U(r.weights.size());
    for(const auto* a:{&r.weights,&r.first_moment,&r.second_moment})for(auto f:*a)b.F(f);}
  b.U(u.requests.size());for(const auto& p:u.requests){b.U(p.learner);b.U(p.next);}
  const auto hash=Hash(b.bytes,b.bytes.size());b.U(hash);return std::move(b.bytes);
}
ReplicatedUpdate DecodeReplicatedUpdate(const std::string& data,std::uint64_t limit) {
  Require(data.size()<=limit && data.size()>=64,"invalid replica record size");
  Binary b{data,0,limit};Require(b.U()==magic,"unsupported replica record format");
  ReplicatedUpdate u;const auto chars=b.U();Require(chars<=256,"replica run exceeds limit");
  Require(chars<=data.size()-b.position,"truncated replica run");u.run=data.substr(b.position,chars);b.position+=chars;
  const auto shard=b.U(),count=b.U();Require(shard<=UINT32_MAX && count<=1024,"invalid replica partition");
  u.shard=shard;u.shards=count;u.epoch=b.U();u.round=b.U();const auto contexts=b.U();Require(contexts<=256,"replica table count exceeds limit");
  for(std::uint64_t i=0;i<contexts;++i) {UpdateContext c{};const auto table=b.U(),home=b.U();
    Require(table<=UINT32_MAX && home<=UINT32_MAX,"invalid replica table id");c.table=table;c.shard=home;
    c.optimizer_sequence=b.U();c.learning_rate=b.F();c.beta1_power=b.F();c.beta2_power=b.F();u.contexts.push_back(c);}
  const auto rows=b.U();Require(rows<=data.size()/32,"replica row count exceeds record");
  for(std::uint64_t i=0;i<rows;++i) {RowState r;const auto table=b.U();Require(table<=UINT32_MAX,"invalid replica row table");
    r.row={TableId(table),b.U()};r.row_updates=b.U();const auto dim=b.U();
    Require(dim>0 && dim<=65536 && dim<=(data.size()-b.position)/12,"replica row dimension exceeds record");
    for(auto* a:{&r.weights,&r.first_moment,&r.second_moment}){a->reserve(dim);for(std::uint64_t d=0;d<dim;++d)a->push_back(b.F());}
    u.rows.push_back(std::move(r));}
  const auto requests=b.U();Require(requests<=1024,"replica learner count exceeds limit");
  for(std::uint64_t i=0;i<requests;++i){const auto learner=b.U(),next=b.U();Require(learner<1024,"invalid replica learner id");u.requests.push_back({std::uint32_t(learner),next});}
  const auto checksum=Hash(data,b.position);Require(b.U()==checksum && b.position==data.size(),"replica record checksum/trailer mismatch");
  Validate(u);return u;
}
}  // namespace eps
