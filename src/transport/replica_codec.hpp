#pragma once
#include "transport/codec.hpp"
#include "eps/replica_log.hpp"
namespace eps::transport {
inline void EncodeScope(const ReplicaLogOptions& o,v1::ReplicaScope* s) {
  s->set_run(o.run);s->set_shard(o.shard);s->set_shards(o.shards);s->set_epoch(o.epoch);s->set_binding(o.binding);
}
inline void CheckScope(const ReplicaLogOptions& o,const v1::ReplicaScope& s) {
  detail::Require(s.run()==o.run && s.shard()==o.shard && s.shards()==o.shards && s.epoch()==o.epoch && s.binding()==o.binding,
                  "replica run/partition/model binding differs",Error::Code::stale_generation);
}
inline ReplicaLease DecodeLease(const v1::ReplicaLease& l){return {l.term(),l.primary()};}
inline void EncodeLease(const ReplicaLease& l,v1::ReplicaLease* out){out->set_term(l.term);out->set_primary(l.primary);}
inline void EncodeHead(const ReplicaHead& h,v1::ReplicaHeadReply* r) {
  EncodeLease(h.lease,r->mutable_lease());r->set_baseline(h.baseline);r->set_base_round(h.base_round);
  r->set_next_round(h.next_round);r->set_bytes(h.bytes);r->set_baseline_bytes(h.baseline_bytes);
}
inline ReplicaHead DecodeHead(const v1::ReplicaHeadReply& r) {
  detail::Require(r.next_round()>=r.base_round() && (!r.baseline() || r.baseline_bytes()>=8),"invalid replica frontier reply");
  return {DecodeLease(r.lease()),r.baseline(),r.base_round(),r.next_round(),r.bytes(),r.baseline_bytes()};
}
}  // namespace eps::transport
