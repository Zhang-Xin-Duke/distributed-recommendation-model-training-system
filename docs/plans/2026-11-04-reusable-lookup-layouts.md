# Reusable sparse lookup metadata

Implement reusable Lookup metadata from the repeated Key-list requirement in
the distributed-training plan.
Cache immutable ID/CSR/pooling layouts at both sides of each parameter-shard
connection. Parameter values, placements, tickets and gradients remain fresh.

The first successful full Lookup may register its validated layout and return
an opaque, process-specific token. Later requests carry that token with their
ordinary request identity, read mode and partition epoch. The server resolves
the token into the same layout before entering the existing parameter actor.
Scope tokens by run, learner, shard and partition epoch. Preserve duplicate ID
occurrences, feature order, offsets, valid lengths and pooling modes exactly.

Bound both registries by bytes, entry count and idle TTL, with LRU eviction.
Expired/evicted/previous-process tokens return an explicit layout miss before
allocating a read ticket. The asynchronous client then sends the original full
layout once with exactly the same request identity; network failures retain
the existing BRPC retry contract. Old servers returning no token keep using
full requests. Cache insertion failure must not invalidate a successful read.

Connect opt-in transport caching to the isolated framework client and enable
it in the model worker, with a disabled ablation. Record actual serialized
request protobuf plus attachment bytes separately from avoided layout bytes.
Compare cached/uncached multi-process training parameters, optimizer state,
losses, clocks and bytes; test expiry, eviction, scope/CSR differences,
concurrent lookups and process restart. Recheck GPU availability separately;
no paid top-up or replacement provider is authorized by this implementation.
