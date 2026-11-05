# Reusable sparse-gradient Key lists

Each synchronous Worker submits fresh gradients every round. When its sorted
Key lists repeat, the first Submit registers table IDs, dimensions and exact Key
order with the parameter shard. Later submissions send an opaque token and new
gradient values. Sample counts, completed-read frontiers and task identities
remain part of every request.

```mermaid
sequenceDiagram
    participant W as Training worker
    participant S as Parameter shard
    participant M as Bounded metadata registry
    W->>S: Task t, Keys, fresh gradients, register layout
    S->>S: Admit contribution to the synchronous round
    S->>M: Retain immutable table/dimension/Key metadata
    S-->>W: Update acknowledged, layout token
    W->>S: Task t+1, token, fresh gradients
    S->>M: Resolve within run/epoch/worker/shard
    S->>S: Collect gradients and update parameters once
    S-->>W: Update acknowledged
    Note over W,M: A token can expire or be evicted
    W->>S: Task t+2, token, fresh gradients
    S-->>W: Metadata miss; no collector admission
    W->>S: Same task t+2, complete Keys, same fresh gradients
    S-->>W: Update acknowledged, replacement token
```

## State and retry boundaries

The registry stores Key metadata only. It retains no gradient values, parameter
values, optimizer state, row placement or completed-round decisions. Resolving
a token reconstructs an ordinary contribution for the existing collector, which
keeps its validation, deterministic reduction, duplicate handling, replication
and one-update-per-round semantics.

Tokens have a process-specific namespace and are scoped to run, partition
epoch, worker and shard. Table IDs, dimensions, table order and exact ordered
Keys form the content identity. Changing any of them sends a full request.
Scopes and schemas are checked before collector admission. Expiry, eviction or
registry replacement produces an explicit metadata miss, also before admission.
The asynchronous client resends the exact original values and task once with
full Keys, within the original completion deadline. A transport failure retains
the usual gradient retry path; it does not turn into a metadata miss.

Both sides bound entry count, conservative metadata bytes and idle TTL, with LRU
eviction. The Lookup and gradient registries share the same typed registry
implementation. Held immutable layouts remain valid through eviction. Optional
registration failure leaves an accepted update usable. Registration piggybacks
on a normal Submit, avoiding a separate RPC and round trip.

The client still fingerprints Keys and prepares a full request for retries.
Token requests share slices of the retained gradient attachment instead of
repacking its numerical values. The optimization reduces repeated network
metadata; it does not remove gradient traffic or local Key preparation. Larger
vector dimensions reduce the fraction of bytes occupied by Keys.

## Configuration

```cpp
BrpcSynchronousGradients gradients(endpoint, run, epoch);
gradients.SetGradientLayoutCache(true);
auto acknowledged = gradients.Submit(contribution);
const auto transport = gradients.TransportStats();
```

C++ clients default to full requests. The TensorFlow training example enables
reuse after configuring training; `reuse_gradient_layouts: false` selects the
ablation. The Python backend exposes `gradient_layout_cache(enabled)` and
`gradient_transport_stats()`. An older server returning no token continues to
receive full requests.

The server's `synchronous_training.layout_cache` object accepts `entries`,
`bytes` and `idle_ttl_ms`. Defaults are 4,096 entries, 64 MiB of charged metadata
and 60 seconds. Zero entries or bytes declines registrations. The C++ client's
optional constructor argument configures its own registry independently.

## Executed model comparison

Two CPU TensorFlow Worker processes train Embedding vectors and a two-layer MLP
through two parameter shards for eight synchronous rounds. Both runs use the
same data and update algorithm: 512 configured rows, eight-dimensional vectors,
16 local examples per Worker and 64 ID occurrences per example.

| Submit request path | Full requests | Token requests | Serialized request bytes |
| --- | ---: | ---: | ---: |
| Full Keys | 32 | 0 | 287,224 |
| Reusable Keys | 4 | 28 | 237,980 |

Request bytes decrease **17.1%** in this configuration. All **503 touched rows**
have identical weights, first/second Adam moments and update counts. Per-round
loss differences are also zero; both jobs complete the same training clocks.
The counter includes Submit protobufs and attachments. It excludes responses,
other RPC paths and BRPC-internal retransmissions; it is not an end-to-end
throughput or speedup measurement.

Native RPC tests additionally cover changing numerical gradients, empty
contributions, completed-round retries, cache rejection, expiry/eviction,
cross-worker/run/epoch misuse, malformed schema/Key payloads and no-admission
misses. Typed registry tests cover process replacement, exact fingerprints,
held views, capacity bounds and concurrent Workers.

```sh
ctest --test-dir build/native --output-on-failure \
  -R '^reusable_(lookup|gradient)_layout_(cache|rpc)$'
python tests/tensorflow_gradient_layouts.py --build build/tensorflow \
  --out build/tf-gradient-layouts.json
```

[Source hashes, checks and model results](validation/2026-11-04-reusable-gradient-layouts.json).
