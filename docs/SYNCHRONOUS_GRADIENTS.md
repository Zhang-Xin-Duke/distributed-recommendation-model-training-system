# Synchronous sparse training

Each parameter shard collects the current round's sparse row gradients from a
fixed worker group, reduces matching Keys and applies one optimizer update. The
coordinator carries task progress; parameter/gradient traffic goes to its owners.
This is an aggregation mode separate from the existing per-worker update path.

```mermaid
sequenceDiagram
    participant W as Worker processes
    participant C as Task coordinator
    participant S as Each parameter shard
    W->>C: Begin iteration t
    C-->>W: Permit
    W->>S: Read-only Pull needed vectors
    S-->>W: Model features
    Note over W: Forward, local mean loss, backward
    W->>S: Submit owned row gradients + sample count
    Note over S: Collect every worker, including empty contributions
    Note over S: Reduce matching Keys; update weights and moments once
    S-->>W: Round update acknowledgement
    W->>C: Complete after every shard acknowledges
```

## Gradient and update semantics

The group fixes its run/epoch, worker count, table IDs/dimensions and reduction.
Contributions contain sorted unique Keys with row-major FP32 gradients. Each
worker performs its local pooling backward first, including repeated-Key
reduction. It sends every table to every shard, using empty rows where it has no
accesses. A missing Key contributes zero; an absent Worker keeps the round open.

| Reduction | Combined gradient for Key k |
| --- | --- |
| `sum` | Sum of the supplied worker gradients |
| `worker_mean` | Sum divided by the full worker count |
| `sample_mean` | Local batch-mean gradients weighted by local sample count, divided by total sample count |

The collector reduces in worker-ID order using double accumulation, then emits
sorted FP32 rows. Changing RPC arrival order cannot change its reduction order.
Different reduction modes describe different training algorithms.

Each complete round executes one Lookup/Push lifecycle in reserved learner
stream 1023. Sequence layouts map row gradients directly onto the existing
parameter/optimizer path, including GPU-resident owners. Empty tables also pass
through this path and advance their optimizer context once, without modifying
absent rows. The configured mode disables ordinary worker writes and permits
read-only Pull; worker IDs range from 0 to 1022.

## Bounded collection and failure

There is one active round and at most one retained contribution per worker.
Conservative byte reservations cover retained input, reduction maps/vectors and
output. Invalid schemas, wrong shard Keys, unsorted/duplicate Keys, non-finite
values, future rounds and oversized contributions reject before admission.
In-flight retries must match their original contents. Completed-round retries
acknowledge the original success without another update.

The application callback runs on one executor outside the admission mutex.
The round acknowledges only after the parameter update completes. A finite
timeout starts at the first contribution; if a Worker never contributes, the
group fails without applying that incomplete round. Apply errors and operator
abort also fail the group. Already-dispatched work drains before collector
destruction; there is no rollback of an externally applied update.

The RPC layer retains at most one pending response per worker. Duplicate calls
cannot occupy another worker's response slot. A second in-flight RPC returns a
capacity error and may retry the same task; accepted gradients remain in the
round. Stats/Abort use independent immediate replies. Shutdown fails pending
calls before the server joins.

Workers submit all shards before waiting and complete their task after every
shard acknowledges. This keeps subsequent model reads behind the prior round's
updates. Failures during cross-shard application can leave partial state;
replication, persistent round progress and fenced recovery remain in the full
training plan. The standalone server currently rejects synchronous snapshot
restore without a matching group checkpoint.

## Configuration

Declare the tables in the normal parameter-server config and add:

```json
"synchronous_training": {
  "workers": 4,
  "reduction": "sample_mean",
  "gradient_bytes": 67108864,
  "round_timeout_ms": 30000
}
```

The example `eps_training_worker` selects `"update_mode": "synchronous"`.
It uses the same worker/coordinator/shard configuration as the existing example.
`eps_training_control inspect` adds collector counts when its config includes
`"synchronous": true`.

## Verification

Core checks cover sums, averaging with missing Keys and unequal batch sizes,
empty contributions, invalid packets, memory rejection, retries, timeout/abort,
held application acknowledgements and eight concurrent workers over 40 rounds.

The BRPC check holds a worker response while 64 duplicate calls reject, then
admits the missing workers and finishes the round. It verifies write guards,
stale administrative identities, shutdown and missing-worker timeout. Full
weights/moments match a single averaged update exactly. An empty middle round
also advances the optimizer context while leaving the row update count unchanged.

The process check runs four worker processes with unequal local batch sizes and
three parameter shards. Two modes each execute 40 rounds / 160 worker batches.
The third shard has no Keys but participates in every round. Each shard applies
40 optimizer updates; 160 completed-round retries per shard add no updates.
All final weights and Adam moments match an independent FP32 model reference.
The reference covers fused and separate moment arithmetic, choosing one whole
model reference to match the compiler's floating-point contraction.

```sh
cmake --build build/native -j 4
ctest --test-dir build/native -V \
  -R '^(synchronous_gradient_rounds|synchronous_gradient_rpc|synchronous_training_processes)$'
```

[Source hashes and executed records](validation/2026-11-04-synchronous-gradients.json).
The current execution record is CPU correctness; native GPU model execution and
the [beyond-single-GPU capacity workload](plans/2026-11-04-multi-gpu-training-capacity.md)
are the next integration and measurement stage.
