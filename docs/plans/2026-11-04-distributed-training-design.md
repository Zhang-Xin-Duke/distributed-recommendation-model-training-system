# Distributed training execution

The project is a distributed parameter server for training one model across
multiple workers. The user's Lecture 16, `16-parameter-server.pdf`, is the
reference for the training mechanisms and evaluation priorities. Existing
Entropy integration remains a requirement; replacing its private Redis client
is an integration task alongside this training work.

The principal multi-GPU scenario is training when parameter/optimizer state
outgrows one GPU. Shard that state across owners and execute real GPU model
forward/backward with sparse Pull/Push. [Capacity acceptance](2026-11-04-multi-gpu-training-capacity.md)
defines actual residency, host-memory placement, correctness and communication
measurements; existing small-state multi-owner results do not establish it.

## Roles and requirements

- Workers own partitioned examples and run model forward/backward (slides 4,
  8, 11-12). Parameter shards own the shared parameters and optimizer updates.
- Sparse Push/Pull route each request to its parameter owners (slides 9-14).
  Preserve sparse layouts, binary values, update scaling and aggregation rules.
  Repeated key-list caching (slide 23) reuses key metadata across requests and
  must preserve run/ownership epochs and bounded cache lifetimes.
- Training-task consistency selects sequential, eventual or bounded-delay
  dependencies (slides 15-18). Asynchronous RPC completion alone does not
  establish a training-consistency policy.
- Logical task progress and range clocks identify completed updates, including
  retries and recovery (slides 21-22). Transport sequence deduplication and
  checkpoint restart have their existing, narrower contracts.
- Replication precedes successful update acknowledgement; a server manager
  fences ownership changes and restores values and clocks (slides 24-26).
  Checkpoint restore and live replica failover have separate execution evidence.
- Compare time to a common loss/quality target, worker waiting, communication,
  staleness and recovery (slides 20 and 27-30). GPU cache/update throughput
  remains a supporting measurement. The paper's dataset/machine counts are
  examples, not implementation acceptance requirements.

## Existing components and missing execution mechanisms

The current service implements shard routing, sparse feature pooling and
unpooling, LazyAdam state, live-run request deduplication, CPU/GPU ownership,
bounded requests and coordinated checkpoint/restore. Native TensorFlow graphs
execute real model gradients. Source-method checks and peer checkpoint checks
have explicit original-native environment substitutions.

Task-clock policies now run through an independent RPC coordinator, with actual
worker processes computing sparse gradients against network parameter shards.
Replica-before-ack, live shard/worker failover and a measured common-quality
comparison across training consistency policies remain open. Full original
Entropy execution also remains open.

## Training-task progress controller

Use a dedicated controller with a fixed worker group and a fresh run identity.
Workers obtain a permit before starting iteration `t` and report completion only
after all parameter updates required by that task are acknowledged. Each worker
has at most one active or waiting iteration; completed counts are contiguous.
The controller never waits synchronously for another worker inside an RPC
completion thread. A blocked admission is represented by a shared future, with
one bounded pending request per worker.

For a completed count `c[w]`, worker `w` has finished iterations `[0,c[w])`.
The global completion floor is `min(c)`:

- Sequential: admit `t` when `min(c) >= t`.
- Eventual: impose no finite inter-worker delay bound.
- Bounded delay `tau`: admit `t` when `t <= tau` or
  `min(c) >= t - tau`.

This is exactly slide 18's rule: all iterations `j < t - tau` must finish.
In particular, with `tau=1`, iteration 1 can start while iteration 0 is
unfinished, and iteration 2 must wait for iteration 0. Permit metadata records
the observed floor at admission; it is a task-progress observation, not a
parameter version or an atomic cross-shard model snapshot.

A server-side controller provides one dependency rule across processes. A
client-only barrier is convenient for strict synchronization, but does not
provide a shared bounded-delay clock or recoverable progress. Framework
collectives remain useful for dense gradient synchronization and are separate
from this controller's task-admission responsibility.

Repeated admission for the same in-flight task shares its result. Duplicate
completion cannot advance progress twice. Run/epoch mismatches reject. Pause
rejects new work, releases waiting callers with an explicit error, and lets
already-admitted tasks finish. A progress checkpoint is exportable only after
pause/drain. Restore requires a fresh run and preserves the configured policy,
worker group and clocks; it remains paused until the job controller verifies
the matching parameter/input checkpoint. Abort wakes blocked callers and is
terminal. Worker replacement and replica-backed task replay are subsequent
protocol work, not guarantees of the initial fixed-group controller.

## Implementation and verification sequence

1. Implement and verify the C++ task-progress controller: slide-18 boundaries,
   duplicate/malformed tasks, pause/drain, fresh-run restore, abort and concurrent
   workers. This establishes the control component for network integration.
2. Expose progress RPCs, connect real worker processes to the controller and
   existing parameter shards, and execute a real sparse training workload.
   Define gradient aggregation/normalization and optimizer-step semantics
   explicitly. A synchronization barrier alone does not implement averaging
   multiple workers' gradients into one optimizer update. Add repeated key-list
   handles separately from the existing single-consumption preparation tokens.
3. Implement persistent per-range update progress, duplicate-safe retries across
   ownership changes, replication-before-ack and fenced primary promotion.
   Inject lost acknowledgements and process failures during training.
4. Replace the missing Redis SDK with an operational public Redis backend,
   preserving the caller's command, batching, ordering, TTL and error contracts;
   build and run the complete original Entropy training/control path.
5. Run common-quality comparisons for sequential/eventual/bounded delay, with
   controlled stragglers and failures. Retain parameter-state checks, rejected
   points, worker idle time, bytes transferred and recovery time.

Policy comparisons keep the update algorithm, gradient normalization and
optimizer-step unit fixed. Round-averaged synchronous SGD is a separate
aggregation mode from one-update-per-worker training; changing that mode must
not be attributed solely to a change in the delay policy.

The eventual end-to-end acceptance requires the complete training path and its
failure behavior; component tests do not substitute for that acceptance.

## Current execution record

Step 1's C++ core and step 2's RPC/worker execution are implemented. Four worker
processes train through two parameter-server processes and a separate coordinator
under each policy. Each policy completes 160 local-mean-gradient/per-worker
LazyAdam updates, checks actual row-update counts and lowers controlled loss.
The RPC overload check also verifies that a saturated dependency-wait list
leaves ready slow-worker admission and completion responsive.

All 32 native checks pass. The modified admission core passes ASan/UBSan and
TSan, including eight-thread progress and four-worker/two-actor sparse updates.
Source-hashed records are `../validation/2026-11-04-training-progress.json` and
`../validation/2026-11-04-training-rpc.json`.

Shard-local synchronous gradient aggregation now adds sum, worker mean and
sample-weighted mean modes, finite collection timeout and one update per complete
round. Four real worker processes train through three parameter shards for 40
rounds under both averaging modes, including an empty shard. Weights and Adam
moments match an independent model reference, and completed retries do not add
updates. The record is `../validation/2026-11-04-synchronous-gradients.json`.
All 35 native tests pass; the collective and task-progress cores pass ASan/UBSan
and TSan. No GPU rental was used for these stages.

The TensorFlow worker now trains shared Embedding and MLP parameters through
the synchronous mode, including verified GPU forward/backward and CUDA owner
updates on an A6000. Its first global gradient and complete touched state match
a single-worker reference within the recorded tolerances. Extend this path
across GPU instances and execute the beyond-single-GPU capacity acceptance.
Reusable key-list handles, durable
round/range progress, replicated failover, original-framework execution and
common-quality measurements remain in the full implementation scope.

Node orchestration now has a Host-backed end-to-end lifecycle check covering
distributed backing/residency, model rounds, sampled optimizer state and owned
process cleanup. The planned two-node GPU capacity run is pending provider
availability: its first A6000 creation is rejected and no instance is created.
The record is `../validation/2026-11-04-distributed-training-orchestration.json`.

Joint synchronous-training restart now couples complete parameter/Adam snapshots,
gradient next-round counters, task-progress clocks, model configuration and
worker input positions in one published manifest. Fresh processes remain paused
through verification and existing-row owner rehydration. The two-shard/two-Host-
owner TensorFlow test compares all 32 rows and continuation losses against
uninterrupted training after restarting inside a three-batch input cycle; both
maximum absolute differences are zero. Injected partial exports do not publish
a manifest. The record is `../validation/2026-11-04-joint-training-checkpoint.json`.
This completes coordinated round-boundary restart; replica-backed in-flight
replay and fenced live replacement remain required work.
