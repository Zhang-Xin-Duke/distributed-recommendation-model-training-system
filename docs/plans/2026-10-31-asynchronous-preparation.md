# Bounded asynchronous input preparation and phase timing

**Goal:** Prepare upcoming input metadata concurrently with current model/RPC
work, preserve explicit parameter-read boundaries, and expose service stage time.

**Architecture:** A move-only prepared lookup owns original/shard-local layouts
and position maps. Its builder is pure and reads no parameter state. The adapter
owns one bounded preparation worker and consumes accepted tokens in FIFO order.
Lookup IDs are assigned only when a prepared input is consumed. Existing
Pull/Push/Join and framework model computation retain their responsibilities.

## Choices and invariants

Preparing actual parameter tensors early could read weights before the caller's
previous update. The queue instead compacts padding and routes IDs only. A
background PS migration thread would require a new ownership protocol; this
change keeps state access on the existing request path.

Bound both accepted-job count and conservative retained/scratch bytes. Cancelled
running work retains its reservation until the worker releases the input; ready
and queued cancellation reclaim their buffers. A claimed input cannot be
cancelled. Save requires both live parameter batches and pending prepared inputs
to be consumed/cancelled, then atomically closes preparation admission. Join
continues to describe live parameter work and permits future metadata inputs.

The worker uses an independent mutex from the adapter's model/RPC lock, so
submission and routing can proceed while a previous Pull is waiting on RPC.
Consumption is explicit and FIFO. Synchronous Pull requires no queued prepared
input; it reuses the same builder. Failure after a successful Lookup releases
known tickets before closing the adapter.

## Implementation

1. Add pure `PreparedLookup` builder with input/derived byte bounds and exact
   shard reservations. Refactor `DistributedClient` to consume prepared routing.
2. Add bounded queue and adapter Prepare/PullPrepared/Cancel APIs with lifetime,
   ordering, shutdown and save behavior. Expose Python IPC/TensorFlow operations.
3. Add stage counters for CPU lookup preparation, gradient preparation, GPU
   packet building/dispatch, CPU optimizer work, agent wait and migration.
   Nested wait/migration scopes overlap and must not be added as disjoint time.
4. Check sum/mean/sequence/padding and optimizer equivalence against synchronous
   execution; test bounded admission, queued/running cancellation, FIFO errors,
   blocked-RPC overlap, no early PS calls, save/Join and failure cleanup.
5. Run native/RPC/TensorFlow and core ASan/UBSan checks. Include a real graph using
   next-input preparation while current-step execution runs. Record evidence,
   update docs and push. No GPU rental before local implementation is ready.

## Local implementation evidence

Implemented the pure prepared lookup, padding/shard preparation, bounded FIFO
worker, cancellation/close/save lifecycle, adapter/Python/TensorFlow operations
and eight inclusive service wall-time counters. Native/RPC/TensorFlow checks
pass 24/24; ASan/UBSan 11/11; the two focused ThreadSanitizer checks pass 2/2.
The graph includes real dense/sparse updates with prepared tokens and automatic
cache/coalesced paths, output-schema cleanup and configured queue-byte rejection.
Source `040ae90` also passes the complete A6000 CPU/CUDA/TensorFlow suite 26/26
and four forced-CUDA process/graph checks 4/4. Six fixed-trace runs retain 14 phase
records and 112 updates with all eight counters and exact final CPU replay.
The records retain one nonfatal cache-maintenance failure; training-update
errors are zero. The single rental was deleted after archive hash verification.
[Source-hashed evidence](../validation/2026-10-31-asynchronous-preparation.json).
