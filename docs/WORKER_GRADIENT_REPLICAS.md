# Durable worker gradient replicas

Workers replicate original gradients to another node before submitting them to
parameter shards. If a worker loses its local journal, the manager can retrieve
the acknowledged original records from that replica and reconcile the same
training round. Replacement workers and replica generations are seeded from the
recovered history.

```mermaid
sequenceDiagram
    participant W as Model worker
    participant L as Worker-local journal
    participant R as Gradient replica on another node
    participant P as Parameter shards
    participant C as Training coordinator
    W->>W: Forward / backward
    W->>L: Publish original gradient record
    L-->>W: Local record durable
    W->>R: Begin / bounded chunks / commit
    R->>R: Validate original bytes; fsync and publish
    R-->>W: Durable record identity and SHA-256
    W->>P: Submit the same sparse gradient
    P->>P: Aggregate; update parameters and replicate after-images
    P-->>W: Required shard updates acknowledged
    W->>C: Complete training task
```

## What is retained

The replica keeps the existing WorkerJournal binary records: iteration, sample
count, completed-read frontier, loss, table IDs, exact uint64 Keys and FP32
gradients. A model/input identity binds the worker group, table configuration,
input batches and base round. Raw-record import validates the existing record
checksum, identity, dimensions, finite values and monotonic progress before
publishing the original bytes.

Parameter replicas retain committed parameter/Adam after-images. Original
worker gradients supply the missing contributions when only part of a round
committed. The worker-gradient replica makes those original contributions
available after the worker-local copy disappears.

## Publication and retries

One service stores one worker's history under a fixed writer generation. Its
configuration and exclusive journal lock bind that storage to the expected
identity. Data requests carry the identity digest, generation and writer name;
replacement stores reject delayed requests from former generations.

A persistent length-framed TCP connection carries begin, chunk and commit
requests. Frames are capped at 1 MiB and chunks at 256 KiB. Begin declares the
record size and SHA-256. Staging retains only the next contiguous record and
resumes at its durable byte offset. Repeated chunks compare existing bytes.
Commit checks the complete file hash and WorkerJournal record, then fsyncs the
immutable publication before acknowledgement.

A lost successful commit reply causes the client to retry the same record.
The replica returns its existing durable identity without adding another
record. After a process stops between record publication and staging cleanup,
restart recognizes the committed file and removes the remaining staging data.
Committed-history and per-record budgets bound storage; at most one bounded
record is staged in addition to committed history.

The worker waits for that remote acknowledgement before `Submit`. If remote
durability cannot be established after bounded transport retries, it retains
its local record and fails the training attempt before any gradient submission
for that iteration. It does not substitute a newly computed gradient after an
uncertain reply. The protocol adds network round trips and file synchronization
to the training path; persistent connections and bounded chunks avoid repeated
connection setup and transferring the entire journal at each step.

## Managed recovery

Set `mirror_worker_journals: true` in the managed fleet. Each node supplies a
`journal_mirror_listen` address and ordered `journal_mirrors` routes; the manager
supplies its own ordered `journal_mirrors` routes. Worker 0's replica is placed
on node 1, and worker 1's replica on node 0.

The manager starts gradient-replica roles before workers. On failure it stops
former writers and replica services, fetches the replica journals and runs the
existing replica-backed parameter reconciliation. Each recovered journal seeds
both a replacement worker and a fresh gradient-replica store. The new generation
starts before task admission resumes. Successful completion checks both replica
histories reach the final round.

The framework worker's optional descriptor is:

```json
{
  "journal_directory": "/worker/job/journal",
  "journal_workers": 2,
  "journal_mirror": {
    "endpoint": "127.0.0.1:19401",
    "generation": 1,
    "writer": "worker-0-g0",
    "timeout_ms": 5000
  }
}
```

Node deployment uses `scripts/journal_replica_server.py --config replica.json`.
Its config specifies `directory`, `listen`, the complete original
`worker_config`, `generation` and `writer`. The service rents no resources and
accepts only its bound worker's records. Administrative placement and generation
assignment remain on the managed node channel.

## Executed verification

Protocol checks cover exact raw import, immutable duplicates, corrupted records,
round gaps, identity/generation rejection, a lost successful commit reply,
restart during partial staging and restart after publication before cleanup.
A record larger than 1 MiB streams across multiple bounded chunks. A fresh
replacement store rejects the former writer generation.

A real TensorFlow failure check points the worker at an unavailable replica.
The worker retains its original local gradient and exits; both parameter shards
record zero optimizer updates and the task clock remains zero.

Managed model checks delete both former worker-local journals after stopping
roles. Recovery fetches only the opposite-node gradient replicas, restores all
64 parameter/Adam rows and continues eight rounds. Full state and every loss
match an uninterrupted baseline without gradient replication exactly. The cases
cover live-read interruption, partial shard commit and lost task completion,
including manager reentry after journal staging.

The CPU execution uses independent node processes and real TCP aliases. A
follow-up execution runs the same three faults across **two rented A6000 GPU
instances with distinct physical GPU UUIDs**. Workers execute forward/backward
on GPU, and parameter owners retain CUDA state. Both former local journals are
deleted in every fault case; the opposite-node replicas supply the originals.
All 64 rows' complete weights/Adam state and every loss match uninterrupted GPU
training exactly. Against the CPU reference, maximum state/loss differences are
`2.98e-8` / `5.96e-8`. Both replacement replica histories finish at round 8 under
generation 2. Manager reentry after journal staging also passes across these
nodes. Role census confirms all 75 recorded remote process groups are drained,
and both rented instances are deleted.

These results establish worker-journal storage-loss recovery with accessible
parameter replicas and replacement nodes. Complete physical-node loss also
requires surviving parameter authority and replacement placement/routing.
Original Entropy native execution and the remaining quality/scaling work retain
their acceptance requirements. The complete local framework suite passes 72/72.

[Validation record](validation/2026-11-04-worker-gradient-replicas.json)
retains full trajectories, protocol/failure reports and source/binary hashes.
