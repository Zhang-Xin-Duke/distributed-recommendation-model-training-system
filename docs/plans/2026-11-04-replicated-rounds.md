# Replicated synchronous training rounds

## Decision and scope

Persist exact post-update weights, both Adam moments, optimizer clocks and
completed transport frontiers before acknowledging a synchronous shard round.
Use a separate disk-backed replica process and a local primary after-image log.
The replica owns a durable, monotonic primary term. An explicit manager performs
compare-and-swap term advancement; old terms cannot commit new records.

This extends the actual Embedding/MLP training path. Keep the existing
coordinator and workers alive during primary replacement so workers retain the
original in-flight gradients. Worker-loss replay and automatic discovery are
separate remaining mechanisms. Multi-GPU capacity and complete Entropy
integration remain required in the overarching plan.

One operating-system advisory lock protects each local log directory across
processes. An independent replica using the same directory fails startup; after
process termination a new instance reopens and verifies the durable frontier.

## Normal execution

1. Before serving, install an immutable CPU snapshot as the replica baseline,
   verify matching model/run/partition configuration and check the granted term.
2. Apply one deterministic aggregated round through the current CPU/GPU owner
   path. Collect updated rows, all table optimizer clocks and contiguous
   completed request frontiers while the collector still holds the round.
3. Fsync the primary's bounded after-image record. Stream the same record to the
   replica in bounded chunks. The replica checks term, contiguous round,
   checksum and completed image, then fsyncs and publishes the record.
4. Only after replica acknowledgement may the collector acknowledge workers.

After-images cost three parameter vectors per touched row. They preserve actual
floating-point results and avoid running Adam again on recovery. Whole-table
copies per update would make sparse training expensive; gradient-only logs
would require optimizer replay and identical numerical execution. Baselines and
records are streamed and disk bounded; the standby does not load the entire
parameter table into RAM. The existing distributed CPU backing still consumes
memory on active parameter shards.

## Failure and recovery

On replacement, first advance the replica term durably, then restore a fresh
primary from its baseline plus contiguous committed after-images. Restore
optimizer and request frontiers, recover the collector's next round and open
admission only after checking the current lease. A round committed before a
lost acknowledgement is acknowledged on retry without another update. A
locally applied but unreplicated round is absent from recovery and is applied
once when workers retry their retained gradients.

Fencing applies to update commits. Endpoint selection is the manager's
responsibility; the protocol does not promise linearizable reads from an old
endpoint. A replaced primary must use an independent/restarted parameter owner;
the first process-failure test uses CPU ownership. Reusing a live old GPU owner
across competing primaries requires additional owner fencing before GPU
failover can be claimed.

## Verification

Check core log durability/reopen, partial uploads, retries, stale terms,
compare-and-swap promotion, corruption, gaps and bounded disk/payloads under
sanitizers. Exercise independent primary/replica processes and real model
workers, inject primary loss and lost acknowledgement, and compare complete
weights/moments, row counters, round/task clocks and loss with uninterrupted
training. Retain failure points and exact execution limits.
