# Shard-local synchronous gradient rounds

Each parameter shard collects one gradient contribution from every member of
a fixed worker group, combines matching table/Key rows and applies one update
for the round. This keeps sparse gradient traffic on parameter owners; the
training coordinator continues to carry task progress, not model-sized tensors.

Workers provide sorted unique row gradients after local pooling backward. The
schema fixes tables/dimensions for the group; every worker sends every table,
including empty contributions to shards it did not access. An absent Key is a
zero gradient in that worker, so worker averaging divides by the complete group
size. Sample-weighted averaging combines local batch-mean gradients using each
worker's sample count. Sum is also available with explicitly supplied gradients.
Reduction uses fixed worker order and double-precision accumulation, then emits
FP32 rows in sorted order. The selected reduction is a training-algorithm choice.

The collector retains one contribution per worker and one live round. Duplicate
in-flight submissions must have identical content and share completion. A retry
of a completed round acknowledges its original success without applying again.
Invalid/future rounds reject; a missing worker triggers a finite round timeout.
Conservative memory reservations cover retained gradients and reduction scratch.
The application callback runs outside the admission lock, and completion follows
the parameter/optimizer acknowledgement. Apply failure or abort fails the group.
An already-dispatched update drains; abort provides no parameter rollback.

The parameter process exposes a separate synchronous-gradient RPC service when
configured for this mode. It reserves learner 1023 for round application, disallows
ordinary worker writes, and keeps read-only Pull available. Its application path
uses the existing row ownership, CPU/GPU update and optimizer code through a
sequence-layout Lookup/Push. Empty tables still advance their optimizer context
once per round. A fixed worker group supports up to 1023 workers in this mode.

The RPC layer keeps at most one pending response per worker. A second in-flight
retry receives a capacity error and can retry the same identity; it cannot consume
the missing worker's response slot. All shards must acknowledge before the worker
completes its task. A strict task-progress policy keeps the next round's reads
behind the preceding round's updates on all shards.

Verification covers averaging with missing Keys/unequal sample counts, sums,
arrival-order determinism, duplicates and invalid packets, bounded memory,
timeout/abort during collection, held application acknowledgements and real
multi-process training compared with an uninterrupted single-model reference.
Replication, crash-persistent replay, worker replacement and original Entropy
integration remain in the overarching training plan.
