# Joint synchronous-training checkpoint

The executor stops input at a complete round boundary and retains each worker's
actual next input position. The job controller pauses task admission, verifies
equal completed clocks/no active tasks, then pauses each gradient collector at
an empty round boundary. All group next-round counts and parameter optimizer
sequences must match. Incomplete rounds reject checkpointing.

Save parameter weights/moments through the existing frozen snapshot protocol.
Export bounded chunks, verify the native checksum, compute SHA-256 and fsync
local files. Publish an immutable joint manifest last, after parameter files,
group/progress metadata, model configuration and worker input states validate.
A failed export must leave no committed manifest and admission remains frozen.

Restore requires a fresh run identity. Each shard validates its loaded file's
source/run/partition/size/checksum, exact table definitions and optimizer clock
against group metadata. Gradient collectors and parameter admission start
paused. Retrying a pre-restore iteration in the new run rejects; completed
updates from the new run retain their existing idempotent acknowledgement.
The job controller verifies every shard and the restored task clocks before
resuming groups and finally the coordinator.

The TensorFlow example uses a deterministic cyclic sequence of local batches,
with a dataset fingerprint, next iteration and next-batch position. Resume
validates this state and continues at that position. Compare a multi-process
save/restart continuation against uninterrupted training, including complete
touched weights/moments and next-round/input counters. Inject missing/corrupt
files, clock/schema mismatch and partial publication failures.

This establishes coordinated checkpoint restart. Crash-persistent same-run
replay, replicated acknowledgement and fenced live replacement remain separate
requirements in the full distributed-training plan.
