# Worker gradient replicas

**Goal:** Durably replicate original worker gradients before parameter updates,
then recover training when worker-local journal storage is lost.

**Architecture:** Each worker has a TCP journal-replica service on the opposite
node. The service binds one model/input identity and writer generation to its
storage. A worker publishes each immutable binary gradient record through bounded
chunks and waits for fsync-backed acknowledgement before gradient Submit. The
managed recovery workflow stops replica writers, fetches the remote journal and
seeds both replacement workers and fresh replica generations.

**Tech stack:** Existing WorkerJournal record format and validation, Python framed
TCP streams, C++ BRPC training path, managed node roles and journal handoff.

## Design choices

Keep original binary records to preserve FP32 and uint64 values. Local-only
journals cannot recover a partially committed round after their disk is lost.
Synchronous remote publication adds a network/disk acknowledgement to the training
path in exchange for a retained copy of every gradient allowed to update a shard.
The source worker never substitutes a newly computed gradient after an uncertain
mirror acknowledgement; it retries the same record identity/content.

TCP data requests are bound to the configured worker identity and writer
generation. Administrative process placement and generation assignment use the
existing manager/node channel. Fresh generation stores are seeded from the
recovered journal. Former replica processes must be stopped before their storage
is sealed; generation checks reject delayed old requests at replacements.

## Implementation and verification

1. Extend WorkerJournal with validated raw-record import; verify exact bytes,
   immutable duplicates, contiguous rounds, read progress, corruption and budgets.
2. Implement the bounded TCP store/client, resumable staging and durable duplicate
   acknowledgement. Check lost successful commit acknowledgement, restart during
   staging, identity/generation mismatch and large records.
3. Integrate remote acknowledgement before Submit in the TensorFlow worker. Run a
   real failed-mirror model check proving zero parameter updates and no completed
   training task when remote durability fails.
4. Integrate opposite-node mirror roles and recovery sources in managed training.
   Delete both former worker-local journals after stopping roles, then recover
   original gradients from replica storage. Compare all 64 parameter/Adam rows,
   complete losses, clocks and cleanup against uninterrupted training.
5. Publish source/binary-hashed evidence and update deployment preparation.

This verifies worker-journal storage loss with accessible parameter replicas and
replacement nodes. Complete physical-node loss additionally requires surviving
parameter authority and replacement placement/routing. Real cross-host CUDA,
original Entropy native execution, compaction, gradient metadata reuse and common
quality/scaling measurements remain in the full project scope.

## Executed milestone

All five implementation steps are complete for worker-journal loss. Local
framework checks pass 72/72. Protocol checks include lost successful commit
acknowledgement and both staging/publication restart windows; the failed-mirror
model check retains a gradient but performs zero shard updates. Two A6000
instances run the three complete GPU recovery faults with both local journals
deleted. Complete 64-row state and per-round loss match uninterrupted GPU
training exactly, including manager reentry after staging. Remote process-group
census and cloud inventory verify cleanup; estimated rental cost is USD0.1451.
Full physical-node loss and original Entropy integration retain their scope.
