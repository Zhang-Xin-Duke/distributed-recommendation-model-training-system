# Periodic checkpoint integration

**Goal:** Let the existing training controller save sparse state, copy it into its checkpoint store, and continue the same run after the joint save boundary.

**Architecture:** Parameter shards retain a successful checkpoint identity. Bounded reads export only that frozen snapshot. A run/epoch/checkpoint-matched resume reopens shard admission; the adapter reopens input only after every shard acknowledges. Failed or partial saves keep input closed. Prepared-input and request counters remain monotonic.

**Tech stack:** C++20, BRPC/Protobuf, versioned binary snapshots and Python/TensorFlow checkpoint storage.

1. Implemented checkpoint identity checks, bounded immutable-file reads and idempotent resume on each shard.
2. Extended BRPC and the native adapter/isolated bridge to export chunks and resume after coordinated commit.
3. Added export through the existing framework filesystem API, embedded checksum verification and per-shard metadata.
4. Added repeated save/resume, stale run/epoch/checkpoint tokens, live-ticket refusal, partial-save fencing, prepared-input reopening, exported-file restore and uninterrupted optimizer-equivalence checks. A portable TensorFlow graph saves and continues mid-run. Executor shutdown now drains while public PImpl pointers remain valid; a suspended-update regression covers the discovered destruction race.
5. Pending: install the backend at the original table factory and Pull/Push/Join boundaries, preserving the existing model/data paths. Connect periodic checkpoint orchestration after the lifecycle support is verified.

Original native-application build, original controller end-to-end execution and in-process GPU tensor interop remain completion requirements. The new lifecycle APIs do not by themselves establish those integrations.
