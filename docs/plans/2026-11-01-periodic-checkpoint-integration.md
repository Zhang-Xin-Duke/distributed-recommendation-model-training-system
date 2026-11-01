# Periodic checkpoint integration

**Goal:** Let the existing training controller save sparse state, copy it into its checkpoint store, and continue the same run after the joint save boundary.

**Architecture:** Parameter shards retain a successful checkpoint identity. Bounded reads export only that frozen snapshot. A run/epoch/checkpoint-matched resume reopens shard admission; the adapter reopens input only after every shard acknowledges. Failed or partial saves keep input closed. Prepared-input and request counters remain monotonic.

**Tech stack:** C++20, BRPC/Protobuf, versioned binary snapshots and Python/TensorFlow checkpoint storage.

1. Implemented checkpoint identity checks, bounded immutable-file reads and idempotent resume on each shard.
2. Extended BRPC and the native adapter/isolated bridge to export chunks and resume after coordinated commit.
3. Added export through the existing framework filesystem API, embedded checksum verification and per-shard metadata.
4. Added repeated save/resume, stale run/epoch/checkpoint tokens, live-ticket refusal, partial-save fencing, prepared-input reopening, exported-file restore and uninterrupted optimizer-equivalence checks. A portable TensorFlow graph saves and continues mid-run. Executor shutdown now drains while public PImpl pointers remain valid; a suspended-update regression covers the discovered destruction race.
5. Implemented named-table hooks and original Runner checkpoint coordination. Source-level checks execute the original class, InputBuilder dispatcher and Runner/module save functions against external shards, including continued training and failed-dense-save fencing.

Remote restore staging, native original-application build/execution, multi-peer framework checkpoint execution and in-process GPU tensor interop remain completion requirements. Source-level controller checks and portable lifecycle checks retain their documented scopes.
