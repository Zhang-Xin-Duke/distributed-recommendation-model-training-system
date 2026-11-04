# Worker Journal Handoff Implementation Plan

**Goal:** Move stopped workers' original-gradient journals between nodes so the
existing replica-backed recovery manager can reconcile and restart training
without a shared filesystem.

**Architecture:** Seal a source journal while holding its exclusive worker lock,
then copy its immutable identity/gradient files through bounded, resumable chunks.
A destination validates the complete journal before atomically publishing it.
The manager fetches worker journals, runs the existing recovery algorithm, then
stages the reconciled journals on replacement worker nodes. Source seals prevent
accidental restart against the old copy. Replica term fencing retains parameter
ownership authority.

**Tech Stack:** Python, existing WorkerJournal binary records and filesystem
fsync/publication, JSON/stdin subprocess or SSH control, existing TensorFlow and
BRPC model recovery tests.

## Design choices

A shared filesystem would avoid copying but requires shared storage on every
worker. Copying a live directory can mix publication points and leave two
writable copies. Explicit stopped-source sealing makes one immutable transfer
snapshot and supports independent replacement-node storage. Sealing is durable
and intentionally requires recovery from the destination once published.

The transport moves original binary records; decoding and re-encoding gradients
could change exact FP32/uint64 values. Per-file SHA-256 checks plus the existing
record/identity/progress validator verify the complete received journal. Persistent per-node streams reuse the SSH/Python connection across chunks. A
bounded chunk protocol resumes the same staging identity after an uncertain
reply. It never merges into an existing unrelated worker journal.

## Task 1: Source sealing and destination publication

Files: `adapters/entropy/training_journal.py`, new
`adapters/entropy/journal_handoff.py`, `tests/journal_handoff_tests.py`.

1. Test rejection of live source locks, immutable seals, changed worker/input,
   corrupt chunks, out-of-order chunks and unrelated existing destinations.
2. Add source sealing under the existing exclusive lock; ordinary worker opens
   and writes reject sealed journals.
3. Implement begin/read/write/finish with bounded chunks, exact duplicate
   acknowledgement, full record validation and atomic destination publication.
4. Verify interrupted/repeated transfer preserves exact original gradients,
   monotonic read frontiers, source immutability and receiver budgets.

## Task 2: Node transport and recovery integration

Files: new `scripts/transfer_worker_journals.py`,
`tests/tensorflow_worker_recovery.py`, `CMakeLists.txt`.

1. Invoke the same node protocol through JSON/stdin subprocesses and SSH using
   the fleet's prepared Python interpreter; keep credentials outside reports.
2. Expose a CLI for fetch/stage with a private fleet and original worker config.
3. Extend full model fault recovery to fetch independent worker journals into
   manager storage, reconcile there, then stage to distinct replacement-worker
   directories before resuming the original training input/gradients.
4. Check all 64 model rows/Adam state, exact per-round losses, duplicate
   reconciliation, old source rejection and process cleanup.

Commands: `build/tensorflow-env/bin/python tests/journal_handoff_tests.py`;
`ctest --test-dir build/tensorflow --output-on-failure -R
'^(durable_worker_gradients|worker_journal_handoff|tensorflow_worker_journal_handoff|tensorflow_worker_recovery|tensorflow_replica_owner_recovery_host)$'`.
Use the actual Python executable recorded by CMake if that environment differs.

## Task 3: Evidence and next integration

Publish source-hashed protocol/model reports and document stopped-source,
retained-storage and receiver-publication semantics. Keep real multi-node CUDA
fault execution, automatic role supervision, full original Entropy training,
log compaction, gradient metadata reuse and common-quality measurements in the
full project scope. The subprocess checks verify serialized node boundaries on
one host; actual SSH/GPU execution requires its own recorded run.

## Execution progress

Protocol implementation, persistent node streams, CLI and model-recovery
integration are complete. The 67-check framework regression suite and the final two handoff checks pass.
Final transfer tests also cover a record larger than 1 MiB and reconnect after
a successful write acknowledgement is lost. The exact source-hashed model
report is retained with the final handoff validation record. No GPU instance
is rented in this phase. Real cross-host GPU failure execution remains next.
