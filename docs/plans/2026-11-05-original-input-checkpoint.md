# Original Input Checkpoint Implementation Plan

**Goal:** Restore original ORC/FileFeeder input at the same batch boundary as sparse parameters, native dense state and training progress.

**Architecture:** TensorFlow iterator serialization preserves pending pipeline buffers. Extend the isolated original native build with actual ORC row positions and stable FileFeeder assignment IDs. Every serialized stream reports its assigned-file history and active cursor; the restored master excludes these assignments because their remaining records reside in restored iterator buffers/current readers. Files assigned by speculative prefetch after a stream snapshot return to the remaining queue. Publish iterator files, stream summaries and the master restore plan under the existing joint-checkpoint commit.

**Tech Stack:** C++17 native cursor codec, original BRPC FetchFile response attachments, original ORC seek API, TensorFlow iterator serialization and the existing Python checkpoint adapters.

---

1. Implement a standalone native cursor/files codec and restore-cut planner in `adapters/entropy/input_checkpoint.h`. Check a distributed cut with differently advanced prefetch streams, duplicate filenames, an active reader and completed files; preserve original assignment indices across repeated restore.
2. Add an idempotent source-overlay installer for the private disposable native bundle. Extend FetchFile with assignment metadata, retain per-stream histories, implement actual ORC seek/save and export capture/source manifests through local native functions. Preserve original source copies and record exact overlay hashes.
3. Compile the changed original CPU/CUDA targets using the existing native build cache while the matching TensorFlow Python build continues. Retain the previous libraries separately. Probe actual native iterator save/restore against multiple ORC files before claiming input recovery.
4. Add Python-side capture/staging and joint-commit binding. On save, serialize each learner iterator and export its stream summary; on restore, validate the input group, restore the master queue and Python input-shot position, then restore iterator buffers/current readers before admitting training.
5. Run complete original multi-peer training, a joint save, fresh-process restore and failure/replay. Compare consumed sample identities and complete parameter/optimizer/progress state. Keep the full original integration requirement open until this execution passes.

The restored queue uses stable assignment indices rather than filenames, so
repeated occurrences of the same file remain distinguishable. A stream's saved
history includes files whose rows have already entered serialized prefetch
buffers. Excluding only its current file would duplicate those buffered rows;
retaining the master's later allocation offset would lose post-snapshot reads.

Native stages 1–3 have executed, including direct FileFeeder recovery at eight
positions and six-stream parallel interleave recovery. The registry also keeps
completed reader histories after leaf destruction; a larger-buffer case
restored all remaining records after all six leaves were released. Stages 4–5
now also execute Python input-shot restoration and the full training joint
commit through original main/Executor processes, as recorded below.

## Original prepared batch and partition notifications

`PushPartitionDataset::Initialize` reads the first batch eagerly. Its original
Save/Restore methods only traverse the upstream iterator: they omit the prepared
batch, EOF flag and `GlobalQueue<CpuPartitionType>` notifications. Extend this
native boundary with TensorFlow tensor/scalar state and a locked queue snapshot,
including `last_peek`. During TensorFlow restore initialization, skip the eager
read when `IteratorContext::is_restoring()` is true.

Fresh original Runners also initialize their dataset before model restore. Add
a FileFeeder source hold: held sources return EOF without advancing allocation.
Keep the hold across source initialization, initialize the correct saved input
shot while held, then restore the ordered source roster/assignment exclusion
plan. After all peers finish held initialization, release allocation and restore
their iterators. This prevents the fresh pipeline from claiming files before
the restored readers and buffers take ownership.

## Joint publication and restore order

Add an opt-in `input_checkpoint` job option with the native planner path and a
fresh-restore flag. The FileFeeder graph factory attaches deterministic per-
learner/train-or-eval/source scopes to its URIs and returned dataset. Build
`SerializeIterator`/`SaveV2` and `RestoreV2`/`DeserializeIterator` operations before
the Runner creates its Session.

After DRAINED/FROZEN/PATH_READY, each CPU learner serializes its active iterator
under the process capture lock and publishes a worker manifest containing the
next input-shot index, EOF state, input specification digest, scope, source
handle and hashed iterator/capture files. At the global input-saved barrier,
the chief exports each actual source roster, stages worker captures locally,
runs the native planner and publishes an input group manifest. Bind that
manifest's digest and completed training clocks to the existing joint commit.

Restore verifies the committed input group, every local learner's staged files,
input specification, scope and topology before the global verification barrier.
Reinitialize the selected input shot while source allocation stays held; then
the chief restores source plans. Restore dense state and each learner's iterator
and Python shot position before the final global barrier releases training
progress. An unsuccessful stage retains frozen parameter/progress admission.

Test native prepared-batch/notification recovery at the start, mid-stream and
EOF, including held fresh initialization. The complete original multi-peer
application now passes the opt-in save, mid-file joint checkpoint, abrupt exit
and fresh-process continuation. Two peers finish 128 committed sample marks at
clocks `[16, 16]`; eight uncommitted attempts are replayed. Restored complete
sparse/Adam state has error zero and native DenseAdam files are byte-identical.
The one-peer continuation also matches uninterrupted final state and every loss
exactly. Execution records are linked from `docs/ORIGINAL_INPUT_CHECKPOINT.md`.
