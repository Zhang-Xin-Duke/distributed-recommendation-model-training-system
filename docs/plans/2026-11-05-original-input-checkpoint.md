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
continue with Python input-shot state and the full training joint commit.
