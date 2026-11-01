# Periodic parameter checkpoints

A training controller can freeze sparse parameter updates, export their state
into its checkpoint store, commit that state together with the model and input
position, and continue training in the same run. Parameter shards expose bounded
snapshot reads and a checkpoint-matched resume operation; the table adapter
coordinates them across all shards.

```mermaid
sequenceDiagram
    participant T as Training controller
    participant A as Table adapter
    participant S as Parameter shards
    participant F as Framework checkpoint store
    T->>A: Complete batches and cancel unused prepared inputs
    T->>A: Save(checkpoint_id)
    A->>S: Drain accepted work and save weights + Adam state
    S-->>A: Successful snapshot metadata from every shard
    T->>A: Export checkpoint through framework filesystem
    loop Bounded chunks per shard
        A->>S: Read frozen snapshot (offset, limit)
        S-->>A: Bytes + total size + EOF
        A->>F: Write unpublished shard file
    end
    Note over A,F: Verify snapshot checksum; publish files and manifest
    T->>F: Commit model, optimizer, input position and sparse manifest
    T->>A: Resume(checkpoint_id)
    A->>S: Resume(run, partition_epoch, checkpoint_id)
    S-->>A: All shard acknowledgements
    A-->>T: Reopen prepared-input and Pull admission
```

## Save and resume

`TableAdapter::Save` requires every live batch to be completed or released and
every queued prepared input to be consumed or cancelled. It closes local
admission before dispatching shard saves. A shard save is an execution barrier:
earlier accepted operations finish before the shard checks outstanding lookup
tickets and writes state.

Saving moves resident GPU rows, including Adam moments and update counters, back
to the CPU owner. This produces a single canonical snapshot per parameter shard.
Continued training starts with CPU-owned rows; the configured placement policy
can admit them again. This choice simplifies the save boundary at the cost of
writeback and cache warmup around a checkpoint.

Each shard remembers only its current successful checkpoint identity. Export
requires the same run, partition epoch and checkpoint ID while the shard remains
frozen. Resume requires that identity and is idempotent for repeated
acknowledgements. The adapter opens input only after every shard acknowledges;
a partial resume can be retried with the same ID. Request handles and
prepared-input tokens continue increasing across resume.

A failed or partial save leaves local input closed and has no successful adapter
checkpoint token. A controller must fail that save attempt and restart from a
previously committed checkpoint. An older token cannot reopen the adapter after
a newer save attempt. Snapshot IDs name immutable files, so each save uses a
fresh ID.

## Export into the framework store

`export_checkpoint` accepts the existing `tf.io.gfile` interface. The portable
graph test uses that interface directly. Each RPC returns at most 1 MiB, so
export does not materialize the whole snapshot in Python or in an RPC response.
The native client verifies the snapshot's trailing FNV checksum incrementally;
the exporter records SHA-256 and byte length for each published shard file.

The exporter writes unique pending files, closes them, checks their stored
lengths, and publishes them without overwriting existing content. Identical
published content is accepted on retry. A content conflict raises an error.
`parameter-service.json` appears only after all shard files are published. Failed
exports remove the current pending file; already published shard files remain
available for a retry. Filesystem durability and rename semantics follow the
selected framework filesystem.

```python
backend.save("step-2000")
manifest = backend.export_checkpoint(
    "step-2000", checkpoint_directory + "/parameters", tf.io.gfile)
# The controller saves its dense model, optimizer and input position, then
# publishes its existing joint checkpoint commit marker.
backend.resume("step-2000")
```

Export keeps admission closed. The parameter manifest identifies the sparse
files; the training controller owns the joint commit with dense and input state.
The [restore stager](ENTROPY_RESTORE.md) streams committed shard files into local
storage and starts fresh servers. Original restore hooks verify loaded snapshot
identities before restoring dense/input state through the existing framework.

## Lifecycle verification

The core test compares eight sparse updates with uninterrupted training across
two save/resume boundaries and a restore from exported bytes. It checks exact
weights, Adam moments and row counters, stale run/epoch/ID rejection, seven-byte
chunk reads, monotonic request/preparation identities and partial-save fencing.
A failed resume acknowledgement is retried while local input stays closed.

The TensorFlow graph test executes Pull, model loss, gradient Push and Join
against two external parameter servers. Mid-run it exports through `tf.io.gfile`,
saves dense optimizer and input state, publishes a joint marker, resumes, and
continues without retracing. Corrupting a disposable snapshot's checksum makes
export fail before the parameter manifest is published. Filesystem contract
tests cover retry, immutable conflicts, short writes and malformed chunks.

ThreadSanitizer exposed a shutdown race when a partial save left accepted work
running during service destruction. Public service, Host/CUDA cache and Agent
destructors now drain their executors while their implementation pointers remain
valid. A gated update plus an accepted checkpoint verifies that destruction
waits for both continuations.

```sh
ctest --test-dir build/tensorflow --output-on-failure \
  -R '(periodic_checkpoint_lifecycle|checkpoint_filesystem_contract|tensorflow_graph_periodic_checkpoint|resumable_shard_operations)'
# On a CUDA-enabled build, select the actual CUDA Agent for the graph check:
EPS_CACHE_BACKEND=cuda ctest --test-dir build/cuda -V \
  -R tensorflow_graph_periodic_checkpoint
```

The [original table/Runner hooks](ENTROPY_TABLE_HOOKS.md) now install this lifecycle
at the original source boundaries. Their source-level Session checks preserve
the original directory preparation, dense/input save flow and gradient dispatcher.
The native original-application build, original-reader/multi-peer framework
execution and in-process device-tensor integration remain separate work.

Executed checks at source `dc72662`: 30/30 local native/RPC/TensorFlow,
15/15 ASan/UBSan, 15/15 ThreadSanitizer, 24/24 A6000 native CPU/CUDA,
33/33 complete native/CUDA/TensorFlow, and 6/6 forced-CUDA lifecycle cases.
The [validation artifact](validation/2026-11-01-periodic-checkpoints.json)
records source/binary/log hashes and verified rental deletion.
