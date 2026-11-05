# Original input checkpoints

The input checkpoint joins the original FileFeeder/ORC iterator to the parameter,
dense optimizer and training-progress commit. A restored job resumes its saved
batch boundary, including records already read into pipeline buffers.

## Enable capture

Install the native source overlay into a disposable original source bundle and
compile its CPU/GPU targets. Native capability version 3 adds prepared-batch,
EOF and partition-notification state to the original iterator serialization.
The overlay preserves original source bytes and records input/output hashes.

Add this option to the native parameter-adapter configuration:

```json
{
  "input_checkpoint": {
    "planner": "/path/to/eps_input_checkpoint_tool",
    "max_retries": 20
  }
}
```

The native backend and training coordinator must also be configured. Add
`"restore": true` for a fresh process that will restore a committed input group.
This holds FileFeeder allocation before the original Runner initializes input.

## Save order

```mermaid
sequenceDiagram
    participant R as Original Runners
    participant P as Parameter/progress services
    participant I as Input iterators
    participant C as Chief
    R->>P: Drain updates and freeze admission
    R->>I: Serialize buffers, prepared batch and ORC cursors
    I-->>C: Hashed per-worker manifests
    C->>C: Build remaining-assignment plans
    R->>R: Save original dense/model state
    C->>C: Commit parameters + progress + input-group digest
    C->>P: Resume admission
```

Each worker manifest identifies its input scope, source handle, input
specification, input shot and next-shot index. Its files contain the serialized
TensorFlow iterator and native stream capture. The chief binds all worker
manifests, ordered source plans and completed training clocks into one input
group. The joint commit stores that group's digest.

File assignment IDs distinguish repeated occurrences of the same filename.
The planner excludes assignments retained by restored readers and buffers;
speculative allocations outside the saved cut return to the remaining queue.
Completed reader histories survive leaf destruction until the next generation.

## Fresh-process restore order

1. Start model-process GPU owners before restored shards register their tables.
   `entropy_launch --startup-gate gate.json` publishes owner readiness and waits
   for a release record containing the same run, agent, endpoints and nonce.
2. Verify the joint commit, group digest, worker membership, input specification
   and payload hashes before the global verification barrier.
3. Initialize the saved input shot while source allocation remains held.
4. Restore ordered source assignment plans, then release source holds.
5. Restore original dense/model state, iterator buffers, ORC positions and
   Python input-shot position. Resume training progress after the final barrier.

Gate configuration uses fresh local paths:

```json
{
  "schema_version": 1,
  "ready": "/private/run/owner-ready.json",
  "release": "/private/run/parameters-ready.json",
  "timeout_seconds": 120
}
```

The orchestrator reads the ready record, starts the configured parameter shards,
waits for their readiness and publishes the identical record to the release
path. A release from an earlier owner generation is rejected.

## Executed checks

The original native TensorFlow C API verifies six-stream prepared-input recovery
at start, mid-stream and EOF, including held fresh initialization and a larger
prefetch buffer. The manifest tests reject missing workers, changed training
clocks, changed manifests and changed payloads; chunked staging covers a
1.5 MB payload. The startup-gate process test verifies generation identity.

The complete input-aware main/Executor save passes with two original training
peers at completed clocks `[16, 16]` (128 records, 32 batches on one physical
GPU). Fresh-process mid-file restore and post-checkpoint failure replay also
pass through the original application entry point. Custom global shuffle and label-partition
buffers require their own serialization before this mode accepts those input
configurations.

## Full-application interruption and replay

The recovery fixture runs the original `main.py`, Executor, reader, decoder,
graph builder, GPU forward/backward and native dense optimizer. A validation
entry hooks the existing checkpoint boundary and records sample marks; the
original training loop continues to execute every model batch.

```sh
python scripts/validate_original_executor_recovery.py \
  --native-root /path/to/matching-native-environment \
  --build /path/to/eps-build \
  --op-library /path/to/eps-tensorflow-ops.so \
  --peers 2 --uncommitted-replay \
  --work /private/new-recovery-run \
  --out /private/new-recovery-result.json
```

Both peers commit after two batches each. The fault run executes another batch
before losing a peer. Because the original dense parameter service lives in
those peers, the manager aborts old task admission and stops all old peer
processes. It creates fresh parameter shards, GPU owners and a paused
coordinator using the committed clocks, restores the original model and input,
and resumes training.

Before continuing, the restored job makes another original Runner checkpoint.
Both sparse partitions' complete weights, Adam moments, beta powers and update
counters match the saved cut with maximum absolute error zero. Native
DenseAdam checkpoint files are byte-identical. The two-peer run replays eight
uncommitted sample attempts and finishes all 128 committed sample marks with
no omissions or duplicates, at clocks `[16, 16]`. The validation distinguishes
discarded attempts from the final committed input trajectory.

An additional synchronized-Pull case delays one peer before task completion.
At failure, the old coordinator observes clocks `[3, 2]` and one admitted
unfinished task. The manager aborts that old run and stops its native peers;
the fresh run restores committed clocks `[2, 2]` and finishes at `[16, 16]`
with zero active/waiting tasks, unchanged saved optimizer state, and complete
sample coverage. Select `--uncommitted-replay --inflight-delay-ms 1500` to run
this case.

The one-peer continuation additionally matches an uninterrupted reference:
all four losses have absolute difference zero, both complete sparse/Adam
partitions have maximum error zero, and the final native dense optimizer files
are byte-identical. Pass both `--reference-checkpoint` and `--reference-report`
to request this comparison with a matching uninterrupted run.

[Two-peer fresh restore](validation/2026-11-05-original-executor-recovery-two-peer.json) ·
[Uninterrupted equivalence](validation/2026-11-05-original-executor-recovery-reference.json) ·
[Post-checkpoint failure replay](validation/2026-11-05-original-executor-uncommitted-replay.json).

[In-flight native peer failure](validation/2026-11-05-original-executor-inflight-replay.json).
