# Joint training checkpoint and restart

A training restart restores the same model **and the next input batch**. The
joint checkpoint couples parameter weights, Adam moments and optimizer clocks
with each shard's gradient round, the coordinator's completed worker clocks,
model configuration and worker input positions.

[Replica/journal generation rollover](TRAINING_LOG_ROLLOVER.md) uses the committed
checkpoint as a new full baseline. `verify-job` checks snapshot provenance and
the restored training group while admission remains paused; `resume-job` opens
the validated generation.

```mermaid
sequenceDiagram
    participant W as Model workers
    participant C as Job controller
    participant T as Task coordinator
    participant P as Parameter shards
    participant F as Checkpoint directory
    W->>W: Finish round N-1; retain next input position N
    C->>T: Pause and export completed clocks
    C->>P: Pause empty gradient collectors
    P-->>C: Next round N; optimizer sequence N
    C->>P: Drain and save weights + Adam state
    C->>F: Export bounded chunks; verify and fsync files
    C->>F: Publish manifest.json last
    Note over C,P: Start fresh processes with a new run identity
    C->>P: Restore matching parameters and gradient clocks, paused
    C->>T: Restore completed clocks, paused
    C->>P: Rehydrate existing rows on assigned owners
    C->>P: Verify all shards, then resume groups
    C->>T: Resume task admission
    W->>W: Continue input sequence at N
```

## Save a completed round

Stop scheduling new input after a chosen complete round and collect each worker's
`input_state` from its actual training report. `save-job` requires equal completed
counts, no active training tasks, empty gradient collectors and matching optimizer
sequences. An incomplete round cannot be checkpointed.

A save configuration supplies `run`, `epoch`, `coordinator`, ordered `shards`,
exact `tables`, `model`, `workers` and a fresh `checkpoint_directory`. `workers`
is an array of `{ "worker": ID, "input_state": REPORT_INPUT_STATE }` records.
For the executable TensorFlow example, the model descriptor is:

```json
{"kind": "embedding_tanh_mlp", "dimension": 8, "hidden": 4}
```

Each input state contains a deterministic cyclic batch sequence, its SHA-256,
next iteration, next batch and model descriptor. The save controller verifies
that its position equals the completed training round. Applications with another
input reader need a reader-specific input-state codec.

```sh
build/eps_training_control --config save-job.json --action save-job
```

The command creates one immutable directory. It streams each parameter snapshot
in chunks of at most 1 MiB, verifies its native checksum, computes SHA-256 and
fsyncs the output. `manifest.json` is published after all shard files and metadata
validate. A partial export has no committed manifest. Save failures leave task
admission paused for inspection; an existing checkpoint directory cannot be
overwritten. Use a fresh directory for the next checkpoint or a failed attempt.

Parameter snapshots retain complete table definitions, weights, both Adam
moments, row update counts and optimizer sequences/beta powers. Existing GPU
resident state is drained to its distributed CPU backing before serialization.
This means a save can temporarily evict the owner's resident rows; it does not
preserve live CUDA allocations as a file image.

## Restore and reopen training

1. Generate configurations with a fresh `run` and the committed `manifest` path:

   ```sh
   build/eps_training_control --config restore-job.json --action restore-config > restored.json
   ```

   The output contains `coordinator`, ordered `shards`, `model` and worker input
   states. It verifies every parameter file before returning configurations.

2. Launch each `eps_server` with its generated shard configuration and
   `--restore FILE`, using the output's `restore_file` as `FILE`. Attach fresh
   cache agents through the normal `cache`/`agents` configuration if using GPU
   parameter owners. Configure state-memory limits for the actual shard size.
   Launch `eps_training_coordinator` with its generated configuration. Both
   parameter and training admission remain paused.

3. Rehydrate existing rows before opening training, when GPU residency is
   required. `eps_training_prepare` accepts the normal preparation settings plus
   `"restore_existing": true`. It checks restored provenance, a paused gradient
   group and the loaded table dimension. It only migrates existing rows, never
   initializes absent rows or consumes training read tickets. A retry skips rows
   already on the assigned owner and verifies complete residency counts.

4. Supply the fresh `run`, actual coordinator/shard addresses and committed
   `manifest` to the activation command:

   ```sh
   build/eps_training_control --config resume-job.json --action resume-job
   ```

   It checks all restored file identities, table definitions, optimizer/group
   rounds and coordinator clocks, then resumes every group before task admission.
   A swapped shard, wrong source or mismatched clock fails verification.

5. Start each worker with its saved `input_state`, `start_iteration` equal to the
   saved next iteration, the same `batches`, model and table definition. The
   example's `iterations` specifies how many additional rounds to execute.
   A dataset, model or cursor mismatch fails before opening the parameter client.

Pre-checkpoint gradient iterations sent under the new run identity return
`retired_request`. A completed gradient retry from the new run keeps its ordinary
idempotent acknowledgement. New run identities separate pre-restart read-ticket
and transport histories from restored model progress.

## Verification

```sh
ctest --test-dir build --output-on-failure -R 'synchronous_gradient_(rounds|rpc)|tensorflow_joint_training_checkpoint'
```

The process test uses two parameter shards, two Host-backed cache owners and two
TensorFlow model workers. It fully loads a 32-row table, executes eight rounds
on changing input batches, and compares uninterrupted training with a save at
round four followed by fresh processes, owner rehydration and four further
rounds. The split is inside a three-batch cycle. It compares all table rows'
weights, moments and update counts, continuation losses and exact input states.

Negative checks cover incomplete/mismatched rounds, changed optimizer settings,
wrong snapshot identity, corrupt/missing files, altered SHA-256, wrong table
schema, wrong input fingerprint/cursor, reversed live shards, immutable
publication and an injected partial-export write failure. The core lifecycle is
also exercised under address/undefined-behavior and thread sanitizers.

The executed Host-backed run matches all 32 rows and all continuation losses
exactly (maximum absolute differences: 0). The 35-check native suite and final
coverage of 53 framework checks pass, including retests of the affected model
and node-orchestration paths. [Source hashes, reports and verification logs](validation/2026-11-04-joint-training-checkpoint.json)
retain the exact execution scope and the corrected preparation-binary retest.

This protocol provides coordinated checkpoint restart. Replica-before-ACK,
crash-persistent in-flight task replay and fenced live owner replacement are
tracked in the [distributed-training implementation plan](plans/2026-11-04-distributed-training-design.md).
