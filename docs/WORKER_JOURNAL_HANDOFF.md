# Worker journal handoff between nodes

A recovery manager can retrieve stopped workers' original-gradient journals,
reconcile replica-backed training, and stage the journals on replacement worker
nodes. Workers do not need a shared filesystem. The handoff preserves original
FP32 gradient bytes, uint64 Keys, per-round loss and completed-read frontiers.

```mermaid
sequenceDiagram
    participant O as Former worker node
    participant M as Recovery manager
    participant P as Parameter replicas / replacement primaries
    participant N as Replacement worker node
    Note over O: Worker exits; journal lock is released
    M->>O: Seal journal with transfer identity
    O-->>M: Immutable file manifest and hashes
    loop Bounded chunks on one persistent connection
        M->>O: Read sealed file at offset
        O-->>M: Original binary bytes and chunk hash
        M->>M: Validate and publish complete local journal
    end
    M->>P: Reconcile original gradients and durable round
    P-->>M: Common round and replacement owner identities
    M->>N: Stage reconciled journal through the same chunk protocol
    N->>N: Validate full journal; atomically publish destination
    M->>N: Start worker at recovered input/read position
    N->>P: Resume Pull / forward / backward / gradient update
```

## Transfer lifecycle

**Seal the stopped source.** The node acquires the same exclusive lock held by a
running model worker. It validates the journal, records the ordered identity and
gradient files with sizes and SHA-256 hashes, then durably publishes a seal bound
to the transfer identity. A live worker's lock prevents sealing. After sealing,
ordinary worker opens and every journal write reject that source directory.
Recovery tools can still read a sealed journal to repeat reconciliation.

**Move exact records.** The protocol reads at most 256 KiB per chunk. A persistent
JSON stream carries the binary data as base64; local subprocess and SSH nodes use
the same request/response protocol. Transport advances one bounded chunk at a
time; full record validation uses the journal's 32 MiB record limit. Each file
retains its original binary encoding and checksum. Repeated chunks compare the bytes already present; offset gaps and
changed duplicate contents reject. A disconnected stream reconnects and retries
the same operation, with at most three attempts.

**Publish a complete receiver.** Incoming files remain in an unpublished sibling
staging directory. The receiver checks its own per-record/disk budgets, file
hashes, run/model/input identity, contiguous rounds and monotonic read progress.
It then publishes a receipt and renames the complete directory into place while
holding the receiver's destination lock. Worker processes start only after the
manager receives completion. Repeated publication returns the matching receipt;
it preserves any later rounds already appended by the replacement worker.
An unrelated existing destination rejects without merging or replacing its data.

A stopped transfer resumes with the same transfer identity and paths. Sealing
is durable: after the source is sealed, continue or repair that handoff rather
than restarting a worker against the old directory. Chunk-level progress is
recoverable from receiver file lengths even when the manager lost its last
acknowledgement.

## Integrating training recovery

Use two handoffs around the existing recovery protocol:

1. Stop all former training roles as required by
   [worker recovery](WORKER_RECOVERY.md). Fetch each worker journal to a separate
   manager directory and use those returned worker configs in the recovery job.
2. Recover replacement primaries under fresh replica terms, plan the common
   round, start a paused coordinator and run `recover_training.py --action
   reconcile`. The manager reads the copied original gradients and handles a
   partially committed round exactly as before.
3. Transfer each returned worker journal to its replacement node. Start the
   returned worker config with the new `journal_directory`, recovered input/read
   position and parameter-owner identities. Resume task admission only after
   reconciliation and receiver publication succeed.

The Python `transfer` function returns the updated worker config and a compact
receipt. A private CLI plan can perform a whole worker group:

```json
{
  "transfer": "job-recovery-fetch-1",
  "workers": [{
    "config": {"journal_directory": "/worker/job/journal", "...": "original worker configuration"},
    "source_node": {"source": "/worker/project", "python": "/worker/env/bin/python", "...": "private SSH fields"},
    "destination_node": {"local": true, "source": "/manager/project", "python": "/manager/env/bin/python"},
    "destination_directory": "/manager/recovery/worker-0"
  }]
}
```

The ellipses stand for the full original worker config and normal fleet SSH
fields (`host`, `port`, `user`, `identity_file`, `known_hosts`). Keep this plan
private. SSH paths and credentials are not included in the CLI's result.

```sh
python scripts/transfer_worker_journals.py \
  --plan private-handoff-plan.json --out fetched-worker-journals.json
```

Source and destination need a prepared Python environment with NumPy and the
current `adapters/entropy` files. The tool rents no resources. It uses retained,
accessible source storage; loss of the only original-gradient journal still
requires a redundant journal copy. Parameter ownership remains fenced by replica
terms and replacement-owner identities. Automatic role discovery/supervision
continues in the distributed-training implementation plan.

## Executed verification

The protocol checks live-source exclusion, durable seals, identity/budget
validation, interrupted/repeated chunks, corrupted data, reconnect after a lost
successful write acknowledgement, exact uint64/FP32 preservation and an already
continued destination. A record larger than 1 MiB crosses multiple full-sized
chunks through a persistent node subprocess.

The full model test fetches journals into manager storage, reconciles there,
stages independent replacement-worker directories and continues eight rounds.
It injects a live-read interruption, a partial shard commit and a lost task
completion. Every one of 64 rows' full weights/Adam state and every worker loss
matches uninterrupted Host training exactly. It also repeats reconciliation
against the manager's sealed copy after staging and checks that former worker
journals cannot be reopened for training.

```sh
ctest --test-dir build/tensorflow --output-on-failure \
  -R '^(worker_journal_handoff|tensorflow_worker_journal_handoff)$'
```

These runs execute serialized node subprocesses with independent storage on one
host. Real cross-host SSH/CUDA fault execution retains a separate acceptance
requirement. [Validation record](validation/2026-11-04-worker-journal-handoff.json).
