# Distributed training-generation retirement

Long-running training accumulates parameter-replica logs, original-gradient
journals and checkpoint exports. After a replacement generation owns an
independent durable baseline, the manager retires those obsolete payloads on
their storage hosts. Workers retain the current input position, model state and
optimizer progress while older recovery data leaves the active storage window.

## Roles and ordering

```mermaid
sequenceDiagram
    participant M as Training manager
    participant A as Storage node A
    participant B as Storage node B
    participant C as Checkpoint controller
    M->>M: Verify replacement checkpoint and replica authority
    M->>A: Seal source journals and primary logs
    A-->>M: Durable certificate and writer fences
    M->>B: Seal source mirrors and replica logs
    B-->>M: Durable certificate and writer fences
    M->>C: Certify immutable checkpoint exports
    C-->>M: Durable certificate
    M->>M: Fsync all-participant commit decision
    M->>A: Delete certified payload paths
    M->>B: Delete certified payload paths
    M->>C: Delete certified checkpoint paths
    A-->>M: Replayable completion receipt
    B-->>M: Replayable completion receipt
    C-->>M: Replayable completion receipt
```

The manager owns the immutable source-resource assignment and group decision.
Each participant opens only its local files. Persistent SSH streams carry the
control messages; checkpoint and journal payloads remain on their hosts.

Worker and gradient-mirror inspection uses the production journal codec,
input/model identity and exact retained round interval. Native replica
inspection verifies the checkpoint frontier and checksums. Primary-parent and
storage locks exclude concurrent writers. A local certificate records the
inspected file identities before retirement markers fence the old roots.

A controller holding only published immutable checkpoint exports needs no
stopped training-process handle. Mutable worker, mirror and replica storage
requires terminal local writer handles, including the host and process birth
recorded by the role supervisor, plus exclusive storage locks.

## Interrupted operation

Sealing publishes no deletion. If a participant is unavailable, previously
sealed payloads remain present and the group has no commit decision. Once all
sealing receipts are durable, the manager publishes the commit decision before
sending any delete request.

Deletion completes per participant. Each node checks its certificate, marker
and remaining file identities, removes the certified names, fsyncs their parent
directories and publishes a completion receipt. Notes and other unregistered
files remain. A changed payload rejects the operation before the first unlink.

Restarting the same manager command reuses the immutable intent. Lost successful
replies return the existing node receipt. After the group decision is durable,
reentry uses those certificates without reopening replacement-authority RPCs;
completed retirement remains replayable after later generation cleanup. A
transport timeout leaves that participant unresolved.

## Deployment

An uninterrupted two-node managed training run can export the source generation
directly:

```sh
python scripts/managed_training_recovery.py \
  --fleet /private/training-fleet.json --run training-prefix \
  --out /controller/training-prefix --export-checkpoint
```

After both Workers finish, the native controller pauses admission and streams
the joint parameter/optimizer checkpoint to the controller. The manager then
stops its owned processes and publishes
`/controller/training-prefix/source-generation/source.json`. This descriptor
contains the source job's worker, primary, replica and mirror configurations,
exact resource ownership, and per-host terminal process handles. Remote journal
payloads remain on their original hosts. The controller checkpoint is a separate
immutable participant.

Reentry adopts the recorded training phases. A checkpoint published before the
manager recorded its phase is validated through the native snapshot reader and
matched against the input progress before adoption. The export descriptor omits
SSH authentication and connection fields. Join its deployment node IDs with the
original private fleet transport configuration, and provide local Python/source
paths for the checkpoint-controller participant. Nonlocal training nodes must
have explicit stable `node_id` values. This export entry currently covers a
fixed-placement, uninterrupted generation; fault recovery uses the existing
managed recovery entry.

First use the existing checkpoint-rollover planner and readiness verifier to
publish the replacement `plan.json` and `ready.json`. Supply a private deployment
file containing:

| Field | Meaning |
| --- | --- |
| `nodes` | Stable node IDs, Python/source/bin paths, retirement-state directories and SSH connection options |
| `resource_nodes` | Exactly one owner for every source resource key |
| `handles` | Per-node terminal source writer handles from the role supervisor |

The standard source keys are `worker-N`, `mirror-N`, `primary-N`, `replica-N`
and `checkpoint`. Repeated rotations also register `staged-checkpoint` for the
former generation's initial baseline. Map each key to its actual storage host.
Checkpoint-only controllers can provide an empty handle list.

Node descriptors use the existing `NodeSessions` transport fields: `source`,
`python`, `host`, `port`, `user`, `identity_file` and `known_hosts`; local
participants use `local: true`. Add `node_id`, `bin_dir` and
`retirement_directory`. Every participant needs the project modules, NumPy and
the native inspection binaries. Generation-specific state directories and
small receipts remain available for retries. The persisted group intent
excludes SSH identity and connection fields.

```sh
python scripts/distributed_retirement.py \
  --plan /controller/new-generation/plan.json \
  --deployment /private/retirement-deployment.json \
  --build /controller/eps-build
```

The host-local `rotate_training_checkpoint.py --action retire` entry remains
available and shares the same storage inspector.

## Continue training after retiring a remote source

`continue_training_generation.py` accepts the exported source and original
private fleet. It stages an independent checkpoint on the controller, starts
replacement primaries, replicas, owners, gradient mirrors and coordinator,
rehydrates the parameter owners, and verifies readiness before retiring source
payloads. Workers resume from the saved input cursor and journal baseline.
The replacement currently executes on the controller CPU; the source can use
remote GPU owners and model Workers.

```sh
python scripts/continue_training_generation.py \
  --source /controller/training-prefix/source-generation/source.json \
  --fleet /private/training-fleet.json --build /controller/eps-build \
  --out /controller/continued-generation --run training-continued \
  --end-round 8 --lose-commit-reply
```

The optional fault drops successful responses from one participant's commit
request until the retirement manager exhausts its transport retries. Reentry
uses the persisted group decision and node receipts, then training resumes.
Each source participant has its own state directory even when two logical
Workers share a storage host. SSH keys stay on the controller.

The end-to-end comparison command runs an uninterrupted reference and a split
training job, then compares every small-model row's weights, Adam moments,
update counts, loss trajectory and final input cursors:

```sh
python tests/training_generation_continuation.py \
  --fleet /private/training-fleet.json --build /controller/eps-build \
  --work /controller/continuation-trial --out /controller/continuation.json \
  --run continuation-trial --tolerance 0.00001
```

Omit `--fleet` to execute the local process regression with zero tolerance.
The output records source host identities, GPU UUIDs, SSH participant count and
the number of distinct hosts that actually acknowledge retirement. The
zero-tolerance CPU regression passes at clocks `[8, 8]`, with all 32 rows and
losses matching the continuous reference exactly. Remote GPU/CPU comparisons
record their explicit numerical tolerance.

## Executed verification

The full CPU TensorFlow model regression uses two parameter shards, two Workers,
replicated optimizer updates and three separate retirement participants: two
training-storage processes and the immutable-checkpoint controller. Replacement
readiness is verified by real native checkpoint and replica RPCs. The test
interrupts sealing, loses successful commit replies, reenters the manager and
then continues the model.

Two rotations at rounds 4 and 8 remove the old payloads. At committed round 10,
SIGKILL removes Workers, primaries, owners and coordinator before all task
completions arrive. Surviving replicas rebuild the state and training finishes
at clocks `[12, 12]`. All 32 model rows' weights, Adam moments, counters and
continued losses match uninterrupted training with maximum measured error zero.
The older completed receipt is replayed after the next generation's cleanup.

The protocol checks additionally cover a live local process or journal lock,
future gradient, omitted resource, interrupted node seal/unlink, changed payload,
changed completion receipt, and preservation of unrelated files.

The managed source-export check runs two real TensorFlow Workers, two parameter
shards and opposite-node gradient/parameter replicas for four rounds. It
interrupts the manager after checkpoint publication, reenters, validates the
checkpoint and every participant's retained storage, and checks terminal process
handles. A changed input cursor cannot adopt the checkpoint. This local check
retains 20 certified payload paths on each training participant and three on the
checkpoint controller; it performs no retirement or GPU execution.

```sh
ctest --test-dir build/tensorflow --output-on-failure \
  -R '^(distributed_retirement_protocol|tensorflow_distributed_retirement|tensorflow_distributed_retirement_recovery)$'
```

The checks above use independent participant processes on one physical host.

## Actual SSH-host model execution

A rented NVIDIA L40 and the controller execute two source placements. Both use
two logical Workers, two parameter shards, cross-owner gradient/parameter
replicas, 32 complete parameter rows and eight training rounds. A joint round-4
checkpoint starts an independent controller CPU replacement; the former hosts
retire certified payloads over SSH before the model continues. One successful
commit response is deliberately lost through all manager retries, then replayed
from durable receipts. All final input cursors and per-row update counts match
the uninterrupted reference.

| Source placement | Training hosts | Physical GPUs | Maximum weight/Adam error | Maximum loss error |
| --- | ---: | ---: | ---: | ---: |
| Two Workers/shards on rented L40 host | 1 | 1 | 2.98e-8 | 5.96e-8 |
| L40 Worker/shard plus controller CPU Worker/shard | 2 | 1 | 2.98e-8 | 5.96e-8 |

Retirement receipts come from two actual hosts in both cases. Each removes 43
certified old payload paths while preserving the staged replacement baseline.
The floating-point acceptance tolerance is 1e-5. These are correctness and
continuation trials; fixed-workload physical multi-GPU scaling remains a separate
acceptance item. The second L40 rental request was rejected for unavailable
inventory, and the first instance was deleted after both successful trials.

The Linux Python client also passes direct TensorFlow/native co-loading in both
import orders after localizing its static BRPC/protobuf symbols. The normal
training adapter continues to isolate its native client in a dedicated process.

[Cross-host execution, native-runtime evidence and cleanup](validation/2026-11-05-cross-host-training-continuation.json).

[Commands, source hashes and measured results](validation/2026-11-05-distributed-training-retirement.json).

[Managed source-export validation](validation/2026-11-05-managed-training-source-export.json).
