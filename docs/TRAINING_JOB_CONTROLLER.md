# Multi-job training controller

The Python controller places parameter-server training jobs on 2–8 registered
compute nodes, using 2–8 fixed logical worker/shard groups per job. Model tensors
and gradients retain their direct C++/RPC paths. SQLite/WAL stores job intent,
state transitions, and events; a process lock permits one active controller.

## Admission and lifecycle

Queued jobs are scanned by descending priority and FIFO submission order within
a priority. Placement filters CPU, host memory, disk, GPU availability, and backup
separation, then scores bounded candidates by estimated transfer cost and memory
fragmentation. Cooperative node-local reservations are atomic. GPUs are exclusive
between jobs; CPU/RAM reservations do not provide operating-system isolation.
There is no priority preemption in this version.

The controller reserves the full placement before launching the managed executor.
Worker startup gates release after readiness. Reentry adopts durable executor
handles rather than restarting live work. Cancellation stops owned processes
before releasing reservations; unresolved outcomes retain their reservations.

Confirmed node loss triggers replacement planning. Recovery selects surviving
primary/replica state and worker/mirror journals, rebuilds redundancy, reconciles
training, and starts replacement roles on the new placement. Initial measurement
gates belong to the former process generation and are removed from replacement
worker configurations. Recovery admission follows the restored coordinator.

## CLI

Use prepared nodes with matching source, Python, native binaries, and native
module paths. Keep the inventory's connection details outside version control.
Job fields and the inventory fixture are demonstrated in
[`tests/training_job_controller_tests.py`](../tests/training_job_controller_tests.py).

```sh
python scripts/training_job_controller.py --state /private/controller \
  --inventory /private/inventory.json job plan /private/job.json
python scripts/training_job_controller.py --state /private/controller \
  job submit /private/job.json
python scripts/training_job_controller.py --state /private/controller \
  --inventory /private/inventory.json controller serve
python scripts/training_job_controller.py --state /private/controller job status
```

The CLI also provides job cancellation, node probes, node draining, and explicit
terminal-loss receipt recording. A transport timeout is not a terminal-loss receipt.

## Executed local checks

On November 5, 2026, the native-service/TensorFlow checks passed on **one physical
host with zero GPUs**:

- Atomic competing reservations, idempotent reserve retries, live-role release
  rejection, external-port conflicts, and durable submission identity.
- Real model training with **2, 4, and 8 logical groups**, concurrent jobs,
  priority/FIFO ordering, controller restart adoption, and cancellation cleanup.
- Four simulated node/storage scopes: destroy one scope, select surviving mixed
  parameter/gradient authorities, rebuild placement, restart the controller, and
  complete eight rounds. All 64 model rows, Adam state, update counters, and
  losses match uninterrupted training exactly; lost storage stays absent.

These are local process/network checks. They do not establish physical 4/8-node
or multi-job GPU execution, preemptive scheduling, controller-disk-loss recovery,
or recovery after loss of every authoritative state copy.

```sh
python tests/job_resources.py /tmp/job-resources.json
python tests/training_job_controller_tests.py --build build/tensorflow \
  --out /tmp/job-controller.json
python tests/job_controller_recovery.py --build build/tensorflow \
  --out /tmp/job-controller-recovery.json
```

Run these with the Python environment used to build the native TensorFlow client.
[Machine-readable check summary](validation/2026-11-05-training-job-controller-local.json).
