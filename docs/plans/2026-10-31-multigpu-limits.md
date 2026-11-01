# Multi-GPU capacity and automatic-cache validation

Compare 1, 2 and 4 distinct CUDA devices with fixed total parameters, resident rows, request dimensions and coordinator CPU allocation. Sweep Learners until throughput plateaus or falls while P99 and queue wait rise; confirm the useful knee and an overloaded point with three longer runs. Keep process/RPC time, CPU utilization, GPU utilization and VRAM samples.

Generalize the fixed-trace policy experiment to multiple cache agents. Replay identical stable, shifted and scan-mixed inputs against CPU/static/frequency/reuse-cost modes, then compare all touched weights, both Adam moments and update counters with an independent CPU replay. Check each agent returns its full slots after drain. Calibrate saved-access cost independently per topology.

Use a separate worktree so ongoing asynchronous-preparation edits remain intact. Do not combine old deployment rates with the new fleet. Publish source/binary/raw hashes, workload definitions and negative results. These are parameter-service steps with synthetic gradients; dense-model forward/backward and physical GPU-memory exhaustion are outside this run.

Run local Host protocol checks before rental. Use existing authorization, a bounded lease and a deletion guard; retrieve artifacts and delete all rented resources after execution.

## Deployed topology

The authorized credit supports single-GPU A6000 instances. This round uses four
such instances: the coordinator runs four local PS processes and one GPU agent;
additional agents use host-to-host SSH-forwarded BRPC. All topology sizes retain
the same six-vCPU coordinator allocation. Every agent binary hash is compared
and all four CUDA UUIDs must be distinct. A matched Host cache-backend control
and selectable gather coalescing are measured after the main capacity schedule.
Network transfer controls run on the same loopback/forwarded endpoints.

The independent policy traces keep 10,240 total configured slots and 8,192
initially resident rows; each agent reserves 32 slots. Each GPU count repeats
CPU/static calibration and two blocks of all four cache modes. Numerical replay
checks all touched rows after drain.
