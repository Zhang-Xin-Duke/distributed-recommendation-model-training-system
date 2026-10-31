# Validation record

## Executed correctness checks — October 31, 2026

| Environment | Executed suite | Result |
|---|---|---|
| macOS arm64, AppleClang 21, BRPC 1.18.0, Protobuf 36.2 | Native core, RPC, cache protocol, adapter and TensorFlow graph | 10/10 passed |
| macOS arm64, core-only ASan/UBSan | Core, policies, ownership, control replay and adapter | 7/7 passed |
| Ubuntu 22.04, GNU 11.4, BRPC 1.18.0, Protobuf 3.12.4, CUDA 13.0.88, L40 48 GB | Native CPU/CUDA suite | 11/11 passed |
| Three separately rented RTX A6000 48 GB devices, compute capability 8.6 | CUDA state/device-result check on each device | All three passed; distinct GPU UUIDs verified |

The L40 driver was 610.43.02, compute capability 8.9. CUDA code was subsequently
built for both 86 and 89 to run the same backend on the A6000 workers. The Ubuntu
image lacked a compiler toolkit; project-local NVIDIA pip components supplied it.
The provider driver and preload library were retained. BRPC's optional pthread
contention-profiler hook was disabled because its bootstrap conflicted with that
preload library.

## What the checks establish

| Check | Evidence covered |
|---|---|
| Core correctness | Sum/mean/sequence, empty bags, repeated IDs across features, gradient validation, row references and request retry suppression |
| Optimizer and storage | Numerical preflight before mutation, weights/moments/counters, bounded state/tickets, corrupt snapshot rejection and continued optimizer trajectory after restore |
| Concurrent Learners | Eight caller threads alternate training/evaluation and retry Push; shared rows receive the expected 192 updates |
| Real PS processes | Two BRPC shard processes reconstruct features and updates against a single-shard reference; restart under a new run rejects old-run requests |
| Cache control | Original command replay, retired-command fencing, slot reclamation after acknowledged writeback and ownership generation checks |
| Mixed placement | CPU and cache-owned portions of one Push share the same original optimizer context; moving a referenced row is rejected |
| CUDA execution | Multi-table gather, mean partials and sequences, complete optimizer state, overflow rejection, eviction/reload generations and device results that outlive staging/backend reuse |
| Compact gradient updates | Ordered GPU reduction of feature-gradient references, cross-feature and mean semantics, overflow rejection, duplicate command replay and optimizer trajectory through RPC save/restore |
| Cold access route | Unique CPU-owned weights can be transferred for GPU pooling without admitting full optimizer state; compared with the host reference |
| CUDA process lifecycle | Two PS processes plus a CUDA cache process exercise mixed reads/updates, two-way migration, cache-draining snapshot and restore |
| Embedding adapter | Padding exclusion, valid mean denominators, sequence scatter/gather, completed-handle replay and Join ordering |
| TensorFlow graph | TensorFlow 2.20.0 graph with two external PS processes, dense variable/optimizer, five sparse updates, evaluation and two-shard save; loss decreases without retracing |

The TensorFlow check exercises the portable host-callback bridge. It does not
rebuild the full private Entropy application. Model/device-native integration
and a complete application training-quality evaluation remain outstanding.

## Checked CUDA read stress

The larger L40 run used:

| Item | Value |
|---|---:|
| Parameter rows | 1,048,576 |
| Vector dimension | 128 FP32 |
| Weight array | 512 MiB |
| Device weights + Adam moments | 1.5 GiB |
| Concurrent callers | 16 |
| Checked batches | 2,048 |
| ID occurrences | 8,388,608 |
| Bag length | 16 |
| Maximum absolute error | 0 |
| Read-phase backend output allocations | 0 |

Every pooled result was compared with an independent CPU formula. The workload
chooses 95% of IDs from a 32,768-row hot set, with the remainder from the complete
table. The checked wall time includes input preparation and CPU comparisons and
is not a pure-kernel or training-throughput measurement. Backend allocation
counters are used for allocation assertions; device-wide free memory can also
change because of the driver, provider and other device users.

## Capacity evaluation

`scripts/capacity_sweep.py` and `scripts/fleet_capacity_sweep.py` run the same
acknowledged sparse-update workload on one or more agents. Timed steps include
distributed lookup, feature reconstruction, gradient payload preparation and
completed sparse updates. Dense-model computation is outside this subsystem
benchmark.

The generator checks sampled per-row update counts and finite state after every
load level. Results include successful updates/s, ID occurrences/s, latency,
sequencer occupancy and queue delay. Fresh topologies use the same initialization,
row count, cache fraction, dimensions and workload distribution. Kernel warmup
and admission are outside timing; each load level is repeated.

The current fleet uses one A6000 per instance and direct host-to-host SSH
forwarding for BRPC. Measurements therefore include that transport and the
coordinator's CPU allocation. A measured plateau establishes the capacity of
this software/configuration/workload combination. Additional shape, CPU-shard
and transport controls are needed before attributing it to GPU compute.

Commands are documented in [RUNNING.md](RUNNING.md). Raw capacity records and
the resulting figures are added after the complete sweep is checked.
