# Validation record

## Executed correctness checks — October 31, 2026

| Environment | Executed suite | Result |
|---|---|---|
| macOS arm64, AppleClang 21, BRPC 1.18.0, Protobuf 36.2 | Current native core, RPC, cache protocol, adapter, both gather policies and TensorFlow graphs | 14/14 passed |
| macOS arm64, core-only ASan/UBSan | Current core, policies, ownership, control replay, adapter and coalesced gathers | 8/8 passed |
| Ubuntu 22.04, GNU 11.4, BRPC 1.18.0, Protobuf 3.12.4, CUDA 13.0.88, L40 48 GB | CPU/CUDA suite before cross-shard gather coalescing | 12/12 passed |
| Ubuntu 22.04, A6000 coordinator, shared server/agent at `5a4cdcb` | Initial native CPU/CUDA checks | 13/13 passed |
| Ubuntu 22.04, current client at `9f36ca3` | Native CPU checks | 11/11 passed |
| Ubuntu 22.04, current client, CUDA agents | Two-shard process lifecycle, both gather policies; fresh and optimizer-exact restored state | Both modes passed |
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
| Cross-shard cache gathers | Two shards and three reference agents cover mixed resident/transfer/CPU routes, repeated IDs, empty bags, sum/mean/sequence, one gather per destination, concurrent dispatch and matching weights/moments/counters; 23 input occurrences with four CPU-only positions send exactly 19 keys across all destination layouts |
| Gather failure cleanup | Missing/malformed/failing resolvers and partial shard capacity rejection release known successful tickets; a promise-backed completion proves a fast destination failure waits for other gather futures |
| Per-shard overlap | A deliberately delayed second shard cannot reply until the first shard's gather starts; the default path completes and matches CPU features, while explicit coalescing separately retains one gather per destination |
| Configured backpressure | With a live-ticket limit of 32, 64 concurrent Learners admit 32 and reject 32; cached reads and row pins remain valid, and a rejected Learner retries successfully after releases |
| Cache control | Original command replay, retired-command fencing, slot reclamation after acknowledged writeback and ownership generation checks |
| Mixed placement | CPU and cache-owned portions of one Push share the same original optimizer context; moving a referenced row is rejected |
| CUDA execution | Multi-table gather, mean partials and sequences, complete optimizer state, overflow rejection, eviction/reload generations and device results that outlive staging/backend reuse |
| Compact gradient updates | Ordered GPU reduction of feature-gradient references, cross-feature and mean semantics, overflow rejection, duplicate command replay and optimizer trajectory through RPC save/restore |
| Cold access route | Unique CPU-owned weights can be transferred for GPU pooling without admitting full optimizer state; compared with the host reference |
| CUDA process lifecycle | Two PS processes plus a CUDA cache process exercise mixed reads/updates, two-way migration, cache-draining snapshot and restore |
| Embedding adapter | Padding exclusion, valid mean denominators, sequence scatter/gather, completed-handle replay and Join ordering |
| TensorFlow graph | TensorFlow 2.20.0 graph with two external PS processes, dense variable/optimizer, five sparse updates, evaluation and two-shard save; loss decreases without retracing; separate CPU-only and transferred-vector/cache-RPC host-reference routes |

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
this software/configuration/workload combination. Completed shard, batch-size
and CPU-only transport controls expose the communication/coordinator path as the
next scaling target.

The new fleet completes **253 capacity trials and 153,703 updates with zero
reported errors**. An additional 207 trials from the initial deployment remain
archived. At three GPUs, four shards and eight Learners, three 30-second trials
complete 10,841 updates at **122.95 steps/s** median and **104.93 ms** median trial
P99. Controls extend to 128 Learners, sixteen shards and a million-row,
512-dimensional table with 6 GiB of logical weights and Adam moments.

The projected candidate also performs an untimed numerical preflight per fresh
topology. It captures 96 sampled CPU-owned rows before migration and compares
sum/mean/sequence features after migration, including repeated IDs and empty
bags. The small workload checks 17,024 values; the 512-dimensional control
checks 68,096. Maximum absolute error across the new candidate records is
**4.470348358154297 × 10⁻⁸**. Preflight precedes warmup and does not advance the
optimizer or timed-input RNG.

The [benchmark report](BENCHMARKS.md) contains within-fleet paired comparisons,
1/2/3-GPU curves, shard and transport controls, source provenance and the limits
revealed by the tests. Its data index links every raw manifest and summary.
The [validation artifact](benchmarks/2026-10-31-capacity-controls/validation.json)
records suite counts, source revisions and source-log hashes. Commands are
documented in [RUNNING.md](RUNNING.md).
