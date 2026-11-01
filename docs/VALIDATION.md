# Validation record

## Executed correctness checks — October 31, 2026

The [current multi-GPU study](CURRENT_MULTIGPU_LIMITS.md) records runtime
`b6d97d3` on four distinct RTX A6000 devices: **22/22** native CPU/CUDA checks,
**86 completed capacity trials / 52,173 verified updates**, and **2,752 sampled
row-update counters**. Mean/sum/sequence preflight has maximum absolute error
**3.91e-08**. Three Agent RPC admission failures are retained separately from
completed trials. The [validation artifact](validation/2026-10-31-current-multigpu.json)
includes raw-record hashes, failed points and verified rental deletion.

| Environment | Executed suite | Result |
|---|---|---|
| macOS arm64, AppleClang 21, BRPC 1.18.0, Protobuf 36.2, source `0661e52` | Complete native/RPC/TensorFlow suite, including dependency caches, invalidation and resumable operations | 26/26 passed |
| macOS arm64, core-only ASan/UBSan, source `0661e52` | Complete core suite | 13/13 passed |
| macOS arm64, ThreadSanitizer, source `0661e52` | Automatic cache, prepared inputs, resumable actor/service and concurrent owners | 5/5 passed |
| Ubuntu 22.04, GNU 11.4, A6000, CUDA 13.0.88, TensorFlow 2.20.0, source `0661e52` | Complete native/RPC/CUDA/TensorFlow suite | 29/29 passed |
| Same A6000 build, forced CUDA | Automatic cache, migration/save/restore, prepared graph and delayed resumable CUDA operations | 5/5 passed |
| Ubuntu 22.04, GNU 11.4, BRPC 1.18.0, Protobuf 3.12.4, CUDA 13.0.88, L40 48 GB | CPU/CUDA suite before cross-shard gather coalescing | 12/12 passed |
| Ubuntu 22.04, A6000 coordinator, shared server/agent at `5a4cdcb` | Initial native CPU/CUDA checks | 13/13 passed |
| Ubuntu 22.04, client at `9f36ca3` | Native CPU checks | 11/11 passed |
| Ubuntu 22.04, client at `9f36ca3`, CUDA agents | Two-shard process lifecycle, both gather policies; fresh and optimizer-exact restored state | Both modes passed |
| Three separately rented RTX A6000 48 GB devices, compute capability 8.6 | CUDA state/device-result check on each device | All three passed; distinct GPU UUIDs verified |

The L40 driver was 610.43.02, compute capability 8.9. CUDA code was subsequently
built for both 86 and 89 to run the same backend on the A6000 workers. The Ubuntu
image lacked a compiler toolkit; project-local NVIDIA pip components supplied it.
The provider driver and preload library were retained. BRPC's optional pthread
contention-profiler hook was disabled because its bootstrap conflicted with that
preload library.

The new automatic-controller checks are recorded in the
[policy validation artifact](validation/2026-10-31-cache-policy.json), with
source/log hashes. Two external PS processes and one Host reference agent cover
both gather modes, changing hot sets, RPC metrics, configuration rejection and
continued optimizer state after restart. Native checks additionally cover
multiple agents/tables, bounded history, generation age, pin protection, byte and
batch budgets, atomic reserve enforcement, and failed admission after eviction.
Two additional TensorFlow graph checks train five dense/sparse steps while
parameters migrate automatically, verify policy metrics through Python IPC,
assert decreasing loss without retracing, and drain cache at save.
The earlier policy revision passes 17/17 CPU/CUDA checks and three additional
forced-CUDA process checks. The [current CUDA validation artifact](validation/2026-10-31-cache-policy-cuda.json)
records that evidence separately from the earlier records. Two fixed-trace
profiles complete 36 independent runs / 6,528 updates with full CPU replay;
[the policy report](CACHE_POLICY_RESULTS.md) retains phase records and checks.

## What the checks establish

The [resumable-shard artifact](validation/2026-10-31-cached-shards.json) records
current source/binary/log hashes, bounded queued-plan caching and ownership-epoch
invalidation, conflict fairness, same-table optimizer ordering, exclusive
maintenance upgrades, checkpoint barriers and cross-continuation failure fencing.
The CUDA delay-injection check proves independent progress and excludes moving
rows until publication/eviction acknowledgement. A queued read planned before a
move must acquire the newly published Agent stream after that move.

The latest [same-binary scheduling study](SCHEDULING_RESULTS.md) retains 64 main
capacity trials / 23,407 updates plus six cache-ablation trials / 2,583 updates,
all without training-update errors. A separate six-run trace checks all touched
weights, moments and row counters over 112 updates with zero final-state error;
one nonfatal cache-maintenance failure remains in its records. Earlier source
`c52f516` and its negative uncached scheduling results retain separate artifacts.

The [asynchronous-preparation artifact](validation/2026-10-31-asynchronous-preparation.json)
records source/binary/log hashes, executed suite results and graph records for
source `040ae90`. Native checks cover count/byte bounds, FIFO consumption,
queued/running cancellation, worker shutdown, blocked-RPC overlap, current-weight
reads after the preceding Push and release of unpublished successful lookups.
The prepared graph trains five dense/sparse steps through two PS processes and a
real CUDA cache agent: loss falls from 0.66015625 to 0.59706187 without retracing.
Dense model computation uses the TensorFlow CPU host-callback path.

A [small fixed-trace CUDA check](benchmarks/2026-10-31-preparation-phase-trace/manifest.json)
adds six runs, 14 phase records and 112 updates with all eight service-phase
counters. Full touched-row weights, moments and counters match CPU replay;
maximum state error is zero and feature error is 9.31e-10. Frequency placement
admits 66 rows and evicts 38. One nonfatal maintenance failure is retained in the
records; training updates complete without errors. This check validates metrics
and lifecycle behavior; the multi-GPU capacity figures retain their original
source revisions. Its single rental is deleted after archive hash verification.

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
| Automatic placement | Bounded decayed request reuse, frequency/reuse-cost modes, cost feedback, minimum residence, hot-set replacement, pins, reserved slots, shared multi-agent/table budgets, read-only completions and rejected-admission fallback before/after eviction |
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

## Four-agent capacity and cache-policy replay

The [latest study](MULTIGPU_LIMITS.md) retains 140 main-matrix capacity trials
(79,163 updates), including four Host-reference controls, and a separate 63-trial
first-fleet scan (24,263 updates). All 203 completed capacity trials report zero
training-update errors. The main matrix checks 4,480 sampled row-update counters,
finite state and live-ticket cleanup.

The first-fleet fixed traces retain 35 completed runs, 81 phase records and 4,800
updates. Every touched row's weights, both moments and row-update counter match
independent CPU replay; maximum state error is zero and feature error is at most
5.22e-8. One driver-timeout run is excluded and recorded in the manifest. A
separate four-agent positive-cost correctness case completes 192 updates with
actual admission/eviction and zero state error. Four nonfatal maintenance events
occur in the completed calibrated traces and one in the positive-cost check.
Concurrent frequency/reuse-cost controls report no maintenance failures.

Each GPU fleet passes the 17-test Ubuntu CPU/CUDA suite; each worker separately
passes the CUDA state check. The local native suite passes 15/15, including the
expanded one-/four-agent Host fixed-trace smoke, with additional concurrent
frequency and reuse-cost Host smoke checks.

[Counts, source/binary hashes and verified rental cleanup](validation/2026-10-31-four-agent-limits.json).
