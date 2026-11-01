# Distributed Parameter Server implementation plan

**Goal:** Extract a distributed parameter server from the reference training
lifecycle, initially supporting sparse embedding parameters, then optimize CPU/GPU
access and migration while retaining the training framework.

**Architecture:** Independent CPU parameter-service shards own row routing,
storage, update sequencing and cache decisions. A thin Entropy adapter preserves
the existing graph; an agent in each existing GPU worker executes device work.

**Tech stack:** C++20, BRPC/Protobuf, TensorFlow custom-operation adapter and CUDA
in the GPU agent. Initial row layout is FP32 weights and unshared LazyAdam moments.

Implementation started on 2026-10-31. Local and rented-GPU verification are
authorized. GPU checks run within the existing USD 20 rental budget.

| Phase | Current evidence |
|---|---|
| 1. CPU parameter service | Implemented; two independent BRPC shard processes exercised by a client |
| 2. Sparse updates and saving | Implemented; repeated requests, bounds and exact restore trajectory checked |
| 3. Framework integration | C++ Pull/Push/Join adapter, portable TensorFlow bridge, optimizer mapping, original table/Runner hooks, periodic export/resume and committed restore staging implemented; original-method checks restart two PS processes and compare complete selected Adam state; native TensorFlow CPU/CUDA operations execute model forward/backward, prepared Pull, acknowledged Push and checkpoint continuation against independent PS processes; native table installation and shared learner contexts are implemented; complete original-application/reader execution and multi-peer framework barriers remain pending |
| 4. GPU cache agent | Standalone CUDA resident gather/Adam, full-state transfer, fencing and restore verified; native worker owner, shared local/RPC admission, model-stream device contributions and native TensorFlow device tensors verified on A6000; original-worker installation remains pending |
| 5. Independent policies | CPU pooling/unique-vector GPU paths, route selection, destination projection and selectable gather coalescing execute; automatic frequency/reuse-cost admission/replacement, bounded statistics and migration budgets connected; bounded asynchronous input preparation and resumable shard operations implemented; dependencies, barrier upgrades, original optimizer order and real CUDA lifecycle checked; Agent-wide count/byte admission and real 128/192-Learner CUDA backpressure now verified |

See `../VALIDATION.md` for the executed checks and `../RUNNING.md` for commands.

The [overload repair and multi-GPU follow-up](../OVERLOAD_SCHEDULING.md) verify
explicit ticket-budget rejection, drain and responsive statistics at 1,022
Learners. [Native table installation](../NATIVE_TABLE_INSTALLATION.md) now connects
the original table/Runner boundary to native CPU/CUDA operators with shared
learner contexts and pre-entry GPU owner startup. Complete original application,
native reader and original C++ multi-peer runtime execution remain open.
[Peer checkpoint checks](../PEER_CHECKPOINTS.md) now execute preserved Python
barriers across two processes/four learner threads with a test network
rendezvous; repeated commits and dense-save failure preserve admission state.

## 1. Build a standalone CPU parameter service

**Create:** `src/store/shard_store.{hpp,cpp}`,
`src/access/batch_access.{hpp,cpp}`, `src/server/service.{hpp,cpp}`,
`src/server/main.cpp`.

- Implement fixed `(table,key)` routing, the key-to-row index and contiguous
  weights/moment arrays. Grow in chunks within an explicit memory limit.
- Apply table definitions and seeded missing-ID initialization. Reject unsupported
  optimizer and feature-creation settings.
- Implement unique-key lookup and retain occurrence maps in bounded lookup tickets.
- Produce sum, mean and ordered sequence features. Combine partial sums across
  shards before applying the original mean denominator.
- Map the original `proto/embedding_service.proto` to BRPC attachment slices;
  validate types, lengths, offsets and allocation bounds.
- Add a launcher that parses `configs/service.example.json`.

**Exit condition:** an external client can create a table and obtain feature
tensors from separately running CPU shards through the documented API.

## 2. Complete sparse updates and saving

**Create:** `src/update/update_engine.{hpp,cpp}`,
`src/update/lazy_adam.{hpp,cpp}`, `src/server/request_tracker.{hpp,cpp}`,
`src/store/snapshot.{hpp,cpp}`.

- Unpool feature gradients with the retained layouts; merge repeated IDs within
  their original update request.
- Sequence each table's updates and advance its beta powers exactly once at the
  configured original-update boundary. Keep row-local update counters separate.
- Track accepted/completed requests, contiguous retired watermarks and out-of-order
  gaps; keep retry identity stable. A completion ack follows optimizer completion.
- Implement read-only ticket release and coordinated abort. A cancelled RPC alone
  cannot release rows still used by model backward or GPU operations.
- Implement run drain, full-state snapshot metadata and restore with partition
  epoch, table configuration and optimizer globals.

**Exit condition:** the service provides the complete lookup → gradient → update →
release → save lifecycle. It can be integrated without a GPU cache.

## 3. Connect the existing training framework

**Create:** `adapters/entropy/table_adapter.{hpp,cpp}`,
`adapters/entropy/completion_bridge.{hpp,cpp}`,
`adapters/entropy/README.md`.

- Map supported table configuration and feature layouts at Pull/Push/Join.
- Batch by destination shard; retain original bag lengths and sequence positions.
- Reuse the existing tensor transport and graph completion dependencies.
- Preserve loss/gradient scaling, dense collectives and independent Learner updates.
- Connect existing checkpoint control to service drain/save/restore.

**Exit condition:** an existing model can select this embedding backend through
configuration while retaining its data, model and job-control paths.

## 4. Add the in-process GPU cache agent

**Create:** `adapters/entropy/gpu/cache_agent.{hpp,cpp}`,
`adapters/entropy/gpu/embedding_kernels.cu`,
`src/cache/placement_directory.{hpp,cpp}`,
`src/cache/migration_coordinator.{hpp,cpp}`.

- Allocate complete row states in the existing GPU worker.
- Implement resident gather/pooling and unique-key sparse updates using the
  service-issued optimizer context.
- Implement CPU → Loading → GPU → Draining → CPU transitions. Close references
  during both move directions; publish generations after acknowledged transfer.
- Batch state copies and return current weights/moments before reclaiming slots.
- Add static placement and frequency/window cache modes as comparison baselines.

**Exit condition:** cached and uncached rows follow one parameter lifecycle;
placement does not alter which IDs train or the number of optimizer updates.

## 5. Implement the independent optimizations

**Create:** `src/cache/access_cost_model.{hpp,cpp}`,
`src/cache/reuse_admission.{hpp,cpp}`, `src/cache/migration_budget.{hpp,cpp}`,
`src/access/prepare_queue.{hpp,cpp}`.

- Record phase times, occurrence/unique/pooled counts and actual transfer bytes.
- Choose CPU pooling or unique-vector transfer for CPU-backed contributions using
  moving estimates, minimum observation counts and a switching margin.
- Score admission by expected saved access cost minus state load/writeback cost.
- Enforce network/staging/in-flight budgets and free-slot reserves by row layout.
- Prepare routing and mappings ahead; coalesce compatible transfers. Keep current
  parameter reads behind the existing ownership/update boundary.
- Expose policy metrics and configuration without changing model semantics.

**Exit condition:** each policy can be enabled independently against the same
CPU-only, static and frequency/window baselines.

## Validation coverage to complete

| Question | Proposed evidence |
|---|---|
| Do lookup and backward preserve layouts? | Small reference cases for sum/mean/sequence, repeated IDs, empty bags and multi-shard contributions |
| Is the optimizer preserved? | Fixed input/update trace comparing weights, moments and counters across CPU and GPU placements |
| Can a retry duplicate an update? | Reordered replies and duplicate Push requests within a live run |
| Is migration complete? | In-flight backward, two-way moves, delayed writeback, generation mismatch and drain/restore cases |
| Does the policy save useful work? | Same model/data/batch size/optimizer across baseline and independently enabled policies |
| Does it scale? | Per-shard CPU, network, PCIe and GPU phase time alongside end-to-end throughput and validation quality |

Stress later workloads with long pooled bags, short bags, repeated IDs, cold IDs,
changing hot sets, constrained cache capacity and multiple Learners. Report
metadata/planning overhead and peak state memory alongside throughput. No speedup
target is used as a substitute for measured results.

## Multi-GPU capacity record — October 31

The new fleet completes 253 trials and 153,703 acknowledged sparse updates
without reported errors, covering 1/2/3 A6000 agents, up to 128 Learners,
2/4/8/16 PS shards and million-row 512D state. At three GPUs, four shards and
eight Learners, three 30-second trials reach 122.95 updates/s and 104.93 ms
median trial P99. Larger shard/concurrency controls identify coordination and
communication as the next scaling targets. Paired results retain negative
coalescing results; per-shard overlap with destination projection is the default.

Current native/TensorFlow checks pass 14/14, core ASan/UBSan 8/8, Linux CPU
11/11, and real CUDA migration/save/restore passes with both gather modes.
[The capacity report](../BENCHMARKS.md) links source revisions and all raw data.
All three instances were deleted after results were fetched and audited.

## Automatic placement integration — October 31

Implemented bounded decayed request reuse, manual/frequency/reuse-cost modes,
minimum residence and replacement margin, live-ticket exclusion, shared migration
budgets, migration-cost feedback and atomic Agent-side reserved-slot enforcement.
The server JSON launcher and RPC Stats expose the controller. Completed requests
trigger coarse decisions on the existing sequencer; cache ownership and optimizer
updates retain their existing lifecycle. Save drains residency; restore rebuilds
performance history from new requests.

At the automatic-controller integration revision, native/RPC/TensorFlow checks
passed 19/19 and core ASan/UBSan 9/9, including
two-shard external reference-agent replacement and exact restart in both gather
modes. [Policy details](../CACHE_POLICY.md) and [source-hashed validation](../validation/2026-10-31-cache-policy.json)
record the scope. Real CUDA-controller execution and changing-hot-set performance are now
recorded in [the policy report](../CACHE_POLICY_RESULTS.md). Bounded asynchronous
preparation is now implemented; native original-framework integration remains.


## Fixed-trace CUDA evaluation

Source `3a2cd7e` passed macOS native/TensorFlow 20/20, Ubuntu CPU/CUDA 17/17 and
three forced-CUDA process checks. Two profiles retain 36 independent runs,
84 phase records and 6,528 updates. Every feature and final touched row is checked
against CPU replay; final-state maximum absolute error is zero. The moderate
profile rejects new admissions after a negative access-savings calibration;
the larger profile retains its small positive calibration and actual policy
behavior without retuning. All raw results, including nonfatal policy events,
are published. The single A6000 instance was deleted after fetch/hash checks.


## Agent admission — November 1

A per-Agent FIFO queue bounds dispatched/unfinished RPCs, pending requests and
retained serialized input. Its completion-owned permit prevents an RPC handler's
return from releasing execution capacity prematurely. Queueing replaces the
transport's 128-RPC rejection boundary; explicit application budgets remain.

The latest local suites pass native 26/26 and native/TensorFlow 40/40, including
expanded RPC mutation and CLI/HTTP checks; the admission component passes
ASan/UBSan and ThreadSanitizer checks. The Linux/CUDA build from `426c27b`
passes 28/28 and its identical binaries pass per-worker CUDA checks on the
final two-A6000/two-L40 fleet. The candidate completes 23 trials, 12,950 updates
and 736 sampled counters at 1/2/4 GPUs, including 128/192 Learners, with no update
or admission errors in completed records. Same-fleet baseline 128-Learner
rejection remains in the evidence. This completes the RPC admission subsystem;
original native application execution and in-process device integration remain
open requirements of phases 3 and 4.

[Design](../AGENT_ADMISSION.md) · [CUDA evidence](../AGENT_ADMISSION_RESULTS.md)


## Native worker ownership and device contributions — November 1

Source `e88ce65` embeds the Agent runtime/backend and RPC listener in a native
training-process owner. Local host/device gathers share the RPC admission queue;
Stop drains accepted work and releases the backend. The standalone executable
uses that same owner. CPU protocol checks cover shared budgets, a paused close,
late-call rejection, backend release and endpoint reuse. A real A6000 consumer
executes a model kernel on a nondefault stream, compares selected full optimizer
state, checks output-budget reclamation and stale locations, then uses retained
results after worker/backend destruction. Native CPU/RPC 27/27, TensorFlow 41/41
and Ubuntu CPU/CUDA 31/31 pass. The rented instance is deleted after retrieval.

This completes the native ownership/device-contribution boundary. Installing it
in the original GPU worker, a TensorFlow device operator, original application
and reader execution, and multi-peer framework barriers remain in phases 3/4.

[Worker API](../WORKER_AGENT.md) ·
[Validation](../validation/2026-11-01-worker-agent.json)


## Complete device feature Pull — November 1

At `9353800`, the native adapter returns complete model features on CUDA while
retaining the original handle lifecycle. Projected native contributions remain on
device; CPU and remote-host contributions join them in the final assembly. Mean
denominators apply once, sequence output projections restore padding and Push
compacts padded gradients. Prepared-device Pull uses the existing input queue.
The A6000 test compares five steps, seven feature layouts and all eight touched
optimizer rows with CPU replay; a native CUDA model kernel produces feature
gradients. Mid-run save/resume, retained outputs, read-error draining and
output-budget failure cleanup pass. Native 28/28, TensorFlow 42/42, ASan/UBSan
18/18 and Ubuntu CPU/CUDA 33/33 pass. Artifacts are retrieved and the GPU deleted.

This supplies the native feature boundary required by a framework device
operator. Original-worker/operator installation, original application/reader and
multi-peer framework execution remain in the main scope.

[Design and API](../DEVICE_PULL.md) ·
[Verification](../validation/2026-11-01-device-pull.json)


## Observable overload and retained scheduling plans — November 1

Runtime `0e46929` retains bounded dependency plans under metadata pressure and
adds bounded actor observations with independent Stats RPC completion waiters.
Held-owner regressions verify control-plane response while data slots and all
four data reply waiters are occupied, then compare complete optimizer state.
Local native/RPC 31/31, portable TensorFlow 43/43, core ASan/UBSan 18/18 and
resumable TSan 2/2 pass. Ubuntu TensorFlow/CUDA executes 48 passing checks and
skips three original-source checks. Same-instance overload and multi-owner
follow-up retains 21 completed windows, 12,227 acknowledged updates, three
explicit budget failures and the incomplete baseline. All 6,292 running-phase
statistics queries succeed on the repair. Three rented GPUs are deleted after
verified retrieval. [Overload scheduling](../OVERLOAD_SCHEDULING.md) links all records.

Original application/reader execution and multi-peer framework barriers remain
open integration requirements.


## Native table and Worker installation — November 1

Source `fc64812` passes the complete 56-test Ubuntu TensorFlow/CUDA suite with the
recorded selected-original-definition fixture. Source `c5a7f93` retains the same
native binaries and passes five targeted checks after making controller Sessions
CPU-only. Native table graphs use Eps Pull/Push/Release operations without Python
callbacks; three recreated model graphs retain one learner request sequence.
The launcher starts a C-ABI GPU owner before the original entry point without
creating a TensorFlow Session, and multiple native learner contexts share it.
Closing one learner leaves another operational. Original native table CPU/GPU
method traces, prediction changes, full snapshot export and native restore
continuation pass. The restored selected optimizer state and dense weights match
the reference exactly.

One startup-order failure from the initial 55-test run remains in the validation
record; the final suite restores reference-table creation before model graph
construction and verifies the pre-entry owner is already available. The GPU is
deleted after evidence retrieval; base-rate duration estimate USD 0.21.

The source-method environment still supplies explicit native feeder/configuration
and barrier stand-ins. Complete original-application/native-reader and multi-peer
native framework execution remain requirements of the full goal.

[Implementation](../NATIVE_TABLE_INSTALLATION.md) ·
[Validation](../validation/2026-11-01-native-table-installation.json)
