# Validation record

## Distributed node orchestration — November 4, 2026

The per-node job lifecycle now starts independent parameter shards, owners and
TensorFlow workers, performs local shard preparation, verifies schema/vector
width, samples process/device resources and collects training state. The local
Host-backed case passes with 1,024 initialized/resident rows split 512/512,
eight synchronized rounds, 16 completed worker tasks and eight updates per
shard. All 37 touched rows have the expected update counters and finite state;
weighted loss decreases from 0.693552 to 0.605325. Cleanup verifies that both
nodes' owned wrapper/child processes are gone. The new CTest lifecycle case also
passes with dynamically assigned ports.

The GPU run is not started: the first A6000 creation for the planned two-node
fleet returns configuration unavailable. The provider UI lists A6000 and L40
as sold out, and the account remains at zero instances. No GPU-capacity result
or new charge is attributed to this phase. SSH relays and cached Linux runtime
installation are implemented but await actual node deployment.

[Node roles and execution commands](DISTRIBUTED_MODEL_TRAINING.md) ·
[Local report, hashes and provisioning rejection](validation/2026-11-04-distributed-training-orchestration.json)

## TensorFlow model training with CUDA parameter owners — November 4, 2026

Workers now train both Embedding vectors and packed two-layer MLP weights/biases
through the synchronous parameter service. The model computes pooled features,
tanh activations, binary cross-entropy and automatic gradients on the requested
device. Its native client runs in a separate IPC process; the RPC/model boundary
uses host staging. GPU placement of inputs, loss and both gradients is checked.

The A6000 run passes the eight-round/two-worker model comparison. A second case
fully initializes and loads 16,384 rows × 64 dimensions on the CUDA owner and
trains scattered ID bags. All 137 touched rows' final weights and Adam moments
agree with the single-worker reference within 4.15e-6; first aggregated gradients
agree within 2.24e-8. Weighted loss decreases from 0.695597 to 0.275646.

The local framework suite passes all 51 checks. The deployed CUDA/framework
suite passes 53 checks, with four original-source checks skipped because private
references are excluded from deployment. The separately executed scattered CUDA
case also passes. Source hashes, hardware, reports and logs are retained before
the owned instance is deleted and verified absent. A compatible compiled runtime
bundle is cached locally for subsequent GPU nodes, with no billed snapshot.

These runs verify one-GPU training execution and small-table residency. The
multi-node capacity case with resident global state above one physical GPU's
memory remains the next required GPU experiment.

[Model flow and parameter layout](TENSORFLOW_SHARDED_TRAINING.md) ·
[Canonical execution record](validation/2026-11-04-tensorflow-sharded-training.json)

## Synchronous sparse gradient rounds — November 4, 2026

Each configured parameter shard now collects one sparse contribution per worker,
reduces matching Keys and acknowledges one optimizer update for the round. Sum,
worker mean and sample-weighted local-mean reductions have core checks. The
network path reserves the round writer stream, keeps read-only Pull available
and rejects ordinary worker writes. Empty contributions participate; an empty
round advances optimizer context without modifying absent rows.

The process check runs four independent worker processes and three parameter
shards, with unequal local batch sizes and a controlled slow worker. Worker-mean
and sample-mean runs each complete 40 rounds / 160 worker batches, with exactly
40 updates per shard. One shard has zero Keys throughout; it still completes
every round. Each shard receives 160 completed-round retries without another
update. Complete final weights and Adam moments match an independent FP32 model
reference with zero maximum absolute error on this build. The reference accounts
for fused versus separate moment arithmetic as a whole-model choice.

The BRPC check also rejects 64 duplicate pending calls while leaving other
workers' response slots available. It checks stale control identities, missing
worker timeout, shutdown, write guards and exact full optimizer state against
one averaged update. All 35 native checks pass. The new collective core and task
progress checks pass under ASan/UBSan and TSan; installed BRPC binaries remain
outside those sanitizer builds. No GPU rental occurred in this stage.

[Implementation and contracts](SYNCHRONOUS_GRADIENTS.md) ·
[Source hashes, process results and logs](validation/2026-11-04-synchronous-gradients.json)

The principal next GPU acceptance is actual training state exceeding one GPU's
memory, with parameter/optimizer shards resident on multiple GPUs and real model
forward/backward. Existing small-state multi-owner measurements remain separate
from that [capacity requirement](plans/2026-11-04-multi-gpu-training-capacity.md).

## Distributed training RPC and worker processes — November 4, 2026

A separate BRPC coordinator now admits tasks from independent training-worker
processes. Four workers pull sparse parameters from two parameter-server
processes, compute binary logistic gradients with local batch-mean reduction,
wait for all Push acknowledgements and then report task completion. Sequential,
bounded-delay (`tau=2`) and eventual policies each complete 160 worker updates.
The controlled model's loss decreases from 0.693 to approximately 0.28; exact
touched-row update counts, finite optimizer state and drained tickets verify
the connection between task progress and parameter changes. This is a CPU
correctness workload, not a same-quality training-speed comparison.

The RPC overload case holds 16 waiting admissions while 64 more return capacity
errors. Stats and slow-worker progress continue, including a ready Begin while
the waiting list is full. Additional cases cover stale administrative identities,
duplicate completion, timeout/retry, pause/checkpoint/resume, abort and shutdown
with a pending caller. A fresh coordinator restores the process run's task
clocks paused and resumes explicitly; this check restores progress only.

All 32 native checks pass. The modified core admission path also passes its
ASan/UBSan and TSan checks; installed BRPC libraries are not sanitizer-built.
No GPU was rented for this stage. Cross-worker gradient aggregation, replicated
update acknowledgement/failover and full original Entropy execution remain
implementation requirements.

[Training execution](TRAINING_EXECUTION.md) ·
[Canonical process results, source hashes and logs](validation/2026-11-04-training-rpc.json)

## Training-task consistency core — November 4, 2026

The fixed-worker-group progress controller implements the lecture's sequential,
eventual and bounded-delay task dependency rules. Native, ASan/UBSan and TSan
configurations pass the targeted C++ check. Cases include the exact slide-18
boundary, duplicate completion, pause/drain, fresh-run restore, malformed clock
snapshots, terminal failure and eight concurrent workers under each policy.

Each configuration also executes three controlled sparse-logistic training
cases: four worker threads, two in-process parameter-service actors and 160
worker updates per policy. Actual gradients use pulled parameter features;
clocks advance after Push acknowledgements. Loss decreases from approximately
0.693 to 0.278; touched-row update counts, finite Adam state and drained tickets
are checked. These are correctness probes with per-worker optimizer updates.
They provide no multi-process/GPU, gradient-aggregation, replica failover or
common-quality performance result.

[Training execution contracts](TRAINING_EXECUTION.md) ·
[Source hashes, records and sanitizer logs](validation/2026-11-04-training-progress.json)

## Original native build preflight — November 1, 2026

The separate Bazel workspace repository supplies the previously missing build
rules. Matching TensorFlow `2.12.1+NVIDIA` headers/libraries are also available;
all three compressed libraries pass their recorded SHA-256 checks. A local
Bazel 7.7.0 workspace, independent repository overlays and legacy rule versions
load the original native C++ dependency graph with downloads disabled.

The final partial query exits 3 and emits 39,338 label records. Its unresolved
repository is the internal Redis SDK, reached through `costa_op`, bgraph's
`ops_utils` and common's Redis layer. The required SDK interface headers are
absent from the available Code tree. No native library compilation, original
reader/cluster execution or additional GPU rental occurred in this preflight.

[Build inputs and dependency path](ORIGINAL_NATIVE_BUILD.md) ·
[Inventory and archive hashes](validation/2026-11-01-native-build-preflight.json)

## Peer/thread checkpoint coordination — November 1, 2026

The subsequent restore check restarts both parameter shards and both peer
processes, with four learner threads. The administrative-only chief verifies
the loaded joint checkpoint; original Python restore barriers complete eight
global rounds. Two more acknowledged updates per shard produce complete
two-row parameter/Adam state identical to six uninterrupted direct-client
updates. Dense state changes from the test initializer `-0.5` to saved `0.5`.
The source fixture still uses the test network rendezvous and native-environment
stand-ins. [Restore evidence](validation/2026-11-01-peer-restore.json).

Seven focused configuration, filesystem, original-method, restore and peer
checkpoint tests pass on macOS arm64. The new two-peer/four-learner case executes
preserved Python local/global barrier definitions with a bounded test network
rendezvous and real TensorFlow Sessions/two BRPC shards. A chief without an
external learner table uses a separate administrative backend. Two repeated
checkpoints complete 24 global barrier rounds; the pre-commit hold verifies
frozen admission and an absent joint marker, and all exported shard hashes verify
after commit. Injected dense-save failure remains uncommitted and frozen.

The previous adapter reproduces `the checkpoint leader has no external parameter
backend` under the same chief placement. The native administrative client's
CPU-only configuration and close order have a mock lifetime check. Full original
C++ cluster/reader execution still needs its absent native runtime artifacts.
No additional GPU was rented for these control-path checks.

[Protocol and commands](PEER_CHECKPOINTS.md) ·
[Structured results and source hashes](validation/2026-11-01-peer-checkpoint-control.json)

## Native original-table installation — November 1, 2026

Source `fc64812` passes 56/56 Ubuntu TensorFlow/CUDA tests with the selected
original-definition fixture. Source `c5a7f93` uses identical native binaries and
passes five targeted follow-ups after making controller Sessions CPU-only. The
controller reports no TensorFlow GPU devices; CPU/CUDA model Sessions share its
native context across graph recreation. The GPU owner starts through the C ABI
before the original entry point and is shared across learners in the worker.

Original table-method CPU/GPU graphs have no Python callbacks. Five compared
training steps have zero feature difference from their direct-table reference,
and the original Runner performs a mid-run sparse/dense checkpoint. The native
restore path rejects fresh/wrong state before dense restore, resumes the saved
cursor from 3 to 6, and compares all five selected rows' weights, both Adam
moments, row counters, optimizer sequence and beta powers with zero difference.
The cursor is supplied by the recorded feeder stand-in.

The initial suite's cache-startup failure is retained; the final suite checks
owner readiness before reference-table creation. The source contract retains
native environment/reader/barrier stand-ins and does not prove execution of the
complete original native application. All GPU evidence is fetched and the
rented A6000 deleted, with account instance count zero.

[Native installation](NATIVE_TABLE_INSTALLATION.md) ·
[Machine-readable results, scopes and hashes](validation/2026-11-01-native-table-installation.json)

## Observable overload and three-owner follow-up — November 1, 2026

Runtime `0e46929` retains dependency plans under metadata pressure and separates
actor observations and Stats RPC completion from data waits. Local native/C-ABI
31/31, portable TensorFlow 43/43, core ASan/UBSan 18/18 and resumable TSan 2/2 pass.
Ubuntu TensorFlow/CUDA executes 48 passing checks and skips three private-source
checks. Native two-/three-A6000 model trajectories match complete touched
optimizer state and model outputs with zero difference.

The new study preserves 21 completed large-input windows, 12,227 acknowledged
updates and 672 sampled row-counter checks. Three explicit ticket-byte capacity
failures and one incomplete baseline probe remain separate. The repaired runtime
answers all 6,292 running-phase statistics queries. At 1,022 Learners, increasing
the per-shard ticket budget from 256 to 512 MiB enables completed single- and
three-owner windows. Four twenty-second three-owner windows at 128/512 Learners
also complete. All three rentals are deleted after hash-verified retrieval.

[Results and bounds](OVERLOAD_SCHEDULING.md) ·
[Structured validation and hashes](validation/2026-11-01-overload-repair.json) ·
[Local regression records](validation/2026-11-01-overload-scheduling/local-validation.json)

## Earlier native multi-GPU operating envelope — November 1, 2026

Runtime `5a69840` and corrected probe `9fe0ac6` retain **63 completed windows /
62,363 acknowledged sparse updates / 2,016 sampled row-counter checks** across
one/two/four physical GPU owners. All three topologies complete three 20-second
windows at 1,022 Learners on the smaller input. The million-row, 512D, 8192-ID
workload completes 512 Learners on one/four owners; its 1,022-Learner cases yield
one incomplete probe and one native RPC-timeout record. The failed cohorts are
excluded from throughput and full optimizer-trajectory validation. Live health,
memory/OOM, later four-owner read recovery and terminal timeout evidence remain
separate records. All four archives are hash checked and all rentals deleted.

[Results and known overload issue](NATIVE_MULTIGPU_RESULTS.md) ·
[Structured validation and artifact hashes](validation/2026-11-01-native-multigpu.json)

## Native TensorFlow CPU/CUDA operations — November 1, 2026

Runtime `5a69840` supplies an isolated C ABI and native resource, Pull, prepared
Pull, Push and control operations. The Ubuntu TensorFlow 2.20.0/CUDA build
executes **47 passing tests**, with three original Entropy source tests skipped
because private reference code is excluded from the uploaded archive. Separate
macOS suites pass **30/30 native/C-ABI checks** and **42/42 portable TensorFlow
checks**. macOS native runtime loading has an explicit platform guard.

The native CPU and CUDA graphs each run five model/automatic-gradient steps
against two external parameter shards, with two additional CPU reference shards.
Complete weights, both Adam moments and row update counters match CPU replay.
CUDA model/loss differences are at most 7.45e-9. Graph inspection confirms Eps
operations without Python callbacks. Prepared input, mid-run dense/input and
sparse save/resume, retained output lifetime and rejected-schema cleanup execute.
The Linux runtime exports only `eps_runtime_api_v1`; dynamic BRPC/Protobuf/protoc
dependencies are absent.

Fixture `1c8f060` repeats the native model trajectory with one, two and four
physical GPU parameter owners. The local TensorFlow GPU executes the model;
remote CUDA owners retain and update their parameter rows. Every remote owner
has resident rows and drains all accepted work. Complete sparse optimizer state
matches the independent CPU replay in all three topologies.

[Native design and build](NATIVE_TENSORFLOW.md) ·
[Native test records](validation/2026-11-01-native-tensorflow/native-checks.json) ·
[Multi-owner model records](validation/2026-11-01-native-tensorflow/multigpu-model-checks.json)

## Complete device Pull — November 1, 2026

Source `9353800` adds destination-projected device reads and CUDA assembly to the
distributed client and C++ TableAdapter. Local device blocks, CPU parameters and
host-returning remote Agent contributions produce complete device features;
global mean normalization and padded sequence reconstruction execute on CUDA.
Device and prepared-device Pull retain ordinary Push/Release/Join handles.

The native suite passes **28/28**, native/TensorFlow **42/42**, core ASan/UBSan
**18/18**, and Ubuntu CPU/CUDA **33/33**. The A6000 scenario uses two CPU shard
objects, two in-process Worker Agents with real BRPC control/remote gather, and
the native device resolver for the model-local Agent. It covers seven feature
layouts over five training steps, repeated IDs, means, empty bags, all-padding
and empty sequences, mixed resident/CPU-pool/unique-transfer routes, two prepared
inputs and mid-run save/resume with cache writeback/reload.

Each step compares all complete features and all eight touched rows' weights,
both Adam moments and update counters with CPU replay (absolute tolerance 5e-6).
A CUDA model kernel consumes assembled features on a nondefault stream to
generate gradients; the existing host/RPC Push acknowledges updates. Thirty
native-device contributions and ten remote-host contributions enter assembly.
Retained initial outputs survive later training and save/reload. A held native
read verifies error draining; a real output-budget rejection leaves zero live
lookup tickets and zero charged output. These checks establish a native device
training boundary. Original framework device-operator/application execution and
cross-node model execution remain separate requirements.

All results and source/archive/binary hashes are retrieved before deleting the
single rented A6000; final account inventory is empty.

[Device Pull design](DEVICE_PULL.md) ·
[Structured results and raw logs](validation/2026-11-01-device-pull.json)

## Native in-process worker Agent — November 1, 2026

Runtime `e88ce65` adds a worker-owned Agent listener, runtime and injected cache
backend. Native host/device gathers and remote control RPCs share admission and
ordered execution. The standalone Agent executable uses the same owner.

The macOS CPU/RPC suite passes **27/27** and native/TensorFlow suite **41/41**.
The Ubuntu CPU/CUDA suite passes **31/31** on an RTX A6000 with CUDA 13.0.88,
GNU C++ 11.4 and BRPC 1.18. The native lifecycle check combines real RPC control
and local reads, enforces count and native retained-input budgets, closes during
a held operation, drains both sources, releases the backend and reuses its port.

The CUDA consumer test reads owner contributions with a kernel on a nondefault
model stream, checks complete weights, both Adam moments and counters for two
rows against Host reference (absolute tolerance 1e-6), enforces/reclaims output
budgets and rejects a stale location. It consumes the original results again
after updates, eviction, slot reuse and Worker/backend destruction. Source archive,
relevant source files and deployed binary hashes are retained. All artifacts are
retrieved before deleting the single rented instance; account inventory is empty.

These checks cover the native ownership and device-contribution interface.
Original application/reader execution, multi-peer framework barriers and a native
TensorFlow device operation retain their separate integration requirements.

[Worker API and lifecycle](WORKER_AGENT.md) ·
[Source-hashed validation and raw logs](validation/2026-11-01-worker-agent.json)


## Bounded Agent admission — November 1, 2026

Runtime `426c27b` adds a shared FIFO count/byte budget before Agent dispatch and
retains its permit through asynchronous response completion. The native suite
passes 26/26; the native/TensorFlow suite passes 40/40. The admission component
passes its ASan/UBSan and ThreadSanitizer checks. A real BRPC paused-backend test
admits 256 requests across eight client channels, observes four dispatched and
252 waiting, then verifies every returned value and unchanged complete row state.
Separate count/byte cases reject before backend execution.

The same Linux/CUDA binary build passes 28/28 checks and is reused for the final
fleet. Each final worker independently passes the CUDA state/device check. The
fleet contains two A6000 and two L40 devices; every deployed Agent hash matches
its selected baseline/candidate build. The candidate completes **23 CUDA trials /
12,950 acknowledged updates / 736 sampled counters**, including 1/2/4-agent
128/192-Learner cases and three sustained four-agent 192-Learner windows. Maximum
feature preflight error is 3.91e-8. Sampled counters, finite parameters and ticket
cleanup pass. Admission snapshots show at most 32 dispatched, unfinished RPCs,
195 total pending requests, 6.07 MiB retained input and zero rejections, dispatcher
failures or remaining work after drain. The same-fleet old Agent rejects 128
Learners with E2004; that failed point remains separate.

Two earlier transport-failed deployments are excluded from the primary matrix.
All rented instances, including those attempts, are deleted after retrieval;
final account inventory is empty. Runtime/binary provenance, workload conditions,
health checks and cleanup records are linked from the report.

[CUDA admission report](AGENT_ADMISSION_RESULTS.md) ·
[Validation index](validation/2026-11-01-agent-admission.json)


## Multi-GPU capacity and post-overload state — November 1, 2026

Sources `955520e` and `b7b87cf` run 1/2/4-GPU fleets with distinct CUDA device
UUIDs and identical Agent binaries within each fleet. The second revision changes
the probe's maximum Learner count and failure diagnostics; service execution
code is unchanged. Each fleet passes the 26/26 native CPU/CUDA suite, and every
worker separately passes the CUDA state/device-result check.

The study retains **89 completed trials / 67,425 acknowledged updates / 2,848
sampled row-counter checks**, including eight Host-reference control trials.
Completed records report zero update errors, zero remaining tickets and a maximum
preflight feature error of **3.91e-8**. Tests reach 192 concurrent Learners and a
million-row, 512-dimensional table with 6 GiB of logical parameter/Adam state.

Eight failed load points retain separate diagnostics. RPC rejection can leave a
usable read path with no tickets, or a failed parameter-work run with outstanding
tickets despite responsive statistics. Failed optimizer trajectories remain
unverified and are excluded from completed-update totals. All eight rented
instances are deleted after hash-verified retrieval; final account inventory is
empty.

[Report and workload contract](MULTIGPU_BOUNDARIES.md) ·
[Combined validation index](validation/2026-11-01-multigpu-study.json)

## Committed restore and loaded-state verification — November 1, 2026

Source `ce17f18` connects committed parameter snapshots to fresh service startup
and the original table/Runner restore boundary. The stager verifies the joint
manifest, file SHA-256 and partition header; the native loader validates the
full snapshot and publishes its provenance only on success. Restore hooks require
matching loaded files in a pristine new run before original dense restore.

| Environment | Executed checks | Result |
| --- | --- | --- |
| macOS arm64, native/RPC/TensorFlow | Complete suite, including two-service staging/restart and original restore flow | 36/36 |
| macOS arm64, ASan/UBSan | Complete core suite | 16/16 |
| macOS arm64, ThreadSanitizer | Complete core suite | 16/16 |
| Ubuntu, GNU 11.4, A6000, CUDA 13.0.88 | Native CPU/CUDA suite | 25/25 |
| Same A6000, TensorFlow 2.20.0 | Complete native/CUDA/TensorFlow suite | 39/39 |
| Same build, forced CUDA Agent | Original table/restore paths, periodic save, cache lifecycle and delayed operations | 8/8 |

The restore regression commits after step 3, terminates both PS processes,
stages their files through `entropy_stage`, starts fresh services with a new run,
and continues three steps through the original source-level Runner/worker/table
restore flow. The CUDA path matches uninterrupted six-step execution with
**zero maximum absolute error** for selected weights, both Adam moments and beta
powers; row counters and optimizer sequence match exactly. The native comparator
checks five rows across two selected tables. Dense parameter error is also zero.

Independent phases reject a fresh un-restored service and a different loaded
snapshot before dense restore. A subsequent restore attempt after training begins
is rejected. Core checks additionally cover failed-load publication, duplicate
definition, first admission, schema change and repeated empty-snapshot restore.
Filesystem checks cover immutable retry, bounded reads, missing joint markers,
mismatched manifests and corrupt files.

Parameter-cache execution uses real CUDA on the A6000; the TensorFlow model runs
on CPU. Original methods are source-executed with explicit native-environment,
save/feeder stand-ins and single-participant barrier records. Input continuation
uses the saved test-feeder cursor. Staging exercises local paths through actual
`tf.io.gfile`; configured remote plugins retain their own deployment validation.
Full native application/reader, multi-peer framework execution and in-process
device tensors remain separate requirements.

[Restore commands and protocol](ENTROPY_RESTORE.md) ·
[Source, binary/log hashes, comparison records and verified rental deletion](validation/2026-11-01-entropy-restore.json).

## Original Entropy table and Runner hooks — November 1, 2026

Source `ea743b9` installs the external backend on selected original table names,
preserves `SparseTable` class identity and registration, and leaves the original
`InputBuilder` gradient dispatcher in place. The Runner hook freezes external
updates, executes original directory preparation before export, runs the original
checkpoint body, commits the parameter-manifest hash and resumes training.

The complete local native/RPC/TensorFlow suite passes **33/33**. Both new
original-method process checks execute, with two external BRPC shards and either
CPU access or the Host cache Agent. Five paired steps have **zero maximum
absolute feature error** against direct reference tables. Six train steps cross
an original-Runner mid-run checkpoint; paired-step loss decreases from
0.1697656214237213 to 0.09437486529350281. The saved dense variable matches an
actual TensorFlow checkpoint reader exactly.

Additional cases preserve valid zero IDs, uint64 padding, mean/sequence shapes,
bucket aliases, missing-gradient policy, all-unused handle release, prediction
release, graph recreation and separate learner IDs. Callbacks run while original
learner TLS is deliberately unavailable. An injected dense-save failure leaves
parameter admission frozen and publishes no external joint-commit marker.

The source contract executes preserved original classes/functions with explicit
native environment/save/feeder stand-ins and single-participant barrier event
records. The complete native module and multi-peer framework barriers are not
executed in this check. Public configuration/launcher tests contain four cases;
missing private source is separately verified to return skip code 77. No GPU is
rented for this Python integration revision; the native C++/CUDA source is
unchanged from `dc72662`.

[Installation and protocol](ENTROPY_TABLE_HOOKS.md) ·
[Source provenance, binary/log hashes and execution scope](validation/2026-11-01-entropy-hooks.json).
Remaining work includes remote restore staging, the original selected-table
restore connection, native original-application execution, multi-peer framework
checkpoint execution and in-process device tensors.

## Periodic checkpoint lifecycle — November 1, 2026

Source `dc72662` adds frozen snapshot reads, framework-filesystem export and
checkpoint-matched same-run resume. The adapter admits new input after all
parameter shards acknowledge the current run/epoch/checkpoint identity. A
partial save keeps admission closed; a partial resume acknowledgement can be
retried with the same identity. Snapshot export remains bounded to 1 MiB per
RPC and verifies the embedded checksum before manifest publication.

| Environment | Checks | Result |
| --- | --- | --- |
| macOS arm64, native/RPC/TensorFlow | Complete suite, including periodic checkpoint and filesystem contract | 30/30 |
| macOS arm64, ASan/UBSan | Complete core suite | 15/15 |
| macOS arm64, ThreadSanitizer | Complete core suite, including accepted-work drain during destruction | 15/15 |
| Ubuntu, GNU 11.4, A6000, CUDA 13.0.88 | Native CPU/CUDA suite | 24/24 |
| Same A6000, TensorFlow 2.20.0 | Complete native/CUDA/TensorFlow suite | 33/33 |
| Same build, forced CUDA | External shard/cache lifecycle, automatic placement, prepared graph, periodic save/resume and delayed operations | 6/6 |
| Same build, focused checks | Bounded optimizer and actual CUDA state/results | 2/2 |

The core regression compares eight updates against uninterrupted training across
two periodic checkpoints and a fresh-run restore from exported bytes. Weights,
Adam moments and row counters match exactly. It also exercises stale identities,
partial-save fencing, failed resume acknowledgement retry and monotonic request
and prepared-input identities.

The forced-CUDA graph uses two external PS processes and an A6000 cache Agent.
It saves dense/input state and exports sparse state through actual `tf.io.gfile`
mid-run, commits a joint marker, then resumes without retracing. Five train steps
reduce loss from 0.66015625 to 0.5970618724822998. Automatic placement admits rows
again after checkpoint writeback. A corrupted disposable snapshot fails its
native checksum check before export can publish a manifest. Model computation
in this host-callback graph remains on CPU; the forced path executes parameter
cache work on CUDA.

ThreadSanitizer identified a PImpl destruction race during a partially failed
save. Public service, Host/CUDA cache and Agent destructors now drain accepted
work before their implementation pointers are cleared. A gated suspended update
and an accepted checkpoint cover that close order on Host and CUDA owners.

[Save/export/resume design](PERIODIC_CHECKPOINTS.md) ·
[Source, binary and log hashes; verified rental deletion](validation/2026-11-01-periodic-checkpoints.json).
The original Entropy factory/controller installation, native application build,
in-process device-tensor integration and automatic remote-store restore staging
remain open work.

## Entropy optimizer bounds — November 1, 2026

Source `ab22ca3` preserves optional sparse Adam weight bounds through CPU row
initialization, CPU/Host/CUDA updates, JSON/Protobuf table configuration and
versioned snapshots. The original optimizer constructor is source-executed for
five mapping checks; its options storage is a local stand-in, with the complete
native framework import still pending.

| Environment | Checks | Result |
| --- | --- | --- |
| macOS, native/RPC/TensorFlow | Complete suite, including zero-valued bounds in the graph/JSON/native/RPC path | 27/27 |
| macOS, ASan/UBSan | Complete core suite, including bounded state and legacy restore | 14/14 |
| Ubuntu, GNU 11.4, A6000, CUDA 13.0.88 | Native CPU/CUDA suite | 23/23 |
| Same A6000, TensorFlow 2.20.0 | Complete native/CUDA/TensorFlow suite | 30/30 |
| Same build, forced CUDA | External cache lifecycle, automatic placement and prepared graph | 5/5 |
| Same build, focused checks | Bounded optimizer and actual CUDA result/state checks | 2/2 |

CUDA assertions check that updated weights reach a configured boundary, remain
inside the interval and match Host weights, moments and counters. Imported rows
retain their supplied state until updated. Overflow validation precedes clipping
and leaves rejected updates unchanged. Snapshot checks preserve the bounds,
restore optimizer state and read an independently encoded format-1 fixture.

[Configuration and remaining framework hooks](ENTROPY_CONFIGURATION.md) ·
[Source/binary/log hashes and verified rental deletion](validation/2026-11-01-entropy-bounds.json).

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
