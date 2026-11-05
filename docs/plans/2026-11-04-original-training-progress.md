# Original Entropy Training Progress Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make original Entropy training steps participate in the distributed parameter service's training-task admission and completion protocol.

**Architecture:** The original Runner arms one learner's native context before executing a training fetch. The first non-prediction native Pull, after its input tensors exist, obtains a coordinator permit. Successful Runner return drains acknowledged parameter work and completes the task; input EOF without a Pull cancels the armed scope without advancing progress. An error after admission aborts the group. Original dense optimization, model execution and framework barriers stay in the original framework.

**Tech Stack:** C++20 C runtime, BRPC training coordinator, native TensorFlow operations, Python original-Runner installation hooks.

---

### Task 1: Native step scopes

Files: `adapters/entropy/runtime_c.cpp`, `adapters/entropy/native_session_backend.py`, new `tests/native_training_progress.py`.

1. Add optional coordinator/worker/initial-iteration configuration and explicit arm/finish/cancel/status commands to the existing native context.
2. Obtain admission only on the first training Pull in an armed scope; support prepared Pulls too. Keep prediction reads outside admission.
3. Complete only after Join and the model step's successful return. Reject overlapping scopes and abort the group on uncertain admitted work.
4. Run real two-Worker native TensorFlow model steps against a real coordinator and two parameter shards. Verify completion counts, no live tickets, EOF-style cancellation, prediction exclusion and error propagation.

### Task 2: Original Runner hooks

Files: `adapters/entropy/entropy_install.py`, `tests/entropy_hook_configuration.py`, original-source integration fixtures.

1. Validate opt-in training configuration and preserve each original learner's identity.
2. Wrap original single-/multi-scene Runner training fetches. Arm before execution, finish after acknowledged completion, cancel/abort on exceptions; leave eval and administrative fetches alone.
3. Exercise the original Runner body with the installed hook and native runtime; preserve the current table/controller regressions.

### Task 3: Complete original multi-peer execution

Files: existing private native-build acceptance drivers, public validation records and integration documentation.

1. Continue the live matching NVIDIA TensorFlow build without replacing the active build on observation timeout.
2. Execute original reader, model, checkpoint/restore and main/Executor gates; repair actual failures.
3. Run two original peers with real learner/controller services and parameter shards, verify per-Worker progress and joint checkpoint boundaries, then test continuation and recovery.
4. Bind task progress to framework checkpoint/restore before claiming full training-progress recovery. Fixed-group sequential/bounded-delay admission requires aligned Worker steps; use eventual admission for unbalanced finite original FileFeeder workloads until explicit Worker retirement is implemented.

The existing independent replicated-gradient recovery and GPU-capacity tests retain their scope. Full original multi-peer execution, its task-progress checkpoint/restore, physical-node loss, cross-node retirement and GPU scaling remain required until executed and verified.

## Execution record

Tasks 1 and 2 pass real native C ABI/coordinator checks and original-method
TensorFlow model checks. The final Linux run passes six checks, including CPU
and GPU original Runner models, native graph/context regressions and C ABI
admission/error cases. Both original-method model variants finish at `[3, 3]`,
with six acknowledged updates per shard and all six rows updated twice. The
parameter learner ID stride is eight while coordinator Worker IDs remain zero
and one. Nine configuration/hook unit cases and four existing original-source
table/restore/peer-checkpoint regressions pass locally.

The verified-idle original main/Executor gate now uses the current native
runtime and a real progress coordinator, asserting its completed-step count
after execution. Its full-module execution is still pending the live matching
TensorFlow build and original reader/training gates. Task 3's multi-peer
execution remains open. Evidence:
`../validation/2026-11-04-original-runner-training-progress.json`.

Progress-bound original-method checkpoint/restore is now implemented. The chief
pauses admission, exports the exact completed-count vector, binds it into the
joint commit and resumes after original saves finish. Fresh coordinators start
paused with restored clocks; table and Runner hooks reject wrong local clocks
before dense restoration and reopen admission after global restore barriers.
Two real native TensorFlow Workers save off the four-batch input-cycle boundary,
restore `[3, 3]` and continue to `[6, 6]`. All nine learned parameter rows, full
Adam state and continued losses match uninterrupted execution exactly in CPU and
one-physical-A6000 GPU model runs. Incorrect starting clocks are rejected, and
injected original dense-save failure leaves admission paused with no commit.
Three Linux checks, seven checkpoint-metadata unit cases, nine configuration
unit cases, two local C ABI checks and four original-source regressions pass.
Evidence: `../validation/2026-11-05-original-training-checkpoint.json`.

Full original-module/native multi-peer execution remains required; these model
checks execute preserved methods with explicit framework environment stand-ins.
