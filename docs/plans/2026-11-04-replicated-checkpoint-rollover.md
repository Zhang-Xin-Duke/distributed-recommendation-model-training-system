# Replicated Checkpoint Rollover Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Advance the retained replica/journal baseline at a committed joint training checkpoint while preserving model updates and recovery.

**Architecture:** Pause and save the existing shared model, optimizer and Worker input positions with the joint checkpoint controller. A fresh training generation loads that snapshot, starts its replica logs at the saved round and starts new Worker journals at the same boundary. Validate the new baseline and resumed model before retiring old-generation storage.

**Tech Stack:** C++20 parameter services and BRPC replicas, TensorFlow Embedding/MLP Workers, fsync-backed binary snapshots and Python orchestration.

---

### Task 1: Verify replicated snapshot rollover

**Files:**
- Modify: `tests/tensorflow_checkpoint_training.py`
- Modify: `CMakeLists.txt`

1. Add an opt-in replicated-rollover fixture using two independent replica processes and actual Worker journals.
2. Run the uninterrupted, pre-checkpoint and restored model phases with identical input batches and optimizer semantics.
3. At restored startup set the new replica/journal base round to the committed joint round; require pristine baseline-only replica heads before admission.
4. Verify every full row, moment, optimizer clock, Worker input cursor and loss against uninterrupted execution; require that new logs contain only post-boundary records.
5. Run `tensorflow_joint_training_checkpoint` and `tensorflow_replicated_checkpoint_rollover` without changing the default fixture contract.

### Task 2: Add an operational generation planner

**Files:**
- Create: `scripts/rotate_training_checkpoint.py`
- Extend: `tests/tensorflow_checkpoint_training.py`

1. Build replacement configs from a C++-validated committed joint manifest, exact original Worker/model definitions and explicit replacement routes.
2. Preserve declared parameter capacity, aggregation and optimizer settings; create separate new-generation replica, primary and journal storage.
3. Validate real baseline heads and paused group state before publishing a generation-ready receipt and reopening admission.
4. Exercise incomplete/corrupt checkpoints, wrong base rounds, active journals and interrupted publication.

### Task 3: Retire old storage and repeat rollover

**Files:**
- Extend: `scripts/rotate_training_checkpoint.py`
- Extend: `tests/training_checkpoint_rollover.py`
- Create: `docs/TRAINING_LOG_ROLLOVER.md`

1. Require confirmed old-role shutdown and exclusive journal/log locks before retirement.
2. Bind retirement to the validated replacement generation; reject storage containing any update beyond the checkpoint frontier.
3. Retain the committed checkpoint and all new-generation data; remove only validated old-generation log payloads.
4. Repeat rollover and a subsequent replica/Worker failure to prove bounded retained history and continued recovery.
5. Publish source-hashed evidence and exact temporary-space requirements. Keep cross-node retirement within the existing explicit role-manager lifecycle.

## Execution record

Tasks 1 and 2 are implemented. The Host-backed model fixture compares all 32
rows and continued losses with zero difference, verifies real parameter and
gradient replicas, survives an interrupted plan publication and cross-device
streaming-copy fallback, and continues after the old checkpoint route vanishes.
The original joint-checkpoint regression passes too. Linux control binaries
compile on the owned A6000 node and complete four-Worker/two-shard native
training under all three consistency policies. Source-hashed evidence is
`../validation/2026-11-04-training-checkpoint-rollover.json`.

Task 3 remains open; current prepare/verify operations retain old payloads.
