# Physical Training Node Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Continue distributed model training after an entire Worker/primary/parameter-owner node becomes unavailable, using surviving parameter and original-gradient replicas and replacement role placement.

**Architecture:** Keep two logical Workers/shards while describing physical role placement separately for each generation. Parameter replicas, gradient mirrors and the coordinator can live on an independent surviving authority node. A durable terminal-node receipt suppresses further access to lost storage; replacement logical roles use new endpoint routes and surviving physical nodes. Existing fencing, owner rehydration and journal reconciliation remain the recovery protocol.

**Tech Stack:** Existing Python node-role transport, BRPC parameter/gradient replicas, CUDA owners and TensorFlow model Workers; provider-scoped deletion receipts for the physical trial.

---

1. Add generation-specific physical role placement, uniform installation paths and per-physical-node storage roots to the role client. Persist terminal-node receipts and reject unknown/unconfirmed loss.
2. Extend managed recovery to fetch replicas from their actual hosts, activate replacement routes before reconciliation and avoid contacting lost nodes during cleanup/reentry.
3. Run a local three-node-scope test: destroy every initial role and storage path on one scope, place its replacement roles on the survivor, and compare all 64 model/Adam rows and losses with uninterrupted training. Retain existing managed-recovery regressions.
4. Only after local readiness, allocate the minimum extra GPU nodes within the existing 20 USD gift-credit budget. Use the current node for surviving CPU authority while its original Entropy build continues. Delete only the test's explicitly owned failure node; verify provider inventory absence before recording terminal loss.
5. Verify distinct physical GPUs before failure, replacement placement, complete model/state/loss continuation, no reads of lost local journals and exact cleanup/deletion receipts. Preserve current original Entropy build and full integration requirements.

Uniform source/binary/Python paths keep role specifications identical across machines; separate physical storage roots make ownership explicit. Node loss requires a terminal receipt from the deployment observer. A transport timeout alone leaves role state unresolved and does not authorize replacement.

## Execution record

Steps 1--3 pass. A three-scope local model check destroys all training roles and
storage on the failed scope, preserves replicas on an independent scope and
reassigns replacement roles to the survivor. Three fault positions and manager
reentry after staging finish eight rounds with all 64 model/Adam rows and losses
identical to uninterrupted training. Completed-job reentry never recreates the
lost storage. The original three managed-role/recovery regressions also pass.
Evidence: `../validation/2026-11-05-physical-node-scope-recovery.json`.

Steps 4--5 pass with an owned L40 failure target and the existing A6000 survivor.
Provider deletion occurs at shard frontiers `[4, 3]`. Original-gradient replay
reconciles round three, and both replacement Workers finish eight rounds with
all 64 model/Adam rows and both complete loss histories identical to the two-GPU
baseline. A manager interpreter/module-path failure also exercises retirement
of a pre-admission recovery cohort and generation-two/term-three reentry.
All 33 registered role groups are drained, and provider inventory confirms the
failure instance is absent. No additional instance is needed for reentry.
Evidence: `../validation/2026-11-05-provider-node-loss-recovery.json`.

The original Entropy native full-module reader/training/Executor/multi-peer
acceptance remains separate and continues on the existing A6000. Cross-node
generation retirement and common-quality GPU/scaling characterization retain
their existing requirements.
