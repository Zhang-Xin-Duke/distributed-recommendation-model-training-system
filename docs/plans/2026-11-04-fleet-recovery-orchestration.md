# Fleet Recovery Orchestration

**Goal:** Run distributed training and retained-journal recovery through node role
APIs, including replicated shards, owner replacement, coordinator recovery and
worker continuation without shared worker directories.

**Architecture:** A durable node role runner starts each named generation once,
reports its actual wrapper/child state and stops only that process group. The
manager sends these operations over the persistent node transport, retains
placement/endpoint identities and combines the existing replica fencing,
owner rehydration and journal handoff protocols. Worker endpoint aliases are
normalized for manager-side reconciliation and mapped back for each worker.

**Tech stack:** Python/POSIX process groups and inherited flock, JSON/stdin/SSH,
existing C++ BRPC roles, TensorFlow model workers and journal handoff.

## Tasks

1. Add `scripts/role_node.py`: immutable intent, wrapper-owned durable handles,
   inherited launch lock until child publication, status/read/stop operations.
   Verify successful-start acknowledgement loss, duplicate start after task
   completion, configuration mismatch and orphan-wrapper process cleanup.
2. Allow the existing persistent `NodeSessions` transport to select its node
   entry point while preserving the journal default. Add a manager role client
   that observes existing handles after transport uncertainty.
3. Add fleet recovery execution using the same real model and replica roles:
   initial training, selected fault point, stop old roles, fetch worker journals,
   advance replica terms, replace owners/primaries, reconcile, stage journals,
   resume workers and inspect complete model/optimizer state.
4. Run local serialized-node checks against uninterrupted training, then prepare
   actual multi-node CUDA execution within the existing rental budget. Preserve
   endpoint-alias scope, physical GPU identities, exact state/loss comparisons,
   process cleanup and deletion receipts.

Source storage/replicas remain accessible for this failure scenario. Redundant
worker-gradient storage for entire node/storage loss, complete original Entropy
execution, compaction, gradient metadata reuse and quality/scaling measurements
remain in the overall project acceptance.

## Executed milestone

The named-role API, durable manager phases, endpoint translation and full model
recovery are implemented. The final four selected checks pass. Distinct real TCP
aliases exercise manager/worker routing; all 64 parameter/Adam rows and loss
histories match uninterrupted training after three faults. Manager reentry after
journal staging and completed-job reentry pass. A process-group census finds
zero surviving non-zombie members across the 63 recorded managed groups.

A6000 and L40 provisioning are both rejected for unavailable inventory; no
instance is created. Current source and matching sm89/sm86 native runtime bundles
are prepared for actual SSH/CUDA execution. That execution and the overall
original-framework/storage-loss requirements remain open.
