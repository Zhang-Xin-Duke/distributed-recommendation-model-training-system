# Distributed Generation Retirement Implementation Plan

**Goal:** Retire obsolete training checkpoints, parameter replica logs and worker
gradient journals on their owning machines after a durable replacement is ready.

**Architecture:** The manager verifies the published replacement generation and
assigns every old storage resource to exactly one node. Each node validates its
local stopped-process evidence, journal frontiers and storage locks, publishes a
certificate, and seals all old roots without deleting payloads. Only after every
node acknowledges sealing does the manager durably publish the commit decision.
Participants then unlink the certified files and return replayable receipts.
SSH node streams carry bounded metadata; payload files stay on their hosts.

**Tech Stack:** Python, existing persistent NodeSessions transport, flock/fsync
publication, original worker-journal codecs and native replica inspection tools.

---

1. Extract the existing storage inspection into `scripts/retirement_storage.py`.
   Preserve worker/mirror identities, primary-parent locks, replica frontiers,
   snapshot inspection and replacement-path exclusions. Keep legacy host-local
   retirement on the shared inspector and run its existing model regression.
2. Add `scripts/retirement_node.py` with inspect/seal/commit operations. Persist
   the certificate before any marker, require all markers before unlinking, and
   recover interrupted sealing/deletion from that certificate. A completed
   receipt must replay without contacting a later retired replacement.
3. Add `scripts/distributed_retirement.py`. Bind source-resource ownership,
   stable node IDs, source process handles and replacement readiness into an
   immutable intent. Record every seal receipt and the all-nodes commit decision
   before sending deletion requests. Reentry and lost replies retry the same
   generation; a transport timeout never proves node loss or retirement.
4. Verify separate participant processes with real journal/replica files,
   missing resource membership, live locks, future gradients, interrupted seal,
   lost successful commit reply, manager reentry, and unchanged unrelated/new
   files. Exercise full TensorFlow model continuation after distributed
   retirement and compare complete parameters, Adam state and loss trajectory.
5. Publish design, commands and measured evidence. Physical-host SSH execution
   follows the local protocol checks when code and deployment are ready; keep
   that evidence distinct from local participant-process execution.

Shared storage would simplify direct unlinking but require filesystem deployment
and common locking semantics. Ad hoc SSH removal would not retain a recoverable
group decision. Per-node sealing uses the existing storage formats and transport,
at the cost of a durable two-stage control protocol and small retained receipts.

## Executed progress

The shared inspector retains the host-local repeated-retirement regression.
The node/manager protocol passes interrupted publication/unlink, changed and
dangling payloads, changed receipts, lost successful responses, partial sealing
and partial commit. The native TensorFlow regression uses two training-storage
participants plus an immutable-checkpoint controller. Two round-4/8 rotations
and later round-10 process loss recover to round 12 with all 32 full parameter/
Adam rows and loss trajectories matching uninterrupted training exactly.

These tests use separate participant processes on one physical host. The SSH
transport path is implemented; actual physical-host execution remains a gate,
followed by fixed-workload physical multi-GPU scaling within the standing budget.
