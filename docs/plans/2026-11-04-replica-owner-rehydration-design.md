# Replica-backed parameter-owner recovery

Recover the same training run after the manager stops all former workers,
primaries, coordinator and GPU owners. Advance the durable replica terms, start
empty replacement owners, and restore primaries paused from their replicas.
Each owner namespace is a SHA-256 identity of training run, partition epoch,
home shard, new primary term and primary identity. Parameter/control/read
clients all select that namespace. Stable optimizer request identities remain
unchanged.

The recovery planner verifies the granted replica roles, deterministic owner
identities, table/cardinality and journal/input identity. Reconciliation keeps
the coordinator paused, pauses gradient collectors and checks their empty
boundaries. For each cached shard, a replica-aware preparation path checks the
same-run restore provenance, live replica term/round and optimizer checkpoint,
then migrates existing full row state to the replacement owner. Missing rows
fail rather than being initialized. Repeated preparation skips rows already
loaded on the correct owner and checks complete resident counts.

Replica after-images intentionally make ordinary pristine-snapshot provenance
false. The new preparation mode validates live replica identity/frontiers and
paused optimizer clocks instead of weakening the existing snapshot rehydration
contract. It rechecks replica frontiers after migration. A partial migration
can resume against the same replacement processes; failed gradient replay may
require replacing primaries from durable replicas again.

Only after every owner is rehydrated does reconciliation resume collectors and
roll a partially committed round forward with the original worker gradients.
The coordinator stays paused until the returned worker configurations select
the new owners and the operator/job manager resumes training. A repeated
reconciliation checks the same durable boundary and does not reapply gradients.

Exercise all existing read/partial-commit/task-completion fault points with two
resident owners. Kill and replace actual owner processes at the same endpoints,
compare every weight/moment/update counter and every round loss with uninterrupted
training, and check bad owner identities/terms/cardinality before training resumes.
Run the same path with CUDA owners and GPU model computation when rental
inventory permits. Distinct-GPU capacity and full original Entropy execution
remain independent parts of overall project acceptance.
