# Physical placement and terminal-node recovery

Training Workers and parameter shards keep their logical identities when their
processes move to another machine. A deployment supplies generation-specific
physical placement and replacement endpoint routes. Parameter and original-
gradient replicas remain on surviving hosts, so recovery can rebuild owners and
continue without opening the lost Worker's local journal.

```mermaid
flowchart LR
    A[Lost node: Worker, primary, parameter owner] --> T[Deployment observer confirms terminal loss]
    P[Surviving parameter replicas] --> F[Fence former primary terms]
    G[Surviving original-gradient mirrors] --> R[Reconcile committed rounds and task progress]
    T --> F
    F --> R
    R --> O[Rebuild owners and primaries at replacement locations]
    O --> W[Stage journals and restart logical Workers]
```

## Deployment configuration

The existing managed training controller has two logical Workers/shards. Its
`nodes` entries describe each logical Worker's initial routes. Physical
deployments additionally provide:

| Field | Purpose |
| --- | --- |
| `physical_nodes` | Node transports, stable `node_id`, storage roots and optional provider UUID |
| `role_placement["0"]` | Initial physical host index for each named role |
| `role_placement["1"]` | Replacement physical host index for each named role |
| `replacement_nodes` | Replacement per-Worker routes and role listeners |
| `replacement_manager` | Manager-visible replacement routes |

For example, initial Worker/primary/owner roles can use physical indices `[0, 1]`
while replicas, gradient mirrors and the coordinator use an independent survivor
at index `2`. The replacement generation can place Worker/primary/owner roles at
`[1, 1]` with distinct listeners. Replica and mirror storage stays on its original
surviving physical host for this transition.

Source, Python, binary and native-module installation paths are uniform across
the physical roster. Storage roots can differ. This keeps immutable role commands
portable while each machine owns its own role state and journals. The declared
routes must resolve to the corresponding host/listener; placement does not
implicitly rewrite network endpoints.

## Confirm loss before skipping a node

`PhysicalRoles.record_loss()` accepts a terminal deployment-observer receipt.
A provider deletion receipt must name the run, stable node identity and the
deployment's previously owned UUID, with fresh inventory evidence that the UUID
is absent. The observer owns the provider operation and verification; the role
client does not delete cloud resources.

A transport timeout leaves the node unresolved. It cannot authorize skipping
its lifecycle checks. Terminal receipts are fsync-published under the manager
state directory and reloaded on manager reentry. The client refuses loss of a
node still required to host a surviving replica/mirror, coordinator or declared
replacement role.

After confirmation, old-role status and stop return the terminal-node evidence
without contacting that machine. Reads and new actions against its lost roles
are rejected. New-generation roles use their declared physical hosts, even when
their logical Worker index formerly belonged to the lost machine.

## Recovery sequence

1. Stop old roles on surviving nodes; retain the terminal receipt for roles on
   the lost node.
2. Activate replacement Worker and manager routes, including on manager reentry.
3. Fetch each original-gradient journal from its actual mirror host.
4. Restart surviving parameter replicas, grant new primary terms and create the
   replacement owners and primaries.
5. Reconcile partially committed rounds, rehydrate complete owner state and
   recover coordinator progress through the existing retained-gradient protocol.
6. Seed fresh Worker and mirror stores, reopen admission and resume logical
   Workers at their recovered input positions.

Replica fencing and owner incarnation checks retain their existing guarantees.
Replacement placement must have enough memory for the recovered state. Placement
can co-locate logical roles when the working set fits; it does not add GPU
capacity automatically.

## Executed local node-scope checks

`managed_physical_node_scope_recovery` uses three local process/storage scopes
and distinct real TCP aliases. At each fault it stops every training role on
scope zero, removes that scope's storage, records an explicit local-test receipt
and moves its replacement roles to the surviving scope. Such receipts require
`allow_test_loss=true` and are rejected for remote production nodes.

The fixture covers loss after parameter Pull, a partially committed round and
updates committed before task completion. In the last case it also interrupts
the manager after journal staging and reenters from durable state. No lost
storage directory is recreated during recovery or completed-job reentry.

All 64 parameter rows, row update counts, Adam moments and every loss match
uninterrupted eight-round training exactly. Role cleanup receipts confirm the
old terminal scope and all remaining managed roles are gone. Actual provider
node termination is a separate physical trial.

[Source hashes, fault cases, replacement placement and cleanup records](validation/2026-11-05-physical-node-scope-recovery.json).

```sh
ctest --test-dir build/tensorflow --output-on-failure \
  -R '^managed_physical_node_scope_recovery$'
```
