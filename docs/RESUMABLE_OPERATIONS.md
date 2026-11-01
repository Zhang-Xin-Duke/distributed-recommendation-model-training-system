# Resumable parameter-shard operations

The shard keeps one thread responsible for parameter state, tickets, placement
and optimizer clocks. When a cache operation is pending, its coroutine suspends;
the same thread can execute another operation whose dependencies are independent.
A completion worker observes the external future and queues its continuation
back onto the shard thread.

```mermaid
sequenceDiagram
    participant A as Learner A
    participant S as Shard actor
    participant G as Cache agent
    participant B as Learner B
    A->>S: Push gradients for rows X
    S->>G: Apply with original Adam context
    Note over S: Suspend A; retain row, optimizer and agent dependencies
    B->>S: Lookup independent rows Y
    S-->>B: Current features / route and ticket
    G-->>S: Update complete
    Note over S: Resume A on the shard thread
    S->>S: CPU portion, optimizer clock and ticket completion
    S-->>A: Update acknowledgement
```

## Dependencies preserve the parameter lifecycle

| Operation | Dependencies retained until completion |
| --- | --- |
| Lookup | Request lifecycle and referenced rows |
| Push | Request lifecycle, referenced rows, each table's optimizer clock and touched cache-agent control streams |
| Release | Request lifecycle; automatic-policy completion stream when enabled |
| ReadState | Referenced rows and their current cache agents |
| MoveRows | Moving rows and the destination/source cache-agent control stream |
| Locations | Inspected rows |
| Stats | No parameter dependency |
| Create/configure/restore/save/resume | Full shard barrier |

A younger operation can pass a blocked request only when their dependencies are
independent. Older conflicting requests retain priority. Same-table updates keep
their original Adam context order, even when they touch different rows. Only one
control operation per home/agent stream is outstanding, preserving its replay
identity. Lookup tickets remain live while their Push is suspended; a retry waits
for the same lifecycle and then receives the retained completion.

Queued requests retain their normalized dependency plan, avoiding repeated key
collection and sorting on every queue wakeup. A placement/policy epoch invalidates
queued plans when their interpretation changes; active operations keep their
acquired dependencies. A Push whose ticket does not exist yet remains uncached
until its preceding Lookup supplies that layout. Optimizer/agent keys are checked
before long row lists, so common control conflicts can stop comparison early.
This uses bounded retained metadata to reduce CPU scheduling work.

Moving rows stay excluded through loading, writeback and acknowledged eviction.
New locations/generations publish at the existing acknowledgement boundary.
Unrelated rows can be read while the move waits. Store/ticket references are
used on the actor thread; map iterators are reacquired after suspension where
other operations may insert entries.

## Automatic maintenance and save

Automatic-policy completers share one ordered maintenance resource. During
ordinary owner waits, independent lookups can proceed. At a decision window the
completing operation upgrades to a full barrier: it closes younger admission,
waits for other active operations to finish, then examines residents and performs
its bounded migrations. Other active continuations can still resume to finish
that drain. The shared maintenance resource permits one upgrader at a time.

Save enters as an ordered full barrier. It follows previously admitted work,
checks live tickets, drains cache state and writes the snapshot. Younger reads
cannot cross the barrier; they encounter the existing draining state. Clients
still complete/cancel prepared inputs and reach the framework save boundary.

## Bounded execution

The JSON launcher accepts these `limits` fields; C++ uses the corresponding
`RuntimeOptions` fields:

| Field | Default | Scope |
| --- | ---: | --- |
| `max_queued_operations` | 1,024 | Accepted operations not started yet |
| `max_active_operations` | 16 | Started, suspended and resumed operations |
| `completion_waiters` | 4 | Native workers observing external futures |
| `queue_payload_bytes` | 64 MiB | Combined queued and active input payload reservations |
| `dependency_payload_bytes` | 64 MiB | Queued and active dependency-vector capacity |
| `cache_dependencies` | true | Reuse stable dependency plans for queued requests |

Suspension retains the input reservation; it cannot be used to admit unlimited
additional payloads. Queued and active plans share the dependency byte budget.
If optional queued plans consume that budget, the scheduler evicts those plans
and retries after active progress; a single oversized plan is rejected.
The scheduler bounds retained scan metadata separately from its temporary builder.
Operation count limits bound headers and coroutine records; these payload budgets
are separate from parameter state, lookup tickets, allocator overhead and agent
staging. `max_active_operations=1` supplies a serialized scheduling control for
matched experiments using the same binary.

The completion pool uses native future notification. It does not poll every
pending RPC. Each operation has at most one external batch wait at a time.
Waiter and result storage are reserved before dispatch; every returned future in
a batch is observed even when a later dispatch or an earlier result fails.
Waiters do not access shard state. Destruction closes new admission and joins
accepted work and completion workers before destroying the state they reference.
External backend methods must return their futures promptly; transport deadlines
and the existing run controller govern unreachable owners.
External futures are completed by transport/agent workers. Nested actor work
composes an `ActorTask` directly, keeping it on the same execution owner.

## Failure and observation

An ambiguous cache-owner failure fences the run. A different suspended operation
checks that fence before committing its CPU portion or acknowledging completion.
Already submitted device work is still observed. Known tickets can be explicitly
released after fencing; checkpoint recovery uses a new run ID.

Stats adds fields 37–40: `active_operations`, `waiting_operations`,
`active_payload_bytes` and `dependency_payload_bytes`. The active count includes
the Stats operation itself. Waiting includes external futures and a pending
maintenance barrier upgrade. Queued payload continues to mean not-yet-started
input; admission considers it together with active payload.
Fields 41–43 expose `dependency_prepare_nanoseconds`, `dependency_rebuilds` and
`dependency_cache_hits`. The dependency-byte counter includes queued cached plans
and active plans under their shared limit. `cache_dependencies=false` provides
a same-binary cache-off control.

`executor_busy_nanoseconds` now measures dependency planning and executed actor
segments, excluding parked external/barrier waits. Older capacity revisions
included owner waits in executor occupancy; their records retain that definition.
New capacity records include `executor_busy_scope`. Service phase times remain
inclusive wall-time scopes: concurrent owner waits and nested maintenance scopes
can overlap and must not be summed as disjoint CPU time.

## Validation and measurement

`resumable_actor_dependencies` covers independent progress, older-conflict
priority, barriers and upgrade drain, bounded input/dependency admission, nested
coroutines, fan-out cleanup and shutdown. `resumable_shard_operations` injects
delayed owner updates and two-way migrations, checks reads and retry order,
compares Adam state with CPU execution, verifies save ordering and fences other
continuations after an owner failure. A CUDA build registers the same integration
case as `resumable_cuda_shard_operations` using real CUDA storage and updates.

The Ubuntu GCC 11 check exposed repeated destruction of a braced owning
aggregate passed directly in an await expression. A standalone string/vector
reproducer failed without any service threads or CUDA; naming the request before
moving it into the child task resolved that case. Migration calls use that form.
The compiler's [temporary-lifetime discussion](https://gcc.gnu.org/pipermail/gcc-bugs/2022-November/805094.html)
provides related background; the service's Linux checks validate its actual path.

Use `scripts/capacity_sweep.py --max-active-operations 1` and `16` under the same
hardware, workload and trial order for a scheduling comparison. The published
four-agent capacity study keeps its original source revision; it is not a
performance claim for this implementation.
