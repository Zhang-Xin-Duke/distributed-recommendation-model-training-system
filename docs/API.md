# Parameter-server API and Entropy integration

Shard operations use a [resumable actor](RESUMABLE_OPERATIONS.md). Independent
requests can progress during owner waits; row/lifecycle/optimizer dependencies
and ordered full barriers preserve the update and save contract. Stats fields
37–40 expose active/waiting operations and retained payload/dependency bytes.

## Public service surface

| Operation | Input | Output |
|---|---|---|
| CreateTable | Table identity, dimension, optimizer, partition | Table handle |
| Lookup | Table, keys, bag/sequence metadata, request identity | Features/route plan and lookup ticket |
| Push | Lookup ticket and gradients | Update completion |
| Release | Read-only or coordinated-abort ticket | Reference release completion |
| ReadState | Row keys and placement generation | Complete row states |
| PlanMigration | Access/cost observations and resource budgets | Admission/eviction batches |
| CompleteMigration | Transfer completion and generation | Published placement changes |
| DrainAndFlush | Save boundary requested by the framework | Shard snapshot metadata |

`PlanMigration` is a planner/coordinator interface; the other operations are in
the service protocol. The adapter groups requests by home shard and holds the
resulting shard-local tickets as one framework operation.

## Lookup contract

The request carries a table ID, feature ID, flat keys, bag offsets, valid lengths
and a sum/mean/sequence mode. The service deduplicates `(table,key)` and retains
an inverse map, so internal order may differ from the caller's feature order.

Offsets delimit the original bags or sequences. The adapter removes padding IDs
and retains original valid lengths before partitioning. Empty bags produce zero
vectors, including mean pooling; sequence lengths and repeated positions remain
unchanged. A feature ID is unique within a request.

CPU and GPU contributions for sum/mean are unnormalized partial sums. The adapter
combines all shard/location contributions and applies the mean denominator once.
Sequence contributions carry their original positions. `Push` receives gradients
of these completed feature tensors; service-side unpooling applies the corresponding
derivative once.

The lookup ticket identifies the original request and the rows/locations it
references. It remains valid until its corresponding update or explicit release
completes. Read-only evaluation requests release after feature consumption.
Cancellation alone does not release a training ticket: the framework must first
finish or abandon dependent forward/backward and GPU work.

`DistributedClient` submits parameter lookups concurrently. By default it
processes replies in shard order and starts a shard's cache work before waiting
for the next reply. The resolver receives shard-local work and layouts.
`SetCacheResolver(resolver, true)` enables cross-shard coalescing: collect all
replies, merge work by `(agent, feature, access route)` and remap occurrences into
original input positions. In either mode, resolver outputs match the supplied
layout and sum/mean contributions remain unnormalized. Resident and transferred
vectors retain separate task payloads.

The default avoids an additional all-shard barrier before cached reads. Coalescing
reduces calls when that saving offsets waiting for the slowest shard. The portable
Python/TensorFlow bridge exposes the same choice through
`cache_endpoints(agents, coalesce=True)`; its default is also per-shard execution.

`MakeCacheResolver` validates destination endpoints, dispatches one gather per
destination and collects every returned future before reporting an error.
Before dispatch it projects key arrays, bag offsets and occurrence maps onto the
positions that destination reads. Original mean denominators remain unchanged;
sequence contributions are scattered back to original positions on return.
Custom resolvers must also collect all submitted read completions before returning.
If lookup, reconstruction or cache resolution fails, the client collects any
already-submitted shard replies and releases every known successful ticket.
The client then remains failed. Unconfirmed remote operations or unknown tickets
after transport failure belong to the run controller's restart boundary.

## Update contract

Push carries the gradients of the returned features. The adapter keeps existing
loss scaling and collective behavior. The service distributes mean-pooling
contributions using the original valid lengths exactly once.

Repeated IDs are merged within this original Push. Independent Learner updates
remain independent optimizer updates. A shard-level update context gives CPU
and GPU portions the same optimizer counter and hyperparameters.

The request identity is `(run, learner, sequence, phase)`. A retry retains that
identity. Accepted updates are acknowledged without adding a duplicate gradient
contribution. A duplicate of an in-flight request joins its completion; an ack
means the update completed. A per-learner retired watermark advances only over
contiguous completed lookup lifecycles, retaining gaps for out-of-order completion.
Push and read-only Release close the same lifecycle sequence, so alternating
training and evaluation does not leave holes in independent phase streams.
Retired identities are rejected without executing them again. This covers a live
run; owner failure follows the framework restart boundary.

The CPU runtime implements this lifecycle. `DistributedClient` fans out to all
configured shards and combines their CPU contributions. If a distributed operation
fails after dispatch, that client run is stopped: cross-shard updates are not a
transaction. Restart uses the framework checkpoint boundary and a new run ID.

The initial optimizer profile uses FP32 sparse LazyAdam with independent w/m/v
arrays. Its formulas, epsilon placement, beta powers, initialization and update
boundary must be mapped explicitly by the adapter. Other optimizer profiles are
rejected until supported. The same original request context is used across CPU
and GPU work; no new global step synchronization is introduced.

## Entropy adapter

The adapter sits at the existing embedding Pull/Push/Join boundary. It:

1. Translates existing table configuration into service table definitions.
2. Sends key arrays and row metadata in batched attachments.
3. Returns the feature tensors expected by existing graph nodes.
4. Routes gradients and completion notifications using the lookup ticket.
5. Connects the existing save hook to `DrainAndFlush`.

No dataset, loss function, dense-model graph or dense synchronization strategy is
replaced. GPU-resident execution uses a cache agent in the existing GPU worker.

## Cache-agent contract

The service can ask the agent to load complete state, read/pool resident rows,
apply a ticketed sparse update, and return rows selected for eviction. The agent
reports completion after its CUDA work finishes. The service publishes the new
location and releases old capacity after the acknowledged move.

Bulk transport uses attachment-relative typed slices, validated against the
attachment length and configured request limits. C++ service contracts use owned
host vectors; the GPU agent contract uses adapter-owned device tensor handles so
resident features and gradients can stay in the existing GPU execution context.

## Service independence

The public contract refers to tables, keys, vectors, feature layouts and gradients.
It does not encode user/video naming, application-specific model layers, or a
particular dataset. Another framework can supply a client adapter to this API.

## Saving in the current runtime

`DrainAndFlush` closes lookup admission first. Outstanding tickets must complete
or be explicitly released; the controller retries drain after reaching that
boundary. Snapshots are immutable shard files published after flush/fsync, with
weights, moments, row counters, table definitions and optimizer globals.
`CpuParameterService::Resume` reopens a local service after coordinated saving.
The standalone server restores with `--restore FILE --run NEW_RUN`; remote resume
control and the Entropy save hook remain integration work.


## Automatic cache placement

After attaching agents, local callers can configure `CachePolicyOptions` with
`CpuParameterService::ConfigureCachePolicy`. The standalone launcher maps the
same settings from `cache.admission` and related JSON fields. Manual is the
default; frequency and reuse-cost modes observe successful distinct-row lookups
and decide after configured completion windows.

`MigrationPlan.reserve_slots` is transported to the destination and checked in
its ordered Load allocation before publishing any row. Updated PS/Agent binaries
are required for this reserve guarantee. Policy byte/batch budgets, read-only
completion behavior and Stats counters are defined in
[CACHE_POLICY.md](CACHE_POLICY.md).

## Prepared input API and stage timing

`PrepareLookupInputs` builds move-only original/shard layouts and position maps
without parameter access. `DistributedClient::LookupPrepared` validates the
prepared topology/tables and performs actual reads. `TableAdapter` adds
`PrepareAsync`, FIFO `PullPrepared`, `CancelPrepared` and `PreparationStatus`;
its worker also handles padding and original sequence positions. Python exposes
`prepare`, `pull_prepared`, `cancel_prepared` and `preparation_stats`.

Queued input count and conservative metadata/scratch reservations are bounded.
Save requires live handles and pending tokens to be closed before atomically
closing input admission. Stage counters are additive Stats fields 29–36;
timing scopes and ownership boundaries are defined in
[asynchronous preparation](ASYNCHRONOUS_PREPARATION.md).
