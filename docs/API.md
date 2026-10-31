# Parameter-server API and Entropy integration

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

`DistributedClient` collects shard replies before resolving cached features.
It merges work by `(agent, feature, access route)` and remaps shard-local
occurrences into original input positions. A `CacheResolver` receives these
merged tasks with the original feature layouts; sequence outputs therefore use
original positions, while sum/mean outputs remain unnormalized contributions.
Resident and transferred vectors retain separate task payloads.

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
