# Configuration examples

`service.example.json` configures a single-shard CPU deployment with manual cache
placement. `service.cache.example.json` enables automatic reuse-cost admission
against a separately running cache agent. Set its `agents[].endpoint` and run
identity to match that agent. Settings are examples, not measured tuning values.

`access.policy` selects the route for **CPU-owned contributions**: `cpu_only`,
`transfer_unique`, or `cost`. Resident rows always use their current cache owner.
Transfer/cost routes require an attached agent. Cost mode can consume explicit
paired CPU/GPU route observations; with insufficient observations, pooled
features keep the CPU baseline. Sequence features preserve positions.

`cache.enabled` attaches the configured agents. `cache.admission` selects manual,
frequency-window or reuse-cost placement. Automatic modes expose bounded access
history, minimum residence, cost priors, migration budgets and free-slot reserve.
See [the policy design](../docs/CACHE_POLICY.md) for units, flow, lifecycle and
metrics. The initial controller sequences migration batches; its batch cap also
limits the total issued per decision window.

`cpu_state_bytes` bounds numeric w/m/v and row-counter arrays.
`ticket_payload_bytes` bounds retained numeric ticket payloads;
`queue_payload_bytes` bounds captured request arrays awaiting execution.
Container metadata is additionally bounded by request, table and ticket counts.
The key-to-row index has metadata proportional to allocated rows and is outside
the numeric state-array budget. Automatic-policy history is bounded by
`max_tracked_rows`; resident age metadata is proportional to cached rows.

Missing IDs are initialized from the configured seeded distribution. Cache
admission changes placement while keeping the same sparse optimizer and update
boundaries. `entropy-adapter.example.json` describes framework-adapter settings;
the portable TensorFlow host bridge and its commands are documented under
[adapters/entropy](../adapters/entropy/README.md).
