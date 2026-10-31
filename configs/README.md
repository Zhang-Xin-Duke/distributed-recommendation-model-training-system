# Configuration examples

`service.example.json` can be used by the implemented `eps_server` launcher.
It configures a local single-shard CPU deployment. Values are examples rather
than measured tuning recommendations.

`service.example.json` selects the table layout, sparse optimizer, access policy,
migration policy settings and resource bounds. The running server currently
requires `access.policy=cpu_only` and `cache.enabled=false`; cost/migration
policies are checked as a library, with GPU integration still pending.
`entropy-adapter.example.json` describes the planned framework adapter.

`cpu_state_bytes` bounds the allocated w/m/v and row-counter arrays.
`ticket_payload_bytes` bounds retained numeric ticket payloads, and
`queue_payload_bytes` bounds captured request-array payloads awaiting execution.
Container metadata is additionally bounded by request, table and ticket counts.
The key-to-row hash index has additional metadata proportional to the number of
allocated rows; it is not included in the numeric state-array budget.

The initial initialization policy creates every requested missing ID from the
configured seeded distribution. Cache admission only changes placement. The
adapter must reject unsupported feature-creation or optimizer settings rather
than silently substitute another training policy.
