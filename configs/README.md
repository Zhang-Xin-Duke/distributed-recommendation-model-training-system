# Configuration examples

These files describe the proposed service and adapter configuration. A parser and
server launcher will be added during implementation. Values illustrate a local
single-shard deployment; they are not measured tuning recommendations.

`service.example.json` selects the table layout, sparse optimizer, access policy,
migration budgets and resource bounds. `entropy-adapter.example.json` specifies
the service endpoints and the existing-worker GPU agent. GPU caching is disabled
in both examples for the first CPU-service milestone.

The initial initialization policy creates every requested missing ID from the
configured seeded distribution. Cache admission only changes placement. The
adapter must reject unsupported feature-creation or optimizer settings rather
than silently substitute another training policy.
