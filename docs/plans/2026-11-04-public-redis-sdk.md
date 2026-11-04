# Public Redis client for original-framework integration

## Decision

Keep the original framework and shared Redis wrapper unchanged. Supply the
observed `redis_sdk/client.h` and `bili_redis/bili_redis_client.h` interfaces from
an independently implemented adapter using Apache BRPC's Redis protocol.
The adapter sends real commands to explicitly configured Redis endpoints.

BRPC already carries the original wrapper's request/response objects. Reusing it
preserves binary command formatting and response types without introducing a
second network/response library. A hiredis adapter would require translating
requests and lifetimes. An in-memory replacement would not exercise the actual
storage dependency and is outside this decision.

## Contract

- Preserve named clients, initialization status, timeout, binary values, Redis
  nil/empty distinctions, server errors and ordered pipeline replies.
- `mget`/`mset` return one response per input record in original order, as the
  retained wrapper's callers require. Batch reads use pipelined GET commands;
  writes use pipelined SET with optional EX. Individual errors stay attached to
  their corresponding record. A batch is not an atomic transaction.
- Optional merged EXEC pipelines expose one array of ordered replies; the
  batch-record APIs always retain their observed per-record response contract.
- Explicit JSON maps logical names to endpoints; initialization replaces private
  corporate discovery with that mapping. Optional passwords are read from named
  environment variables. No implicit default destination or embedded credential.
- Existing outer retry, circuit-breaker and write-protection code stays in place.
  Network automatic retries are disabled: a lost write reply leaves execution
  ambiguous and must not silently repeat INCR or another non-idempotent command.
- Bound request count and bytes. Errors produce both an explicit Status and a
  Redis error response so the existing wrapper can observe them. Transport
  failure does not manufacture nil values. Shared clients support concurrent
  calls; per-request response objects and controllers remain independent.

## Integration and verification

Expose a separate CMake library and a Bazel `bilibili-redis` target backed by real
BRPC/JSON dependencies. Add it as a local repository in the retained isolated
native workspace; keep original code and upstream BUILD files untouched.

Run against isolated real Redis processes bound to loopback, without persistent
background services. Check binary/empty/nil values, repeated and reordered Keys,
TTL, pipeline types, batch errors, authentication, timeouts/disconnection,
non-replayed writes and concurrent callers. Then resolve the original native
build dependency graph and attempt its actual build/training entry points.
Adapter checks alone do not establish complete original Entropy execution.

Beyond-single-GPU training capacity, recovery during outstanding reads,
worker/coordinator loss, GPU-owner fencing and same-quality policy comparisons
remain in the overarching distributed-training scope.
