# Reusable gradient Key metadata

Synchronous workers repeatedly submit sorted sparse Keys and fresh gradient
values. Reuse the table IDs, dimensions and exact ordered Key lists while
preserving the existing round identity, sample weighting and update semantics.

The first full Submit asks its parameter shard to register immutable metadata;
its successful reply carries an opaque token. Later requests retain their task,
sample count, read frontier and fresh gradient arrays, replacing only the Keys
with that token. A separate registration RPC would add a round trip and another
failure boundary. Piggyback registration keeps one normal request per update.

Both registries use bounded entries, conservative byte accounting, idle TTL and
LRU eviction. Extract the existing Lookup registry algorithm into a private
typed metadata registry so Lookup and gradient caches share the same lifetime
and concurrency rules while retaining different layout types. Scope tokens to
run, partition epoch, worker, shard and the server process namespace.

Resolve a token before calling the gradient collector. A miss returns no update
acknowledgement and performs no collector admission. The asynchronous client
resends the same original task and exact retained gradient values once with the
complete metadata, within the original call deadline. Scope/schema errors stay
errors. Network retries retain the existing duplicate-safe round semantics.
No gradient values, optimizer state or round clocks enter the metadata cache.
An old server returning no token continues to receive full requests.

Verify cached versus uncached optimizer state and clocks, fresh gradient values,
empty contributions, worker/epoch isolation, changed layouts, disabled budgets,
expiry/eviction and registry replacement. Count serialized Submit request bytes
(protobuf plus attachment), full/token/fallback calls and registry occupancy.
These counters describe this request path, excluding responses, replica writes,
Agent traffic and retries performed internally by BRPC. Test the shared registry
under native correctness, ASan/UBSan and TSan before publishing measurements.
