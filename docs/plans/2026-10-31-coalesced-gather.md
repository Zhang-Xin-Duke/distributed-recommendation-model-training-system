# Cross-shard GPU gather implementation plan

The measured capacity sweep identified repeated per-shard cache resolution as a
scaling target. Preserve the original parameter-shard tickets and update stream,
while resolving one request's cached features across all shards together.

## Design

Collect every successful shard reply before resolving cached work. Merge tasks
by `(agent, feature, access route)`, concatenate unique-row payloads, and remap
each shard-local occurrence to its original input position. Pass original feature
layouts to the resolver; sequence output then has original positions and pooled
output contains unnormalized sums. Combine CPU contributions and cached results,
then normalize means once using the original valid lengths.

Keep resident and transferred rows in separate tasks because their payload and
ownership contracts differ. Share one resolver between the native capacity probe
and TensorFlow adapter: validate all endpoints, issue one gather per destination,
then collect all completions, including when a destination fails.

Successful lookup tickets remain live until Push/Release. On lookup or resolver
failure, collect already-issued shard replies and release every known ticket
after cache reads complete. Preserve the existing failed-client boundary on an
uncertain or partially completed operation. Unknown tickets after a transport
failure still require the run's failure/restart lifecycle.

## Implementation and verification

1. Add `tests/coalesced_gather_tests.cpp`: two shards, three host-reference agents,
   repeated IDs, sum/mean/sequence, empty bags, mixed CPU/resident/transfer routes,
   per-agent call counters and a rendezvous proving parallel dispatch. Compare
   outputs and optimizer state against a CPU-only baseline. Check ticket cleanup
   after resolver errors and partial capacity rejection.
2. Update `src/client/distributed_client.cpp` to collect, remap, coalesce and
   reconstruct; document the resolver's global-layout contract in its header.
3. Add `include/eps/cache_resolver.hpp` and `src/client/cache_resolver.cpp`; use
   the shared resolver in Python bindings and the native capacity probe.
4. Build/check the native, TensorFlow and core sanitizer configurations. Existing
   process-level cache checks exercise the changed mapping with real BRPC.
5. Update implementation/validation docs and publish the change. Keep the
   existing GPU benchmark curves attached to their measured source commit;
   performance of the new path needs a paired rerun before a speedup claim.

## Execution evidence

Implemented the coalesced client and shared destination resolver. The new
two-shard/three-agent test passes for mixed routes, original sequence positions,
mean denominators, empty bags, repeated IDs and optimizer state. Rendezvous gates
prove parallel destination dispatch; per-agent counters prove one gather each.
A promise-backed gather proves a fast error waits for other destination futures.
Missing/malformed/failing resolvers and partial capacity rejection leave no known
lookup tickets live.

`ctest --test-dir build/tensorflow --output-on-failure`: **13/13 passed**,
including standalone BRPC lifecycle/restore and TensorFlow graphs using both
CPU-only and transferred-vector/cache-RPC host-reference paths.
`ctest --test-dir build/asan --output-on-failure`: **8/8 passed**.
The existing CUDA benchmark data remains associated with `db74e47`.

## Destination projection follow-up

The first alternating CUDA comparison exposes a tradeoff: merging calls leaves
each destination with full original key arrays and dense occurrence maps.
Reduce those payloads before issuing the gather. For each destination/feature,
retain only positions referenced by its tasks, preserve original bag boundaries
and mean denominators, and slice each task's occurrence map to those positions.
Resident and transferred payloads retain their existing row references.

The backend returns pooled contributions with unchanged bag count. Sequence
results use the smaller destination layout; the resolver scatters them back to
original positions before returning to `DistributedClient`. This keeps the
resolver's public contract and all shard tickets unchanged. Verify exact
destination key counts, empty bags, repeated IDs, mixed routes, sequences and
optimizer state. Measure the new client separately on the same fixed fleet.

Implemented destination projection and sequence restoration. Native/TensorFlow
checks pass 13/13 and core ASan/UBSan checks pass 8/8. The mixed-route test sends
exactly 19 destination keys for 23 input occurrences, with four CPU-only
occurrences retained at the parameter shards. Parameter and optimizer state
still matches the CPU reference. The fixed-fleet GPU comparison remains separate
from these correctness checks.
