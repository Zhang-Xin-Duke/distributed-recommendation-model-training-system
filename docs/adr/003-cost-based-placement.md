# 003: Choose access and placement by cost

**Status:** Implemented with native, CUDA-process and fixed-trace validation, 2026-10-31.

## Context

The reference designs already compare CPU pooling with caching. Fewer transferred
bytes can still mean a slower batch if CPU aggregation or queueing dominates.
A high cache hit rate can conceal expensive state migration.

## Decision

For CPU-backed contributions, compare measured CPU-pooling and unique-vector/GPU
routes. For admission, estimate future access savings minus full-state loading,
writeback and capacity cost. Bound migration bytes and in-flight work, reserving
some slots for upcoming admissions. Use decayed observations and a switching
margin to reduce policy oscillation.

Keep static CPU-only and frequency/window cache policies as comparison modes.
Observe original occurrences, unique rows and pooled features separately.

## Consequences

Placement responds to the work saved, rather than a single compression or hit-rate
metric. Statistics and planning consume CPU time; sampled observations and coarse
decision windows keep that overhead bounded. Reserving slots lowers usable cache
capacity in exchange for shorter reclamation waits. Policy benefits require later
end-to-end measurement under fixed model and optimizer settings.


## Implementation

`src/cache/reuse_policy.cpp` maintains bounded observations and cost/residence
state; the service executes windowed decisions through acknowledged Move calls.
`cache.admission` selects manual/frequency/reuse-cost modes. Agent allocation
atomically enforces the free-slot reserve. See [the policy design](../CACHE_POLICY.md)
for lifecycle, budgets, costs, metrics and the current sequencer trade-off.
