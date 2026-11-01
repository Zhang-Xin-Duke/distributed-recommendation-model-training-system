# Automatic cache policy implementation plan

**Goal:** Connect reuse-aware cache admission, replacement and migration budgets
to the running parameter service, with static/manual and frequency baselines.

**Architecture:** Successful fresh lookups feed bounded, decayed per-row request
statistics. Completed Push/Release operations advance decision windows on the
existing shard sequencer. At a window boundary, the controller skips pinned rows,
uses existing acknowledged full-state migration, and retains CPU fallback when
an admission cannot fit. Cache failure and snapshot semantics remain those of
the service. Configuration opts in; existing benchmark defaults remain manual.

**Tech stack:** C++20, existing CacheBackend/BRPC protocol, FP32 sparse LazyAdam,
native reference/backend checks and ASan/UBSan.

## Design choices

An external polling service would require exporting access traces and would race
ownership decisions. An independent background thread inside each PS would need
a new synchronization protocol for tickets, updates and snapshots. Integrating
coarse decisions on the existing sequencer reuses its ownership boundary and
makes migration costs visible in request latency. Asynchronous preparation and
nonblocking agent updates remain separate work.

Track one reuse observation per row per successful fresh lookup, rather than
counting repeated positions as independent future accesses. Bound the observation
map with LRU replacement and lazily decay old observations by decision window.
Use configurable saved-access and migration-cost priors; optionally update
load/writeback estimates from acknowledged migration elapsed time per state byte.
These estimates are policy inputs, not benchmark performance claims.

Only replace a same-table resident when the candidate's predicted net benefit
exceeds the resident's retained benefit and writeback cost by a configured
margin. Minimum residence windows reduce churn. Keep row counters with weights
and moments through the existing Move path. A window has shared network/staging
byte budgets, a strict migration-batch cap and a reserve of free table slots.
The initial controller issues migration batches sequentially; its batch cap is
stricter than an in-flight-only bound.

## Implementation and checks

1. Add public CachePolicyOptions and internal bounded reuse statistics, cost
   scoring, minimum-residence tracking and migration timing feedback.
2. Add ConfigureCachePolicy; collect fresh-lookup observations; run decisions
   after successful ticket completion. Skip draining/failed services. Reuse
   Move for admission/eviction and publish policy metrics through RPC Stats.
3. Parse manual/frequency/reuse_cost modes and explicit budgets in server JSON;
   reject invalid modes/options. Document units and estimated-versus-measured
   costs. Keep manual as the default.
4. Add native service checks against CPU weights/moments/counters for changing
   hot sets, reserved slots, pinned rows, duplicate requests, bounded statistics,
   insufficient budgets, recoverable admission failure, and save/restore.
5. Add a two-shard external BRPC/agent check exercising automatic replacement,
   policy counters, configured bounds and continued optimizer trajectory after
   restart. Use the host protocol backend locally; real GPU execution is a
   distinct later verification requirement for this new policy.
6. Build native/TensorFlow and core sanitizer suites. Update architecture,
   validation and implementation status; commit and push verified changes.

No new GPU instance is rented for these local lifecycle checks. Existing GPU
capacity records stay attached to their measured commits.


## Executed evidence

Implemented the controller, configuration, RPC counters and atomic reserve field.
`ctest --test-dir build/tensorflow --output-on-failure`: **19/19 passed**.
`ctest --test-dir build/asan --output-on-failure`: **9/9 passed**.
The two external-process checks run both gather modes and fresh/restored
optimizer trajectories. Native coverage includes multiple table layouts/agents,
changing hot sets, pins, bounded observations, separate network/staging limits,
failed replacement admission and reserve enforcement inside allocation.

The new CUDA reserve check and CUDA-enabled process driver are supplied but
have not been executed on a device in this iteration. This iteration creates
no GPU instance. Source/log hashes are in
[the validation artifact](../validation/2026-10-31-cache-policy.json).


The portable Python/TensorFlow bridge also exposes runtime Stats. Two graph
checks run automatic residency with per-shard/coalesced gathers, train five
steps with decreasing loss and no retracing, verify policy counters/resident
rows, and drain at save. Together with the other checks, the native/RPC/TensorFlow
suite is now 19/19; core ASan/UBSan remains 9/9.


## Subsequent real-device evaluation

Source `3a2cd7e` subsequently passed 17/17 Ubuntu CPU/CUDA checks and three
additional explicitly CUDA-backed process checks. Two fixed-trace profiles
completed 36 independent runs and 6,528 updates with full feature/final-state
replay. Current local native/TensorFlow coverage is 20/20. These results and
cost-model behavior are in [CACHE_POLICY_RESULTS.md](../CACHE_POLICY_RESULTS.md).
