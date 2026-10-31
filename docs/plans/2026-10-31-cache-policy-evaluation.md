# Cache policy trace and CUDA evaluation

**Goal:** Validate automatic placement on a real CUDA agent and measure adaptation
cost under a fixed sparse-training trace, including migration and tail latency.

**Architecture:** A native probe drives two external PS processes and a separately
running agent. It replays identical seeded stable-hot, shifted-hot and mixed
hot/one-pass-scan phases with one Learner, preserving update order for numerical
comparison. An untimed independent CPU service replays each phase and checks the
captured features; after drain, all touched weights/moments/counters are compared.

Manual static, frequency and reuse-cost policies start from the same first hot
set. A CPU-only mode has no resident rows. Initialization, GPU warmup and initial
static loading are excluded; timed steps include lookup, output capture, feature
gradient preparation, acknowledged updates and automatic migration decisions.
Phase records include hit occurrences, full-state movement, decisions, cache
occupancy, mean/P99 latency and correctness results.

## Cost calibration

Run repeated CPU-only and static-cache stable-hot calibrations before the main
comparison. Estimate saved microseconds per distinct initially hot row from the
median complete-step difference, clamped to zero when caching is slower. Publish
both raw calibration records and the resulting prior. Automatic load/writeback
costs start from explicit byte-cost priors and learn from acknowledged migrations.
This estimate is an input to the heuristic, not a claim of universal cache gain.

## Execution

1. Add `eps_cache_policy_probe` and a process driver with finite shape/run bounds,
   fixed trace fingerprints and original-source/binary metadata.
2. Run a compact Host-reference smoke comparison, checking every touched row and
   recording all phases. Add it to the native integration suite.
3. Commit the prepared source. Rent one A6000 only after the driver is ready;
   install/build once with an instance-UUID cleanup lease and a strict deadline.
   Total project rental remains within the authorized USD 20.
4. Run native CUDA checks, automatic two-shard lifecycle checks in both gather
   modes, and paired/rotated fixed-trace trials. Do not overlap measurements with
   builds or other workloads. Retain negative results and policy failures.
5. Fetch results before deleting the instance; verify deletion. Publish curated
   raw records, source hashes, calibration, plots and a report with tested scope.

The main comparison uses a single Learner to keep row-update order identical.
The earlier multi-Learner capacity report remains a separate experiment. This
trace measures the sparse parameter subsystem with synthetic feature gradients;
model forward/backward remains outside its timed interval.
