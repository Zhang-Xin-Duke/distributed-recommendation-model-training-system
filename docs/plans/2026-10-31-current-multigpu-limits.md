# Current-runtime multi-GPU capacity study

**Goal:** Measure the scaling knee and overload behavior of the resumable,
dependency-cached parameter service on one, two and four physical GPU agents.

**Architecture:** Four single-GPU machines host distinct CUDA agents. One fixed
coordinator hosts parameter shards and the load generator. The same runtime
binaries, total table size and resident-row count are retained across GPU counts.
Remote agents use SSH-forwarded BRPC; transfer controls and process/GPU telemetry
are collected with the capacity records.

**Tech stack:** C++20, CUDA, BRPC, Python orchestration, JSONL evidence.

1. Build the committed runtime on the coordinator; run native/CUDA checks and
   verify identical agent hashes and distinct physical GPU UUIDs on all workers.
2. Sweep 1/2/4 GPUs with 4/8/16/32/64/128 Learners, four parameter shards,
   two five-second windows per point. Preserve numerical preflight, sampled
   update-counter checks, finite-state and live-ticket checks.
3. Select the lowest load within 95% of the measured peak for each topology;
   confirm it with three 20-second windows. Retain the 128-Learner overload point.
4. Repeat selected four-GPU points with active limit 1 to isolate scheduling.
   Test a larger 1,048,576-row, 512-dimensional table on one and four GPUs.
   Record automatic-placement behavior separately from manual placement.
   Add a hot-shard control whose timed IDs all map to home shard 0, while keeping
   table size, vector width and resident-mixture probability fixed.
5. Collect link-transfer controls, CPU/RSS/GPU telemetry, source/binary hashes
   and raw results before deleting all rented instances. Retain failed points
   explicitly; do not reinterpret a failed run as a zero-error completion.
6. Publish an English report and overview figure. State capacity as an observed
   operating envelope for this deployment, with the workload and network path.

The source archive excludes private references and credentials. No provider
payment or credit purchase is part of this study. Each owned rental has an
independent deletion deadline; the study stops before that deadline.
