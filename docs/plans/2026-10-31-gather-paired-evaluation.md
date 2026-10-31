# Paired CUDA evaluation of coalesced gathers

Compare baseline client `db74e47` and coalesced client `5a4cdcb` on the same
prepared instances, keeping the server, CUDA agent, compiler, transport, table,
cache fraction and workload fixed. The existing GPU-agent implementation is
unchanged between these revisions. Verify native CUDA correctness first.

Use three RTX A6000 instances, with one also hosting the load generator and PS
processes. That differs from the earlier L40 coordinator deployment, so compare
the two paths within this new experiment. Keep runtime and source provenance
with the records.

For one, two and three agents, run four concurrency levels (4, 8, 16, 32 Learners)
in three blocks. Each trial has a five-second submission window plus drain.
Alternate baseline/candidate execution order by block and topology; every run
starts fresh agents and parameter shards. Retain all successful trials and their
observed ranges. Report step throughput, lookup/update/step P99 and queue time.
Inspect sampled update counters, finite state and live tickets after each trial.

The controller records its exact run order and completion states. Connection
details remain in ignored local files. Per-instance UUID leases have cleanup
deadlines; the total authorized rental budget remains USD 20. Fetch raw logs and
numeric records before deleting instances, then verify their absence. Add a
longer-window check at the measured operating point if the initial comparison
justifies it and the lease time remains sufficient.

Update the benchmark report with paired results and limitations, regenerate
figures and publish. Preserve the previous dataset and its source revision.

## Destination-projected follow-up

Retain the first 72 trials for the full-layout coalesced client, including
regressions. Evaluate the destination-projected client separately against the
same baseline and fixed server/agent builds. Extend concurrency to 64 and 128
Learners, giving 108 trials across the same three GPU counts and three paired
blocks. This checks whether a rising curve at 32 Learners reaches a plateau or
declines. Compile the new probe into a separate build after the first timed suite
finishes; keep all measured binaries immutable throughout each suite.

The projected probe also performs an untimed numerical preflight: capture 96
sampled rows' CPU weights before migration, then compare mean/sum/sequence
features across resident and cold IDs after migration. Repeated positions and
empty bags are included. At dimension 128 this checks 17,024 output values.
Read-only release precedes warmup and timing; no optimizer update or benchmark
input RNG change is introduced. Each timed record retains the checked-value count
and maximum absolute error. A three-agent host-reference smoke check passed this
preflight locally; those local rates are excluded from GPU results.

## Completed evaluation

The 72 full-layout, 108 projected-coalescing and 36 projected-per-shard paired
trials are complete. An additional 37 shard, larger-table and sustained controls
bring this round to 253 capacity trials and 153,703 completed updates, all with
zero reported errors. The default now retains per-shard overlap with destination
projection; coalescing is explicit. Both paths pass real CUDA lifecycle checks.

Three 30-second trials at three GPUs, four shards and eight Learners establish
122.95 updates/s and 104.93 ms median trial P99. Eight/sixteen-shard and
128-Learner controls expose service/transport limits. Million-row, 512D controls
exercise 6 GiB of logical weights/moments. All raw records and source hashes are
published through [BENCHMARKS.md](../BENCHMARKS.md). All three instances were
deleted after fetching the results; account inventory confirmed zero instances.
