# Multi-GPU boundary verification

**Goal:** Measure the current parameter service's operating envelope on 1, 2 and
4 physical CUDA agents, including observable state after overload rejection.

**Architecture:** Retain one coordinator and four independent single-GPU
workers. Keep the total table, resident fraction, input size and coordinator
allocation fixed for strong scaling. Report the SSH-forwarded transport and
synthetic sparse gradients explicitly. Model training throughput is outside the
scope of this service benchmark.

**Tech stack:** C++20, CUDA, BRPC, Python, JSONL and sampled process/GPU telemetry.

1. Extend `tests/capacity_probe.cpp` to emit failed-point diagnostics before
   exiting nonzero: acknowledged work, error, post-failure shard statistics,
   read-only availability and sampled parameter finiteness. Do not infer exact
   optimizer correctness from partial acknowledgements. Verify the ordinary
   probe and deterministic server-capacity rejection on the local Host backend.
2. Run the native/CUDA suite before measurements; verify distinct GPU UUIDs and
   identical binaries. Lease guards cap each rental at 50 minutes; at the listed
   A6000 rate the four-GPU lease ceiling is about $1.17, within the standing $20
   budget. Retrieve evidence before deletion. No credit purchase is authorized.
3. Sweep 4/8/16/32/64 Learners on 1/2/4 GPUs, two five-second windows per point.
   Confirm each selected knee with three 20-second windows. Probe 128 Learners
   separately with failure diagnostics. A rejected point remains a failure even
   if a subsequent health query succeeds.
4. Repeat a larger 1,048,576-row, 512-dimensional workload at 8/32 Learners on
   1/4 GPUs and a serialized-admission control at 128 Learners on four GPUs.
   This bounds the tested operating region; it does not determine a universal
   row-count or physical VRAM maximum.
5. Publish raw records, hashes, a figure and an English report. Distinguish the
   throughput knee, latency degradation, admission failure and recovery
   observations. Delete the owned fleet and verify its absence.
