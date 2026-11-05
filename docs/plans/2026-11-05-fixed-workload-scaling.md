# Fixed-workload model scaling implementation plan

**Goal:** Compare the same distributed parameter-server training job on one and
two physical GPUs with two logical Workers/shards and identical model/data.

**Architecture:** Reuse the managed training supervisor and real TensorFlow
Workers. Each Worker warms the model and publishes readiness on its owning node.
The manager observes all ready Workers before releasing their local start gates.
Discard a fixed number of complete warmup rounds, then compute aggregate sample
throughput using the slowest Worker's measured-window duration. Per-stage timers
separate parameter reads, model/staging work, journal durability, gradient
replication, synchronous update acknowledgements and task completion. No metric
subtracts clocks from different hosts.

**Tech Stack:** Python monotonic clocks, existing role/SSH protocol, native BRPC
parameter service, CUDA owners and TensorFlow model forward/backward.

1. Add a role-scoped warm-start gate operation and tests for readiness, immutable
   identity, response replay and stopped-role rejection.
2. Add opt-in stage timers to the model Worker. Keep the timed model stage through
   host result materialization so its duration includes actual GPU completion.
3. Connect measurement mode in the managed supervisor, including model warmup,
   gate release, timing/timeout bounds and fail-on-recovery benchmark semantics.
4. Add a seeded workload generator and a benchmark driver. Bind workload/model,
   worker count, batches, update count and replication policy across placements;
   verify actual GPU UUID count and homogeneous device models. Alternate
   placement order across repeated trials. Compare full loss trajectories and
   inspected parameter/Adam state; retain raw per-round timing records.
5. Test the complete flow with native CPU-owned services/TensorFlow, including
   independent batches and warmup exclusion. Report local execution as protocol
   validation, with no GPU scaling result.
6. Refresh provider inventory and gift credits only after the benchmark is ready.
   Rent within the existing total USD 20 cap when both matching GPUs are
   available; retain verified deletion evidence. Publish measured results and
   the explicit SSH-relay transport. Physical multi-GPU scaling remains pending
   until that actual placement comparison has executed.

The metric is slowest-Worker-window sample throughput, analogous to a maximum
rank duration. It excludes process startup and model compilation and includes
parameter access, host/device staging, durability and synchronization. It is
distinct from controller-observed job wall time and from isolated GPU kernel
throughput. A multi-host placement also adds network/relay costs, which the
stage breakdown retains rather than attributing the entire change to the GPU.
