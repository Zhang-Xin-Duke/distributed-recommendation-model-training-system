# Two-node sharded training execution

Place one parameter shard, CUDA owner and TensorFlow model worker on each GPU
node. The task coordinator runs on node 0. Each shard retains only its local
CPU backing rows, and each GPU owner holds the corresponding weights and both
Adam moments. Model workers use the verified synchronous Embedding/MLP path.

The fleet file provides SSH transport and already connected parameter, Agent
and coordinator endpoints from each node. It remains outside version control.
Node jobs have unique run directories and persistent process handles/results;
an existing job cannot be silently restarted. Resource sampling records GPU
memory/utilization, process RSS and Linux cgroup memory while preparation and
training run.

Before allocating the table, verify distinct physical GPU UUIDs, physical
capacity, actual free GPU memory and available/cgroup-limited host memory.
A capacity run requires the global resident FP32 parameter/moment bytes to
exceed the largest single physical GPU. Per-node state must leave explicit
workspace and host-memory headroom. A bounded allocator alone cannot meet it.

The preparation preflight creates/verifies the table schema and checks actual
returned vector width across shards. Preparation runs locally on each node
using `only_shard`, checks all loaded row ownership and final resident counts,
then all workers train. Final acceptance checks task/round completion, sampled
finite weights/moments and row counters, loss reduction, GPU model placement,
resident totals and resource telemetry. A small whole-model reference remains
the separate numerical correctness check already established for this path.

Verify the same process lifecycle locally with Host owners before rental.
Then run a small two-node CUDA case to verify network/model execution, followed
by the beyond-single-device table. Retain successful and failed points, source
and runtime hashes, monitor data and owned-resource cleanup proof. Original
Entropy integration and durable update/replica recovery stay in the full plan.
