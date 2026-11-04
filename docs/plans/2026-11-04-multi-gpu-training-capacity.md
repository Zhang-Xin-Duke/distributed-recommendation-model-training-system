# Training beyond single-GPU memory

The primary multi-GPU acceptance scenario is a model whose parameter and
optimizer state does not fit on one GPU. The project addresses this through
sharded sparse parameters: each GPU owner holds disjoint embedding rows and
their optimizer state, while training workers retrieve required vectors and
return gradients to the corresponding owners. Multiple model replicas alone
do not meet this capacity requirement.

## Execution requirements

1. Partition a fixed embedding table by Key, with explicit per-shard ownership.
   Keep weights and both Adam moments together on their GPU owner. Report row
   counts, state bytes and actual device allocations for every owner.
2. Execute real model forward/backward on GPU workers using the retrieved
   features. Return sparse row gradients to their owners, apply the configured
   per-worker or synchronous-round algorithm and wait for all required updates.
   Preserve original Entropy integration as part of complete acceptance.
3. Allocate a global parameter/optimizer state exceeding the largest physical
   memory capacity of any participating GPU, while each assigned shard and
   required workspace fits its owner's actual free memory. A configured smaller
   allocator budget or another process's memory use cannot establish this
   capacity claim. Do not count unused logical table capacity or CPU offload as
   GPU-resident state. Initialize and verify all claimed resident rows.
4. Distribute parameter-server processes as needed so the entire global CPU
   backing store does not exceed a single coordinator's host-memory limit.
   Record host memory, duplicate backing copies, device workspace and transport
   staging separately from parameter/optimizer bytes.
5. Establish correctness first against a small uninterrupted model: outputs,
   gradients, weights, moments, update counts and round/task progress. Then run
   the capacity workload, checking sampled state and finite gradients/loss,
   complete shard counts and bounded memory throughout actual training steps.
6. Record GPU UUID/model/capacity, peak device/host memory, communication bytes,
   per-owner load, step time and waiting. Evaluate skewed/hot Key access and
   recovery alongside uniform distribution; adding GPUs need not increase
   throughput when communication or one owner dominates.

## Existing evidence and next execution

The recorded A6000 multi-owner workload has 1,048,576 rows by 512 FP32
dimensions, or 6 GiB of logical weights and two moments, with half the rows
resident. It establishes cross-owner updates and an operating envelope. Its
global state fits within one A6000, so it does not establish this capacity case.

The synchronous-gradient collector supplies fixed-group shard-local reduction
and one optimizer update per complete round. Its network/reference checks now
pass, and TensorFlow model forward/backward with CUDA parameter updates passes
on an A6000, including scattered IDs and fully loaded small tables. Extend this
verified training path across GPU nodes and size
the capacity experiment using live device and host-memory availability. Prepare
and verify the workload before renting resources; retain the existing rental
budget and recheck the available credit/instances before starting billing.

The overarching task-progress, repeated-Key metadata, durable range progress,
replication/failover and original-framework requirements remain in
[the distributed-training plan](2026-11-04-distributed-training-design.md).

## Executed capacity milestone

The L40 + A6000 run completes this fixed-workload residency case: 5,242,880
initialized rows × 1,024 FP32 dimensions occupy 60 GiB globally, split into
30 GiB per owner. The largest participating GPU exposes 47.99 GiB. Both workers
execute GPU forward/backward for eight rounds and loss decreases from 0.694 to
0.231. All resident row counts remain complete; 64 touched rows have finite
weights/moments and exactly eight updates. A preceding small cross-node model
comparison has maximum state/loss errors of 8.94e-7 / 5.96e-8.

[Execution and resource measurements](../MULTIGPU_TRAINING_CAPACITY.md) retain
the workload, host backing, SSH transport, sampled memory peaks and cleanup.
This establishes one beyond-single-GPU training capacity point. Hot-Key skew,
cross-node recovery, full original Entropy execution and the remaining
common-quality/scaling measurements retain their requirements above.
