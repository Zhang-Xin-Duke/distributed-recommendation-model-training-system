# Acceleration design

The service aims to reduce parameter-access and update time inside the existing
training pipeline. Model structure, loss, optimizer definition and input ordering
stay fixed when evaluating a policy change.

## Baseline and changes

Entropy already contains partitioning, batched access, pooling, sparse gradient
merging, optimizers and CPU/GPU cache designs. The independent work focuses on
making access and migration choices within an extracted service.

| Baseline | Service optimization | Expected benefit |
|---|---|---|
| Feature-level pooling/cache traffic comparisons | Include observed CPU, network, PCIe, GPU and queue time in route selection | Avoid reducing bytes while increasing CPU-side waiting |
| Frequency/window cache policies | Admit rows when expected reuse pays for full-state loading and writeback | Reduce one-use insertions and repeated replacement |
| Insertion-driven capacity pressure | Use byte budgets, in-flight limits and a free-slot reserve | Smooth migration and reduce waits for slot reclamation |
| Individual request preparation and transfers | Prepare future ID metadata and coalesce compatible payloads | Reduce scheduling, allocation and small-message overhead |

## Access-path cost

For CPU-backed contributions, compare:

```text
CPU route = CPU lookup + CPU pooling + transfer of pooled features
GPU route = CPU lookup + transfer of unique vectors + GPU processing
```

Resident rows are always read from their current owner. A shared table row cannot
be treated as a CPU-owned row simply because a particular feature chooses CPU
pooling. Sequence features preserve positions and do not receive sum compression.

For illustration, 100 FP32 vectors of dimension 128 contain 51,200 bytes; their
sum contains 512 bytes, before metadata. CPU pooling can greatly reduce transfer.
High repetition, CPU saturation, or GPU locality can favor another route. The
planner uses moving averages and a switching margin to avoid frequent changes.

Network transfer and CPU/GPU transfer are measured separately. Communication of
features and gradients to the training worker remains part of route cost.

## Admission and migration cost

Sparse LazyAdam row state includes w, m and v, approximately `12 * dimension`
bytes in FP32 before metadata. An ordinary fallback access needs weights or a
pooled result, so loading full state for every miss can be wasteful.

Admission estimates repeated-use savings minus loading, expected writeback and
capacity cost. Statistics are sampled and decay over time. A minimum residence
interval and replacement margin reduce churn. Unadmitted rows continue to train
through the CPU path.

The scheduler limits remote transfer bytes, host/device staging bytes and request
count in flight. It maintains free slots per compatible row layout. Rows that do
not fit the current migration budget use fallback instead of forcing insertion.
The reserve sacrifices some resident capacity in exchange for smoother admission.

## Prepare and coalesce

Prepare key routing, unique mappings, buffer sizes and migration candidates ahead
of demand. Parameter reads still respect the required update and ownership
boundary. Group compatible state transfers by destination, dimension and optimizer
layout, retaining a map back to rows and requests.

## Keep pooled gradients compact across worker links

A pooled feature produces one gradient vector for a bag. Sending a separate
expanded copy for every ID repeats that vector many times. The compact update
path sends the original feature-gradient rows and a CSR-style map from each
unique parameter row to its contributing gradient rows and mean-pooling scales.
The GPU reduces these ordered references directly into device gradient staging,
then runs the same LazyAdam validation and update kernels.

For a bag of 64 distinct IDs and dimension 128, expanded gradients contain
64 copies of a 128-dimensional vector. The compact representation carries one
vector and 64 references. Repeated IDs retain repeated references; contributions
from different features remain in their original reduction order. Each GPU
receives the home shard's original optimizer context.

Reference metadata adds work and can exceed the already-reduced payload for
short bags, low dimensions or sequence features. The service compares complete
numeric payload sizes and uses expanded gradients when they are smaller. Packet
validation and device staging limits apply before state mutation. CPU-owned rows
retain their existing update path; the home service still validates the original
gradient reduction before dispatching device work.

Updates to different GPU owners are submitted before awaiting acknowledgements.
Commands to one home/agent control stream remain ordered, including multi-table
updates. This removes sequential waits between independent devices while keeping
one original Push boundary. The capacity harness exposes separate ablation flags
for concurrent dispatch and compact gradients.

## Expected workload fit

- Long pooled bags: reduce raw-vector transport through CPU partial pooling.
- Stable repeated IDs: gain from local resident reads and updates.
- Bursty or shifting hot sets: limit churn and transfer bursts.
- Many small transfers: reduce per-message and per-copy setup work.

The first implementation will compare CPU-only, static placement, frequency/window
caching and these policies using the same training computation. Metrics include
end-to-end throughput, phase time, transfer bytes, cache churn, slot wait and
validation quality. Correctness and capacity evidence is recorded in
[VALIDATION.md](VALIDATION.md).
