# Distributed Parameter Server

**Distributed storage, access and updates for machine-learning parameters.**

Training workers pull the parameters needed for a batch and push gradients back
for updates. Distributed Parameter Server partitions those parameters across
server shards and manages batched access and optimizer state. The cache design
extends this path with CPU/GPU placement.

The first implementation targets sparse access to embedding tables: each batch
touches only a small fraction of a large table. This pattern appears in
recommendation, advertising, knowledge graphs, graph learning and word-embedding
models.

The first integration retains **Entropy's training framework**: dataset readers,
model graphs, GPU forward/backward, dense gradient synchronization and training
orchestration continue to run there. The planned adapter connects its embedding
operations to this service.

**Implemented:** independently running CPU shards, a BRPC client/server,
cross-shard batching, sparse LazyAdam updates, bounded request/ticket payloads,
and durable snapshots with exact optimizer restore. Cost-based routing and
migration policies are available as a separately checked library. GPU execution
and Entropy graph integration are the next milestones.

[Build and run](docs/RUNNING.md) · [Validation](docs/VALIDATION.md) ·
[Implementation progress](docs/plans/2026-10-31-implementation.md)

```mermaid
sequenceDiagram
    participant W as Training worker / client
    participant A as Parameter shard 0
    participant B as Parameter shard 1
    W->>A: Batched IDs owned by shard 0
    W->>B: Batched IDs owned by shard 1
    A-->>W: Partial features + ticket
    B-->>W: Partial features + ticket
    Note over W: Combine sums / normalize means / restore sequence order
    Note over W: Model forward and backward
    W->>A: Gradients + original ticket
    W->>B: Gradients + original ticket
    A-->>W: Update completed
    B-->>W: Update completed
```

## Target integration

```mermaid
flowchart LR
    subgraph Entropy[Existing training framework]
        L[Dataset and Learners] --> G["Model graph<br/>GPU forward / backward"]
        G --> D[Dense updates and orchestration]
        A[Embedding adapter] <--> G
        C["GPU cache agent<br/>Existing tensors and CUDA execution"] <--> G
    end
    A <-->|Batched lookup and sparse gradients| S[Distributed Parameter Server]
    S --> T[Sharded parameter storage]
    S --> U[Sparse gradient reduction and optimizer]
    S --> P[Access and cache planner]
    P <-->|Cache plans and completion| C
```

## Service responsibilities

- **Parameter storage:** shard `(table, key)` rows and retain weights and optimizer state.
- **Batched access:** deduplicate requests while preserving bag and sequence positions;
  support sum, mean and ordered sequence features.
- **Sparse updates:** merge repeated-key contributions within the original update
  request and apply its configured sparse optimizer at the current parameter location.
- **Cache management:** choose reusable rows, coordinate CPU/GPU placement and batch
  transfers within memory and bandwidth budgets.
- **Framework integration:** return the tensor layouts expected by the existing
  training graph and expose a drain/flush operation for its save lifecycle.

## Independent optimization focus

| Optimization | What changes | Why it can accelerate training |
|---|---|---|
| Cost-based access planning | Compare CPU pooling with unique-vector transfer and GPU processing | Reduce traffic without moving the bottleneck into CPU pooling |
| Reuse-aware admission | Account for parameter-state loading and writeback when admitting a row | Avoid one-use insertions and repeated cache churn |
| Migration budgets and slot reserve | Bound in-flight transfer work and prepare reusable capacity early | Reduce bursts and waits for old rows to finish writing back |
| Batched preparation and transfer | Prepare ID routing ahead and coalesce compatible payloads | Reduce allocation, scheduling and small-message overhead |

These extend a baseline that already has partitioning, pooling, sparse updates,
cache management and asynchronous transfers. The [acceleration design](docs/ACCELERATION.md)
explains the decision model and expected workload benefits.

## Design and code map

- [Service architecture](docs/ARCHITECTURE.md)
- [API and integration](docs/API.md)
- [中文项目说明](docs/ARCHITECTURE.zh-CN.md)
- [Application families and sources](docs/APPLICATIONS.md)
- [Design decisions](docs/adr/README.md)
- [Implementation plan](docs/plans/2026-10-31-implementation.md)
- [Project scope and provenance](docs/PROVENANCE.md)

```text
include/eps/       C++20 service and cache-agent contracts
src/store/         Chunked row storage and durable binary snapshots
src/access/        Deduplicated lookup and feature-gradient reduction
src/update/        Sparse LazyAdam validation and updates
src/server/        Bounded shard executor and standalone server
src/transport/     BRPC services, client and attachment codec
src/client/        Cross-shard routing, pooling and sequence reconstruction
src/cache/         Cost model and bounded migration proposals
proto/            Service protocol
configs/          Example service/integration configuration
tests/            Core, policy and real-process RPC correctness checks
docs/             Architecture, rationale and delivery plan
references/       Local reference collection guide
```

The implementation is designed around **C++20**, **BRPC/Protobuf** and contiguous
parameter-state storage. The Entropy adapter targets **TensorFlow custom operations** and the existing
CPU/GPU communication path; its GPU cache agent executes the CUDA-facing work in
the training process. A future client can use the same table/key/gradient contract
without depending on video-specific features.

The CPU runtime and BRPC transport are executable. The GPU-agent contract and
target integration remain documented alongside the implementation sequence.

## References

1. Li, M., et al. *Scaling Distributed Machine Learning with the Parameter Server.*
   OSDI, 2014. [Paper](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/li_mu).
2. Zheng, D., et al. *DGL-KE: Training Knowledge Graph Embeddings at Scale.* SIGIR, 2020.
   [Paper](https://www.amazon.science/publications/dgl-ke-training-knowledge-graph-embeddings-at-scale).
3. PyTorch. *FBGEMM: Table Batched Embedding Training Module.*
   [Documentation](https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/tbe_ops_training.html).

Original project contributions use [PolyForm Noncommercial 1.0.0](LICENSE).
See [NOTICE](NOTICE) for attribution. Local upstream source and internal reference
material are excluded from version control.
