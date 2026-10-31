# Embedding Parameter Service

**An independent parameter service for sparse embedding training.**

A training batch often touches only a small fraction of a large embedding table.
The same pattern appears in recommendation, advertising, knowledge graphs, graph
learning and word-embedding models. Embedding Parameter Service manages the
parameter tables, batched lookups, sparse updates and CPU/GPU cache placement
behind a service interface.

The first integration retains **Entropy's training framework**: dataset readers,
model graphs, GPU forward/backward, dense gradient synchronization and training
orchestration continue to run there. A thin adapter connects its embedding
operations to this service.

**Project stage:** service architecture and interface contracts. The implementation
sequence is documented in the [plan](docs/plans/2026-10-31-implementation.md).

```mermaid
flowchart LR
    subgraph Entropy[Existing training framework]
        L[Dataset and Learners] --> G["Model graph<br/>GPU forward / backward"]
        G --> D[Dense updates and orchestration]
        A[Embedding adapter] <--> G
        C["GPU cache agent<br/>Existing tensors and CUDA execution"] <--> G
    end
    A <-->|Batched lookup and sparse gradients| S[Embedding Parameter Service]
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
proto/            Service protocol
configs/          Example service/integration configuration
docs/             Architecture, rationale and delivery plan
references/       Local reference collection guide
```

The implementation is designed around **C++20**, **BRPC/Protobuf** and contiguous
parameter-state storage. The Entropy adapter targets **TensorFlow custom operations** and the existing
CPU/GPU communication path; its GPU cache agent executes the CUDA-facing work in
the training process. A future client can use the same table/key/gradient contract
without depending on video-specific features.

The tracked code currently defines the service, planner and GPU-agent interfaces,
an original Protobuf schema, and example configurations. Runtime modules follow
the implementation plan. The CMake target exposes the interface headers only.

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
