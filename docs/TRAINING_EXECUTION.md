# Distributed training execution

Workers train one shared model: each worker reads its data partition, pulls the
sparse parameters it needs, computes model gradients and pushes updates to the
owning parameter shards. Training-task progress decides when the next iteration
may start. Parameter ownership, update completion and recovery must agree with
that progress.

## Training responsibilities

| Component | Responsibility |
| --- | --- |
| Training worker | Own data/input position, forward/backward, gradient scaling and task completion |
| Parameter shard | Own parameter rows and optimizer state; apply the configured aggregation/update algorithm |
| Training progress controller | Admit tasks according to inter-worker dependencies and retain completed worker clocks |
| Server manager | Track membership, partition ownership and fenced failover |
| Replicas/checkpoint storage | Preserve committed parameter state and update progress for recovery |

The design reference is Duke CPS 512 Lecture 16, *Distributed Machine Learning:
The Parameter Server*, supplied as `16-parameter-server.pdf`. Slides 11-14 cover
roles and sparse communication; 15-22 cover training dependencies and logical
progress; 24-26 cover replication and recovery; 20 and 27-30 motivate comparison
at a common training-quality target.

## Implemented task-progress core

[`TrainingProgressController`](../include/eps/training_progress.hpp) supports a
fixed group of up to 1,024 worker identities. Worker `w` with completed count
`c[w]` has finished iterations `[0,c[w])`. The global completion floor is the
minimum of these counts.

| Policy | Admission for iteration `t` |
| --- | --- |
| Sequential | All iterations before `t` are globally complete |
| Eventual | No finite inter-worker delay bound |
| Bounded delay `tau` | All iterations `j < t - tau` are globally complete |

For slide 18's `tau=1`, iterations 0 and 1 need no completed earlier iteration;
iteration 2 waits for iteration 0, and iteration 3 waits for 0 and 1. Completed
counts use a different offset from iteration indices: a paused/drained group
can have counts differing by at most `tau+1`. Restore validates this invariant.

Admission returns a shared future. Duplicate requests for one in-flight task
share that result, with at most one active/waiting request per worker. Completing
an unadmitted iteration rejects; duplicate completion leaves clocks unchanged.
Run/epoch checks reject old-job identities. Permit metadata records the exact
completion floor observed when admitted.

The executor advances the clock after its parameter update acknowledgements.
The core controls task dependencies; gradient aggregation, optimizer-step
semantics and dense collectives are configured in the training path. In
particular, admitting several workers in one round does not itself average their
gradients into a single optimizer update. A task clock also does not identify an
atomic cross-shard parameter snapshot.

Pause rejects new work, releases waiting callers and lets admitted tasks drain.
A progress checkpoint can be exported only after this drain. Restore requires a
fresh run and matching group/policy, and keeps admission paused until the job
controller finishes matching parameter/input restoration. Abort is terminal and
wakes waiting callers; the training executor still owns cleanup of parameter
handles and any device work already submitted.

## Executed verification

The C++ checks cover the exact bounded-delay boundary, sequential/zero-delay
equivalence, unlimited eventual progress, duplicate completion, malformed task
identities, pause/drain, fresh-run restore, invalid clock snapshots and abort.
Eight concurrent threads each complete 80 iterations under every policy,
including an intentionally slower worker.

A controlled sparse-logistic-loss check connects four worker threads to two
in-process parameter-service actors. Every worker computes actual gradients
from pulled features and waits for its LazyAdam Push acknowledgement before
completing the training task. Each policy completes 160 worker updates, reduces
the controlled loss, verifies all five touched rows' update counts/finite
optimizer state and drains lookup tickets. This is a correctness workload with
equal per-worker task counts. It is not a common-quality benchmark or a
multi-process/GPU training result.

```sh
cmake --build build/native --target eps_training_progress_tests -j 4
ctest --test-dir build/native -V -R '^training_progress_consistency$'
```

The same target runs under the existing ASan/UBSan and TSan core builds.
[Verified records, source hashes and logs](validation/2026-11-04-training-progress.json).

## Work remaining for complete training acceptance

1. Connect the controller through RPC to real worker processes and parameter
   shards, preserving update acknowledgements and input positions.
2. Define and implement synchronous gradient aggregation/normalization and
   optimizer-step boundaries, alongside asynchronous update policies. Reuse
   repeated key-list metadata with bounded, epoch-checked handles.
3. Persist per-range logical update progress; replicate before acknowledgement
   and fence shard/worker replacement. Exercise lost replies and process failure.
4. Replace the private Redis SDK through the retained interfaces and execute
   the complete original Entropy training/control path.
5. Compare time to the same quality target under controlled stragglers,
   communication load and failures. Retain waiting time, bytes, staleness,
   convergence and recovery measurements.

Comparisons across dependency policies must retain the same update algorithm,
gradient normalization and optimizer-step unit. A round-averaged update and
per-worker updates have separate aggregation semantics and reference checks.

[Implementation design](plans/2026-11-04-distributed-training-design.md)
