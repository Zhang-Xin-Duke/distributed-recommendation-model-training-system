# Embedding Pull/Push/Join adapter

[Entropy optimizer configuration](../../docs/ENTROPY_CONFIGURATION.md) maps the
original SparseLazyAdam options, including default weight clipping, to external
table definitions. [Original table and Runner hooks](../../docs/ENTROPY_TABLE_HOOKS.md)
install the service into selected table names while preserving the original class
and gradient dispatcher. [Restore staging](../../docs/ENTROPY_RESTORE.md) validates
the joint checkpoint and loaded shard identities before original dense restore.
The native original-application build, original-reader/multi-peer execution and
device interop retain their own integration work.

This adapter replaces the parameter-access boundary in a training application.
Dataset readers, feature construction, the model, loss scaling, dense optimizer,
collectives and job controller remain with that application.

```mermaid
sequenceDiagram
    participant G as Existing TensorFlow graph
    participant C as Native client
    participant S as Parameter shards
    participant A as CUDA cache agents
    G->>C: Pull IDs and row splits
    C->>S: Route IDs; retain original bag lengths
    S-->>C: CPU contributions and cached-row locations
    C->>A: Gather cached contributions
    A-->>C: Feature contributions
    C-->>G: Reconstructed feature tensors + handle
    G->>G: Existing model, loss and backward
    G->>C: Push feature gradients + handle
    C->>S: Original sparse update request
    S->>A: GPU-owned gradients + optimizer context
    A-->>S: Update completion
    S-->>C: Update acknowledgement
    C-->>G: Join dependency satisfied
```

## Tensor and gradient conventions

`Feature` describes a table, feature ID, pooling mode, flat IDs and row offsets.
Padding IDs are removed before access. Mean pooling uses the valid count after
padding removal; the global denominator is retained across server partitions.
Sequence outputs restore their original positions, filling padding positions
with zero. Backward gathers only valid positions.

Pass gradients of the completed feature tensors to `push`. The service performs
mean unpooling once. When replacing a framework operation that already divides
mean gradients by bag length, connect before that division. Existing loss and
distributed-gradient scaling still apply.

Every Pull handle closes with one Push or an explicit Release. Push waits for
the server acknowledgement, including updates performed by GPU agents. Join
checks that all local handles are closed; TensorFlow control dependencies order
it after the corresponding Push operations. The caller must discard backward
and finish any model-side device work before releasing an abandoned training
handle.

## Build

Use a Python environment containing `pybind11`, NumPy and TensorFlow:

```sh
cmake -S . -B build/tensorflow -DCMAKE_BUILD_TYPE=Release \
  -DEPS_BUILD_BRPC=ON -DEPS_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build/tensorflow -j 6
export PYTHONPATH="$PWD/build/tensorflow/python:$PWD/adapters/entropy"
ctest --test-dir build/tensorflow -R tensorflow_graph_external_shards --output-on-failure
```

`eps_tensorflow.ParameterServer` provides `create_table`, `pull`, `push`,
`release`, `join`, `save` and `close`. `tests/tensorflow_processes.py` is a complete
graph integration example: two external parameter shards, an existing dense
variable and optimizer, five sparse update steps, evaluation and checkpointing.

At the existing train-step boundary:

```python
handle, embeddings = parameter_server.pull(features)
# Build the existing model/loss from embeddings.
gradients = tf.gradients(loss, embeddings)
sparse_ack = parameter_server.push(handle, feature_ids, gradients)
train_done = parameter_server.join([sparse_ack, dense_train_op])
```

The bridge uses stateful `tf.py_function` host callbacks. Its native BRPC client
runs in a private local subprocess, preventing Protobuf/Abseil runtime conflicts
with the TensorFlow wheel. IPC commands are serialized and limited to 64 MiB;
a failed or ambiguous connection ends the Learner session. This portable path
includes host tensor copies and IPC overhead. The C++ `TableAdapter` can be linked
directly when the training binary uses a matching dependency toolchain.

[`WorkerCacheAgent`](../../docs/WORKER_AGENT.md) embeds the Agent listener and
cache lifetime in a native training process. Local host/device reads share its
RPC admission and ordered runtime; `GatherDevice` provides owned device results
for a model stream and is verified on a real CUDA consumer.
The supplied Python bridge currently consumes host-returning contributions;
native TensorFlow CUDA kernels and device-tensor interop remain a separate
integration task.

## Checkpoint boundary

Close input, complete the framework's train/save barrier and call `save`.
It requires zero live handles and drains GPU states into their home shards.
Each shard publishes a durable snapshot containing weights, moments and counters.
The existing checkpoint controller commits the complete set together with dense
weights and dataset progress. An incomplete set must not become a job checkpoint.
Successful save leaves these services drained. Export the frozen files through
`export_checkpoint(..., filesystem=tf.io.gfile)`, commit them with dense and input
state, then call `resume(checkpoint_id)` to continue the same run. Reads are
bounded to 1 MiB, verify the snapshot checksum and publish a parameter manifest
after all shard files are complete. The adapter opens input after every shard
acknowledges the matching run/epoch/checkpoint identity. A fresh-run restart can
also restore the exported files after staging them into the server snapshot
layout. [Periodic save, export and resume](../../docs/PERIODIC_CHECKPOINTS.md).

The private Entropy source collection is used to study this boundary. The full
original training application has not been rebuilt or run by this repository.


## Runtime metrics and automatic placement

`ParameterServer.stats()` returns a list of runtime-metric dictionaries in
configured shard order. It uses the same isolated native client IPC as the
training operations and can be called outside the TensorFlow graph. It includes
cached rows, live tickets, completed updates, migration bytes and the
`policy_*` counters for automatic placement.

Enable automatic placement in each PS configuration with `cache.admission`;
register cache endpoints on the existing adapter so later Pull calls resolve
resident rows. The service keeps original update/ticket boundaries through
migration. [The cache policy guide](../../docs/CACHE_POLICY.md) describes
configuration, costs, bounds and saved-state behavior. The automatic-cache graph
checks verify five dense/sparse updates, resident placement, metrics and cache
drain at save with both per-shard and coalesced gathers.

## Preparing future inputs

`prepare(features)` performs bounded ID/layout preparation on a native worker.
`pull_prepared(token, dimensions)` reads current parameters when the existing
train graph consumes that FIFO token. Pre-enqueue the next input before invoking
the current step to overlap metadata work with model/RPC execution. Unused
inputs must be consumed or cancelled before `save`; `join` permits future
metadata while requiring live Pull handles to be closed.

[Queue lifecycle, limits, graph example and phase-metric definitions](../../docs/ASYNCHRONOUS_PREPARATION.md).
