# Standalone recommendation training on Linux GPUs

Use the same Git revision and absolute runtime paths on the controller and
workers. Each GPU worker host exposes one NVIDIA GPU. Use a private routed
network between the controller and nodes for BRPC and gradient-replica traffic.
On rented hosts, set up a private overlay first and set each node's
`bind_host` and `service_host` to its address on that overlay. SSH is used for
process management. Do not forward the training service ports to the public Internet.

## Install

Install on the controller and each GPU node. Use a GPU controller host for this
installation recipe. The controller can share a surviving training or spare node
as long as it retains its own durable state.

```sh
DPS_WORK_ROOT=/srv/drmts DPS_BUILD_FROM_SOURCE=1 bash scripts/install_training_runtime.sh
```

The source build installs the pinned BRPC, CUDA toolkit, CMake, TensorFlow and
pybind11 versions in the installer. Use a host NVIDIA driver that supports the
configured CUDA toolkit. Use `/srv/drmts/environment/bin/python` to run the CLI.
For the controller's native binaries, export the runtime library paths from the
installation if the system loader cannot find them. Start with
`/srv/drmts/dependencies/lib` and `/srv/drmts/dependencies/lib64`.

Copy [the inventory example](../configs/recommendation-inventory.example.json)
to a private file and fill in SSH and overlay addresses. Keep `source`,
`python`, `bin_dir`, and `module_dir` identical across GPU hosts. The
controller probes node reachability, device and TensorFlow visibility, the source
revision and writable state/checkpoint directories before launching a GPU job.
For node-loss acceptance, register three distinct hosts, leaving capacity for a
replacement worker and its parameter shard.

## Prepare input and train

Mount the prepared dataset read-only at the same absolute path on the
controller and all GPU nodes, including the spare. Mount the checkpoint directory
at the same absolute path, writable by the controller and readable by the nodes.
Use independent durable storage for node-local replica logs.

Download [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) from
GroupLens under its published terms. A rating of at least four becomes a positive
label in the example. To use your own data, omit `--movielens` and pass CSV
with the header `user_id,item_id,label,timestamp`. Labels range from 0 to 1.
The latest time range is held out. Tied timestamps stay on the same side of
the split. The prepared manifest binds each worker's indexed input to the job.

```sh
python scripts/prepare_recommendation_data.py ml-100k/u.data /shared/movielens   --movielens --workers 2 --batch-size 128
```

Copy [the job example](../configs/recommendation-job.example.json) and replace
the dataset version with the value in `/shared/movielens/manifest.json`.
The batch size and group count must match preparation. The GPU path is the
default for versioned recommendation jobs. Every round consumes one batch per
worker. After the last training batch the sequence repeats.

```sh
cli=scripts/training_job_controller.py
python "$cli" --state /srv/drmts/controller --inventory inventory.json node probe
python "$cli" --state /srv/drmts/controller --inventory inventory.json job plan job.json
python "$cli" --state /srv/drmts/controller job submit job.json
python "$cli" --state /srv/drmts/controller --inventory inventory.json controller serve
```

In another terminal, use `job status movielens` to see progress and the final
held-out loss, accuracy and checkpoint. Use `job cancel movielens` to stop the
job and release its reservations after processes exit. To continue a completed
segment with a fresh job identity:

```sh
python scripts/training_job_controller.py --state /srv/drmts/controller   job resume movielens movielens-continued --extra-rounds 100
```

A completed segment publishes its checkpoint at
`<checkpoint_root>/<job_id>/manifest.json`. For planned intermediate
checkpoints, submit bounded segments and continue them with `job resume`.
Resume verifies the snapshot and input identity before reopening training.
Controller state, event history and the full training report remain in the
configured state directory.

A confirmed node loss is recorded via `node lost NODE_ID RECEIPT.json`.
The controller uses surviving parameter copies and original-gradient journals
to reconcile a partial round, then starts the same logical job on replacement
resources. See [the node recovery procedure](PHYSICAL_NODE_RECOVERY.md) for
receipt requirements and failure boundaries. An SSH timeout alone does not prove
node loss. This deployment model assumes a trusted network, one controller,
and a surviving copy of each shard and gradient journal.

## Release verification

Run the above procedure with two physical GPU nodes and one spare. Retain GPU
device placements, validation results, and checkpoint/restart evidence. Compare
a continued job and a continuous run with matching inputs and configuration.
Confirm recovered input cursors and update counters exactly, and compare losses
and weights within a documented tolerance. On disposable acceptance nodes,
inject one node loss during a round, verify replacement and replay, and check
controller restart, competing reservations, and cancellation. Keep local CPU
checks separate from physical GPU acceptance results.
