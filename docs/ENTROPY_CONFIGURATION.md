# Entropy optimizer configuration

The table adapter maps the existing `ts.train.SparseLazyAdam` object to an
external parameter table. Its default `weight_bounds=[-10, 10]` is retained.
CPU and CUDA parameter owners clamp updated weights while preserving Adam
moments, beta-power advancement and row-update counters. CPU row initialization
also applies the configured interval before placement on either owner.

```python
from entropy_configuration import sparse_table_definition

definition = sparse_table_definition(
    table_id=0, dimension=128, optimizer=existing_sparse_optimizer, seed=42)
parameter_server.create_table(definition)
```

`learning_rate`, `beta1`, `beta2`, `epsilon`, initial beta powers and the uniform
initialization range map directly from the optimizer options. Passing an explicit
`initialization=(value, value)` supports the framework's constant initializer.
The service retains its deterministic per-key initializer; it does not reproduce
the reference process's random-number sequence.

`weight_bounds=None` in the original optimizer produces an unbounded table.
`[0, 0]` is an explicit fixed interval and retains presence through JSON, the
native bridge and Protobuf. Equal finite endpoints are valid; reversed or
non-finite bounds are rejected. All parameter-service and Agent processes must
use a build supporting this table option.

The initial mapping supports unshared FP32 LazyAdam. Shared moments, non-optimized
dimensions, weight decay and active feature creation/purge/decay policies require
their corresponding runtime integration and currently raise an error.

## Training-framework connection points

Source inspection identifies these existing boundaries in the preserved Entropy
snapshot:

| Boundary | Existing source location | Responsibility |
| --- | --- | --- |
| Table factory | `inputs.py`, `ts.train.SparseTable` construction | Select the external backend from job configuration |
| Pull | `SparseTable.embedding_lookup_and_pooling` | Preserve pooled and sequence layouts expected by the input layer |
| Gradients | `InputBuilder.apply_sparse_grads` and `SparseTable.apply_gradients` | Pass feature gradients with the framework's existing loss scaling |
| Checkpoint | Training checkpoint controller and `SparseTable.checkpoint` | Drain sparse updates at the existing barrier and commit with dense/input state |
| Restore | `worker_restore_sparse` and `SparseTable.restore` | Restore the matching sparse snapshot before continuing the existing job |

The configuration mapper, Pull/Push/Join bridge and
[original table/Runner hooks](ENTROPY_TABLE_HOOKS.md) are implemented. Source-level
Session graph checks exercise the original methods with real external shards.
A native original-application build, multi-peer framework execution, remote
restore staging and in-process device-tensor interop remain open work.
The [periodic checkpoint bridge](PERIODIC_CHECKPOINTS.md) provides bounded export
through the framework filesystem and guarded same-run resume for that controller.

## Snapshot format

New state files use snapshot format 2, which stores the presence and endpoints of
weight bounds with each table definition. The reader accepts format 1 as an
unbounded table. The legacy fixture retains an independently encoded format-1
file, including its checksum. New readers verify both formats; old readers
reject the new format.

Validation covers initialization outside the interval, saturated updates with
independent moment calculations, CPU/Host-owner parity, bounded restart,
reversed ranges and legacy snapshots. TensorFlow process checks also exercise a
zero-valued interval across the JSON/native/RPC path. CUDA test coverage compares
bounded and unbounded tables with the Host reference and asserts that clipping
occurs; execution evidence is recorded separately.

The A6000 execution passes 23/23 native CPU/CUDA checks, 30/30 complete
native/CUDA/TensorFlow checks and five additional forced-CUDA cases. The
[validation artifact](validation/2026-11-01-entropy-bounds.json) records the
compiled source, device, binary/log hashes and rental cleanup.
