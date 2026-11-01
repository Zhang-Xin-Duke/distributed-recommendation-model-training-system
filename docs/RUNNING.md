# Build and run

## Dependencies

The core requires a C++20 compiler, CMake 3.20 or newer and POSIX file APIs.
The network server additionally uses BRPC, Protobuf and nlohmann/json.
The real-process integration check uses Python 3.9 or newer.

On macOS with Homebrew:

```sh
brew install cmake brpc protobuf nlohmann-json
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Debug -DEPS_BUILD_BRPC=ON
cmake --build build/native -j 4
ctest --test-dir build/native --output-on-failure
```

On Linux, install the compiler, CMake, Protobuf development files, nlohmann/json,
OpenSSL, gflags and LevelDB. Build/install BRPC using its
[official instructions](https://brpc.apache.org/docs/getting_started/).
Expose its installation through `PKG_CONFIG_PATH` and `CMAKE_PREFIX_PATH`, then
use the same CMake commands. The initial development check used BRPC 1.18.0.

For a dependency-light core check with sanitizers:

```sh
cmake -S . -B build/asan -DCMAKE_BUILD_TYPE=Debug \
  -DEPS_BUILD_BRPC=OFF -DEPS_SANITIZERS=ON
cmake --build build/asan -j 4
ctest --test-dir build/asan --output-on-failure
```

Use separate build directories. The installed BRPC/Protobuf binaries are not
sanitizer-built; their Protobuf message layout under ASan differs in the currently
installed version. The project therefore keeps the sanitizer check core-only.

## Start a server

```sh
./build/native/eps_server --config configs/service.example.json
```

The server prints `READY HOST:PORT` after binding. The example uses loopback and
stores snapshots beneath `./snapshots`. For two shards, create two configurations
with the same table definitions, run ID, partition epoch and `shard_count=2`;
set their `shard_id` values to 0 and 1 and use different listen ports.

The client routes `key % shard_count` to the matching endpoint. It retains
original bag lengths, combines partial sums before mean normalization, and
scatters sequence rows back to their original positions.

## Client API

Create one `BrpcParameterService` per server endpoint and pass them in shard order
to `DistributedClient`. Create/register tables with `CreateTable`, submit a
`LookupRequest`, use `DistributedBatch::features()` in the model, then call `Push`
with the gradients of those feature tensors. Read-only batches use `Release`.

Lookup sequence numbers start at zero and increase per Learner, including
evaluation batches. Lookup retries keep the same ID. Push/Release retain the
lookup's learner and sequence and change its phase. Each batch must eventually
close; otherwise it pins rows and prevents a completed drain.

`tests/rpc_checks.cpp` is an executable client example that compares a real
two-shard deployment with a single-shard reference. `tests/rpc_processes.py`
creates temporary configurations, starts the servers, saves them, restarts both
from their shard files and checks the next updates.

## Save and restore

Pause new work in the training controller, finish existing batches and invoke
`DrainAndFlush` on every shard. Each successful response identifies an immutable
binary shard file. The framework coordinates these files with its dense model,
dataset position and other training state.

Restore each server with its matching shard file and a new run ID:

```sh
./build/native/eps_server --config shard-0.json \
  --restore snapshots/checkpoint/shard-0.bin --run resumed-run
```

This preserves weights, LazyAdam moments, per-row update counts and table optimizer
globals. A new run ID rejects packets from the old run. Static shard membership
and fail/restart recovery are used for this version.

## CUDA cache worker

With a CUDA toolkit installed, build with `-DEPS_BUILD_CUDA=ON` and the appropriate
`CMAKE_CUDA_ARCHITECTURES` (89 for L40, 86 for RTX A6000):

```sh
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release \
  -DEPS_BUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build/cuda -j 6
ctest --test-dir build/cuda --output-on-failure
./build/cuda/eps_cache_agent --run example-training-run --backend cuda \
  --agent 0 --device 0 --listen 127.0.0.1:9200
```

Enable `cache.enabled` and add `agents: [{"id": 0, "endpoint": "127.0.0.1:9200"}]`
to each PS config. Clients install a cache resolver with
`DistributedClient::SetCacheResolver`; the TensorFlow bridge exposes
`cache_endpoints([(0, "127.0.0.1:9200")])`.

The default resolves cache work per shard while other shard replies remain in
flight. Use `SetCacheResolver(resolver, true)` or
`cache_endpoints(agents, coalesce=True)` to merge work across all shard replies.
Both paths project input IDs onto each destination and restore sequence positions.
For a local capacity probe, set `EPS_COALESCE_GATHERS=1` to compare coalescing with
the default `0`. The fleet wrapper exposes `--coalesce-gathers 0|1` for prepared
probe builds that support this setting.

`access.policy` selects `cpu_only`, `transfer_unique`, or `cost`.
These choices concern CPU-backed rows. Resident rows always use their current
agent. `cost` uses paired workload measurements supplied through
`access.observations`, or a live `AccessPlanner` supplied to `ConfigureAccess`.
Uncalibrated pooled features retain the CPU route. Migration is explicitly
requested through `MoveRows`, which checks live references, generations, staging
bytes and reusable slots before publishing a new owner.

`scripts/gpu_validate.sh` provisions a project-local compiler/runtime and BRPC
on Ubuntu while retaining the installed NVIDIA driver. It records a complete
build/check log. On Thunder it disables BRPC's optional pthread profiler hook,
which conflicts with the provider's preloaded `dlsym` interposer.

## Scaling and capacity

`eps_capacity_probe` exercises sparse parameter training steps: lookup, mean
pooling, feature gradients and acknowledged updates. It samples parameter update
counters after each load level and rejects runs with mismatched counters,
non-finite results or live tickets left behind. Initialization and kernel warmup
are outside its timed interval. It measures the parameter subsystem; no dense
model computation is included.

For an existing host with three visible devices:

```sh
python3 scripts/capacity_sweep.py --build build/cuda --out results/capacity \
  --devices 0,1,2 --gpu-counts 1,2,3 --ps-shards 2 \
  --rows 262144 --dimension 128 --keys 1024 --bag 32 \
  --learners 1,2,4,8,16,32 --seconds 5 --repeats 3
```

Each topology starts fresh processes. Each load level produces raw JSON metrics
for completed updates/s, ID occurrences/s, P50/P95/P99 latency, lookup/update
latency, sequencer occupancy and queue waiting. Worker/PS CPU and RSS are sampled
separately; GPU telemetry is device-wide. Repeat ranges accompany median rates.

Use `--serial-agent-updates` or `--expanded-gradients` for independent ablations.
They set `cache.concurrent_agent_updates=false` and `cache.compact_gradients=false`
respectively. The default uses concurrent owners and chooses compact gradients
only when their numeric payload is smaller. Runtime statistics report expanded
reference bytes, transmitted numeric gradient bytes and compact-update commands.

`scripts/fleet_capacity_sweep.py` manages agents across separately prepared SSH
hosts and invokes the same load generator. Its private fleet config supplies
connection paths and already prepared forwarding endpoints. It verifies distinct
GPU UUIDs and stops its agent/telemetry processes in `finally` blocks. For an SSH
tunnel deployment, report transport overhead with the results. Rental creation
and resource deletion remain with the experiment controller.

See [the TensorFlow adapter guide](../adapters/entropy/README.md) for its build
and graph integration example.


## Automatic cache policy

The cache example enables reuse-cost placement; set its run ID and agent
endpoint to match the agent. With a CUDA-enabled build:

```sh
./build/cuda/eps_cache_agent --run example-cache-training-run --backend cuda \
  --listen 127.0.0.1:9101 --rows-per-table 1024 --state-bytes 4194304
# In another terminal:
./build/cuda/eps_server --config configs/service.cache.example.json
```

A client must register that agent endpoint to resolve resident feature work.
Use `cache.admission=frequency` for the decayed-frequency baseline, or `manual`
to leave placement to explicit Move calls. Example cost priors need tuning for
the actual workload; [CACHE_POLICY.md](CACHE_POLICY.md) defines their units.

The automatic two-shard lifecycle check can use a CUDA agent on a GPU machine:

```sh
EPS_CACHE_BACKEND=cuda ctest --test-dir build/cuda \
  -R 'automatic_cache_.*external_shards' --output-on-failure
```

Without that environment setting the same tests use the Host reference backend.
Both checks include fresh and restored processes, policy metrics and state
comparisons. `cuda_state_and_device_results` also checks destination slot reserves.


For calibrated CPU/static/frequency/reuse-cost comparisons with fixed hot-set
changes and complete CPU replay, use `scripts/cache_policy_sweep.py`.
[The experiment report](CACHE_POLICY_RESULTS.md) records exact moderate/bulk
commands, source revisions and raw data. A compact Host-reference process check
runs as `cache_policy_fixed_trace_smoke`; its rates are excluded from GPU results.

### Multi-GPU fixed-trace placement evaluation

```sh
python3 scripts/cache_policy_sweep.py --build build/cuda --out results/policy-four-gpu \
  --backend cuda --devices 0,1,2,3 --cache-rows 10240 --hot-rows 8192 \
  --rows 262144 --dimension 512 --keys 8192 --bag 128 --phase-steps 64
```

`--cache-rows` is the total across agents, divided equally. `--reserve-slots`
is per agent. The runner checks distinct physical GPU UUIDs, accounts for each
agent's resident accesses and verifies every agent returns its slots on drain.
The trace is generated independently of topology, so the same IDs and gradients
can be compared across GPU counts. CPU/static calibration is repeated for each
topology. The CPU/reference smoke also exercises four separate agent processes.

### Concurrent automatic-admission load

```sh
python3 scripts/fleet_capacity_sweep.py --fleet /path/to/private-fleet.json \
  --out-name automatic-frequency --gpu-counts 4 --ps-shards 4 \
  --learners 8,16,32 --seconds 5 --repeats 2 --cache-admission frequency
```

`--cache-admission reuse_cost` uses a zero saved-access prior for this
concurrency control. Each successful automatic-admission run writes
`post-run-policy.json` with per-shard policy counters from the service's Stats
endpoint. These counters cover setup, warmup and all timed trials.
The native probe still checks finite parameters, acknowledged row-update
counters and ticket cleanup.

For fixed traces across separately hosted agents, use
`scripts/fleet_cache_policy_sweep.py --fleet /path/to/private-fleet.json
--out results/policy --out-name fixed-policy --gpu-counts 1,2,4
--source-commit BUILD_SOURCE_COMMIT`. It restarts all agents between trials,
repeats CPU/static calibration for each topology and verifies identical trace
fingerprints across all cache modes and GPU counts.
