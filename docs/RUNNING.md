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
