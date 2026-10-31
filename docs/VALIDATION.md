# Validation record

The 2026-10-31 implementation was built with AppleClang 21 on macOS arm64.
Network checks used BRPC 1.18.0 and Protobuf 36.2. These are correctness results;
no training-throughput or GPU speedup measurement is reported.

| Executed check | Evidence covered |
|---|---|
| Core correctness | Sum/mean/sequence, empty bags, repeated IDs across features, gradient validation, retry suppression and row-reference lifetime |
| Storage and recovery | State limits, epoch checks, durable publication, corrupt-file rejection, exact weights/moments/counters after restore and continued optimizer trajectory |
| Resource payload limits | Rejected ticket/queue payloads do not publish a live ticket |
| Concurrent Learners | Eight concurrent caller threads alternate training/evaluation, retry updates and leave no live tickets; shared rows receive the expected 192 updates |
| Numeric and ordering boundaries | Optimizer overflow is rejected without partial updates; out-of-order completions retire without leaving sequence holes |
| Policy checks | Observation threshold, moving cost estimates, route hysteresis, negative-benefit exclusion, reuse ranking, duplicate-candidate suppression, byte budgets, slot reserve and in-flight batch limit |
| Real-process RPC | Two independent server processes plus a client; pooled and sequence outputs, gradients and parameter states match a single-shard reference |
| Process restart | Both servers stop, restore their shard snapshots under a new run ID, reject old-run packets and match the reference's next updates |
| Wire validation | A malformed attachment length/count is rejected before parameter access |

The native CTest suite contains three entries: core correctness, policy checks,
and the real-process lifecycle. The core-only ASan/UBSan suite contains two
entries. Commands are in `RUNNING.md`; detailed output is available in the local
build directory's `Testing/Temporary/LastTest.log`.

GPU residency, CUDA updates, two-way migration under active training, TensorFlow
graph integration, long-running stress and performance comparisons remain to be
implemented or measured. A passing CPU policy check does not establish GPU
execution or a throughput benefit.
