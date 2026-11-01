# Parameter-free checkpoint chief and peer coordination

An original worker chief can participate in training control without creating a
selected external table. It must still freeze/export/resume the shared parameter
service and verify restored provenance. Add a dedicated lazy administrative
backend, separate from learner backends and GPU ownership. Retain the existing
leader backend when it exists so its input-close contract remains unchanged.

Test the original Runner/checkpoint methods with two OS peer processes and two
learner threads per process, real TensorFlow Sessions and two BRPC parameter
shards. Execute the preserved Python local Barrier/MultiThreadingLearner/env
barrier methods; bridge only their native cluster call through a test rendezvous.
Hold the pre-commit global stage to prove sparse admission stays frozen and no
joint marker appears before every peer completes. A later checkpoint must reuse
the lifecycle successfully; a dense-save failure must remain uncommitted/frozen.

The test proves the adapter's peer/thread coordination against an actual network
rendezvous. It does not establish original C++ cluster-barrier or reader execution.
The full native application still needs its missing ts_clib.so/ts_clib_gpu.so
runtime; the copied tree and complete Code repository contain no such artifact.

Implemented and verified: seven focused CTest checks pass, including repeated
two-peer/four-thread checkpoints and an injected dense-save failure. The prior
adapter reproduces the administrative-only-chief failure. The public evidence
index is `../validation/2026-11-01-peer-checkpoint-control.json`. No additional
GPU rental was required. Complete original native application/reader/cluster
execution remains pending the runtime artifact.

Follow-up: restart the two parameter shards from the second joint checkpoint
under a fresh run. Recreate the two peer processes/four learner threads and
execute the original Runner restore, including the parameter-free chief's
administrative provenance check and the preserved Python barriers. Continue
one update per external learner and export the resulting state. Compare all
weights, Adam moments, counters, optimizer sequence and beta powers with six
uninterrupted direct-client updates. Restore a deliberately different dense
initializer from the saved value. Keep the native reader/cluster scope explicit.

Restore follow-up implemented: both PS processes and both peer processes restart;
four learner threads complete the original restore barriers. After two resumed
updates per shard, two rows' complete optimizer state matches six uninterrupted
updates with zero difference. The dense Saver restores `-0.5` to saved `0.5`.
Native cluster/reader artifacts remain the separate application dependency.

Native build preflight: the separate Bazel workspace and matching TensorFlow
2.12.1 NVIDIA SDK are recovered. An independent local workspace loads 39,338
partial dependency labels, with downloads disabled. The remaining unresolved
repository is the internal Redis SDK (`platform/redis-sdk-cpp`), on the original
`ts_clib.so -> costa_op -> bgraph/ops_utils -> common/base:redis` path. Exact SDK
headers are absent from the Code tree. Full compilation/execution needs that
SDK or a compatible prebuilt framework environment; no GPU rental was started.
