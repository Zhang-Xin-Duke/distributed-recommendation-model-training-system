# In-process worker cache agent implementation plan

**Goal:** Let a native training process own its cache Agent, accept the existing
PS control RPCs and consume owned CUDA feature results on its model stream.

**Architecture:** `WorkerCacheAgent` owns the existing backend, `AgentRuntime`,
BRPC service and listener. Local host/device gathers and remote RPCs enter the
same count/byte admission queue and ordered runtime. Closing the worker ends
admission, drains accepted operations, then destroys runtime/backend state.
Returned device allocations keep their own ownership after worker destruction.

**Tech stack:** C++20, BRPC, existing CUDA backend and stream/event interface.

The existing user authorization covers implementation, local checks and a short
single-GPU correctness run within the USD 20 project rental budget. This step
uses the Code, brainstorming and writing-plans workflows in the current checkout.

## Choice of integration boundary

Link the existing Agent service into the worker rather than spawning another
process or replacing its control protocol. This keeps state in the training
process and preserves PS-issued update identities, migration and generation
checks. A separate executable remains useful for deployment and uses the same
owner after this change. TensorFlow custom device operators need a matching
framework ABI; they remain a later integration step. No DLPack dependency is
needed for the native CUDA-stream boundary.

## Steps and evidence

1. Add `AgentRuntime::GatherDevice`; validate run and backend capability on the
   same ordered executor as control commands. Host-only backends fail explicitly.
2. Add asynchronous local gather dispatch to `BrpcAgentServer`, using its existing
   admission queue and completion pool. Charge retained native input capacities;
   keep the permit through future completion. Expose drain and native stats.
3. Create `include/eps/worker_cache_agent.hpp` and
   `src/transport/worker_cache_agent.cpp`. The owner accepts an injected backend,
   binds its endpoint and supports idempotent Stop and direct host/device gather.
   Replace duplicate listener ownership in `src/server/agent_main.cpp`.
4. Add `tests/worker_agent_tests.cpp`: real RPC control plus local reads, shared
   count/byte rejection, wrong-run/capability errors, close during a paused read,
   late-call rejection, backend destruction and endpoint reuse. Run native and
   TensorFlow suites after the focused test.
5. Add `tests/worker_agent_cuda.cu`: remote load/update/eviction and local device
   gather, consume output with a kernel on a nondefault model stream, inspect
   full optimizer state, and retain a result across worker destruction. Execute
   on a rented NVIDIA GPU; retrieve raw logs before deleting the instance.
6. Document the ownership API, shutdown responsibility and measured scope. Record
   source hashes, executed tests and cleanup, then commit/push.

## Completion scope

This closes the native worker ownership/lifecycle boundary. Complete original
Entropy application/reader execution, multi-peer framework barriers and a native
TensorFlow device path continue to be requirements of the main plan. Device
gather returns owner contributions; global cross-shard feature reconstruction
and mean normalization still belong to the client/framework integration.


## Execution record

Implemented at `e88ce65`. macOS CPU/RPC 27/27 and native/TensorFlow 41/41 pass.
The Ubuntu CPU/CUDA build passes 31/31 on one RTX A6000, including both new
worker checks. The CUDA test verifies model-stream consumption, two complete
optimizer rows, output-budget enforcement/reclamation, stale-location rejection
and result ownership after worker destruction. Source/archive/binary hashes and
raw logs are retained; the single rental was deleted and final inventory is empty.

[Verification index](../validation/2026-11-01-worker-agent.json). The main plan's
remaining framework-integration requirements are unchanged.
