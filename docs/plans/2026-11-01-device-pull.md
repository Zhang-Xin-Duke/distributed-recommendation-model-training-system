# Device Pull implementation plan

**Goal:** Produce complete distributed model features on CUDA through the existing
Pull/Push/Join lifecycle, including cross-shard pooling and sequence padding.

**Architecture:** A device resolver returns local device contributions and remote
host contributions after destination projection. The distributed client retains
lookup tickets and CPU contributions, then delegates final feature assembly to
the model worker's CUDA backend. The adapter passes sequence output projections
and returns owned device blocks under its ordinary batch handle.

**Tech stack:** C++20, existing BRPC control, native worker Agent and CUDA.

The brainstorming and writing-plans workflows identify a prerequisite for native
TensorFlow device operations: the current client reconstructs all final features
on CPU. Adding a framework operator before fixing that boundary would retain
the device-to-host round trip. A CPU assembly followed by upload is retained as
the host baseline; the new path assembles local device contributions on CUDA.

## Implementation

1. Define device contributions, sequence projections and assembly requests. Add
   device lookup/resolver configuration to the distributed client, preserving its
   lookup-failure drain and release guarantees. Device batches retain host shape
   metadata, device outputs and the existing per-shard tickets.
2. Extend destination projection to device resolvers. Dispatch all native/remote
   destinations before collecting; drain every submitted read on any error.
   Native device blocks preserve projected positions; remote blocks restore the
   existing host layout. Native callbacks use `WorkerCacheAgent::GatherDevice`.
3. Add CUDA assembly on the backend's ordered executor and model device. Reuse
   its output ownership/budget and stream. Merge host/device contributions,
   scatter projected sequence rows, divide pooled means once and zero padding.
   Retain all inputs until device reads complete, including failure cleanup.
4. Add `TableAdapter::PullDevice` and `PullPreparedDevice`, using the same live
   handle, Push/Release/Join, save and preparation lifecycle. Validate gradients
   by output shape, compact padded sequence gradients before sparse Push.
5. Test native routing/cleanup on CPU with explicit reference device tensors;
   verify on a rented GPU against full CPU training trajectories with repeated
   IDs, multiple shards, mixed CPU/native/remote cache contributions, means,
   empty bags, padded sequences, prepared inputs and save/resume. Verify invalid
   assembly and output-budget failures release live tickets.
6. Record source-hashed results, delete the GPU instance after artifact retrieval,
   update documentation and push. Native TensorFlow device operator and original
   application/reader/multi-peer execution remain in the main implementation plan.

## Validation scope

The native device path supplies complete model features. Framework loss/dense
collectives remain unchanged. Backward still submits feature gradients through
the existing acknowledged host/RPC update path. No throughput gain is assumed.
