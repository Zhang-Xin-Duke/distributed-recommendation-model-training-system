# 004: Separate CPU service; retain the GPU execution context

**Status:** Proposed for implementation, 2026-10-31.

## Context

The initial integration already uses C++, BRPC, Protobuf, TensorFlow operations
and GPU-side communication. A new training framework or mandatory transport stack
would add unrelated integration work.

## Decision

Use a C++20 parameter service with BRPC metadata RPCs and contiguous payload
attachments. Define original protocol messages in this repository. Keep CUDA
allocation, kernels and completion handling inside the existing GPU process.

Use a fixed shard map for each run. Start with loopback development endpoints.
Multi-host deployment binds to an explicitly configured private interface and uses
the deployment's authenticated transport boundary. Maintenance calls and agent
registration are restricted to the run controller; payload limits are validated
before allocation.

## Consequences

The first adapter fits the existing execution chain. The independently defined
API remains usable by other clients, although BRPC limits immediate portability
to ecosystems expecting another RPC protocol. Static membership simplifies
ownership and draining; elastic resharding and live service failover are later work.
