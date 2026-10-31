# 001: Extract the embedding backend

**Status:** Proposed for implementation, 2026-10-31.

## Context

The reference framework combines data ingestion, model construction, sparse
parameters, GPU execution and job control. Rebuilding all of these would make
the independent project much larger and obscure parameter-access improvements.

## Decision

Retain Entropy's training framework. Extract table storage, batched access,
sparse updates and cache coordination into a separately configured service.
Connect it through an adapter at the embedding Pull/Push/Join boundary. Keep the
GPU agent in the existing GPU worker, sharing its device and model environment.

The public API uses table IDs, keys, layouts and gradients. Application-specific
feature extraction remains in the caller.

## Consequences

This gives the project a complete input/output lifecycle and independent CPU
deployment. It also keeps integration changes small enough to compare access
policies under the same model computation. The first adapter remains coupled to
the original graph and completion conventions; another framework needs its own
adapter. The service does not replace dense training or job scheduling.
