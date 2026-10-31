# 002: Preserve updates and move complete state

**Status:** Proposed for implementation, 2026-10-31.

## Context

Sparse optimizer updates depend on their ordering and history. Combining two
independent Adam updates into one changes training. Moving weights without their
moment states also changes subsequent updates.

## Decision

Deduplicate repeated keys and reduce their contributions within one original Push.
Keep different Learner updates separate. A home-shard sequencer supplies one
optimizer context to the CPU and GPU portions of each table update, retaining the
framework's asynchronous request model.

A lookup ticket pins parameter locations through backward and update completion.
Migration closes new references for its rows, drains existing references, copies
weights and optimizer state, then publishes the new generation. Requests waiting
on those rows resume afterwards. Completion is acknowledged before slot reuse.

## Consequences

There is one current state per row, and migration does not reset optimizer history.
The initial design forgoes live replicated writes and transparent agent failover.
Hot-row migration may wait, so admission budgets and free-slot reserves are part
of the performance design. It introduces no global training-step barrier.
