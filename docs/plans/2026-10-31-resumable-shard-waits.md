# Resumable shard operations implementation plan

**Goal:** Let a shard process independent requests while an operation waits for a cache agent, preserving parameter ownership, optimizer order and checkpoint barriers.

**Architecture:** A single actor owns shard state. C++20 coroutines suspend at external futures and resume on that actor. A bounded native completion pool observes futures; resource dependencies keep conflicting operations ordered. Parameter/model computation remains in its existing component.

**Tech stack:** C++20 coroutines, native condition variables, existing BRPC futures and Host/CUDA cache backends.

## Design decisions

Adding unordered service threads would require synchronizing every store, ticket,
policy and optimizer access. The actor retains one state owner and adds explicit
suspension points. A nested queue pump could hold references across reentrant
mutations and deadlock behind a later operation; resumable frames make pending
state and completion ownership explicit.

Initial dependencies retain the existing table-wide optimizer order for Push,
row exclusion during updates/migration, one control stream per home/agent and
lifecycle identity for duplicate requests. Lookups of independent rows can run
while remote work is pending. Automatic-policy completers share a maintenance
resource, then upgrade to a full barrier at an actual decision window after
other active operations drain. Configuration and save/restore also use barriers.
The design therefore preserves existing semantics before relaxing individual
dependencies using further measurements.

## 1. Resumable actor and dependency scheduler

Create `src/common/actor_task.hpp` and `src/common/resumable_actor.hpp`.
Implement move-only tasks, nested coroutine completion, single-owner resumption,
bounded admission/active work, resource ordering, full barriers, a bounded waiter
pool and drain-on-destruction. Allocate completion storage before invoking an
external operation. Observe every submitted future in a fan-out even after one
fails. Queue and active payloads share admission bytes; dependency metadata has
a separate cap. Measure execution segments separately from suspension time.

Create `tests/resumable_actor_tests.cpp` and register it in CMake. Check a blocked
external future, independent progress, conflicting-order fairness, barriers,
fan-out failure cleanup, exception mapping, bounded bytes/jobs and shutdown.
Run native, ASan/UBSan and focused TSan checks before service integration.

## 2. Integrate shard operations

Modify `src/server/service.cpp` to use the actor. Convert agent waits, Push,
ReadState, Move, maintenance and save to resumable tasks. Keep store/ticket access
on the actor; reacquire map iterators after suspension. Derive row/table/agent
dependencies from actual requests and live tickets. Expose bounded scheduling
configuration and active/waiting counters through the existing service/Stats.

Add a delayed Host-backend integration check that proves independent Lookup can
finish during a blocked update/migration, a conflicting Lookup cannot, optimizer
contexts and exact state remain ordered, and Save waits for outstanding work.
Run the complete local native/RPC/TensorFlow suite and sanitizer checks.

## 3. CUDA and measured effect

After local checks, validate the integrated path on one rented GPU. Reuse fixed
traces and numerical replay. Then compare the old and new schedulers under the
same topology/workload, recording queue time and actor active/suspended time.
Keep the earlier four-agent capacity results attached to their source revision;
publish new performance claims only after matched measurements. Fetch/hash
results before deleting rentals. Update architecture, API and validation docs,
commit and push the implementation and evidence.
