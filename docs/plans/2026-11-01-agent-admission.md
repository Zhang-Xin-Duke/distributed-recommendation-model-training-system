# Agent admission and backpressure

**Goal:** Replace the transport's 128-RPC rejection boundary with observable,
bounded Agent-side waiting while preserving parameter command identities and
completion semantics.

**Architecture:** One FIFO admission queue per Agent covers every RPC producer.
It bounds active work, total accepted requests and retained request bytes. A
dispatcher starts work without blocking BRPC workers; its permit remains owned
by the asynchronous completion until response encoding and completion finish.
BRPC's independent transport concurrency rejection is disabled for this service.
The application budget remains finite and rejects before payload unpacking and
backend execution if exhausted. The protobuf envelope is already parsed by BRPC; attachment unpacking
and backend execution follow application admission. An ungated statistics RPC
exposes current/high-water state.

**Tech stack:** C++20, mutex/condition-variable queue, BRPC/Protobuf, existing
future completion executor and CUDA backend.

1. Add a standalone admission component and deterministic tests for count/byte
   bounds, FIFO dispatch, completion-held permits, exception cleanup and shutdown.
   Exercise the same tests under ASan/UBSan and ThreadSanitizer.
2. Route all Agent data/control methods through admission. Keep queued request
   objects and closures alive until dispatch; capture RPC pointers by value.
   Return explicit capacity errors when budgets are exhausted. Do not replay
   mutating commands under new identities or silently discard accepted work.
3. Add CLI budgets and a statistics RPC. Verify external concurrent RPCs with a
   deliberately slow backend, independent clients and a low active limit. Check
   exact optimizer state/counters after drain and that rejection precedes mutation.
4. Run complete native checks and real CUDA high-load verification, including
   the previously failing 128/192-Learner paths on independent GPU agents.
   Record latency as well as completion: backpressure is not a throughput claim.
5. Publish source-hashed results, update integration progress and delete rented
   resources after artifact retrieval. The native Entropy environment and
   in-process device-tensor integration remain separate completion requirements.

## Execution evidence

Runtime `426c27b` implements the complete RPC admission lifecycle and the CLI /
statistics surface. Core tests cover asynchronous permit ownership, FIFO order,
count/byte rejection, concurrent producers, exception cleanup and shutdown.
The component passes ASan/UBSan and ThreadSanitizer. The native RPC test checks
256 queued reads and separately compares complete state for eight home streams
after 256 expanded/compact updates, followed by acknowledged eviction.

The native suite passes 26/26 and the native/TensorFlow suite passes 40/40.
The identical Linux/CUDA binaries pass a 28-test build suite and per-worker CUDA
checks. A final two-A6000/two-L40 fleet completes 23 verified candidate trials,
12,950 acknowledged updates and 736 sampled counters, including 128/192 Learners
on 1/2/4 Agents and sustained four-Agent pressure. No candidate update/admission
errors or outstanding work remain in completed records. The same-fleet previous
Agent fails at 128 Learners and retains 254 tickets. Deployment transport failures
are recorded separately. Every owned rental is deleted after artifact retrieval.

[Design](../AGENT_ADMISSION.md) · [CUDA report](../AGENT_ADMISSION_RESULTS.md) ·
[Validation index](../validation/2026-11-01-agent-admission.json)
