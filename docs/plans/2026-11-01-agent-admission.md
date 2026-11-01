# Agent admission and backpressure

**Goal:** Replace the transport's 128-RPC rejection boundary with observable,
bounded Agent-side waiting while preserving parameter command identities and
completion semantics.

**Architecture:** One FIFO admission queue per Agent covers every RPC producer.
It bounds active work, total accepted requests and retained request bytes. A
dispatcher starts work without blocking BRPC workers; its permit remains owned
by the asynchronous completion until response encoding and completion finish.
BRPC's independent transport concurrency rejection is disabled for this service.
The application budget remains finite and rejects before decoding/execution if
exhausted. An ungated statistics RPC exposes current/high-water state.

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
