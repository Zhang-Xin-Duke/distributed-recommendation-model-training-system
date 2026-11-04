# Cross-shard read recovery

A worker can retry a failed parameter read without restarting training after
every shard confirms that the abandoned read is closed. Identity-based
cancellation handles successful Lookups whose replies were lost and Lookups
that arrive after the client has already timed out. Training task progress and
optimizer updates remain unchanged until the worker submits its gradients.

```mermaid
sequenceDiagram
    participant W as Model worker
    participant A as Parameter shard A
    participant B as Parameter shard B
    W->>A: Lookup, read sequence 3
    A-->>W: Ticket and parameters
    W->>B: Lookup, read sequence 3
    Note over W,B: Successful reply is lost, or original request is delayed
    W->>A: Cancel read identity 3
    W->>B: Cancel read identity 3
    A-->>W: Read closed
    B-->>W: Read closed; late Lookup will be rejected
    W->>A: Retry same input, fresh read sequence 4
    W->>B: Retry same input, fresh read sequence 4
    W->>W: Forward/backward for the same training iteration
    W->>A: Submit gradient
    W->>B: Submit gradient
```

## Closing an uncertain read

The existing Release RPC has an explicit `cancel_unseen` mode and a partition
epoch. The server checks training run, epoch, learner and read sequence. It
releases an existing ticket or records a completed lifecycle if Lookup has not
arrived. Lookup and cancellation share the actor's lifecycle dependency, so
either arrival order leaves that read identity terminal. Retired frontiers
continue rejecting late packets after individual cancellation records leave the
bounded request-history window. Canceling an unseen read creates no parameter
row. Cancellation does not roll back an applied optimizer update and cannot
retire the synchronous optimizer actor's identities.

After a distributed Lookup fails, the client drains every dispatched completion
and device read, then cancels the identity on every shard. It does not need the
lost ticket. Cleanup and ordinary Release retry only unacknowledged shards, at
most three attempts, preserving the same request identity. A lost successful
Release acknowledgement therefore does not free anything twice or advance a
training clock twice.

If any shard's cleanup remains uncertain, the client stops admitting reads.
`Join` and completed-read certification fail, and the existing replica-backed
job recovery path is required. Partial gradient Push retains its existing
failure behavior; read cancellation does not make multi-shard updates atomic.

## Model-worker retry

The TensorFlow model worker retries temporarily unavailable Pull operations
only while its adapter confirms healthy read lifecycles. The input batch and
training iteration stay fixed; each new Pull uses a fresh read sequence. It
records `read_retry_count` separately from gradient retry counts and task clocks.
No forward/backward or gradient submission occurs for the discarded read.

`read_retry_ms` defaults to 2,000 milliseconds after the first failed attempt;
zero disables worker-level retries, and the configured maximum is 10,000 ms.
The attempt's transport timeout still applies. Retry admission checks the
budget before another attempt, so an admitted RPC can finish after that budget.
Input/output validation and reconstruction failures close queued input as
before. Prepared-read failures still close and cancel future prepared inputs;
their framework must restart from its saved input cursor. This prevents a
prefetched later batch from overtaking the failed batch. Synchronous Pull also
requires its preparation queue to be empty before admitting another read.

The wire fields are additive. Existing ordinary Release requests retain their
contract. Runtime installation checks the new `read_lifecycle_cleanup`
capability before deploying the framework worker with identity-based recovery.

## Verification

The core check covers pre-dispatch failure, lost successful Lookup replies,
dispatch exceptions, lost cancellation acknowledgements, partial Release,
bounded failed cleanup, request-window retirement, invalid epochs/tickets and
protection of optimizer actor identities. It verifies zero live tickets and no
optimizer updates after successful cleanup, plus continued reads and completed
read certification. The adapter, preparation and gather paths also run under
ASan/UBSan and TSan.

The real model check uses two workers and two parameter shards for eight rounds:

| Injected fault | Observed action |
| --- | --- |
| Lookup rejected before dispatch | Cancel both identities; retry the same input |
| Successful Lookup reply discarded | Release the unknown ticket by identity; retry |
| Release rejected before execution | Retry only that shard's original Release |
| Successful Release reply discarded | Duplicate Release acknowledges the existing completion |
| Original Lookup forwarded after its deadline | Cancel first; late Lookup returns `RETIRED_REQUEST` |

The test-only BRPC proxy forwards production parameter and gradient RPCs. A
lost-success case replaces the upstream successful result with `UNAVAILABLE`;
the delayed case forwards the original Lookup after the caller's deadline.
After all delayed calls finish, the test checks no leftover tickets, exact
optimizer step counts and all model parameters/Adam state against uninterrupted
training. CPU execution compares all ten initialized model rows and every loss
exactly. CUDA mode uses two GPU parameter owners on one physical card, initializes
all 64 rows and additionally checks GPU model tensor placement and full resident
state. CUDA execution is recorded separately from the CPU checks.

```sh
ctest --test-dir build/tensorflow --output-on-failure \
  -R '^(read_lifecycle_cleanup|tensorflow_read_failures|bounded_asynchronous_preparation)$'
python tests/tensorflow_read_failures.py --build build-cuda --cuda --out read-failures-cuda.json
```

[Validation record](validation/2026-11-04-read-lifecycle-recovery.json) retains
the CPU and L40 CUDA fault trajectories, the CUDA owner-rehydration follow-up,
source/runtime hashes and native/framework/sanitizer logs. Local suites pass
40 native and 65 framework checks, plus four ASan/UBSan, four TSan and two
C-ABI checks. The L40 build initially passes 67 checks and skips four missing
original-source fixtures; separate final rebuilt-runtime checks cover CUDA
read faults and owner recovery.
