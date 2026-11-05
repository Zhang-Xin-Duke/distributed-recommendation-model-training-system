# Training Job Controller implementation plan

Goal: extend the existing parameter-server training path to 2–8 logical groups and multiple queued jobs on 2–8 registered compute nodes.
Architecture: one durable Python controller, SQLite/WAL scheduling ledger, node-local atomic reservations and existing C++/TensorFlow workers and recovery. Parameter and gradient payloads retain their direct data paths.

Accepted decisions: extend the existing framework; priority/FIFO feasible-job scanning without preemption; bounded deterministic placement search (128 candidates); explicit replica/mirror placement; cooperative CPU/RAM reservations and exclusive per-job GPUs; fixed logical membership during a job. Validate 2/4/8 real process/network topologies and separately physical GPUs within an explicit available budget.

1. Generalize managed execution, role placement and continuation; keep existing two-group inputs compatible. Verify original and generalized training/recovery.
2. Implement node resource inspection and durable reservation/verification/release, including role ownership and ports. Verify lost replies, partial gang rollback and conservative unresolved cleanup.
3. Implement JobSpec, resource-aware placement, SQLite state/events and CLI serve/plan/submit/status/cancel/node operations. Persist intent before external actions; adopt existing handles after controller restart.
4. Integrate multi-job execution and recovery-time replanning, confirmed-node-loss fencing, survivor-authority selection and redundancy rebuilding. Verify full model/Adam/input/loss continuation and controller restart.
5. Publish executable examples, architecture/ADR, machine-readable local and hardware validation; retain rejected experiments and precise hardware counts.

Acceptance: resource ownership never overlaps across jobs; no worker starts before gang readiness; no reservation is released while its process outcome is unresolved; failed process/node recovery retains training semantics. Single controller-disk loss or loss of all state authorities is outside v1. Physical GPU verification is a separate required gate, never inferred from local process counts.
