# Parameter-shard overload progress and observation

The multi-GPU study retains two large-input failures at 1,022 Learners. GPU
Agents drain while parameter-shard statistics time out. Source inspection shows
that dependency-cache pressure discards every queued plan and parks the scan;
later wakeups repeatedly rebuild the same blocked prefix. Statistics currently
share that scan and its active-operation limit.

Implement two changes while retaining row/optimizer/Agent conflict ordering:

1. Keep bounded queued dependency plans when a further plan does not fit. Retry
   after existing work makes progress and releases its reservation. Individual
   plans larger than the configured budget still reject before execution.
2. Add a bounded synchronous observation queue on the same state-owning thread.
   Statistics can observe in-flight state without acquiring data dependencies
   or consuming asynchronous active-operation slots. Interleave observations
   with runnable data work; checkpoint and mutation barriers keep their ordering.
   Give statistics RPC replies an independent bounded completion pool so four
   occupied data waiters cannot block an already-ready observation response.
3. Regress tight dependency budgets, a fully occupied active set, a held external
   future, observation admission, data fairness, failure and shutdown/drain. Add
   a service-level held-owner check, then verify ordinary native and framework
   checks. Run CPU/RPC pressure before renting CUDA hardware.
4. Repeat the recorded large-input trajectory on real CUDA, preserving failed
   baseline records and source/binary provenance. Check completed updates,
   post-drain state, observation latency and resource bounds. The global project
   retains original application/reader and multi-peer integration requirements.

Source behavior is established; the contribution of cache churn to the cloud
failure remains a hypothesis until the controlled and same-workload checks run.
