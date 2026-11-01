# Current native integration and multi-GPU operating envelope

Run the native TensorFlow CPU/CUDA graph checks before deploying additional GPU
workers. Reuse the same compiled server, Agent and probe across the fleet.

1. Verify native graph loss, dense weights and complete sparse optimizer state
   against independent CPU parameter shards. Retain failed build records.
2. Extend that graph check with externally owned CUDA Agents. Execute the same
   model trajectory with one, two and four physical GPU owners. Check that every
   remote owner receives work and drains, and compare all touched parameter state.
3. On the same fleet, hold table, input and PS shard count fixed and sweep
   8/32/128/256/512 concurrent Learners on one, two and four GPU Agents. Each
   topology starts fresh. Stop its load scan at the first failed point and retain
   failure diagnostics; continue independent topologies.
4. Confirm the highest useful completed point with three 20-second windows.
   Compare throughput, P99, queue wait, Agent RPC admission, CPU and GPU samples.
   Report a latency operating region separately from the maximum accepted load.
   The service accepts Learner IDs 0–1023. Reserve IDs 1023/1022 for setup and
   health, then scan 768/1022 training Learners on a separately named probe if
   the 512-Learner scan completes. Keep server/Agent binaries fixed and label
   probe provenance independently. Protocol and generator limits remain distinct
   from service throughput and admission failures.
   Confirm successful 1,022-Learner points with three 20-second windows. Then
   scan 128/512/1022 Learners on one/four owners using the million-row, 512D,
   8192-ID workload, stopping each topology at its first failed load point. This
   tests the payload/deadline boundary within the supported Learner ID range.
5. Retrieve and hash raw logs before deleting the owned rentals. Keep the fleet
   under the standing USD20 project budget; never purchase credits. Report source
   revisions and transport limits with the results.

The service pressure test uses synthetic feature gradients. The native model
trajectory separately verifies actual TensorFlow forward/backward execution.
Together they cover capacity and integration; they do not measure distributed
dense all-reduce or establish a universal GPU memory maximum.

## Execution record

The native Linux suite executes 47 passing tests and skips three private-source
checks. Native model trajectories pass on one, two and four physical GPU owners,
with complete touched sparse optimizer state matching CPU replay. The current
fixed, sustained, upper-stream and large-input matrix retains 63 completed
windows, 62,363 acknowledged sparse updates and 2,016 sampled row counters.

All three topologies complete three 20-second windows at 1,022 Learners on the
128D/1024-ID workload. On the million-row 512D/8192-ID workload, one/four owners
complete 512 Learners. At 1,022 Learners the one-owner probe exceeds its 285-second
overall deadline; the four-owner run reports 991 RPC errors and later recovers
statistics/read availability with zero live tickets. Neither failed cohort has
a verified full optimizer trajectory. Live observations retain drained Agent
queues alongside shard-statistics timeouts; the four-owner memory sample reports
zero OOM events. Parameter-service waiting and overload/control-plane response
require further investigation and repair.

All four archives are retrieved and hash checked before deleting the four owned
instances; final account inventory is empty. The corrected probe reserves
initialization/health identities; a preliminary overlapping-ID probe is excluded.

[Full results](../NATIVE_MULTIGPU_RESULTS.md) ·
[Validation](../validation/2026-11-01-native-multigpu.json)
