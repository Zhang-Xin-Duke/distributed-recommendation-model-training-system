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
5. Retrieve and hash raw logs before deleting the owned rentals. Keep the fleet
   under the standing USD20 project budget; never purchase credits. Report source
   revisions and transport limits with the results.

The service pressure test uses synthetic feature gradients. The native model
trajectory separately verifies actual TensorFlow forward/backward execution.
Together they cover capacity and integration; they do not measure distributed
dense all-reduce or establish a universal GPU memory maximum.
