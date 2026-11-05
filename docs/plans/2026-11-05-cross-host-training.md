# Cross-host training continuation implementation plan

**Goal:** Validate a real distributed training prefix, joint checkpoint,
node-owned source retirement and continued model updates as one executable job.

**Architecture:** The existing managed supervisor trains a reference run and a
short source generation on a supplied two-node fleet. It streams the joint
checkpoint to the controller and exports stopped source ownership. A local
replacement generation stages an independent baseline, verifies readiness,
retires the source through the original node transports, and continues the same
model/input sequence. A lost commit response interrupts the retirement manager;
reentry completes the same durable decision. Model parameters, optimizer state,
update counts and losses are compared with uninterrupted training.

**Tech Stack:** Existing C++ replica/checkpoint controls, TensorFlow model
workers, ManagedRoles, NodeSessions SSH transport, native snapshot verification.

1. Add `scripts/continue_training_generation.py` to start a controller-local
   replacement from an exported source, preserve input cursors, verify readiness,
   execute retirement through source-node transports, and resume training.
2. Add an executable integration check that runs an uninterrupted reference and
   a split/continued model. Require complete small-model state inspection;
   record actual host/GPU identities and the transport rather than inferring
   physical topology from the number of participants.
3. Run the integration locally with real native services and TensorFlow first.
   Check lost successful commit replies, manager reentry and full-state/loss
   comparison. Retain source files until the replacement baseline is verified.
4. Build the remote deployment and reusable runtime upload locally, refresh
   provider credits and availability, and rent only after all commands are ready.
   Use actual SSH for source storage participants and retain terminal cleanup
   evidence. Do not treat local execution as physical-host acceptance.
5. Prepare the separate fixed-workload one-versus-two GPU comparison. Keep two
   logical Workers, the table/model, samples and total update count fixed; verify
   device UUIDs and resident-state bytes and report the controller-relay transport.

The local replacement makes checkpoint and data-plane recovery observable with a
single rented GPU plus the controller. It validates cross-host state ownership
and source retirement while limiting rental time. Physical GPU scaling remains
a separate experiment using the same logical workload across placements.

## Executed results

The local continuation/source-export regression passes. On the rented L40, 104
native source files match the deployment manifest; the native client and
services were rebuilt from that source. Real CUDA owner checks pass. Linux
direct TensorFlow/native co-loading initially exposed protobuf symbol collision;
localizing static dependency symbols passes both import orders. The existing
process-isolated model adapter remains the training path.

Two actual-SSH trials pass: two source Workers/shards sharing the L40 host, then
one L40 Worker/shard and one controller CPU Worker/shard on separate hosts.
Both export the round-4 checkpoint, retire old payloads on two actual hosts,
recover a lost successful commit response, and finish the model at [8, 8].
All 32 complete parameter/Adam rows and losses match uninterrupted references
within 2.98e-8 and 5.96e-8. Input cursors and update counts match exactly.
The initial mixed-host configuration used a non-TensorFlow local interpreter;
that failed attempt was stopped and excluded from the accepted references.

The second L40 request was rejected for unavailable inventory. The owned GPU
instance was deleted and provider inventory absence verified after 14.6 minutes.
Fixed-workload physical multi-GPU scaling remains outstanding. The next rental
should start only after that benchmark/deployment is prepared and stock and
remaining gift credits are refreshed.
