# GPU model training integration

Connect the fixed-group synchronous parameter updates to TensorFlow model
forward/backward. Keep all trainable state shared by placing Embedding rows and
packed two-layer MLP weights/biases in the same sharded table. Each model worker
performs read-only Pull, computes local mean-loss gradients, reduces repeated
Keys, submits to all owners and advances its task only after acknowledgements.

Reuse the isolated native client process so the ML wheel and native RPC
dependency versions remain independent. Bound row-gradient payloads and keep
one fixed table schema. A whole-table preparation command initializes/migrates
every claimed resident row in bounded batches and verifies each owner count.
Local per-shard preparation permits distributing the large CPU backing store.

First compare CPU multi-worker execution against a single-worker global-batch
model: first gradients, final complete touched parameter/moment state, exact
step counts, loss and ticket drain. Add actual GPU device checks for model
inputs/loss/gradients and validate against the same model reference. Preserve
raw reports and source hashes before deleting the rented validation instance.

After this path is verified, orchestrate GPU workers and parameter shards across
multiple GPU instances and run actual resident state above the largest physical
single-device capacity. Record device UUIDs and actual host/device memory,
communication and owner load. A smaller allocator budget or small-state
multi-owner run does not substitute for this capacity experiment.
