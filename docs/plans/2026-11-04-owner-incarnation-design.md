# Independent parameter-owner incarnations

Keep the training run stable during replica-backed continuation, while each
replacement GPU parameter owner receives a fresh control/read namespace. The
manager stops the old owners and launches replacements with a new `--run` and
the existing `--training-run`. Parameter shards, remote framework adapters and
preparation tools select that namespace through `agents[].owner_run`.

Reusing the training run as the owner identity permits a delayed request to
match reused row/slot/generation numbers after an owner restart. Changing the
whole training run would also invalidate the existing replica and worker
journals. Separate identities preserve committed training progress and reject
old owner reads/control commands at the replacement endpoint. This uses the
existing run fields in the Agent protocol, so it requires no wire-schema change.

The Agent validates the owner incarnation before dispatching an operation and
validates the stable training identity in optimizer contexts. Neither context
nor original request identities are rewritten. Existing callers can omit the
new fields and retain the original single-run behavior. Incarnation identities
provide replay isolation; deployment authentication remains separate.

Verify real RPC listener reuse with old readers/writers, fresh row/slot reuse,
ordinary and compact updates, rejected foreign model contexts and unchanged
update counters after rejection. Train the full Embedding/MLP model through two
Host owners with distinct incarnations and compare against the global-batch
reference. Existing native TensorFlow CPU tests exercise adapter compatibility.

This change supplies the isolation mechanism needed by GPU job recovery. The
recovery manager continues rejecting GPU-owned jobs until replica replay,
owner rehydration, partial-round reconciliation and resumed model training are
verified together. Old owners must be terminated before replacement; endpoint
replacement does not revoke a still-running owner at another endpoint.
