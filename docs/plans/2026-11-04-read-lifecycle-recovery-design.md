# Cross-shard read lifecycle recovery

Add an explicit identity-based cancellation to the existing Release RPC. It
validates training run and partition epoch, completes an existing lookup or
records a bounded completed lifecycle when Lookup has not arrived. The same
lifecycle dependency serializes Lookup and cancellation, so a late Lookup
cannot allocate a new ticket after cancellation. Ordinary ticket-based Release
retains its existing contract. Cancellation does not undo a committed Push.

On failed distributed Lookup, collect all dispatched completions before any
device reads can be released, then cancel the identity on every shard, including
shards whose ticket was never received. Retry transient cancellation/release
failures a bounded number of times using the exact original identities. A
client remains usable only after every shard acknowledges cleanup. If cleanup
is uncertain it fails closed and uses the existing replica/job recovery path.
Partial gradient Push keeps its existing failure behavior.

Expose read-lifecycle health through the framework adapter. The model worker
can retry the same input before forward/backward when cleanup is confirmed;
the new Pull consumes a fresh read sequence while the training iteration stays
unchanged. A bounded retry budget and retry count make this behavior observable.
Read completion certificates include canceled lifecycles independently of task
clocks and optimizer rounds.

Verify unseen/late cancellation, lost successful Lookup replies, partial
Release, failed cleanup, request-window bounds and unchanged parameters. Use
real BRPC fault proxies to discard successful read/release results and delay
Lookup beyond the client deadline. Train the real TensorFlow Embedding/MLP
through those endpoints and compare weights, Adam state and losses with the
uninterrupted reference; verify no leftover pins after late packets finish.
