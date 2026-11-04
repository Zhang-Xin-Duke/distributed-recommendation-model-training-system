# Durable worker gradients and synchronous job recovery

## Decision

Persist each worker's original sparse gradient, sample count, completed-read
frontier, loss and input/model identity before sending that round to any shard.
Keep immutable checksummed files on retained worker storage. Recover the same
run through replica-backed parameters, with all previous workers, primaries and
the coordinator stopped before replacement.

Sequential task admission and one synchronous optimizer update per round make
shard frontiers differ by at most one. If frontiers differ, every worker's
original gradient for that round must already be durable: a shard could not have
committed without receiving all workers. Replay those packets to every shard;
committed shards acknowledge without another update, and lagging shards commit.
Only then reconstruct the coordinator at the common durable round and resume
workers at their corresponding cyclic input positions.

If no shard committed the interrupted round, workers with a durable gradient
reuse it. Other workers recompute from the unchanged pre-round parameters.
Restarting the primaries removes abandoned Pull/Release tickets. Restore each
worker's transport read sequence from persisted request frontiers and its own
journal; training rounds and read sequences remain separate clocks.

## Why this recovery boundary

Rolling back every failure to a periodic checkpoint discards acknowledged
replica progress. Recomputing missing gradients against partially updated shards
can produce a different model. Durable original gradients permit rolling the
partially committed round forward. This adds disk writes before submission and
retains gradient history; bounded journal storage rejects exhaustion, and log
compaction is subsequent work. Model/data identity and exact input positions
are checked before replay. No stochastic RNG/external reader state is inferred
from the simple deterministic cyclic model used for the first execution.

The explicit manager owns role termination, term advancement, endpoint selection
and restart. Journal locks reject concurrent workers using the same directory.
The recovery planner locks journals, validates the group and replica frontiers,
and publishes a restart plan. It does not infer termination from a stale PID
file. Worker storage must remain accessible; disk/node loss and automatic
multi-node supervision require further execution. GPU-owner fencing, capacity
beyond one physical GPU and full original Entropy training remain required.

## Verification

Test journal corruption, immutable retries, bounds, identity/cursor mismatch and
exclusive writers. Kill actual training processes at a live read ticket, after
one shard commits and before task completion, and after a worker journal is
durable but before submission. Replace primaries under fresh terms, reconstruct
the coordinator and resume fresh worker processes. Compare all parameter/Adam
rows and per-round losses with uninterrupted training and verify cleanup.
