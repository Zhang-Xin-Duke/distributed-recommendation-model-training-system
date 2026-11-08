# Choose a training policy before starting a job

The TensorFlow workload runner can try round-synchronous updates, unrestricted
per-worker updates, and per-worker updates with admission bounds of 1, 2, and
4 steps. Run it with a local BRPC and TensorFlow build:

```sh
python tests/tensorflow_auto_strategy.py --build build/tensorflow \
  --mode auto --patterns balanced --repeats 1 --target-loss 0.60 \
  --out build/auto.json --model-out build/auto-model.json
```

Each candidate gets the same pilot step budget, starting weights, optimizer
state, training batches, and validation loss target. The runner chooses the
fastest pilot that reaches the target. If all miss, it records that outcome
and falls back to round-synchronous training. It starts a fresh run with the
chosen policy, then checks the configured step budgets in ascending order.
Each budget starts from the same initial state. If the final attempt misses
the target, the report marks the run as unsuccessful. Model export writes the
inspected parameter and optimizer state only after a successful final run.

The default workload is a TensorFlow embedding classifier. To provide batches
and held-out examples, pass a JSON file with `--dataset` using the
[workload format](../tests/tensorflow_strategy_workloads.py). To use a second
host, pass `--remote-worker SSH_HOST CHECKOUT PYTHON BUILD`. The remote host
must have the same checkout and a compatible build.

## Measure the cost of selection

```sh
python tests/tensorflow_auto_strategy.py --build build/tensorflow \
  --mode compare --patterns balanced persistent bursty --repeats 3 \
  --bounds 1 2 4 --pilot-steps 8 --endpoints 8 16 --target-loss 0.60 \
  --out build/auto-strategy-results.json
```

The three cases inject no delay, 20 ms on every step of one Worker, or 80 ms
every fourth step of one Worker. The last two inject the same mean delay over
four steps. The delayed Worker alternates between repeats. Within each paired
comparison, dataset and initial state hashes must match.

Wall time includes process startup, model warmup, validation, and teardown.
The auto arm also includes every pilot and the decision. For each policy,
the report records successful pairs and the difference in wall time. Runs
that exhaust their step budget without passing remain visible. Step budgets
are independent runs, so this measures time to a tested passing endpoint,
including the cost of earlier attempts.

The admission bound is the allowed lead over the slowest completed Worker.
Round-synchronous training applies one sample-weighted Adam update per round.
The other candidates apply one Adam update per Worker contribution. Each
candidate uses the same sample budget, and the report records optimizer
update counts separately.

## Research behind the design

- [AutoSync](https://proceedings.neurips.cc/paper/2020/hash/0a2298a72858d90d5c4b4fee954b6896-Abstract.html)
  searches synchronization configurations for a model and cluster.
  [Pollux](https://www.usenix.org/conference/osdi21/presentation/qiao) combines
  throughput and statistical efficiency. Both motivate measuring useful
  training progress on the target workload.
- [MLPerf Training](https://mlcommons.org/benchmarks/training/) measures wall
  time to a fixed quality target and repeats runs to account for variance.
  This experiment includes policy search in the timed interval.
- [Hyperband](https://arxiv.org/abs/1603.06560) limits search cost by eliminating
  weak candidates at intermediate budgets. A short pilot can miss a slower
  starter, so the first version records fallback and final validation instead
  of extrapolating a winner from samples per second.
- [DSSP](https://baoxinchen.github.io/publications/DSSP_ICDCS_2019.pdf) adjusts
  the staleness bound while training. [Sync-Switch](https://arxiv.org/abs/2104.08364)
  switches between synchronous and asynchronous training. [GBA](https://arxiv.org/abs/2205.11048)
  shows why global batch and stale gradients matter when switching. These
  approaches address online changes. The runner selects a policy before the
  final run and keeps it fixed until that run finishes.
