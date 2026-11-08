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

## When to spend time on another search

A fresh search pays off only when the selected policy saves more time than
the pilots cost. For a short or recurring job, one option is to reuse a
previous choice keyed by model, data, optimizer settings, batch size, Worker
count and hardware, and the measured network profile. Measure cached choices
separately from cold searches and revalidate when those inputs change. This
extends AutoSync's use of related configurations and Vizier's reuse of prior
search experience. [Google Vizier](https://research.google/pubs/google-vizier-a-service-for-black-box-optimization/)

When the search space grows, compare candidates at equal sample budgets,
give each a minimum budget, and then allocate more work to the strongest
candidates. Retain candidates whose early scores are too close to distinguish.
Report selection cost and the rate at which the final run reaches its target.
[ASHA](https://proceedings.mlsys.org/paper_files/paper/2020/hash/a06f20b349c6cf09a6b171c71b88bbfc-Abstract.html)
· [Ray Tune trial schedulers](https://docs.ray.io/en/latest/tune/api/schedulers.html)

Occasional slow steps can also be addressed within synchronous training.
Backup workers let a synchronous round proceed using the first required
contributions, which changes resource use and which batches contribute.
Benchmark that separately when spare workers are available. For a worker
that stays slow, compare replacing it with tolerating its delay. Include
checkpoint and restart time when evaluating replacement.
[Revisiting Distributed Synchronous SGD](https://research.google/pubs/revisiting-distributed-synchronous-sgd/)
· [PyTorch elastic launch](https://docs.pytorch.org/docs/main/elastic/run.html)

## Estimate from a matching workload

Pass a complete comparison report from the same deployment to use historical
measurements before launching pilots.

```sh
python tests/tensorflow_auto_strategy.py --build build/tensorflow \
  --mode auto --patterns balanced persistent bursty \
  --profile build/auto-strategy-results.json --out build/auto-estimate.json
```

The runner describes the job in its report with the model, worker count, batch
size, sample budget, validation target and delay pattern. It matches the
profile's device, build, step budgets and delay settings. An external dataset
also has to match its fingerprint. For the built-in workload the seed can
change, while the generator and its dimensions stay the same. The profile
must contain at least three comparable repetitions. A fixed strategy is
eligible for reuse only if it passed all of them.

The estimate uses the median wall time for each eligible strategy and the
median historical pilot cost. If synchronous training is within 10% of the
fastest estimate, the runner favors synchronous training. It skips pilots
when the proposed direct run is no slower than the estimated fastest run plus
pilot cost. Otherwise it runs the ordinary pilot search. The final run always
checks the same validation target. Supply profiles from the current hardware,
network and software deployment because the runner cannot infer changes in
shared load or link quality from a historical report.
