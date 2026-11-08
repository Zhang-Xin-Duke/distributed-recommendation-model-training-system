# DeepLOB training on two RTX A6000 hosts

Unbounded asynchronous training reached the validation target first in both trials. It took 20.9–23.0 s, compared with 27.9–52.3 s for round-synchronous training. Bounded delay kept the faster worker within two steps of the slower worker at admission, but took 62.5–98.4 s to reach the target.

## Dataset and single-GPU baseline

In the initial experiment, the model predicted “unchanged” for every validation window. That class made up 58.8% of training labels and 35.4% of validation labels. A constant predictor that outputs the training class frequencies has a validation cross-entropy of 1.209, close to the 1.208–1.215 measured after 64 steps.

In a separate CPU comparison, we kept the initialization and training schedule fixed and changed the input features. After 320 steps, the original DecPre features yielded a macro F1 of 0.256. Expressing prices relative to the order-book midpoint, applying log1p to volumes, and scaling with training-set statistics raised macro F1 to 0.491.

For the new benchmark, we sampled 2,048 development windows from the final 20% of the FI-2010 CF7 training file and left a 100-row gap at the split. The official test set was held out. A run passes when cross-entropy is at most 0.95, macro F1 is at least 0.45, and recall is at least 10% for each class.

Single-GPU TensorFlow runs used a batch size of 512. Two M1 Pro runs with different seeds and one RTX A6000 run passed at both 200 and 250 steps. On the A6000, cross-entropy fell from 0.616 to 0.597 and macro F1 rose from 0.514 to 0.576 between those checkpoints. In a separate one-step check, the two-host synchronous Adam update matched the single-GPU reference with maximum absolute errors of 1.94e-7 for weights, 1.86e-9 for first moments, and 8.59e-11 for second moments.

## Two-host results

Each host ran one worker on an RTX A6000 with a batch size of 256. Worker 0 shared its host with the coordinator and two CPU parameter shards. Worker 1 connected through an SSH tunnel with a median echo RTT of 15.9 ms. All runs used the same Keras initial weights, input order, and Adam learning rate of 0.001. We switched which worker received a 20 ms delay per step and rotated the strategy order between trials.

| Strategy | Trial 1 first pass | Trial 2 first pass | Mean utilization, GPU 0 / GPU 1 | Max. worker lead at admission |
| --- | ---: | ---: | ---: | ---: |
| Round-synchronous | 27.9 s (100 steps) | 52.3 s (200 steps) | 7.5% / 6.9% | 0 steps |
| Unbounded asynchronous | 23.0 s (100 steps) | 20.9 s (100 steps) | 30.4% / 7.6% | 239 steps |
| Bounded delay | 98.4 s (400 steps) | 62.5 s (300 steps) | 8.3% / 7.8% | 2 steps |

Each checkpoint came from a fresh run with a fixed step budget. The table gives the first observed pass in each trial. The next measured checkpoint also passed. All three strategies ran at 50, 100, 200, and 300 steps per worker. Only bounded delay ran at 400 and 500 steps. The utilization averages use the six main runs per strategy at 100, 200, and 300 steps, with NVML readings collected every 200 ms.

In the second asynchronous trial, recall for the weakest class dropped to 1.6% at 300 steps. Bounded delay reached a macro F1 of 0.740 at 500 steps in its second trial, an endpoint measured for that strategy alone. These runs compare time to a shared validation target on one network topology. Synchronous training applies one Adam update per round. The other two strategies apply one update per worker, so a fixed sample budget gives them twice as many updates.

In the first 200-step synchronous run, model computation and staging took a median of 28 ms per step. Submitting the update and waiting for it took 141 ms. All 28 two-host runs completed without an RPC failure.

[Experiment protocol and baseline commands](plans/2026-11-08-order-book-quality-baseline.md) · [Distributed benchmark](../tests/order_book_strategy_workloads.py)
