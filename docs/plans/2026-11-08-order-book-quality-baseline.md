The previous FI-2010 run stopped at 64 rounds and classified every validation
window as unchanged. Use a chronological 80/20 split of CF7's training file,
with a 100-row gap before validation. Leave all official test files unused.
Input prices are expressed relative to the current bid/ask midpoint; volumes
are log-transformed. Feature scaling is fitted on the training partition only.
Each example retains 100 snapshots, and uses the first FI-2010 horizon.

Prepare and verify the single-device baseline:

    python tests/order_book_baseline.py --prepare data.zip --data fi2010-chrono-relative.npz
    python tests/order_book_baseline.py --data fi2010-chrono-relative.npz --batch-size 512 \
      --steps 500 --eval-every 50 --val-windows 2048 --initial-weights initial.npy --out baseline.json

A passing checkpoint has cross-entropy at most 0.95, macro F1 at least 0.45,
and recall at least 0.10 for each class. Require two consecutive passing
checkpoints for the single-device baseline. The fixed development subset
contains 2,048 evenly spaced eligible windows; it is not the official test set.
Run a second seed to check that the result is not unique to one initialization.

For the distributed comparison, use the identical initial-weight file on both
hosts and 256 windows per Worker. The parameter server keeps FP32 deltas from
the immutable Keras initializer, so its optimizer clocks start at zero. The
Worker adds the delta before computing its gradients. Validate one synchronous
update against a single-device calculation before starting the comparison.
Use the same training-window schedule and checkpoint quality rules across
policies. Report the first **observed** passing endpoint, any subsequent
confirmation, and unsuccessful runs. Checkpoints are independent fixed-budget
trials, so the true first crossing between checkpoints is unobserved. GPU
utilization is a device-level sample collected during the timed run. The
three policies differ in optimizer update counts at fixed sample counts.
