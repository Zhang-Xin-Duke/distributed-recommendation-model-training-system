# L40 large-table training

Two TensorFlow workers train a shared Embedding/MLP parameter table whose
weights and both Adam moments occupy **30 GiB on one NVIDIA L40**. Every declared
row is initialized and migrated to the CUDA owner before training. The run
completes eight synchronous gradient rounds and checks model gradients and
sampled final optimizer state against a CPU global-batch reference.

| Item | Executed configuration/result |
| --- | --- |
| Physical GPU | One NVIDIA L40; 46,068 MiB reported total device memory |
| Parameter table | 2,621,440 rows × 1,024 FP32 dimensions |
| Initialized GPU-resident rows | 2,621,440; zero remaining owner slots |
| Weight / Adam moment storage | 10 GiB weights + 10 GiB first moments + 10 GiB second moments |
| Model | Embedding pooling followed by a trainable two-layer tanh MLP |
| Workers / rounds | Two workers on the same GPU; eight synchronous rounds |
| Local examples / IDs | 16 examples per worker, 16 IDs per example; fixed repeated batches |
| Final task clocks | `[8, 8]`; no live read tickets |
| CPU reference comparison | All weights/moments of 512 selected touched rows; 521 rows touched in total |
| Maximum final-state absolute error | `7.6238066e-6` (acceptance tolerance `5e-5`) |
| Maximum first aggregated-gradient error | `2.3283064e-8` |
| Weighted loss, first → last round | `0.685872 → 0.237314` |
| Peak `nvidia-smi memory.used` | 32,535 MiB (31.77 GiB) |
| Peak host cgroup memory | 32.95 GiB |
| Complete initialization/migration time | 199.04 seconds |

The table contains the Embedding vectors and packed MLP weights/biases. All
trainable state follows the same sharded parameter interface and sparse Adam
update path. CUDA owns the vectors/moments; the workers execute actual GPU
forward/backward and submit their gradients. Host staging remains at RPC
boundaries. The host backing copy is included in the measured cgroup footprint.

The CPU reference uses the combined worker batch and the same initialization,
optimizer and update count. The first aggregated gradient comparison covers all
touched Keys. Final state inspection selects 512 of the 521 touched rows to stay
within the bounded inspection payload, comparing 1,572,864 FP32 values across
weights and both moments. Initialization/residency checks cover the entire
2,621,440-row table.

## Reproduce

After installing the current CUDA/Python build, run:

```sh
python tests/tensorflow_synchronous_training.py --build build-cuda --devices 0 \
  --rows 2621440 --dimension 1024 --hidden 8 --iterations 8 \
  --samples-per-worker 16 --bag-size 16 --out l40-large-training.json
```

The recorded model run uses revision `7e19c8112bf0d24ea813ccf2eaea08fc9715d1fa`,
TensorFlow 2.20.0, NumPy 2.5.4, CUDA architecture 8.9 and NVIDIA driver 610.43.02.
Resource sampling runs through reference preparation, GPU initialization,
training and teardown. The validation record retains sampled row errors, losses,
placement/count checks, resource observations, source/binary hashes and the full
raw report's SHA-256. The complete raw report and a verified reusable Linux
runtime archive are retained in the local experiment directory.

The same Linux/CUDA build executes 62 passing checks, including six CUDA checks;
four original-source fixture checks skip because the private source fixture is
absent from the deployed archive. A separate real Redis 6.0.16 process check
passes after using the version-compatible CLIENT PAUSE syntax in its timeout
fixture. No model or parameter-runtime implementation changes were needed for
that fixture adjustment.

This is the single-GPU large-working-set baseline. The principal acceptance
workload next distributes a global state exceeding the largest physical GPU
across distinct GPU owners. Its candidate two-node layout assigns 30 GiB to each
owner, using the working set established here. The second L40 allocation was
rejected for unavailable inventory; that multi-GPU capacity run remains pending.

The owned cloud instance was deleted after verified download of results and the
runtime cache. [Machine-readable results and cleanup evidence](validation/2026-11-04-l40-large-training.json).
