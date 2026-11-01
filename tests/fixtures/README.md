# Snapshot compatibility fixture

`snapshot-v1-unbounded.bin` independently encodes the original little-endian
format, with the original FNV-1a checksum. It contains one table and one row:

- Run `legacy-unbounded`; shard 0 of 1; partition epoch 1.
- Table 1, dimension 2; LazyAdam learning rate 0.1, beta1 0.9, beta2 0.999,
  epsilon 1e-8 and initial beta powers 0.9/0.999.
- Constant initialization 0.25, seed 42; optimizer sequence 0.
- Key 1, weights `[0.25, 0.25]`, zero moments and zero row updates.

Format 1 has no weight-bounds field. The compatibility test loads it into a new
run and checks that the table remains unbounded with unchanged row values.
