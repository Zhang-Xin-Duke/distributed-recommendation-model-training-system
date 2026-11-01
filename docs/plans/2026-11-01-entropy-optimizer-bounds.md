# Entropy optimizer bounds implementation plan

**Goal:** Preserve Entropy SparseLazyAdam weight clipping when selecting the external parameter backend.

**Architecture:** Optional finite bounds belong to the immutable table optimizer configuration and travel with table definitions to every parameter owner. Initialization and Adam updates clamp weights; moments and update clocks retain their existing rules. New snapshots store bounds and continue reading existing unbounded snapshots.

**Tech stack:** C++20, Protobuf, CUDA, JSON and Python framework configuration.

1. Add a regression covering clipped initialization, alternating saturated updates, unchanged moments, CPU/Host-owner migration, and exact restart. Confirm the current interface fails this requirement.
2. Carry optional bounds through table configuration, BRPC codecs and JSON entry points. Validate the same range at CPU, Host and CUDA owners.
3. Apply bounds after finite Adam calculations on all execution paths. Keep default unbounded behavior and overflow rejection.
4. Version snapshots, read legacy snapshots as unbounded, and verify optimizer-exact bounded restart.
5. Add an Entropy optimizer configuration mapper and test it against the original optimizer constructor loaded from the preserved source. Keep unsupported optimizer features explicit.
6. Run appropriate native/RPC/Python and sanitizer checks. Record CUDA execution separately until verified on an actual device. Framework factory/checkpoint and device interop integration remain open requirements.
