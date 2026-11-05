# Original Executor Multi-Peer Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Execute the complete original training application with two native peers, external sparse parameter shards and real training-progress/checkpoint integration.

**Architecture:** Launch the unchanged original `main.py` through `entropy_launch` in independent processes. Retain the original native dense parameter service, FileFeeder, ORC decoder, Executor, cross-peer barriers and Controller/Learner services. The independent parameter service owns selected sparse tables and acknowledged task progress. A private matching Linux runtime supplies the original native libraries and TensorFlow wheel.

**Tech Stack:** TensorFlow 2.12.1 NVIDIA, original CPU/CUDA libraries, C ABI parameter runtime, BRPC, gRPC, Hadoop FsShell and ORC.

---

1. Extract the existing one-peer acceptance into `scripts/validate_original_executor.py`; accept explicit native-root, native-operation library, parameter runtime, interpreter, work directory and peer count. Use a fresh work directory and retain exact process/log handles.
2. Add the project's own small Embedding/MLP user model in `examples/entropy_executor_model.py`. Model execution uses the full original Executor and native DenseAdam rather than a replacement training loop.
3. Generate several real ORC files through the recovered protobuf schema, use the original InputGlobbing/FileFeeder path and launch one or two complete peers. Give each peer a distinct native cluster endpoint, Learner service and GPU parameter-owner endpoint; share the declared Controller and external parameter group.
4. Verify native Controller/Learner pings, GPU graph placement, finite per-task losses, aggregate sparse update/task counts, complete joint checkpoint hashes, original dense optimizer exports and drained service state. Record the actual per-peer work distribution rather than assuming file assignment is balanced.
5. Gate physical execution on the existing matching-wheel/reader/original-training/one-peer checks. Never restart the live compiler to run this validation. An observation timeout retains the exact live process handles; an actual failed validation retains its logs for diagnosis.
6. Extend the executed multi-peer path with fresh-process joint restore and checkpoint-based failure/replay after the initial original multi-peer run succeeds. Full integration remains open until those executions pass; configuration generation and preserved-method fixtures do not satisfy it.

The fixed checkpoint export scratch files used by the original native code are
created only when absent, with ownership recorded by this validation, and removed
after all owned peer process groups have stopped. Private source and native
artifacts remain outside version control.

## Input recovery source finding

The original `tensorswitch/orc_dataset.cc:327` writes the current filename index,
while its row-offset write and seek are commented out. `FileFeederNextableDataset`
distributes files through the original native `FetchFile` service and keeps its
input-shot index in Python. The checkpoint routine separately writes
`icb_feeder_ckpt.meta.pbtxt` through `IcbTableIncFeeder`; that metadata is for the
ICB feeder and does not establish the ORC input cursor.

The multi-peer execution harness records those auxiliary files under their own
name. It verifies actual input processing through batch/task counts and retains
input restore/failure replay as an explicit acceptance requirement. The next
integration must bind a supported ORC replay position to the same parameter,
dense-model and task-progress checkpoint; a saved filename index alone cannot
establish batch-exact continuation.
