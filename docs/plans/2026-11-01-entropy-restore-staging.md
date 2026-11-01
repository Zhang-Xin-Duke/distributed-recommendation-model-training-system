# Entropy restore staging implementation plan

**Goal:** Restart fresh parameter shards from the original Runner's committed checkpoint, verify their loaded state at the original restore boundary, and continue the same training trajectory with restored dense/input state.

**Architecture:** The server retains provenance only after a fully checked snapshot load and exposes it through a read-only RPC. A bounded framework-filesystem stager verifies the joint marker, manifest SHA-256, file SHA-256 and partition header before publishing a server-local file. Original table/Runner restore hooks verify every loaded shard's current run, source run, byte length, embedded checksum and pristine state before letting the existing framework restore dense/native/input state.

**Tech stack:** C++20 snapshot reader, BRPC/Protobuf, native Python client, TensorFlow filesystem and original source-level Runner restore flow.

1. Implemented restored-snapshot provenance and a pristine-state boundary in CPU service, RPC and Python client. Core checks cover successful/failed loading, duplicate definition, first admission and repeated empty-snapshot restore rejection.
2. Added committed checkpoint validation, bounded local staging and a server-exec launcher. Filesystem checks cover uncommitted/mismatched/corrupt files, immutable retry and streamed copies.
3. Installed original get_reshard/table restore/Runner restore hooks. They require the original checkpoint ID and topology and verify loaded source state before original dense restore.
4. The two-process restart regression stages actual files through the CLI, executes original source restore functions with a real dense Saver, resumes a checkpointed test-feeder cursor and compares complete selected parameter/Adam state against uninterrupted execution. Initial Host-cache execution has zero state error; final verification is recorded separately.
5. Publish source/log provenance. Full native original-application execution, multi-peer framework barriers and in-process device-tensor integration remain requirements until directly executed.
