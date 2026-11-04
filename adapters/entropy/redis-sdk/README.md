# Public Redis SDK adapter

A C++17 BRPC-backed Redis client implementing the SDK interfaces used by the
retained training framework's shared Redis wrapper. It executes real Redis
commands, preserves ordered binary batch results and per-record TTL, and keeps
transport errors distinct from missing values.

[Configuration, execution semantics and verification](../../../docs/PUBLIC_REDIS_SDK.md)

The CMake target is `eps_redis_compat`. In an independent native Bazel workspace,
register this directory as the local repository `bilibili-redis`; its target
uses `@brpc//:brpc` and `@json//:json`. Original source files stay unchanged.
