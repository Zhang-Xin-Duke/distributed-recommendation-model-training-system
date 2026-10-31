# Project scope and references

This repository defines an independent embedding parameter-service design and
original interfaces. Its first integration retains the existing training
framework at the embedding backend boundary.

Research inputs include a locally archived framework source snapshot and related
design documents. The local collection preserves file hashes, document provenance,
attachments and reading records. The collection is excluded from Git. Public
technical references appear in the README and application-scope document.

The reference framework already provides sharding, pooling, gradient merging,
sparse optimizers, cache mechanisms and asynchronous communication. This project's
proposed work is the service extraction and the cost-based access, admission and
migration policies described in `ACCELERATION.md`.

Tracked C++ headers and protocol messages are original interface contracts.
Historical framework measurements are not project benchmark results. Runtime
implementation and performance evaluation follow the implementation plan.
