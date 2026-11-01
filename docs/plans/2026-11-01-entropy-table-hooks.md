# Entropy table hooks implementation plan

**Goal:** Install the external parameter service at the existing SparseTable boundary while preserving InputBuilder, model code, gradient scaling and native tables not selected in configuration.

**Architecture:** Patch selected methods on the original class in place, retaining class identity and framework variable registration. A backend per original learner/thread owns request identities across all selected tables. Graph bindings preserve pooled and sequence layouts; callbacks capture their learner state explicitly. The original controller continues owning training/prediction mode and checkpoint coordination.

**Tech stack:** C++20 service/client already built; Python installation hooks, TensorFlow graph callbacks, existing SparseTable and InputBuilder source, two external BRPC shards.

1. Add `adapters/entropy/entropy_install.py`: validate opt-in job configuration, assign explicit stable table IDs and unique learner/thread identities, and install/uninstall conditional methods without replacing the original class.
2. Preserve original table construction and `add_var_collections`. Connect `_create_table`, `embedding_lookup_and_pooling`, `vars`, `output`, `apply_gradients` and mode setters. Flatten rank-2 sequence IDs, build offsets, retain UINT64_MAX padding, and preserve raw-ID/hash-bucket semantics supported by the selected source path.
3. Test the hook on source-executed original SparseTable/InputBuilder methods, then exercise a TensorFlow v1 Session graph with actual external shards and the unchanged gradient-dispatch method. Cover mixed native/external tables, valid zero IDs, padding, pooled/sequence shapes, automatic gradients, ignored missing gradients, prediction release and learner isolation.
4. Add an entry-point launcher that installs hooks before executing the existing training module; expose original-source provenance and exact validation scope.
5. Added original Runner checkpoint coordination, directory preparation before export, a joint commit marker after all original roles complete, and failure fencing. Source-executed original Runner/module checkpoint checks continue the graph after a mid-run save and reject a failed dense save. Remote restore staging, the native original-application build, multi-peer framework-barrier execution and in-process CUDA tensor interop remain requirements.

Verification commands use the canonical built native extension/server and Python environment while placing this worktree's adapter directory first in `PYTHONPATH`. The original private source remains unchanged and excluded from Git.
