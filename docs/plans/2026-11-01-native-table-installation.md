# Native original-table installation

Preserve the original SparseTable, InputBuilder gradient dispatcher and Runner
checkpoint boundaries while selecting native TensorFlow operations through job
configuration. The portable backend remains the default.

1. Share one native runtime per original learner across a private controller
   Session and every model Session/graph. A process-local weak registry, keyed by
   an installation-generated context ID, retains the original request sequence;
   reject configuration reuse under a conflicting key. The controller Session
   keeps the context alive between model Session lifetimes.
2. Add native controller operations for bounded checkpoint reads, loaded restore
   metadata and per-table training/prediction mode. Preserve the existing checksum
   and committed-export contract. Prediction releases its handle after producing
   independently owned output; mode changes require a drained learner.
3. Install a native Session backend through JobConfiguration. Create selected
   tables in the controller, build Eps Pull/Push/Release operations in the original
   graph, and reuse the existing save/export/restore hooks. Shared GPU ownership
   must retain one worker Agent across learner contexts in the same process.
4. Exercise graph recreation, mixed selected tables, sequence/padding/bucket IDs,
   prediction/training switches, missing gradients and original-controller save
   and restore. Check native graph types and complete touched optimizer state
   against the existing portable trajectory. Run Linux native Session checks;
   macOS retains its explicit native-runtime platform guard.

Original application/reader execution and native multi-peer framework barriers
remain separate requirements. Do not mark them complete from extracted-method
checks. Private reference files remain unchanged and are excluded from publishing.
