# Original native Linux build bundle implementation plan

**Goal:** Move the complete original training framework's native dependency closure
to a Linux/NVIDIA build host without depending on the source disk's absolute paths.

**Architecture:** Read the existing Bazel workspace and successful dependency and
build-file queries. Materialize source inputs and build metadata into a private,
relocatable workspace, retaining the complete original framework tree for Python
training and control entry points. Regenerate compiler and CUDA facts on the
destination from installed tools before building the original shared library.

**Tech stack:** Python standard library, Bazel 7.7.0, original TensorFlow 2.12.1
NVIDIA SDK, Linux GCC/Clang and CUDA.

1. Add `scripts/original_native_bundle.py` with safe literal workspace parsing,
   source materialization, relative repository declarations and SHA-256 inventory.
   Keep all original inputs read-only; generated private artifacts belong in build/.
2. Add fixture tests for relocation, file symlink materialization, missing metadata,
   generated-output classification and rejection of escaping label paths.
3. Run it against the actual original native dependency graph. Move the resulting
   workspace and rerun the original Bazel dependency query. Compare label sets.
4. Configure actual Linux compiler/toolkit fixtures, build `//entropy:ts_clib.so`,
   load it with its matching Python runtime, then run original training/control.
   A successful source bundle/query is evidence for step 3 only, not step 4.

The full distributed-training goal and original Entropy integration remain open
until the native build, library loading and real training/control execution pass.
