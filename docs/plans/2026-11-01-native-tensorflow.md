# Native TensorFlow operation implementation plan

**Goal:** Connect complete Device Pull and acknowledged sparse updates to real
TensorFlow CPU/GPU graphs without host Python callbacks or native-client IPC.

**Architecture:** A versioned POD C function table separates the native C++20
runtime from TensorFlow's C++ ABI. The runtime owns the existing adapter/worker;
TensorFlow resource kernels call that table and wrap owned output buffers.
The Linux runtime links private static BRPC/Protobuf and hides internal symbols.
The TensorFlow operation library is compiled with the installed wheel's flags.

**Tech stack:** C++20 native runtime, C ABI, TensorFlow resource/operation APIs,
existing BRPC and CUDA, Python graph wrapper and process integration tests.

The Code, brainstorming and writing-plans workflows apply. A fresh subprocess
probe (`import tensorflow; import _eps_native`) exits 139 on the local toolchain.
The existing version-isolated Python bridge remains available. For native device
operations, use a C boundary and private transport symbols rather than importing
the existing pybind extension into TensorFlow. The TensorFlow custom-op guide
requires wheel-compatible compile/link flags:
https://www.tensorflow.org/guide/create_op

## Steps

1. Add a C ABI for create/destroy, host/device Pull, prepare/consume, owned
   feature buffers, host/device gradient submission and control commands.
   Keep exceptions inside the native module and validate ABI version/structure
   sizes. Buffers retain their own allocation after runtime destruction.
2. Build `eps_runtime` with position-independent native libraries and a single
   exported entry point. On Linux select static transport/protobuf dependencies
   and bind/hide private symbols. Test it from a plain C/ctypes consumer.
3. Build a separate TensorFlow operation library with resource creation, Pull,
   prepared Pull, Push and control. Register CPU kernels and optional CUDA
   kernels. Device output buffers retain native ownership; stream synchronization
   precedes device gradient upload into the existing host/RPC Push path.
4. Add a graph wrapper preserving the ordinary lifecycle. Run a real graph with
   model loss/automatic gradients and two external PS processes, compare complete
   sparse state with CPU replay and exercise save/resume, prepared inputs and
   output lifetime. Verify graph operation types contain the native operations.
5. Build and run the CUDA graph on a rented NVIDIA GPU within the existing USD20
   project budget. Retrieve hashes/logs before deleting the instance. Record exact
   framework/runtime/toolchain scope and update the main plan and README.

## Scope retained

Original application/reader installation and multi-peer framework barriers remain
part of the original goal. This step implements native TensorFlow operations;
it does not replace model code, dense optimizers or collectives.
