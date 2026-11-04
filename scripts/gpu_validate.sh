#!/usr/bin/env bash
set -euo pipefail

# Reproducible build on an Ubuntu GPU instance. The NVIDIA driver is retained;
# compiler/runtime components are installed into this project's Python environment.
dps_source="$(cd "$(dirname "$0")/.." && pwd)"
dps_work="${DPS_WORK_ROOT:-${HOME}/dps-validation}"
dps_prefix="${dps_work}/dependencies"
mkdir -p "${dps_work}/results"
exec > >(tee "${dps_work}/results/build-and-validation.log") 2>&1
date -u
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader

sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  build-essential git pkg-config libgflags-dev libprotobuf-dev libprotoc-dev \
  protobuf-compiler libssl-dev libleveldb-dev libsnappy-dev zlib1g-dev \
  nlohmann-json3-dev python3-venv

python3 -m venv "${dps_work}/environment"
"${dps_work}/environment/bin/python" -m pip install --disable-pip-version-check \
  'cmake==3.31.6' ninja 'cuda-toolkit[nvcc,cudart,crt,nvvm,cccl]==13.0.3'
dps_native_tf="${DPS_NATIVE_TF:-OFF}"
dps_training_python="${DPS_TRAINING_PYTHON:-OFF}"
dps_python_bridge=OFF
if [[ "${dps_native_tf}" == "ON" || "${dps_training_python}" == "ON" ]]; then
  dps_python_bridge=ON
  "${dps_work}/environment/bin/python" -m pip install --disable-pip-version-check \
    pybind11 'tensorflow[and-cuda]==2.20.0'
fi
dps_python="${dps_work}/environment/bin/python"
dps_cmake="${dps_work}/environment/bin/cmake"
dps_nvcc="$(find "${dps_work}/environment" -type f -name nvcc -print -quit)"
if [[ -z "${dps_nvcc}" ]]; then
  printf 'No nvcc executable was installed by the selected CUDA component packages.\n' >&2
  exit 1
fi
dps_cuda="$(dirname "$(dirname "${dps_nvcc}")")"
# The pip distribution places libraries in lib; nvcc's compiler-identification
# link step uses the conventional Linux toolkit lib64 directory.
if [[ ! -e "${dps_cuda}/lib64" && -d "${dps_cuda}/lib" ]]; then
  ln -s lib "${dps_cuda}/lib64"
fi
if [[ ! -e "${dps_cuda}/lib/libcudart.so" ]]; then
  ln -s libcudart.so.13 "${dps_cuda}/lib/libcudart.so"
fi
"${dps_nvcc}" --version

dps_protobuf_flags=()
if [[ "${dps_native_tf}" == "ON" ]]; then
  # Distribution static archives can contain non-PIC TLS relocations. Build the
  # private runtime dependency with the same protobuf version used by BRPC.
  if [[ ! -d "${dps_work}/protobuf-source/.git" ]]; then
    git clone --depth 1 --branch v3.12.4 https://github.com/protocolbuffers/protobuf.git "${dps_work}/protobuf-source"
  fi
  "${dps_cmake}" -S "${dps_work}/protobuf-source/cmake" -B "${dps_work}/protobuf-build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${dps_prefix}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF -Dprotobuf_WITH_ZLIB=ON
  "${dps_cmake}" --build "${dps_work}/protobuf-build" -j 6
  "${dps_cmake}" --install "${dps_work}/protobuf-build"
  dps_protobuf_flags=(
    "-DProtobuf_INCLUDE_DIR=${dps_prefix}/include"
    "-DProtobuf_LIBRARY_RELEASE=${dps_prefix}/lib/libprotobuf.a"
    "-DProtobuf_LIBRARY_DEBUG=${dps_prefix}/lib/libprotobuf.a"
    "-DProtobuf_PROTOC_EXECUTABLE=${dps_prefix}/bin/protoc"
    "-DEPS_PROTOC_STATIC=${dps_prefix}/lib/libprotoc.a"
  )
fi

if [[ ! -d "${dps_work}/brpc-source/.git" ]]; then
  git clone --depth 1 --branch 1.18.0 https://github.com/apache/brpc.git "${dps_work}/brpc-source"
fi
# Thunder interposes dlsym through its preload library. BRPC's optional pthread
# contention-profiler hook can re-enter its pthread_once initialization through
# that interposer. Disable only this hook in the dependency's mutex source.
if [[ -f /etc/thunder/libthunder.so ]]; then
  "${dps_python}" - "${dps_work}/brpc-source/src/CMakeLists.txt" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
line = '\nset_source_files_properties(bthread/mutex.cpp PROPERTIES COMPILE_DEFINITIONS NO_PTHREAD_MUTEX_HOOK)\n'
text = p.read_text()
if line not in text:
    p.write_text(text + line)
PY
fi
"${dps_cmake}" -S "${dps_work}/brpc-source" -B "${dps_work}/brpc-build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${dps_prefix}" \
  -DBUILD_BRPC_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF -DWITH_DEBUG_SYMBOLS=OFF
"${dps_cmake}" --build "${dps_work}/brpc-build" -j 6
"${dps_cmake}" --install "${dps_work}/brpc-build"

export PKG_CONFIG_PATH="${dps_prefix}/lib/pkgconfig:${dps_prefix}/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="${dps_prefix}/lib:${dps_prefix}/lib64:${dps_cuda}/lib:${dps_cuda}/lib64:${LD_LIBRARY_PATH:-}"
"${dps_cmake}" -S "${dps_source}" -B "${dps_work}/build-cuda" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${dps_prefix};${dps_cuda}" \
  -DCMAKE_CUDA_COMPILER="${dps_nvcc}" -DCUDAToolkit_ROOT="${dps_cuda}" \
  -DCMAKE_CUDA_ARCHITECTURES="${DPS_CUDA_ARCHITECTURE:-89}" \
  -DEPS_BUILD_BRPC=ON -DEPS_BUILD_CUDA=ON -DEPS_BUILD_TESTS=ON -DEPS_SANITIZERS=OFF \
  -DEPS_BUILD_RUNTIME="${dps_native_tf}" -DEPS_BUILD_PYTHON="${dps_python_bridge}" \
  -DPython3_EXECUTABLE="${dps_python}" "${dps_protobuf_flags[@]}"
"${dps_cmake}" --build "${dps_work}/build-cuda" -j 6
"${dps_work}/environment/bin/ctest" --test-dir "${dps_work}/build-cuda" --output-on-failure
cp "${dps_work}/build-cuda/Testing/Temporary/LastTest.log" "${dps_work}/results/LastTest.log"
date -u
