#!/usr/bin/env bash
set -euo pipefail

# Install a previously validated Linux runtime bundle on an owned GPU node.
# The caller supplies its checksum-verified archive; no credentials enter it.
task_root="${DPS_WORK_ROOT:?Set DPS_WORK_ROOT}"
if [[ "${DPS_BUILD_FROM_SOURCE:-0}" == "1" ]]; then
  export DPS_TRAINING_PYTHON=ON
  if [[ -z "${DPS_CUDA_ARCHITECTURE:-}" ]]; then
    task_capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.[:space:]')"
    export DPS_CUDA_ARCHITECTURE="${task_capability}"
  fi
  exec bash "$(dirname "$0")/gpu_validate.sh"
fi
task_archive="${DPS_RUNTIME_ARCHIVE:?Set DPS_RUNTIME_ARCHIVE}"
mkdir -p "${task_root}"
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  libgflags-dev libprotobuf-dev libprotoc-dev libssl-dev libleveldb-dev \
  libsnappy-dev zlib1g-dev python3-venv
python3 -m venv "${task_root}/environment"
"${task_root}/environment/bin/python" -m pip install --disable-pip-version-check \
  'tensorflow[and-cuda]==2.20.0' 'numpy==2.5.4' 'pybind11==3.1.0'
python3 - "${task_archive}" "${task_root}" <<'PY'
import pathlib,sys,tarfile
archive,root=map(pathlib.Path,sys.argv[1:])
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        path=pathlib.PurePosixPath(member.name)
        if not path.parts or path.parts[0]!='portable-runtime' or '..' in path.parts or member.issym() or member.islnk():
            raise RuntimeError('unexpected runtime archive member')
    tar.extractall(root,filter='data')
PY
export LD_LIBRARY_PATH="${task_root}/portable-runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${task_root}/portable-runtime/python:${task_root}/source/adapters/entropy"
"${task_root}/environment/bin/python" - "${task_root}" <<'PY'
import pathlib,sys,hashlib,json,subprocess,importlib.metadata
root=pathlib.Path(sys.argv[1])
import _eps_native
assert hasattr(_eps_native.Backend,'submit_synchronous')
assert hasattr(_eps_native.Backend,'lookup_layout_cache'), 'Runtime bundle predates reusable lookup metadata; rebuild from current source'
assert hasattr(_eps_native.Backend,'lookup_transport_stats')
binary=root/'portable-runtime/bin'
for p in binary.iterdir():
    text=subprocess.check_output(['ldd',str(p)],text=True)
    if 'not found' in text:raise RuntimeError('missing runtime dependency: '+p.name)
record={'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in ['tensorflow','numpy','pybind11']},
        'binary_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in binary.iterdir()},
        'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version','--format=csv,noheader'],text=True)}
(root/'runtime-installed.json').write_text(json.dumps(record,indent=2)+'\n')
print('Validated runtime installed')
PY
