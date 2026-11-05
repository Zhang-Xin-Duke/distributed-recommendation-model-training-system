"""Hold model startup after its GPU owner binds, until parameter shards are ready."""
import hashlib
import json
import os
import pathlib
import time
import uuid

from checkpoint_store import LocalFileSystem, _publish


def publish(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True) + '\n').encode()
    pending = path.with_name(path.name + '.pending-' + uuid.uuid4().hex)
    try:
        with pending.open('xb') as stream:
            os.chmod(pending, 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(LocalFileSystem, str(pending), str(path), len(raw), hashlib.sha256(raw).hexdigest())
    finally:
        pending.unlink(missing_ok=True)


def wait_for_parameters(gate_path, configuration):
    with pathlib.Path(gate_path).open() as stream:
        gate = json.load(stream)
    if (not isinstance(gate, dict) or set(gate) != {'schema_version', 'ready', 'release', 'timeout_seconds'} or
            gate['schema_version'] != 1 or not all(isinstance(gate[k], str) and gate[k] for k in ('ready', 'release')) or
            type(gate['timeout_seconds']) is not int or not 1 <= gate['timeout_seconds'] <= 3600):
        raise ValueError('invalid parameter startup gate')
    if configuration.backend != 'native' or 'cuda' not in configuration.native:
        raise ValueError('parameter startup gate requires the model process GPU owner')
    ready, release = pathlib.Path(gate['ready']), pathlib.Path(gate['release'])
    if ready == release or ready.exists() or release.exists():
        raise ValueError('startup gate requires fresh ready/release paths')
    value = dict(schema_version=1, run=configuration.run, agent=configuration.native['cuda'].get('agent', 0),
                 endpoints=list(configuration.endpoints), nonce=uuid.uuid4().hex)
    publish(ready, value)
    deadline = time.monotonic() + gate['timeout_seconds']
    while time.monotonic() < deadline:
        if release.exists():
            with release.open('rb') as stream:
                raw = stream.read(65537)
            if len(raw) > 65536 or json.loads(raw) != value:
                raise ValueError('parameter startup release differs from this owner generation')
            return
        time.sleep(.05)
    raise TimeoutError('parameter shards did not release this GPU owner before startup deadline')
