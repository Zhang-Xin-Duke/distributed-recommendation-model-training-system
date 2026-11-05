"""Exercise startup release identity and stale-generation rejection."""
import json
import pathlib
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters/entropy'))
from startup_gate import publish


def main():
    with tempfile.TemporaryDirectory(prefix='eps-startup-gate-') as temporary:
        root = pathlib.Path(temporary)
        for mismatch in (False, True):
            directory = root / str(mismatch)
            directory.mkdir()
            gate = directory / 'gate.json'
            ready, release = directory / 'ready.json', directory / 'release.json'
            gate.write_text(json.dumps(dict(schema_version=1, ready=str(ready), release=str(release), timeout_seconds=5)))
            code = '''import sys,types
sys.path.insert(0,sys.argv[1])
from startup_gate import wait_for_parameters
config=types.SimpleNamespace(backend='native',run='restore-run',native={'cuda':{'agent':1}},endpoints=['127.0.0.1:9000'])
wait_for_parameters(sys.argv[2],config)
print('model-released')
'''
            process = subprocess.Popen([sys.executable, '-c', code, str(ROOT / 'adapters/entropy'), str(gate)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                deadline = time.monotonic() + 4
                while not ready.exists():
                    if process.poll() is not None:
                        raise AssertionError('startup child exited before readiness: ' + process.stderr.read())
                    if time.monotonic() > deadline:
                        raise AssertionError('startup child did not publish readiness')
                    time.sleep(.01)
                assert process.poll() is None
                owner = json.loads(ready.read_text())
                assert owner['run'] == 'restore-run' and owner['agent'] == 1
                if mismatch:
                    owner['nonce'] = 'previous-owner-generation'
                publish(release, owner)
                output, error = process.communicate(timeout=4)
                if mismatch:
                    assert process.returncode != 0 and 'differs from this owner generation' in error
                    assert 'model-released' not in output
                else:
                    assert process.returncode == 0 and 'model-released' in output, error
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=3)
        print(json.dumps(dict(test='parameter_startup_gate', valid_generation_released=True,
            stale_generation_rejected=True, model_waits_before_release=True,
            filesystem_protocol_only=True)))


if __name__ == '__main__':
    main()
