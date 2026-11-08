"""Launch one training Worker behind reverse SSH forwards, using a stdin start gate."""
import json
import os
from pathlib import Path
import selectors
import shlex
import subprocess
import sys
import tempfile
import time


def ssh_command():
    return ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=10',
            '-o', 'ServerAliveCountMax=2']


def probe_remote(host, checkout, python, _build):
    script = ('import sys; print("READY", flush=True); '
              '[print(line.strip(), flush=True) for line in sys.stdin]')
    command = [*ssh_command(), '--', host, shlex.join([python, '-u', '-c', script])]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True) as process, selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)

        def receive(expected):
            if not selector.select(timeout=15) or process.stdout.readline().strip() != expected:
                raise RuntimeError(f'SSH echo failed for {host}')

        try:
            receive('READY')
            samples = []
            for index in range(12):
                value = str(index)
                start = time.monotonic()
                process.stdin.write(value + '\n')
                process.stdin.flush()
                receive(value)
                samples.append(1000 * (time.monotonic() - start))
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    commit = subprocess.check_output([*ssh_command(), '--', host, shlex.join(
        ['git', '-C', checkout, 'rev-parse', 'HEAD'])], text=True, timeout=20).strip()
    return {'ssh_echo_rtt_ms': samples, 'git_commit': commit}


def worker_command(host, checkout, python, build, endpoints):
    command = ssh_command()
    for endpoint in endpoints:
        address, port = endpoint.rsplit(':', 1)
        if address != '127.0.0.1':
            raise ValueError(f'expected loopback endpoint, got {endpoint}')
        command += ['-R', f'127.0.0.1:{port}:{endpoint}']
    return [*command, '--', host, shlex.join(
        ['timeout', '1800', python, str(Path(checkout) / 'tests/ssh_training_worker.py'),
         checkout, build])]


def main():
    checkout, build = map(Path, sys.argv[1:])
    config = json.loads(sys.stdin.readline())
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(build / 'python'), str(checkout / 'adapters/entropy')]),
        CUDA_VISIBLE_DEVICES='0' if config['device'] == '/GPU:0' else '')
    with tempfile.TemporaryDirectory(prefix='eps-ssh-worker-') as directory:
        gate = Path(directory) / 'start'
        config['start_gate'] = str(gate)
        path = gate.with_suffix('.json')
        path.write_text(json.dumps(config))
        marker = Path(f'{gate}.worker-{config["worker"]}.ready')
        worker = subprocess.Popen([sys.executable, str(checkout / 'examples/tensorflow_training_worker.py'),
                                   '--config', str(path)], env=env, stdin=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 60
            while not marker.exists():
                if worker.poll() is not None:
                    raise RuntimeError(f'Worker exited before warmup: {worker.returncode}')
                if time.monotonic() >= deadline:
                    raise TimeoutError('remote Worker warmup timed out')
                time.sleep(.01)
            print('READY ' + marker.read_text(), flush=True)
            if sys.stdin.readline() != 'start\n':
                raise RuntimeError('remote Worker start gate was not released')
            gate.touch()
            return worker.wait()
        finally:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()


if __name__ == '__main__':
    sys.exit(main())
