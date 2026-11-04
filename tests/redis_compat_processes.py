"""Exercise the public SDK adapter against real isolated Redis processes."""
import argparse
import json
import os
import pathlib
import secrets
import socket
import subprocess
import tempfile
import threading
import time


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client', type=pathlib.Path, required=True)
    p.add_argument('--redis-server', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path)
    a = p.parse_args()
    children, logs, reports = [], [], []
    password = secrets.token_hex(24)
    env = dict(os.environ, EPS_REDIS_TEST_PASSWORD=password, EPS_REDIS_WRONG_PASSWORD='incorrect-test-password')
    with tempfile.TemporaryDirectory(prefix='eps-real-redis-') as tmp:
        root = pathlib.Path(tmp)

        def start(name, authenticated=False):
            address = port()
            config = root / (name + '.conf')
            config.write_text(f'bind 127.0.0.1\nport {address}\nsave ""\nappendonly no\nprotected-mode yes\n' +
                              (f'requirepass {password}\n' if authenticated else ''))
            config.chmod(0o600)
            log = open(root / (name + '.log'), 'w+')
            logs.append(log)
            child = subprocess.Popen([str(a.redis_server), str(config)], stdout=log, stderr=log)
            children.append(child)
            until = time.monotonic() + 5
            while True:
                assert child.poll() is None, 'Redis process exited during startup'
                try:
                    with socket.create_connection(('127.0.0.1', address), timeout=.1):
                        break
                except OSError:
                    assert time.monotonic() < until, 'Redis startup timed out'
                    time.sleep(.02)
            return child, address

        def run(mode):
            r = subprocess.run([str(a.client.resolve()), mode], env=env, capture_output=True, text=True, timeout=15)
            if r.returncode:
                # Synthetic ephemeral credentials never enter diagnostics.
                raise RuntimeError((r.stderr + r.stdout).replace(password, '<test-password>')[-4000:])
            reports.append(json.loads(r.stdout))

        try:
            plain, plain_port = start('plain')
            auth, auth_port = start('authenticated', True)
            fault_socket = socket.socket()
            fault_socket.bind(('127.0.0.1', 0))
            fault_socket.listen(8)
            fault_socket.settimeout(.1)
            fault_port = fault_socket.getsockname()[1]
            stop = threading.Event()
            observations = {'connections': 0, 'incr_packets': 0}
            failures = []

            def drop_reply():
                try:
                    while not stop.is_set():
                        try:
                            client, _ = fault_socket.accept()
                        except socket.timeout:
                            continue
                        observations['connections'] += 1
                        with client, socket.create_connection(('127.0.0.1', plain_port), timeout=1) as upstream:
                            client.settimeout(1)
                            packet = b''
                            while b'lost-reply-counter' not in packet:
                                data = client.recv(65536)
                                if not data:
                                    break
                                packet += data
                            if packet:
                                observations['incr_packets'] += packet.count(b'INCR')
                                upstream.sendall(packet)
                                upstream.recv(65536)
                                # Real Redis executed the write. Drop its response.
                except Exception as e:
                    failures.append(type(e).__name__)

            thread = threading.Thread(target=drop_reply, daemon=True)
            thread.start()
            config = root / 'clusters.json'
            config.write_text(json.dumps({'clusters': {
                'plain': {'endpoint': f'127.0.0.1:{plain_port}'},
                'database-one': {'endpoint': f'127.0.0.1:{plain_port}', 'database': 1},
                'authenticated': {'endpoint': f'127.0.0.1:{auth_port}', 'password_env': 'EPS_REDIS_TEST_PASSWORD'},
                'wrong-password': {'endpoint': f'127.0.0.1:{auth_port}', 'password_env': 'EPS_REDIS_WRONG_PASSWORD'},
                'fault': {'endpoint': f'127.0.0.1:{fault_port}'}}}))
            env['EPS_REDIS_CONFIG'] = str(config)
            for name, content in [('malformed', '{'), ('empty-clusters', '{"clusters":{}}'),
                                  ('missing-password', json.dumps({'clusters': {'x': {
                                      'endpoint': f'127.0.0.1:{auth_port}',
                                      'password_env': 'EPS_REDIS_UNSET_TEST_PASSWORD'}}}))]:
                bad_config = root / (name + '.json')
                bad_config.write_text(content)
                env['EPS_REDIS_CONFIG'] = str(bad_config)
                run('config-rejected')
                reports[-1]['case'] = name
            env['EPS_REDIS_CONFIG'] = str(root / 'nonexistent.json')
            run('config-rejected')
            reports[-1]['case'] = 'absent-file'
            del env['EPS_REDIS_CONFIG']
            run('config-rejected')
            reports[-1]['case'] = 'unset-config'
            env['EPS_REDIS_CONFIG'] = str(config)
            run('basic')
            run('lost-reply')
            stop.set()
            thread.join(timeout=3)
            fault_socket.close()
            assert not thread.is_alive() and not failures, failures
            assert observations['incr_packets'] == 1, observations
            run('inspect-counter')
            plain.terminate()
            plain.wait(timeout=5)
            run('unavailable')
        finally:
            if 'stop' in locals():
                stop.set()
            if 'thread' in locals():
                thread.join(timeout=3)
            if 'fault_socket' in locals():
                fault_socket.close()
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=5)
            for log in logs:
                log.close()
            assert all(child.poll() is not None for child in children)
    result = {'test': 'public_redis_sdk', 'server_version': subprocess.check_output([str(a.redis_server), '--version'], text=True).strip(),
              'reports': reports, 'lost_reply_proxy': observations, 'cleanup_verified': True}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
