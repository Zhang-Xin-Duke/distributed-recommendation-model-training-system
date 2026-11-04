"""Fetch/stage immutable worker journals without shared worker filesystems.

Private transfer plans contain entries with config, source_node,
destination_node and destination_directory. Nodes supply source/python and
SSH fields, or local=true for the same serialized subprocess protocol.
"""
import argparse
import json
import pathlib
import os
import selectors
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapters/entropy'))
from journal_handoff import CHUNK_BYTES


class NodeSessions:
    """One bounded JSON stream per node; reconnect only after a failed observation."""
    def __init__(self, timeout=60):
        self.timeout, self.sessions = timeout, {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for key in list(self.sessions):
            self.drop(key)

    def drop(self, key):
        entry = self.sessions.pop(key, None)
        if entry is None:
            return
        process, errors = entry
        process.stdin.close()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=3)
        process.stdout.close(); errors.close()

    def invoke(self, node, request):
        key = json.dumps(node, sort_keys=True)
        if key not in self.sessions:
            python = node.get('python', sys.executable if node.get('local') else 'python3')
            module = str(pathlib.PurePath(node['source']) / 'adapters/entropy/journal_handoff.py')
            argv = [python, module, '--serve']
            if not node.get('local'):
                argv = ['ssh', '-o', 'IdentityAgent=none', '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
                        '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=15', '-o', 'UserKnownHostsFile='+node['known_hosts'],
                        '-i', node['identity_file'], '-p', str(node['port']),
                        node.get('user', 'ubuntu')+'@'+node['host'], shlex.join(argv)]
            errors = tempfile.TemporaryFile()
            process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, bufsize=0)
            os.set_blocking(process.stdin.fileno(), False)
            self.sessions[key] = process, errors
        process, errors = self.sessions[key]
        try:
            pending = memoryview(json.dumps(request).encode()+b'\n')
            deadline, output = time.monotonic()+self.timeout, bytearray()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdin, selectors.EVENT_WRITE)
                while pending:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise ConnectionError('journal node request timed out')
                    written = os.write(process.stdin.fileno(), pending[:65536])
                    if not written:
                        raise ConnectionError('journal node stream stopped accepting its request')
                    pending = pending[written:]
                selector.unregister(process.stdin)
                selector.register(process.stdout, selectors.EVENT_READ)
                while not output.endswith(b'\n'):
                    remaining = deadline-time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise ConnectionError('journal node response timed out')
                    data = os.read(process.stdout.fileno(), 65536)
                    if not data:
                        raise ConnectionError('journal node stream ended before its response')
                    output.extend(data)
                    if len(output) > 8 << 20:
                        raise ConnectionError('journal node response exceeds budget')
            response = json.loads(output)
        except (OSError, ConnectionError, ValueError):
            self.drop(key)
            raise ConnectionError('journal node stream interrupted; retry the same transfer request') from None
        if not response.get('ok'):
            raise ValueError(response.get('error', 'journal node request failed'))
        return response['result']


def transfer(config, source_node, destination_node, destination_directory, transfer_id,
             *, invoke_node=None, chunk_bytes=CHUNK_BYTES):
    if type(chunk_bytes) is not int or not 1 <= chunk_bytes <= CHUNK_BYTES:
        raise ValueError('invalid journal transfer chunk size')
    if invoke_node is None:
        with NodeSessions() as sessions:
            return transfer(config, source_node, destination_node, destination_directory, transfer_id,
                            invoke_node=sessions.invoke, chunk_bytes=chunk_bytes)
    def call(node, request):
        for attempt in range(3):
            try:
                return invoke_node(node, request)
            except (ConnectionError, subprocess.TimeoutExpired, RuntimeError):
                if attempt == 2:
                    raise
    manifest = call(source_node, dict(op='seal', config=config, transfer=transfer_id))
    destination = dict(config, journal_directory=str(destination_directory))
    state = call(destination_node, dict(op='begin', config=destination, manifest=manifest))
    chunks, sent = 0, 0
    if not state['complete']:
        for file in manifest['files']:
            offset = state['offsets'][file['name']]
            while offset < file['bytes']:
                chunk = call(source_node, dict(op='read', config=config, transfer=transfer_id,
                             name=file['name'], offset=offset, size=chunk_bytes))
                result = call(destination_node, dict(op='write', config=destination, manifest=manifest, **chunk))
                if not offset < result['offset'] <= file['bytes']:
                    raise RuntimeError('journal receiver did not advance its file offset')
                sent += result['offset'] - offset
                offset = result['offset']; chunks += 1
    result = call(destination_node, dict(op='finish', config=destination, manifest=manifest))
    return dict(result, config=destination, transferred_bytes=sent, chunks=chunks,
                journal_bytes=sum(f['bytes'] for f in manifest['files']),
                records=len(manifest['files'])-1, worker=config['worker'],
                source_sealed=True, transfer=transfer_id)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    a = p.parse_args()
    if a.out.exists() or not a.out.parent.is_dir():
        p.error('output must be a new file in an existing directory')
    plan = json.loads(a.plan.read_text())
    entries = plan['workers']
    if not 1 <= len(entries) < 1024 or [e['config']['worker'] for e in entries] != list(range(len(entries))):
        raise ValueError('invalid worker handoff group')
    results = []
    for entry in entries:
        results.append(transfer(entry['config'], entry['source_node'], entry['destination_node'],
                                entry['destination_directory'], plan['transfer']))
    # This result contains worker configs and storage locations, but never SSH credentials.
    result = dict(kind='eps-worker-journal-group-v1', transfer=plan['transfer'], workers=results)
    from journal_handoff import publish
    publish(a.out, result)
    print(json.dumps(dict(workers=len(results), records=sum(r['records'] for r in results), complete=True)))


if __name__ == '__main__':
    main()
