"""Checksummed, bounded transfer of stopped native replica/primary logs."""
import base64
import hashlib
import json
import os
import pathlib
import re

CHUNK = 256 << 10


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def operation(root, run, role, action, term=1, name=None, offset=0, data=None, manifest=None):
    from role_node import directory, locked, snapshot, write
    if not re.fullmatch(r'(primary|replica)-\d+-g\d+', role) or type(term) is not int or term < 1:
        raise ValueError('invalid replica transfer role')
    path = directory(root, run, role)
    with locked(path):
        state = snapshot(path)
        if state['live_child'] or state['live_wrapper']:
            raise ValueError('replica transfer requires stopped owned roles')
        folder = path / ('storage' if role.startswith('replica') else 'primary-log/term-' + str(term))
        receipt = path / 'replica-handoff.json'
        allowed = lambda n: n in ('configuration', 'lease', 'baseline.eps') or re.fullmatch(r'round-\d+\.epsr', n)
        if action == 'seal':
            if receipt.exists():
                return json.loads(receipt.read_text())
            files = [dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p))
                     for p in sorted(folder.iterdir()) if p.is_file() and allowed(p.name)]
            if not {'configuration', 'lease', 'baseline.eps'}.issubset({p['name'] for p in files}):
                raise ValueError('surviving replica lacks a complete baseline')
            value = dict(run=run, files=files)
            write(receipt, value)
            return value
        if action == 'begin':
            if not isinstance(manifest, dict) or manifest.get('run') != run:
                raise ValueError('replica handoff run differs')
            files = manifest.get('files', [])
            if not 3 <= len(files) <= 10003 or len({p['name'] for p in files}) != len(files):
                raise ValueError('invalid handoff membership')
            if any(not allowed(p['name']) or type(p['bytes']) is not int or not 8 <= p['bytes'] <= 64 << 30
                   or not re.fullmatch('[0-9a-f]{64}', p['sha256']) for p in files):
                raise ValueError('invalid handoff file')
            if receipt.exists() and json.loads(receipt.read_text()) != manifest:
                raise ValueError('replica handoff changed')
            folder.mkdir(parents=True, exist_ok=True); write(receipt, manifest)
            return dict(begun=True)
        value = json.loads(receipt.read_text()); files = {f['name']: f for f in value['files']}
        if action == 'finish':
            for item in files.values():
                p = folder / item['name']
                if p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
                    raise ValueError('replica handoff checksum differs')
            return dict(verified=True, files=len(files))
        if name not in files or type(offset) is not int or not 0 <= offset < files[name]['bytes']:
            raise ValueError('replica chunk outside manifest')
        p = folder / name
        if action == 'read':
            with p.open('rb') as f:
                f.seek(offset); chunk = f.read(min(CHUNK, files[name]['bytes'] - offset))
            return dict(data=base64.b64encode(chunk).decode(), sha256=hashlib.sha256(chunk).hexdigest())
        if action == 'write':
            chunk = base64.b64decode(data['data'], validate=True)
            if not 0 < len(chunk) <= CHUNK or offset + len(chunk) > files[name]['bytes'] or hashlib.sha256(chunk).hexdigest() != data['sha256']:
                raise ValueError('invalid replica chunk checksum/size')
            with p.open('r+b' if p.exists() else 'w+b') as f:
                f.seek(0, 2); size = f.tell()
                if offset > size:
                    raise ValueError('replica transfer has a gap')
                if offset < size:
                    f.seek(offset)
                    if f.read(len(chunk)) != chunk:
                        raise ValueError('repeated replica transfer changed bytes')
                else:
                    f.write(chunk); f.flush(); os.fsync(f.fileno())
            return dict(offset=offset + len(chunk))
        raise ValueError('unknown replica transfer operation')


def transfer(sessions, source, target, run, source_role, target_role, term=1):
    def call(node, role, action, **extra):
        return sessions.invoke(node, dict(op='replica_handoff', root=node['root'], run=run,
            role=role, action=action, term=term, **extra))
    manifest = call(source, source_role, 'seal')
    call(target, target_role, 'begin', manifest=manifest)
    for item in manifest['files']:
        for offset in range(0, item['bytes'], CHUNK):
            chunk = call(source, source_role, 'read', name=item['name'], offset=offset)
            call(target, target_role, 'write', name=item['name'], offset=offset, data=chunk)
    return call(target, target_role, 'finish')
