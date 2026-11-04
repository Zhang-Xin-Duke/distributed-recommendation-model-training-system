"""Sealed original-gradient journal transfer between stopped and replacement nodes."""
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import re
import sys
import uuid

from training_journal import WorkerJournal, identity, canonical, sync_directory

CHUNK_BYTES = 256 << 10
MANIFEST_BYTES = 4 << 20
CONTROL_FILES = {'handoff.epsh', 'handoff-receipt.json', '.transfer.json'}


def budgets(config):
    record = config.get('journal_record_bytes', 32 << 20)
    disk = config.get('journal_disk_bytes', 1 << 30)
    if type(record) is not int or type(disk) is not int or not 4096 <= record <= 32 << 20 or disk < record:
        raise ValueError('invalid receiver journal budgets')
    return record, disk


def publish(path, value):
    data = canonical(value)
    if len(data) > MANIFEST_BYTES:
        raise ValueError('journal transfer manifest exceeds budget')
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('immutable handoff metadata changed')
        return
    pending = path.parent / ('.handoff-pending-' + uuid.uuid4().hex)
    try:
        with pending.open('xb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.link(pending, path)
        sync_directory(path.parent)
    finally:
        pending.unlink(missing_ok=True)


def load(path):
    if path.stat().st_size > MANIFEST_BYTES:
        raise ValueError('journal transfer manifest exceeds budget')
    return json.loads(path.read_bytes())


def validate(config, manifest):
    if len(canonical(manifest)) > MANIFEST_BYTES or manifest.get('kind') != 'eps-worker-handoff-v1':
        raise ValueError('invalid journal handoff manifest')
    transfer = manifest.get('transfer', '')
    if not isinstance(transfer, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,96}', transfer):
        raise ValueError('invalid journal handoff identity')
    digest = hashlib.sha256(canonical({k:v for k,v in manifest.items() if k != 'manifest_sha256'})).hexdigest()
    if digest != manifest.get('manifest_sha256') or manifest.get('identity') != identity(config):
        raise ValueError('journal handoff identity/checksum differs')
    record, disk = budgets(config)
    files = manifest.get('files', [])
    if not 1 <= len(files) <= 16385 or files[0].get('name') != 'identity.epsj':
        raise ValueError('invalid journal handoff files')
    names = ['identity.epsj'] + [f"round-{manifest['identity']['base_round']+i}.epsw" for i in range(len(files)-1)]
    if [f['name'] for f in files] != names:
        raise ValueError('journal transfer rounds are not contiguous')
    for f in files:
        if type(f.get('bytes')) is not int or not 48 <= f['bytes'] <= record or not re.fullmatch(r'[0-9a-f]{64}', f.get('sha256', '')):
            raise ValueError('journal transfer file exceeds record budget or has invalid hash')
    if sum(f['bytes'] for f in files) > disk:
        raise ValueError('journal transfer exceeds receiver disk budget')
    return {f['name']:f for f in files}


def seal(config, transfer):
    record, disk = budgets(config)
    with WorkerJournal(config['journal_directory'], identity(config), record, disk, allow_sealed=True) as journal:
        files = []
        for name in ['identity.epsj', *[f'round-{n}.epsw' for n in journal.rounds()]]:
            data = (journal.root / name).read_bytes()
            files.append(dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        manifest = dict(kind='eps-worker-handoff-v1', transfer=transfer, identity=journal.identity, files=files)
        manifest['manifest_sha256'] = hashlib.sha256(canonical(manifest)).hexdigest()
        validate(config, manifest)
        publish(journal.root / 'handoff.epsh', manifest)
        return manifest


def read(config, transfer, name, offset, size):
    root = pathlib.Path(config['journal_directory'])
    manifest = load(root / 'handoff.epsh')
    files = validate(config, manifest)
    if manifest['transfer'] != transfer or name not in files:
        raise ValueError('read is outside the sealed worker journal')
    if type(offset) is not int or type(size) is not int or not 0 <= offset < files[name]['bytes'] or not 1 <= size <= CHUNK_BYTES:
        raise ValueError('invalid journal transfer chunk')
    with (root / name).open('rb') as f:
        f.seek(offset); data = f.read(min(size, files[name]['bytes']-offset))
    if not data:
        raise ValueError('sealed journal file was truncated')
    return dict(name=name, offset=offset, data=base64.b64encode(data).decode(), data_sha256=hashlib.sha256(data).hexdigest())


@contextlib.contextmanager
def receiving(config, manifest):
    files = validate(config, manifest)
    target = pathlib.Path(config['journal_directory'])
    target.parent.mkdir(parents=True, exist_ok=True)
    # All cooperating receiver operations serialize publication to this destination.
    with (target.parent / ('.' + target.name + '.handoff.lock')).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stage = target.parent / ('.' + target.name + '.handoff-' + manifest['transfer'])
        if target.exists():
            receipt = target / 'handoff-receipt.json'
            if not receipt.is_file() or load(receipt) != manifest:
                raise ValueError('destination already contains another journal')
            yield target, files, True
            return
        stage.mkdir(exist_ok=True)
        publish(stage / '.transfer.json', manifest)
        yield stage, files, False


def offsets(stage, files):
    result = {}
    for name, spec in files.items():
        complete, partial = stage / name, stage / (name + '.part')
        p = complete if complete.exists() else partial
        result[name] = p.stat().st_size if p.exists() else 0
        if result[name] > spec['bytes'] or (complete.exists() and result[name] != spec['bytes']):
            raise ValueError('receiver journal file size differs')
        if complete.exists() and hashlib.sha256(complete.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('receiver journal checksum differs')
    return result


def begin(config, manifest):
    with receiving(config, manifest) as (stage, files, complete):
        return dict(complete=complete, offsets={} if complete else offsets(stage, files))


def write(config, manifest, name, offset, data, data_sha256):
    if not isinstance(data, str) or len(data) > 4*((CHUNK_BYTES+2)//3):
        raise ValueError('journal transfer chunk exceeds budget')
    raw = base64.b64decode(data, validate=True)
    if not 1 <= len(raw) <= CHUNK_BYTES or hashlib.sha256(raw).hexdigest() != data_sha256:
        raise ValueError('journal transfer chunk checksum differs')
    with receiving(config, manifest) as (stage, files, complete):
        if name not in files or type(offset) is not int or offset < 0 or offset + len(raw) > files[name]['bytes']:
            raise ValueError('journal transfer chunk is outside its file')
        if complete:
            return dict(complete=True, offset=files[name]['bytes'])
        final = stage / name
        part = stage / (name + '.part')
        p = final if final.exists() else part
        with p.open('r+b' if p.exists() else 'w+b') as f:
            length = f.seek(0, os.SEEK_END)
            if offset < length:
                f.seek(offset)
                if f.read(len(raw)) != raw:
                    raise ValueError('repeated journal chunk differs')
            elif offset == length and not final.exists():
                f.write(raw); f.flush(); os.fsync(f.fileno()); length += len(raw)
            else:
                raise ValueError('journal transfer chunk has an offset gap')
        if length == files[name]['bytes'] and not final.exists():
            if hashlib.sha256(part.read_bytes()).hexdigest() != files[name]['sha256']:
                raise ValueError('completed journal file checksum differs')
            part.rename(final); sync_directory(stage)
        return dict(complete=False, offset=length)


def finish(config, manifest):
    with receiving(config, manifest) as (stage, files, complete):
        if complete:
            return dict(complete=True, manifest_sha256=manifest['manifest_sha256'])
        if offsets(stage, files) != {n:s['bytes'] for n,s in files.items()} or not all((stage/n).exists() for n in files):
            raise ValueError('journal handoff has incomplete files')
        record, disk = budgets(config)
        with WorkerJournal(stage, identity(config), record, disk) as journal:
            if ['identity.epsj', *[f'round-{n}.epsw' for n in journal.rounds()]] != list(files):
                raise ValueError('received journal contents differ from manifest')
        publish(stage / 'handoff-receipt.json', manifest)
        (stage / '.transfer.json').unlink()
        sync_directory(stage)
        stage.rename(pathlib.Path(config['journal_directory']))
        sync_directory(stage.parent)
        return dict(complete=True, manifest_sha256=manifest['manifest_sha256'])


def dispatch(request):
    op = request['op']
    arguments = {k:v for k,v in request.items() if k != 'op'}
    handlers = dict(seal=seal, read=read, begin=begin, write=write, finish=finish)
    if op not in handlers:
        raise ValueError('unknown journal handoff operation')
    return handlers[op](**arguments)


def response(request):
    try:
        return dict(ok=True, result=dispatch(request))
    except Exception as error:
        return dict(ok=False, error=type(error).__name__ + ': ' + str(error))


if __name__ == '__main__':
    if sys.argv[1:] == ['--serve']:
        for line in sys.stdin:
            print(json.dumps(response(json.loads(line))), flush=True)
    else:
        result = response(json.load(sys.stdin))
        print(json.dumps(result))
        sys.exit(0 if result['ok'] else 1)
