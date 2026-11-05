"""Publish and verify learner input state under a joint training checkpoint."""
import hashlib
import json
import os
import pathlib
import posixpath
import re
import uuid

from checkpoint_store import LocalFileSystem, _publish

LIMIT = 1 << 20


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def _read_json(fs, path):
    with fs.GFile(path, 'rb') as stream:
        raw = stream.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError('input checkpoint manifest exceeds its size limit')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('input checkpoint manifest is not an object')
    return raw, value


def _json_publish(fs, target, value):
    raw = _json_bytes(value)
    if len(raw) > LIMIT:
        raise ValueError('input checkpoint manifest exceeds its size limit')
    pending = target + '.pending-' + uuid.uuid4().hex
    try:
        with fs.GFile(pending, 'wb') as stream:
            stream.write(raw)
        digest = hashlib.sha256(raw).hexdigest()
        _publish(fs, pending, target, len(raw), digest)
        return digest
    finally:
        if fs.exists(pending):
            fs.remove(pending)


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _name(value):
    return isinstance(value, str) and re.fullmatch('[a-zA-Z0-9_.-]+', value) is not None and value not in ('.', '..')


def _file(item):
    if (not isinstance(item, dict) or set(item) != {'file', 'bytes', 'sha256'} or
            not _name(item['file']) or type(item['bytes']) is not int or item['bytes'] < 1 or
            not _digest(item['sha256'])):
        raise ValueError('invalid input checkpoint file descriptor')


def publish_file(fs, source, directory, name):
    if not _name(name):
        raise ValueError('invalid input checkpoint filename')
    source = pathlib.Path(source)
    target = posixpath.join(directory, name)
    pending = target + '.pending-' + uuid.uuid4().hex
    digest = hashlib.sha256()
    length = 0
    try:
        with source.open('rb') as reader, fs.GFile(pending, 'wb') as writer:
            while data := reader.read(LIMIT):
                written = writer.write(data)
                if written is not None and written != len(data):
                    raise RuntimeError('short input checkpoint write')
                digest.update(data)
                length += len(data)
        item = dict(file=name, bytes=length, sha256=digest.hexdigest())
        _file(item)
        _publish(fs, pending, target, length, item['sha256'])
        return item
    finally:
        if fs.exists(pending):
            fs.remove(pending)


def stage_file(fs, directory, item, destination):
    _file(item)
    destination = pathlib.Path(destination)
    pending = destination.with_name(destination.name + '.pending-' + uuid.uuid4().hex)
    digest = hashlib.sha256()
    length = 0
    try:
        with pending.open('xb') as writer, fs.GFile(posixpath.join(directory, item['file']), 'rb') as reader:
            os.chmod(pending, 0o600)
            while data := reader.read(LIMIT):
                length += len(data)
                if length > item['bytes']:
                    raise ValueError('input checkpoint file exceeds its committed length')
                writer.write(data)
                digest.update(data)
            writer.flush()
            os.fsync(writer.fileno())
        if (length, digest.hexdigest()) != (item['bytes'], item['sha256']):
            raise ValueError('input checkpoint file differs from its committed digest')
        _publish(LocalFileSystem, str(pending), str(destination), length, item['sha256'])
        return destination
    finally:
        pending.unlink(missing_ok=True)


def _worker(value, worker):
    if (value.get('schema_version') != 1 or type(value.get('worker')) is not int or value['worker'] != worker or
            type(value.get('source_handle')) is not int or not 0 <= value['source_handle'] < 4096 or
            not isinstance(value.get('scope'), str) or not re.fullmatch('[a-zA-Z0-9_-]{1,256}', value['scope']) or
            value.get('mode') not in ('train', 'eval') or not _digest(value.get('input_spec_sha256')) or
            type(value.get('shot_count')) is not int or value['shot_count'] < 1 or
            type(value.get('next_index')) is not int or not 1 <= value['next_index'] <= value['shot_count'] or
            type(value.get('end')) is not bool or (value['end'] and value['next_index'] != value['shot_count']) or
            not isinstance(value.get('ident'), str)):
        raise ValueError('invalid learner input position')
    files = value.get('files')
    if not isinstance(files, list) or len(files) != 3:
        raise ValueError('input checkpoint requires iterator index/data and a native capture')
    for item in files:
        _file(item)
    if {item['file'] for item in files} != {'iterator.index', 'iterator.data-00000-of-00001', 'capture.bin'}:
        raise ValueError('input checkpoint has unexpected iterator files')


def publish_worker(root, worker, metadata, files, *, filesystem=None):
    fs = filesystem or LocalFileSystem
    if type(worker) is not int or not 0 <= worker < 1024:
        raise ValueError('invalid input checkpoint worker ID')
    directory = posixpath.join(os.fspath(root), 'input_state', 'worker-' + str(worker))
    fs.makedirs(directory)
    record = dict(metadata, schema_version=1, worker=worker,
                  files=[publish_file(fs, path, directory, name) for name, path in sorted(files.items())])
    _worker(record, worker)
    return _json_publish(fs, posixpath.join(directory, 'manifest.json'), record)


def publish_group(root, *, run, epoch, completed, source_files, filesystem=None):
    fs = filesystem or LocalFileSystem
    directory = posixpath.join(os.fspath(root), 'input_state')
    fs.makedirs(directory)
    workers = []
    for worker in range(len(completed)):
        raw, value = _read_json(fs, posixpath.join(directory, 'worker-' + str(worker), 'manifest.json'))
        _worker(value, worker)
        workers.append(dict(worker=worker, sha256=hashlib.sha256(raw).hexdigest()))
    sources = []
    for handle, files in sorted(source_files.items()):
        sources.append(dict(handle=handle, **{
            name: publish_file(fs, files[name], directory, f'source-{handle}-{name}.bin')
            for name in ('roster', 'plan')}))
    record = dict(schema_version=1, run=run, partition_epoch=epoch,
                  completed=list(completed), workers=workers, sources=sources)
    digest = _json_publish(fs, posixpath.join(directory, 'manifest.json'), record)
    load_group(root, digest, run=run, epoch=epoch, completed=completed, filesystem=fs)
    return digest


def load_group(root, expected_digest, *, run, epoch, completed, filesystem=None):
    fs = filesystem or LocalFileSystem
    directory = posixpath.join(os.fspath(root), 'input_state')
    raw, group = _read_json(fs, posixpath.join(directory, 'manifest.json'))
    if (hashlib.sha256(raw).hexdigest() != expected_digest or group.get('schema_version') != 1 or
            group.get('run') != run or group.get('partition_epoch') != epoch or
            group.get('completed') != list(completed) or not 1 <= len(completed) <= 1024):
        raise ValueError('input group differs from the joint training checkpoint')
    workers = group.get('workers')
    if not isinstance(workers, list) or len(workers) != len(completed):
        raise ValueError('input checkpoint learner membership differs')
    records = []
    sources = {}
    for worker, item in enumerate(workers):
        if (not isinstance(item, dict) or set(item) != {'worker', 'sha256'} or
                type(item['worker']) is not int or item['worker'] != worker or not _digest(item['sha256'])):
            raise ValueError('invalid input checkpoint learner manifest')
        raw, value = _read_json(fs, posixpath.join(directory, 'worker-' + str(worker), 'manifest.json'))
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('learner input manifest differs from the input group')
        _worker(value, worker)
        position = tuple(value[k] for k in ('input_spec_sha256', 'mode', 'next_index', 'shot_count', 'end', 'ident'))
        if value['source_handle'] in sources and sources[value['source_handle']] != position:
            raise ValueError('learners sharing an input source have different input shots')
        sources[value['source_handle']] = position
        records.append(value)
    source_records = group.get('sources')
    if not isinstance(source_records, list) or len(source_records) != len(sources):
        raise ValueError('input checkpoint source membership differs')
    for handle, item in zip(sorted(sources), source_records):
        if not isinstance(item, dict) or set(item) != {'handle', 'roster', 'plan'} or item['handle'] != handle:
            raise ValueError('input source order differs from learner membership')
        for name in ('roster', 'plan'):
            _file(item[name])
            if item[name]['file'] != f'source-{handle}-{name}.bin':
                raise ValueError('input source file identity differs')
    return dict(directory=directory, manifest=group, workers=records)
