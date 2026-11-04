"""Immutable worker gradients for replica-backed synchronous training recovery."""
import fcntl
import hashlib
import json
import math
import os
import pathlib
import struct
import uuid

import numpy as np

MAGIC = b'EPSWAL1\0'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def identity(config):
    batches = config.get('batches', [config.get('samples', [])])
    return dict(run=config['run'], epoch=config.get('epoch', 1), worker=config['worker'],
                workers=config['journal_workers'], shards=len(config['shards']), table=config['table'],
                base_round=config.get('journal_base_round', 0),
                model=dict(kind='embedding_tanh_mlp', dimension=config['dimension'], hidden=config.get('hidden', 8)),
                dataset_json=canonical(batches).decode(), dataset_sha256=hashlib.sha256(canonical(batches)).hexdigest())


class WorkerJournal:
    def __init__(self, directory, expected, record_bytes=32 << 20, disk_bytes=1 << 30):
        if not 4096 <= record_bytes <= 32 << 20 or disk_bytes < record_bytes:
            raise ValueError('invalid worker journal budgets')
        self.root = pathlib.Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = open(self.root / 'lock', 'a+b')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.record_bytes, self.disk_bytes = record_bytes, disk_bytes
            self.identity = expected
            if not expected['run'] or expected['epoch'] <= 0 or not 0 <= expected['worker'] < expected['workers'] < 1024:
                raise ValueError('invalid journal worker identity')
            if not 1 <= expected['shards'] <= 1024 or not 0 <= expected['base_round'] < (1 << 63):
                raise ValueError('invalid journal partition/base round')
            self.digest = hashlib.sha256(canonical(expected)).hexdigest()
            for p in self.root.glob('.pending-*'):
                p.unlink()  # Never submitted before immutable publication.
            path = self.root / 'identity.epsj'
            if path.exists():
                header, payload = self._read(path)
                if payload or header != dict(kind='identity', identity=expected):
                    raise ValueError('worker journal model/input/run identity mismatch')
            else:
                if list(self.root.glob('round-*.epsw')):
                    raise ValueError('worker gradients exist without journal identity')
                self._publish(path, self._encode(dict(kind='identity', identity=expected), b''))
            previous = -1
            for iteration in self.rounds():
                record = self.get(iteration)
                if record['completed_reads'] <= previous:
                    raise ValueError('worker read progress is not monotonic')
                previous = record['completed_reads']
            if self._bytes() > self.disk_bytes:
                raise ValueError('worker journal disk budget exceeded')
            sync_directory(self.root)
            sync_directory(self.root.parent)
        except Exception:
            self.lock.close()
            raise

    def close(self):
        self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _bytes(self):
        return sum(p.stat().st_size for p in self.root.iterdir() if p.is_file())

    def _encode(self, header, payload):
        meta = canonical(header)
        data = MAGIC + struct.pack('<Q', len(meta)) + meta + payload
        if len(data) + 32 > self.record_bytes:
            raise ValueError('worker record exceeds budget')
        return data + hashlib.sha256(data).digest()

    def _read(self, path):
        if path.stat().st_size > self.record_bytes:
            raise ValueError('worker record exceeds budget')
        data = path.read_bytes()
        if len(data) < 48 or data[:8] != MAGIC or hashlib.sha256(data[:-32]).digest() != data[-32:]:
            raise ValueError('worker record checksum/format mismatch')
        length, = struct.unpack('<Q', data[8:16])
        if length > len(data) - 48:
            raise ValueError('truncated worker metadata')
        header = json.loads(data[16:16 + length])
        return header, data[16 + length:-32]

    def _publish(self, path, data):
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('immutable worker record changed')
            return
        if self._bytes() + len(data) > self.disk_bytes:
            raise ValueError('worker journal disk budget exhausted')
        pending = self.root / ('.pending-' + uuid.uuid4().hex)
        try:
            with pending.open('xb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.link(pending, path)
            pending.unlink()
            sync_directory(self.root)
        finally:
            if pending.exists():
                pending.unlink()

    def rounds(self):
        rounds = []
        for p in self.root.glob('round-*.epsw'):
            n = int(p.name[6:-5])
            if p.name != f'round-{n}.epsw':
                raise ValueError('invalid worker journal filename')
            rounds.append(n)
        rounds.sort()
        if any(n != self.identity['base_round'] + i or n >= (1 << 63) for i, n in enumerate(rounds)):
            raise ValueError('worker journal has a round gap')
        return rounds

    def get(self, iteration):
        path = self.root / f'round-{iteration}.epsw'
        if not path.exists():
            return None
        header, payload = self._read(path)
        if len(payload) > 16 << 20:
            raise ValueError('worker gradient exceeds submission budget')
        if header['kind'] != 'gradient' or header['identity_sha256'] != self.digest or header['iteration'] != iteration:
            raise ValueError('worker gradient identity mismatch')
        if not math.isfinite(header['loss']) or not 1 <= header['sample_count'] <= 65536 or not 0 < header['completed_reads'] < (1 << 64):
            raise ValueError('invalid worker gradient metadata')
        tables, offset, seen = [], 0, set()
        if not 1 <= len(header['tables']) <= 256:
            raise ValueError('invalid gradient table group')
        for table in header['tables']:
            t, rows, dim = table['table'], table['rows'], table['dimension']
            if not 0 <= t < (1 << 32) or t in seen or not 0 <= rows <= (1 << 24) or not 1 <= dim <= 65536:
                raise ValueError('invalid gradient dimensions')
            seen.add(t)
            size = rows * 8 + rows * dim * 4
            if size > len(payload) - offset:
                raise ValueError('truncated gradient payload')
            keys = np.frombuffer(payload, '<u8', rows, offset).copy()
            offset += rows * 8
            values = np.frombuffer(payload, '<f4', rows * dim, offset).reshape(rows, dim).copy()
            offset += rows * dim * 4
            if not np.isfinite(values).all() or np.any(keys[1:] <= keys[:-1]):
                raise ValueError('invalid/nonunique gradient values or Keys')
            tables.append(dict(table=t, keys=keys, values=values))
        if offset != len(payload):
            raise ValueError('worker gradient has trailing bytes')
        return dict(header, tables=tables)

    def put(self, iteration, sample_count, completed_reads, loss, tables):
        rounds = self.rounds()
        next_round = rounds[-1] + 1 if rounds else self.identity['base_round']
        if iteration > next_round or iteration < self.identity['base_round']:
            raise ValueError('worker gradient round is not contiguous')
        layout, pieces = [], []
        for table in tables:
            keys = np.asarray(table['keys'], dtype='<u8')
            values = np.asarray(table['values'], dtype='<f4')
            if keys.ndim != 1 or values.ndim != 2 or len(keys) != len(values):
                raise ValueError('invalid gradient array shapes')
            layout.append(dict(table=int(table['table']), rows=len(keys), dimension=values.shape[1]))
            pieces.extend((keys.tobytes(), values.tobytes()))
        header = dict(kind='gradient', identity_sha256=self.digest, iteration=iteration,
                      sample_count=sample_count, completed_reads=completed_reads, loss=float(loss), tables=layout)
        data = self._encode(header, b''.join(pieces))
        if sum(map(len,pieces)) > 16 << 20:
            raise ValueError('worker gradient exceeds submission budget')
        # Validate before the file can become a submitted training record.
        if not math.isfinite(float(loss)) or not 1 <= sample_count <= 65536 or not 0 < completed_reads < (1 << 64):
            raise ValueError('invalid gradient metadata')
        if not tables or len(tables) > 256 or any(not np.isfinite(t['values']).all() or
                np.any(np.asarray(t['keys'])[1:] <= np.asarray(t['keys'])[:-1]) for t in tables):
            raise ValueError('invalid gradient payload')
        if len({t['table'] for t in layout}) != len(layout) or any(not 0 <= t['table'] < (1 << 32) or
                not 0 <= t['rows'] <= (1 << 24) or not 1 <= t['dimension'] <= 65536 for t in layout):
            raise ValueError('invalid gradient table layout')
        if iteration == next_round and rounds and completed_reads <= self.get(rounds[-1])['completed_reads']:
            raise ValueError('worker read progress is not monotonic')
        self._publish(self.root / f'round-{iteration}.epsw', data)
