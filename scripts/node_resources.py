"""Node-local durable resource reservations for cooperative training jobs."""
import contextlib
import json
import math
import os
import pathlib
import re
import shutil
import socket
import subprocess
import time


def resources(root, gpu=False):
    from training_node import hardware
    result = hardware(gpu)
    if result['host_available_bytes'] is None:
        result['host_available_bytes'] = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']))
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    quota = pathlib.Path('/sys/fs/cgroup/cpu.max')
    if quota.exists():
        values = quota.read_text().split()
        if values[0] != 'max':
            cpus = min(cpus, int(values[0]) / int(values[1]))
    path = pathlib.Path(root)
    while not path.exists():
        path = path.parent
    return dict(cpu_cores=cpus, memory_bytes=result['host_available_bytes'],
                disk_bytes=shutil.disk_usage(path).free,
                gpus=[] if not result['gpu'] else [result['gpu']], observed_epoch=time.time())


def store(root):
    path = pathlib.Path(root) / '.training-resources'
    path.mkdir(parents=True, exist_ok=True)
    return path


def valid_token(token):
    if not isinstance(token, str) or not re.fullmatch('[a-z0-9_-]{1,96}', token):
        raise ValueError('invalid reservation token')


def records(path):
    return [json.loads(p.read_text()) for p in path.glob('reservation-*.json')]


def requirements(value):
    if set(value) != {'cpu_cores', 'memory_bytes', 'disk_bytes', 'gpu_bytes', 'gpu_uuid', 'ports'}:
        raise ValueError('reservation requires all resource dimensions')
    for key in ('cpu_cores', 'memory_bytes', 'disk_bytes', 'gpu_bytes'):
        v = value[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError('invalid resource requirement: ' + key)
    if not isinstance(value['gpu_uuid'], str) or bool(value['gpu_bytes']) != bool(value['gpu_uuid']):
        raise ValueError('GPU identity and bytes must be supplied together')
    ports = value['ports']
    if not isinstance(ports, list) or len(set(ports)) != len(ports) or any(type(p) is not int or not 1024 <= p <= 65535 for p in ports):
        raise ValueError('invalid reservation ports')
    return value


def capacity(snapshot, limits):
    result = dict(snapshot)
    for key in ('cpu_cores', 'memory_bytes', 'disk_bytes'):
        if key in limits:
            v = limits[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError('invalid node capacity limit')
            result[key] = min(result[key], v)
    return result


def available(snapshot, active, limits):
    result = capacity(snapshot, limits)
    for key in ('cpu_cores', 'memory_bytes', 'disk_bytes'):
        # Observed free memory/disk already includes running allocations. Compare
        # both remaining declared capacity and fresh free capacity, not their sum.
        total = limits.get(key, snapshot[key])
        result[key] = max(0, min(snapshot[key], total - sum(r['resources'][key] for r in active)))
    result['occupied_gpus'] = sorted({r['resources']['gpu_uuid'] for r in active if r['resources']['gpu_uuid']})
    result['reserved_ports'] = sorted({p for r in active for p in r['resources']['ports']})
    return result


def operation(root, run, role, action, token=None, request=None, gpu=False, limits=None):
    from role_node import locked, write, snapshot as role_snapshot
    path = store(root)
    with locked(path):
        all_records = records(path)
        active = [r for r in all_records if r['state'] == 'reserved']
        if action == 'probe':
            raw = resources(root, gpu)
            return dict(snapshot=raw, available=available(raw, active, limits or {}), reservations=all_records)
        valid_token(token)
        target = path / ('reservation-' + token + '.json')
        existing = json.loads(target.read_text()) if target.exists() else None
        if existing and existing['run'] != run:
            raise ValueError('reservation belongs to a different job')
        if action == 'reserve':
            wanted = requirements(request)
            intent = dict(run=run, resources=wanted)
            if existing:
                if any(existing[k] != v for k, v in intent.items()):
                    raise ValueError('reservation intent changed')
                if existing['state'] != 'reserved':
                    raise ValueError('released reservation cannot be reopened')
                return existing
            raw = resources(root, bool(wanted['gpu_uuid']))
            free = available(raw, active, limits or {})
            if any(wanted[k] > free[k] for k in ('cpu_cores', 'memory_bytes', 'disk_bytes')):
                raise ValueError('node resource capacity is insufficient')
            if wanted['gpu_uuid']:
                devices = [g for g in raw['gpus'] if g['uuid'] == wanted['gpu_uuid']]
                if not devices or wanted['gpu_uuid'] in free['occupied_gpus'] or wanted['gpu_bytes'] > devices[0]['free_bytes']:
                    raise ValueError('GPU is unavailable or already reserved')
            if set(wanted['ports']) & set(free['reserved_ports']):
                raise ValueError('port is already reserved')
            with contextlib.ExitStack() as sockets:
                for port in wanted['ports']:
                    s = sockets.enter_context(socket.socket())
                    s.bind(('0.0.0.0', port))
            value = dict(intent, token=token, state='reserved', roles=[], created_epoch=time.time())
            write(target, value)
            return value
        if existing is None:
            if action == 'release':
                return dict(token=token, state='absent')
            raise ValueError('reservation does not exist')
        if action == 'verify':
            return existing
        if action == 'release':
            for name in existing['roles']:
                p = pathlib.Path(root) / 'results' / run / 'roles' / name
                with locked(p):
                    state = role_snapshot(p)
                    if state['live_child'] or state['live_wrapper']:
                        raise ValueError('reservation still owns live processes')
            existing.update(state='released', released_epoch=time.time())
            write(target, existing)
            return existing
        raise ValueError('unknown resource operation')


@contextlib.contextmanager
def launch_guard(root, run, role, token):
    from role_node import locked, write
    valid_token(token)
    path = store(root)
    with locked(path):
        target = path / ('reservation-' + token + '.json')
        value = json.loads(target.read_text())
        if value['run'] != run or value['state'] != 'reserved':
            raise ValueError('launch requires this job\'s active reservation')
        if role not in value['roles']:
            value['roles'].append(role)
            write(target, value)
        yield
