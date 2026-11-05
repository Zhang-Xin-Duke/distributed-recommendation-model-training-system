"""Bounded deterministic gang placement for parameter-server training roles."""
import copy
import math
import re


def job_spec(value):
    spec = copy.deepcopy(value)
    if not isinstance(spec.get('id'), str) or not re.fullmatch('[a-z0-9_-]{1,48}', spec['id']):
        raise ValueError('job requires a stable lowercase id')
    count = spec.get('groups', 2)
    if type(count) is not int or not 2 <= count <= 8:
        raise ValueError('job requires 2–8 fixed logical groups')
    spec['groups'] = count
    spec.setdefault('priority', 0)
    if type(spec['priority']) is not int or not -1000 <= spec['priority'] <= 1000:
        raise ValueError('invalid priority')
    spec.setdefault('device', 'cpu')
    if spec['device'] not in ('cpu', 'gpu'):
        raise ValueError('device must be cpu or gpu')
    workload = spec.setdefault('workload', {})
    if 'batches' in workload and len(workload['batches']) != count:
        raise ValueError('input partition count differs from logical membership')
    rows = workload.get('rows', 64)
    dim = workload.get('table', {}).get('dimension', 8)
    hidden = workload.get('hidden', 4)
    rounds = workload.get('rounds', 8)
    if any(type(x) is not int or x <= 0 for x in (rows, dim, hidden, rounds)) or dim > 4096 or hidden > 256 or rounds > 10000:
        raise ValueError('invalid model size')
    if rows <= (dim * hidden + 2 * hidden + 1 + dim - 1) // dim + 6:
        raise ValueError('table does not hold model and sparse inputs')
    defaults = dict(group_cpu_cores=1., group_workspace_bytes=512 << 20,
                    gpu_workspace_bytes=1 << 30, replica_cpu_cores=.1,
                    replica_workspace_bytes=64 << 20, journal_disk_bytes=1 << 30,
                    safety_fraction=.15, network_bytes_per_second=100 << 20, max_groups_per_node=8)
    resources = spec.setdefault('resources', {})
    if set(resources) - set(defaults):
        raise ValueError('unknown resource option')
    for key, default in defaults.items():
        resources.setdefault(key, default)
        v = resources[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError('invalid resource estimate: ' + key)
    if resources['network_bytes_per_second'] <= 0 or resources['safety_fraction'] > 1:
        raise ValueError('invalid bandwidth or safety fraction')
    if type(resources['max_groups_per_node']) is not int or not 1<=resources['max_groups_per_node']<=8:
        raise ValueError('invalid group packing limit')
    access = spec.get('access_profile')
    if access is not None and (len(access) != count or any(len(row) != count for row in access)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for row in access for v in row)):
        raise ValueError('access_profile must be a nonnegative Worker-by-shard byte matrix')
    return spec


def role_demands(spec):
    n = spec['groups']; w = spec['workload']; r = spec['resources']
    rows = w.get('rows', 64); dim = w.get('table', {}).get('dimension', 8)
    result = []
    for h in range(n):
        state = ((rows + n - 1 - h) // n) * (dim * 12 + 8) + 4096
        margin = 1 + r['safety_fraction']
        result.append(dict(kind='group', home=h, cpu_cores=r['group_cpu_cores'],
            memory_bytes=math.ceil((state + r['group_workspace_bytes']) * margin),
            disk_bytes=math.ceil(2 * state + r['journal_disk_bytes']),
            gpu_bytes=math.ceil((state + r['gpu_workspace_bytes']) * margin) if spec['device'] == 'gpu' else 0))
        result.append(dict(kind='backup', home=h, cpu_cores=r['replica_cpu_cores'],
            memory_bytes=math.ceil((state + r['replica_workspace_bytes']) * margin),
            disk_bytes=math.ceil(2 * state + r['journal_disk_bytes']), gpu_bytes=0))
    return sorted(result, key=lambda x: (-x['gpu_bytes'], -x['memory_bytes'], x['kind'], x['home']))


def score(spec, assigned, remaining):
    groups = {h: node for (kind, h), node in assigned.items() if kind == 'group'}
    backups = {h: node for (kind, h), node in assigned.items() if kind == 'backup'}
    dim = spec['workload'].get('table', {}).get('dimension', 8)
    access = spec.get('access_profile', [[dim * 4] * spec['groups'] for _ in range(spec['groups'])])
    traffic = sum(access[w][h] for w in groups for h in groups if groups[w] != groups[h])
    traffic += sum(dim * 12 for h in backups if h in groups and backups[h] != groups[h])
    fragmentation = sum((r['memory_bytes'] / max(1, r['initial_memory'])) ** 2 for r in remaining)
    tie = tuple((kind, h, node) for (kind, h), node in sorted(assigned.items()))
    return traffic / spec['resources']['network_bytes_per_second'], -fragmentation, tie


def place(spec, snapshots):
    spec = job_spec(spec)
    nodes = sorted((s for s in snapshots if not s.get('draining')), key=lambda x: x['node_id'])
    if len(nodes) < 2:
        return dict(feasible=False, reason='insufficient_healthy_nodes')
    demands = role_demands(spec)
    remaining = []
    for node in nodes:
        a = node['available']
        gpu = next((g for g in a.get('gpus', []) if g['uuid'] not in a.get('occupied_gpus', [])), None)
        remaining.append(dict(cpu_cores=a['cpu_cores'], memory_bytes=a['memory_bytes'],
            initial_memory=a['memory_bytes'], disk_bytes=a['disk_bytes'], gpu_bytes=gpu['free_bytes'] if gpu else 0,
            gpu_uuid=gpu['uuid'] if gpu else ''))
    candidates = [({}, remaining)]
    truncated = False
    for demand in demands:
        expanded = []
        for assigned, free in candidates:
            for i, node in enumerate(nodes):
                if demand['kind']=='group' and sum(k=='group' and node==i for (k,h),node in assigned.items())>=spec['resources']['max_groups_per_node']:
                    continue
                other = 'backup' if demand['kind'] == 'group' else 'group'
                if assigned.get((other, demand['home'])) == i:
                    continue
                if any(demand[k] > free[i][k] for k in ('cpu_cores', 'memory_bytes', 'disk_bytes', 'gpu_bytes')):
                    continue
                next_free = copy.deepcopy(free)
                for k in ('cpu_cores', 'memory_bytes', 'disk_bytes', 'gpu_bytes'):
                    next_free[i][k] -= demand[k]
                mapping = dict(assigned); mapping[demand['kind'], demand['home']] = i
                expanded.append((mapping, next_free))
        if not expanded:
            return dict(feasible=False, reason='bounded_search_exhausted' if truncated else 'insufficient_resources')
        expanded.sort(key=lambda c: score(spec, *c))
        truncated |= len(expanded) > 128
        candidates = expanded[:128]
    assigned, free = candidates[0]
    allocation = []
    for i, node in enumerate(nodes):
        wanted = dict(cpu_cores=0., memory_bytes=0, disk_bytes=0, gpu_bytes=0, gpu_uuid='', ports=[])
        for d in demands:
            if assigned[d['kind'], d['home']] == i:
                for k in ('cpu_cores', 'memory_bytes', 'disk_bytes', 'gpu_bytes'):
                    wanted[k] += d[k]
        if wanted['gpu_bytes']:
            wanted['gpu_uuid'] = remaining[i]['gpu_uuid']
        if wanted['cpu_cores'] or wanted['memory_bytes']:
            allocation.append(dict(node_id=node['node_id'], resources=wanted))
    group = [nodes[assigned['group', h]]['node_id'] for h in range(spec['groups'])]
    backup = [nodes[assigned['backup', h]]['node_id'] for h in range(spec['groups'])]
    return dict(feasible=True, groups=group, backups=backup, coordinator=group[0], allocations=allocation,
        estimated_transfer_seconds=score(spec, assigned, free)[0], search_width=128,
        search_truncated=truncated, access_estimate='provided' if spec.get('access_profile') is not None else 'uniform',
        reason='resource_feasible_then_communication_and_fragmentation')
