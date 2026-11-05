"""Reservation races, uncertain replies, live ownership and placement policies."""
import concurrent.futures
import copy
import json
import pathlib
import socket
import sys
import tempfile
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from node_resources import operation
from job_placement import job_spec, place
from training_job_controller import JobStore
from transfer_worker_journals import NodeSessions


def main():
    checks = []
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        request = dict(cpu_cores=.75, memory_bytes=100, disk_bytes=100, gpu_bytes=0, gpu_uuid='', ports=[])
        limits = dict(cpu_cores=1)
        def reserve(token):
            try:return operation(str(root), token, 'allocation', 'reserve', token=token, request=request, limits=limits)
            except ValueError:return None
        with concurrent.futures.ThreadPoolExecutor(2) as pool:results = list(pool.map(reserve, ['first', 'second']))
        assert sum(r is not None for r in results) == 1
        winner = next(r for r in results if r); token = winner['token']
        assert reserve(token) == winner
        checks += ['atomic_competing_reservations', 'lost_successful_reserve_reply_is_idempotent']
        node = dict(local=True, source=str(ROOT), python=sys.executable)
        with NodeSessions(entrypoint='scripts/role_node.py') as sessions:
            base = dict(root=str(root), run=token, role='live')
            sessions.invoke(node, dict(op='ensure', **base,
                spec=dict(command=[sys.executable, '-c', 'import time;time.sleep(120)'], reservation_token=token)))
            try:operation(str(root), token, 'allocation', 'release', token=token)
            except ValueError:pass
            else:raise AssertionError('live allocation released')
            assert sessions.invoke(node, dict(op='stop', **base))['owned_processes_gone']
        assert operation(str(root), token, 'allocation', 'release', token=token)['state'] == 'released'
        try:reserve(token)
        except ValueError:pass
        assert reserve(token) is None
        checks += ['live_process_prevents_release', 'stopped_process_allows_release', 'released_token_cannot_reopen']
        with socket.socket() as busy:
            busy.bind(('127.0.0.1', 0)); busy.listen()
            request = dict(request, cpu_cores=.1, ports=[busy.getsockname()[1]])
            try:operation(str(root), 'busy-port', 'allocation', 'reserve', token='busy-port', request=request)
            except OSError:pass
            else:raise AssertionError('busy port was reserved')
        checks.append('external_port_conflict_rejected')
        db = JobStore(root / 'db')
        value = dict(id='identity', groups=2)
        assert db.submit(value)['seq'] == db.submit(value)['seq']
        try:db.submit(dict(value, priority=1))
        except ValueError:pass
        else:raise AssertionError('changed submission reused id')
        db.close(); checks.append('durable_submit_identity')
        snapshot = dict(cpu_cores=10, memory_bytes=10 << 30, disk_bytes=10 << 30,
            gpus=[dict(uuid='synthetic-gpu', free_bytes=10 << 30)], occupied_gpus=[])
        nodes = [dict(node_id='a', available=copy.deepcopy(snapshot)), dict(node_id='b', available=copy.deepcopy(snapshot))]
        gpu = job_spec(dict(id='placement', groups=2, device='gpu'))
        plan = place(gpu, nodes); assert plan['feasible'] and plan['access_estimate'] == 'uniform'
        for node in nodes:node['available']['occupied_gpus'] = ['synthetic-gpu']
        assert not place(gpu, nodes)['feasible']
        checks += ['synthetic_gpu_placement_only', 'gpu_exclusivity_admission']
    value = dict(test='job_resources', checks=checks, success=True, physical_gpus=0)
    if len(sys.argv) > 1:pathlib.Path(sys.argv[1]).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))


if __name__ == '__main__':main()
