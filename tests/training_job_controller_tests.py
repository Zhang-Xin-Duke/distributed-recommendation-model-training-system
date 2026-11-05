"""Real native services and TensorFlow training through the multi-job controller."""
import argparse
import copy
import json
import pathlib
import socket
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from training_job_controller import Controller, TERMINAL
from job_placement import job_spec, place
from role_node import stop


def inventory(build, root, count):
    return dict(manager=dict(bin_dir=str(build), module_dir=str(build / 'python')),
        nodes=[dict(node_id=f'node-{i}', local=True, root=str(root / f'node-{i}'),
            source=str(ROOT), python=sys.executable, bin_dir=str(build), module_dir=str(build / 'python'), gpu=False,
            capacity=dict(cpu_cores=1, memory_bytes=1 << 30, disk_bytes=8 << 30)) for i in range(count)])


def spec(name, count, priority=0):
    return dict(id=name, groups=count, priority=priority, device='cpu',
        workload=dict(rows=64, rounds=4, hidden=4, inspect_keys=list(range(64))),
        resources=dict(group_cpu_cores=.7, replica_cpu_cores=.1, group_workspace_bytes=256 << 20,
            replica_workspace_bytes=32 << 20, journal_disk_bytes=8 << 20, max_groups_per_node=1))


def await_jobs(control, ids, timeout=180):
    deadline = time.monotonic() + timeout
    while True:
        control.tick()
        jobs = [control.store.get(i) for i in ids]
        if all(j['state'] in TERMINAL for j in jobs):
            if not all(j['state'] == 'SUCCEEDED' for j in jobs):
                raise AssertionError([(j['id'], j['state'], j['detail']) for j in jobs])
            return jobs
        if time.monotonic() > deadline:
            raise AssertionError('controller timeout: ' + str([(j['id'], j['state']) for j in jobs]))
        time.sleep(.1)


def run(build, out):
    reports = []; checks = []
    with tempfile.TemporaryDirectory(prefix='eps-job-controller-') as tmp:
        root = pathlib.Path(tmp)
        for n in (2, 4, 8):
            inv = inventory(build, root / str(n), n)
            control = Controller(inv, root / str(n) / 'controller')
            try:
                value = spec(f'topology-{n}', n)
                proposed = control.plan(value)
                assert proposed['feasible'] and len(set(proposed['groups'])) == n
                assert all(a != b for a, b in zip(proposed['groups'], proposed['backups']))
                control.store.submit(value); assert control.store.submit(value)['seq'] == 1
                with control.leadership():
                    result = await_jobs(control, [value['id']])[0]
                raw = json.loads((control._runtime(value['id']) / 'training/summary.json').read_text())
                assert raw['model']['state']['completed'] == [4] * n
                assert len(raw['model']['state']['rows']) == 64 and result['detail']['cleanup_verified']
                assert all(r['state'] == 'released' for node in control.probe() for r in node['reservations'])
                reports.append(dict(groups=n, physical_hosts=1, physical_gpus=0, result=raw))
                checks.append(f'{n}_groups_real_training_and_gang_cleanup')
            finally:
                for j in control.store.jobs():
                    if j['state'] not in TERMINAL:
                        control.store.cancel(j['id'])
                for _ in range(2):control.tick()
                control.close()
        inv = inventory(build, root / 'queue', 4)
        state = root / 'queue/controller'; control = Controller(inv, state)
        try:
            low = spec('low-priority', 2, 0); high = spec('high-priority', 2, 10)
            same = spec('same-priority', 2, 10)
            for value in (low, high, same):control.store.submit(value)
            control.tick()
            assert control.store.get(high['id'])['state'] == 'RUNNING'
            assert control.store.get(same['id'])['state'] == 'RUNNING'
            assert control.store.get(low['id'])['state'] == 'WAITING_RESOURCES'
            event_ids = [r['job'] for r in control.store.db.execute("SELECT job FROM events WHERE kind='RESERVING' ORDER BY seq")]
            assert event_ids[:2] == [high['id'], same['id']]
            handles = {j['id']: j['detail']['executor_handle']['child'] for j in control.store.jobs() if j['state'] == 'RUNNING'}
            control.close(); control = Controller(inv, state)
            control.tick()
            assert all(control.store.get(k)['detail']['executor_handle']['child'] == v for k, v in handles.items())
            await_jobs(control, [low['id'], high['id'], same['id']])
            checks += ['priority_and_equal_priority_fifo', 'two_jobs_concurrently_running',
                       'capacity_queue_handoff', 'controller_restart_adopts_executor_handles']
            cancel = spec('cancel-running', 2)
            control.store.submit(cancel); control.tick()
            assert control.store.get(cancel['id'])['state'] == 'RUNNING'
            control.store.cancel(cancel['id']); control.tick()
            assert control.store.get(cancel['id'])['state'] == 'CANCELLED'
            assert control.store.get(cancel['id'])['detail']['cleanup_verified']
            checks.append('running_job_cancel_stops_owned_groups_before_release')
        finally:
            for j in control.store.jobs():
                if j['state'] not in TERMINAL:control.store.cancel(j['id'])
            for _ in range(2):control.tick()
            control.close()
    value = dict(test='training_job_controller', checks=checks, physical_hosts=1, physical_gpus=0,
                 logical_group_counts=[2, 4, 8], cleanup_verified=True, reports=reports)
    out.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'reports'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--build', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True); a = p.parse_args(); run(a.build.resolve(), a.out)
