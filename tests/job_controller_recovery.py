"""Lose a whole node scope, rebuild from mixed authorities and reenter control."""
import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from training_job_controller_tests import inventory, spec, await_jobs
sys.path.insert(0, str(ROOT / 'scripts'))
from training_job_controller import Controller


def main(build, out):
    with tempfile.TemporaryDirectory(prefix='eps-controller-recovery-') as tmp:
        root = pathlib.Path(tmp); inv = inventory(build, root, 4); inv['allow_test_loss'] = True
        control = Controller(inv, root / 'controller')
        try:
            reference = spec('recovery-reference', 2); reference['workload']['rounds'] = 8
            control.store.submit(reference); await_jobs(control, [reference['id']])
            expected = json.loads((control._runtime(reference['id']) / 'training/summary.json').read_text())
            value = spec('node-recovery', 2); value['workload']['rounds'] = 8
            value['_test_fault'] = 'after_update'
            control.store.submit(value); control.tick()
            directory = control._runtime(value['id']); deadline = time.monotonic() + 90
            request = directory / 'training/recovery-request.json'
            while not request.exists():
                if time.monotonic() > deadline:raise AssertionError('recovery gate was not reached')
                time.sleep(.05)
            fleet = json.loads((directory / 'fleet.json').read_text())
            placement = fleet['role_placement']['0']; lost = placement['worker'][0]
            node = fleet['physical_nodes'][lost]
            for kind in ('worker', 'owner', 'primary', 'replica', 'journal-mirror'):
                for h in range(2):
                    if placement[kind][h] == lost:
                        result = control.sessions.invoke(node, dict(op='stop', root=node['root'], run=value['id'], role=f'{kind}-{h}-g0'))
                        assert result['owned_processes_gone']
            if placement['coordinator'] == lost:
                assert control.sessions.invoke(node, dict(op='stop', root=node['root'], run=value['id'], role='coordinator-g0'))['owned_processes_gone']
            shutil.rmtree(node['root'])
            control.record_loss(node['node_id'], dict(node_id=node['node_id'], kind='test_role_scope_destroyed',
                observed_epoch=time.time(), storage_removed=True))
            control.tick(); control.tick()
            job = control.store.get(value['id'])
            assert job['state'] == 'STARTING' and job['detail']['recovery_prepared'], job
            sources = job['detail']['recovery_sources']
            assert {s['parameter_source'] for s in sources} == {'primary', 'replica'}, sources
            assert {s['gradient_source'] for s in sources} == {'worker', 'journal-mirror'}, sources
            state = control.store.root; control.close(); control = Controller(inv, state)
            await_jobs(control, [value['id']])
            recovery_state = json.loads((directory / 'training/state.json').read_text())
            assert all('start_gate' not in item['config']
                       for item in recovery_state['steps']['journals_staged'])
            actual = json.loads((directory / 'training/summary.json').read_text())
            assert actual['model']['state']['completed'] == [8, 8]
            maximum = loss_error = 0.
            for got, want in zip(actual['model']['state']['rows'], expected['model']['state']['rows']):
                assert got['key'] == want['key'] and got['row_updates'] == want['row_updates']
                for field in ('weights', 'first_moment', 'second_moment'):
                    maximum = max(maximum, max(abs(a-b) for a,b in zip(got[field], want[field])))
            for got, want in zip(actual['model']['loss_history'], expected['model']['loss_history']):
                loss_error = max(loss_error, max(abs(a-b) for a,b in zip(got, want)))
            assert maximum == loss_error == 0
            assert not pathlib.Path(node['root']).exists(), 'lost storage was recreated'
            result = dict(test='job_controller_node_recovery', physical_hosts=1, physical_gpus=0,
                simulated_node_scopes=4, source_selection=sources, all_64_rows_verified=True,
                max_state_absolute_error=maximum, max_loss_absolute_error=loss_error,
                controller_restart_after_redundancy_rebuild=True, lost_storage_not_recreated=True,
                cleanup_verified=True, reference=expected, actual=actual)
            out.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({k:v for k,v in result.items() if k not in ('reference','actual')}))
        finally:
            for job in control.store.jobs():
                if job['state'] not in ('SUCCEEDED','FAILED','CANCELLED'):control.store.cancel(job['id'])
            control.tick(); control.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=pathlib.Path,required=True)
    p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();main(a.build.resolve(),a.out)
