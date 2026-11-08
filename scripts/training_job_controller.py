"""Durable priority/FIFO controller for cooperative parameter-server jobs.

Only trusted, preinstalled nodes are accepted. The controller stores metadata;
training tensors and gradients remain on the existing native data path.
"""
import argparse
import contextlib
import copy
import fcntl
import json
import hashlib
import pathlib
import sqlite3
import subprocess
import sys
import time

from job_placement import job_spec, place
from job_fleet import make_fleet, initial_port
from role_node import ensure, snapshot, stop, write
from transfer_worker_journals import NodeSessions

ROOT = pathlib.Path(__file__).resolve().parents[1]
TERMINAL = {'SUCCEEDED', 'FAILED', 'CANCELLED'}


class JobStore:
    def __init__(self, root):
        self.root = pathlib.Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.root / 'jobs.sqlite', timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS jobs(seq INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT UNIQUE NOT NULL, spec TEXT NOT NULL, state TEXT NOT NULL,
            desired TEXT NOT NULL DEFAULT 'RUN', created REAL NOT NULL, detail TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
            job TEXT, time REAL NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY, draining INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS node_losses(id TEXT PRIMARY KEY, receipt TEXT NOT NULL);
        ''')
        (self.root / 'jobs.sqlite').chmod(0o600)

    def close(self):
        self.db.close()

    @contextlib.contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def event(self, job, kind, detail):
        self.db.execute('INSERT INTO events(job,time,kind,detail) VALUES(?,?,?,?)',
                        (job, time.time(), kind, json.dumps(detail, sort_keys=True)))

    def submit(self, value):
        spec = job_spec(value); encoded = json.dumps(spec, sort_keys=True)
        with self.transaction():
            existing = self.db.execute('SELECT spec FROM jobs WHERE id=?', (spec['id'],)).fetchone()
            if existing:
                if existing['spec'] != encoded:
                    raise ValueError('job id already binds a different specification')
            else:
                self.db.execute('INSERT INTO jobs(id,spec,state,created,detail) VALUES(?,?,?,?,?)',
                                (spec['id'], encoded, 'QUEUED', time.time(), '{}'))
                self.event(spec['id'], 'submitted', {})
        return self.get(spec['id'])

    def get(self, job):
        row = self.db.execute('SELECT * FROM jobs WHERE id=?', (job,)).fetchone()
        if row is None:
            raise ValueError('unknown job')
        value = dict(row)
        value['spec'] = json.loads(value['spec']); value['detail'] = json.loads(value['detail'])
        return value

    def resume(self, source, new_id, extra_rounds):
        if type(extra_rounds) is not int or extra_rounds < 1:
            raise ValueError('extra rounds must be positive')
        previous = self.get(source)
        spec = copy.deepcopy(previous['spec'])
        workload = spec['workload']
        if 'datasets' not in workload:
            raise ValueError('resume requires a standalone recommendation job')
        manifest = pathlib.Path(workload['checkpoint_root']) / source / 'manifest.json'
        raw = manifest.read_bytes()
        checkpoint = json.loads(raw)
        boundary = checkpoint['progress']['completed'][0]
        if checkpoint['source_run'] != source or boundary != workload['rounds']:
            raise ValueError('checkpoint belongs to another job or round')
        spec['id'] = new_id
        workload.update(rounds=boundary + extra_rounds,
                        resume_checkpoint=dict(manifest=str(manifest), sha256=hashlib.sha256(raw).hexdigest()))
        return self.submit(spec)

    def status(self, job):
        value = self.get(job)
        runtime = self.root / 'jobs' / job / 'training'
        progress = runtime / 'progress.json'
        summary = runtime / 'summary.json'
        if progress.is_file():
            value['progress'] = json.loads(progress.read_text())
        if summary.is_file():
            result = json.loads(summary.read_text())
            value['training'] = dict(checkpoint=result['steps'].get('job_checkpoint'),
                workers=[{key: w.get(key) for key in ('worker','input_state','validation','samples_per_second','runtime')}
                         for w in result['model']['workers']])
        return value

    def jobs(self):
        return [self.get(r[0]) for r in self.db.execute('SELECT id FROM jobs ORDER BY seq')]

    def update(self, job, state, **detail):
        with self.transaction():
            value = self.get(job)['detail']; value.update(detail)
            self.db.execute('UPDATE jobs SET state=?,detail=? WHERE id=?', (state, json.dumps(value), job))
            self.event(job, state, detail)

    def cancel(self, job):
        with self.transaction():
            value = self.get(job)
            if value['state'] not in TERMINAL:
                self.db.execute("UPDATE jobs SET desired='CANCEL' WHERE id=?", (job,))
                self.event(job, 'cancel_requested', {})
        return self.get(job)

    def drain(self, node, enabled=True):
        with self.transaction():
            self.db.execute('INSERT INTO nodes(id,draining) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET draining=excluded.draining',
                            (node, int(enabled)))
            self.event(None, 'node_drain', dict(node=node, enabled=enabled))


class Controller:
    def __init__(self, inventory, state):
        self.inventory = copy.deepcopy(inventory)
        nodes = self.inventory['nodes']; ids = [n.get('node_id') for n in nodes]
        if not 2 <= len(nodes) <= 8 or len(set(ids)) != len(ids) or any(not i for i in ids):
            raise ValueError('inventory requires 2–8 distinct stable nodes')
        # A node's root is its durable reservation authority; aliases cannot
        # advertise the same authority twice and double its capacity.
        authorities = [((None, None) if n.get('local') else (n['host'], int(n.get('port', 22))),
                        pathlib.PurePath(n['root'])) for n in nodes]
        if len(set(authorities)) != len(nodes):
            raise ValueError('inventory repeats a node resource authority')
        for key in ('source','python','bin_dir','module_dir'):
            if any(not n.get(key) or n[key]!=nodes[0].get(key) for n in nodes):
                raise ValueError('registered nodes require uniform '+key+' paths')
        self.store = JobStore(state); self.sessions = NodeSessions(entrypoint='scripts/role_node.py')
        self.nodes = {n['node_id']: n for n in nodes}

    def close(self):
        self.sessions.__exit__(); self.store.close()

    @contextlib.contextmanager
    def leadership(self):
        with (self.store.root / 'controller.lock').open('a+b') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError('another controller is already active') from error
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def node_call(self, node_id, action, run='controller-probe', **kwargs):
        node = self.nodes[node_id]
        if action == 'reserve':
            kwargs['bind_host'] = '127.0.0.1' if node.get('local') else node.get('bind_host', '0.0.0.0')
        return self.sessions.invoke(node, dict(op='resources', root=node['root'], run=run,
            role='allocation', action=action, limits=node.get('capacity', {}), **kwargs))

    def probe(self):
        result = []
        draining = {r['id']: bool(r['draining']) for r in self.store.db.execute('SELECT * FROM nodes')}
        lost={r[0] for r in self.store.db.execute('SELECT id FROM node_losses')}
        for name in sorted(self.nodes):
            if name in lost:continue
            try:
                value = self.node_call(name, 'probe', gpu=self.nodes[name].get('gpu', False))
                result.append(dict(value, node_id=name, draining=draining.get(name, False)))
            except (ConnectionError, OSError, RuntimeError, ValueError) as error:
                self.store.event(None, 'node_observation_failed', dict(node=name, error=str(error)))
        return result

    def plan(self, value):
        return place(job_spec(value), self.probe())

    def record_loss(self, node_id, receipt):
        if node_id not in self.nodes or receipt.get('node_id') != node_id:
            raise ValueError('terminal loss node identity differs')
        node = self.nodes[node_id]
        if receipt.get('kind') == 'provider_instance_deleted':
            if (not node.get('uuid') or receipt.get('uuid') != node['uuid']
                    or receipt.get('inventory_absent') is not True or receipt.get('previously_owned') is not True):
                raise ValueError('provider terminal loss requires owned UUID and inventory evidence')
        elif receipt.get('kind') == 'test_role_scope_destroyed':
            if (not self.inventory.get('allow_test_loss') or not node.get('local')
                    or pathlib.Path(node['root']).exists() or receipt.get('storage_removed') is not True):
                raise ValueError('local loss receipt requires explicitly destroyed test storage')
        else:
            raise ValueError('timeout does not establish terminal loss')
        if type(receipt.get('observed_epoch')) not in (int, float) or receipt['observed_epoch'] <= 0:
            raise ValueError('terminal loss requires observation time')
        with self.store.transaction():
            old = self.store.db.execute('SELECT receipt FROM node_losses WHERE id=?', (node_id,)).fetchone()
            encoded = json.dumps(receipt, sort_keys=True)
            if old and old[0] != encoded:
                raise ValueError('terminal loss evidence changed')
            self.store.db.execute('INSERT OR IGNORE INTO node_losses(id,receipt) VALUES(?,?)', (node_id, encoded))
            self.store.event(None, 'terminal_node_loss', receipt)

    def _runtime(self, job):
        return self.store.root / 'jobs' / job

    def _start_executor(self, job):
        detail = job['detail']; directory = self._runtime(job['id'])
        serial = detail.get('executor_serial', 0)
        name = 'executor-' + str(serial)
        config = dict(fleet=str(directory / 'fleet.json'), run=job['id'], out=str(directory / 'training'))
        state = ensure(str(self.store.root), job['id'], name,
            dict(command=[sys.executable, str(ROOT / 'scripts/training_job_executor.py')], config=config,
                 environment=dict(PYTHONPATH=self.inventory['manager']['module_dir'] + ':' + str(ROOT / 'adapters/entropy'))))
        self.store.update(job['id'], 'RUNNING', executor_role=name, executor_serial=serial,
                          started_epoch=detail.get('started_epoch', time.time()), executor_handle=state)

    def preflight(self, spec, plan):
        workload = spec['workload']
        if 'datasets' not in workload or spec['device'] != 'gpu':
            return
        revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        for allocation in plan['allocations']:
            name = allocation['node_id']
            node = self.nodes[name]
            workers = [workload['datasets'][w] for w, host in enumerate(plan['groups']) if host == name]
            try:
                self.sessions.invoke(node, dict(op='training_preflight', revision=revision, gpu=bool(workers),
                    datasets=workers, checkpoint_root=workload['checkpoint_root'],
                    **{key: node[key] for key in ('root','source','bin_dir','module_dir')}))
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError(f'preflight failed on {name}: {error}') from error

    def _reserve(self, job):
        detail = job['detail']; spec = job['spec']; plan = detail['placement']
        tokens = detail['tokens']; directory = self._runtime(job['id'])
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fleet, ports = make_fleet(spec, plan, self.inventory, job['id'], tokens, detail['port_base'])
        if spec.get('_test_fault'):
            if not self.inventory.get('allow_test_loss'):raise ValueError('test fault requires a test inventory')
            fleet['controller_test_fault']=spec['_test_fault']
        self.preflight(spec, plan)
        # Persist complete intent before issuing any node-side reservation.
        write(directory / 'fleet.json', fleet); (directory / 'fleet.json').chmod(0o600)
        for item in plan['allocations']:
            wanted = dict(item['resources'], ports=ports[item['node_id']])
            self.node_call(item['node_id'], 'reserve', run=job['id'], token=tokens[item['node_id']], request=wanted)
        self.store.update(job['id'], 'STARTING', reserved_epoch=time.time())
        self._start_executor(self.store.get(job['id']))

    def _cleanup(self, job):
        detail = job['detail']; directory = self._runtime(job['id'])
        executor = detail.get('executor_role')
        if executor:
            receipt = stop(str(self.store.root), job['id'], executor)
            if not receipt['owned_processes_gone']:
                raise RuntimeError('job executor is still live')
        fleet_path = directory / 'fleet.json'
        if fleet_path.exists():
            from job_recovery import effective_fleet
            fleet = effective_fleet(directory)
            state_file = directory / 'training/state.json'
            saved = json.loads(state_file.read_text()) if state_file.exists() else {}
            generations = range(saved.get('recovery_generation', 1) + 1)
            from physical_roles import PhysicalRoles
            with PhysicalRoles(fleet['nodes'], job['id'], fleet, directory / 'training') as roles:
                for generation in reversed(list(generations)):
                    for kind in ('worker', 'primary', 'owner', 'replica', 'journal-mirror'):
                        for h in range(job['spec']['groups']):
                            receipt = roles.stop(h, f'{kind}-{h}-g{generation}')
                            if not receipt['owned_processes_gone']:
                                raise RuntimeError('owned role remains live')
                    if not roles.stop(0, f'coordinator-g{generation}')['owned_processes_gone']:
                        raise RuntimeError('coordinator remains live')
        lost={r[0] for r in self.store.db.execute('SELECT id FROM node_losses')}
        intent_path = directory / 'recovery-intent.json'
        pending_tokens = json.loads(intent_path.read_text())['tokens'] if intent_path.exists() else {}
        for allocation in [*detail.get('reservation_history',[]),detail.get('tokens',{}),pending_tokens]:
            for name,token in allocation.items():
                if name not in lost:self.node_call(name,'release',run=job['id'],token=token)
        target = 'CANCELLED' if job['desired'] == 'CANCEL' else detail.get('terminal_target', 'FAILED')
        self.store.update(job['id'], target, cleanup_verified=True, finished_epoch=time.time())

    def tick(self):
        # Reconcile existing intents/handles before making fresh allocations.
        for job in self.store.jobs():
            if job['state'] in TERMINAL:
                continue
            try:
                if job['desired'] == 'CANCEL':
                    if job['state'] in ('QUEUED', 'WAITING_RESOURCES') and not job['detail'].get('tokens'):
                        self.store.update(job['id'], 'CANCELLED', cleanup_verified=True)
                    else:
                        self.store.update(job['id'], 'CLEANUP_PENDING')
                        self._cleanup(self.store.get(job['id']))
                elif job['state'] == 'RESERVING':
                    try:
                        self._reserve(job)
                    except (ValueError, OSError) as error:
                        self.store.update(job['id'], 'CLEANUP_PENDING', error=str(error), terminal_target='FAILED')
                        self._cleanup(self.store.get(job['id']))
                elif job['state'] == 'STARTING':
                    self._start_executor(job)
                elif job['state'] in ('RECOVERING','WAITING_RECOVERY_RESOURCES'):
                    from job_recovery import prepare
                    try:
                        prepare(self, job)
                    except ValueError as error:
                        if 'datasets' not in job['spec']['workload']:
                            raise
                        self.store.update(job['id'], 'CLEANUP_PENDING', terminal_target='FAILED', error=str(error))
                elif job['state'] == 'RUNNING':
                    lost={r[0] for r in self.store.db.execute('SELECT id FROM node_losses')}
                    assigned={a['node_id'] for a in job['detail']['placement']['allocations']}
                    if lost & assigned and not job['detail'].get('recovery_prepared'):
                        self.store.update(job['id'],'RECOVERING')
                        continue
                    path = self.store.root / 'results' / job['id'] / 'roles' / job['detail']['executor_role']
                    state = snapshot(path)
                    if 'result' in state:
                        code = state['result']['exit_code']
                        if code==76:
                            self.store.update(job['id'],'RECOVERING')
                        elif code == 75:
                            self.store.update(job['id'], 'STARTING', executor_serial=job['detail'].get('executor_serial', 0) + 1)
                        else:
                            report = self._runtime(job['id']) / 'training/summary.json'
                            target = 'SUCCEEDED' if code == 0 and report.exists() else 'FAILED'
                            self.store.update(job['id'], 'CLEANUP_PENDING', terminal_target=target,
                                error=(path/'stderr').read_text(errors='replace')[-5000:] if target=='FAILED' and (path/'stderr').exists() else None)
                    elif not state['live_child'] and not state['live_wrapper']:
                        # A dead observer does not prove that its training roles
                        # died. Reenter the same run with the same managed intent.
                        self.store.update(job['id'], 'STARTING', executor_serial=job['detail'].get('executor_serial', 0) + 1)
                elif job['state'] == 'CLEANUP_PENDING':
                    self._cleanup(job)
            except (ConnectionError, TimeoutError, OSError, RuntimeError, ValueError) as error:
                self.store.event(job['id'], 'reconciliation_pending', dict(error=str(error)))
        queued = sorted((j for j in self.store.jobs() if j['state'] in ('QUEUED', 'WAITING_RESOURCES') and j['desired'] == 'RUN'),
                        key=lambda j: (-j['spec']['priority'], j['seq']))
        for job in queued:
            plan = self.plan(job['spec'])
            if not plan['feasible']:
                if 'datasets' in job['spec']['workload'] and job['spec']['device'] == 'gpu' and plan['reason'] == 'insufficient_healthy_nodes':
                    self.store.update(job['id'], 'FAILED', error='fewer than two reachable GPU-capable nodes')
                else:
                    self.store.update(job['id'], 'WAITING_RESOURCES', waiting_reason=plan['reason'])
                continue
            attempt = job['detail'].get('allocation_attempt', 0)
            tokens = {a['node_id']: job['id'] + '-a' + str(attempt) for a in plan['allocations']}
            self.store.update(job['id'], 'RESERVING', placement=plan, tokens=tokens,
                              allocation_attempt=attempt, port_base=initial_port(job['id']))
            # Reservation is synchronous; the next candidate probes the updated
            # node ledgers rather than planning against this tick's old snapshot.
            try:
                self._reserve(self.store.get(job['id']))
            except (ConnectionError, TimeoutError) as error:
                self.store.event(job['id'], 'reservation_observation_pending', dict(error=str(error)))
            except (ValueError, OSError, RuntimeError) as error:
                self.store.update(job['id'], 'CLEANUP_PENDING', terminal_target='FAILED', error=str(error))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=pathlib.Path, required=True)
    p.add_argument('--inventory', type=pathlib.Path)
    commands = p.add_subparsers(dest='kind', required=True)
    controller = commands.add_parser('controller').add_subparsers(dest='action', required=True)
    serve = controller.add_parser('serve'); serve.add_argument('--once', action='store_true')
    jobs = commands.add_parser('job').add_subparsers(dest='action', required=True)
    for name in ('plan', 'submit'):
        jobs.add_parser(name).add_argument('spec', type=pathlib.Path)
    jobs.add_parser('status').add_argument('id', nargs='?')
    jobs.add_parser('cancel').add_argument('id')
    resume = jobs.add_parser('resume')
    resume.add_argument('source')
    resume.add_argument('id')
    resume.add_argument('--extra-rounds', required=True, type=int)
    nodes = commands.add_parser('node').add_subparsers(dest='action', required=True)
    nodes.add_parser('list'); nodes.add_parser('probe')
    drain = nodes.add_parser('drain'); drain.add_argument('id'); drain.add_argument('--off', action='store_true')
    lost = nodes.add_parser('lost'); lost.add_argument('id'); lost.add_argument('receipt', type=pathlib.Path)
    a = p.parse_args()
    store = JobStore(a.state)
    try:
        if a.kind == 'job' and a.action == 'submit':
            value = store.submit(json.loads(a.spec.read_text()))
        elif a.kind == 'job' and a.action == 'cancel':
            value = store.cancel(a.id)
        elif a.kind == 'job' and a.action == 'status':
            value = store.status(a.id) if a.id else store.jobs()
        elif a.kind == 'job' and a.action == 'resume':
            value = store.resume(a.source, a.id, a.extra_rounds)
        elif a.kind == 'node' and a.action == 'drain':
            store.drain(a.id, not a.off); value = dict(node=a.id, draining=not a.off)
        else:
            if not a.inventory:
                p.error('this operation requires --inventory')
            control = Controller(json.loads(a.inventory.read_text()), a.state)
            try:
                if a.kind == 'job':
                    value = control.plan(json.loads(a.spec.read_text()))
                elif a.kind == 'node' and a.action == 'lost':
                    control.record_loss(a.id, json.loads(a.receipt.read_text()))
                    value = dict(node=a.id, terminal_loss_recorded=True)
                elif a.kind == 'node':
                    value = control.probe()
                else:
                    with control.leadership():
                        while True:
                            control.tick()
                            if a.once:
                                break
                            time.sleep(1)
                    value = dict(reconciled=True)
            finally:
                control.close()
        print(json.dumps(value, indent=2))
    finally:
        store.close()


if __name__ == '__main__':
    main()
