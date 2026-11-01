"""Two framework peers, two learner threads each, and real parameter shards.

Execute the preserved Python barriers and checkpoint methods. Only their native
cluster rendezvous is replaced by a bounded test coordinator. Native feeder and
legacy sparse persistence remain the source-contract stand-ins.
"""
import json
import hashlib
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
from multiprocessing.managers import BaseManager

from entropy_source_contract import source_root
from rpc_processes import ready, stop


class Rendezvous:
    def __init__(self):
        self.condition = threading.Condition()
        self.generation = 0
        self.name = None
        self.arrivals = set()
        self.history = []
        self.held = False
        self.released = False
        self.error = None

    def abort(self, error):
        with self.condition:
            self.error = str(error)
            self.condition.notify_all()

    def wait(self, peer, name):
        with self.condition:
            generation = self.generation
            if self.name is None:
                self.name = name
            if self.name != name or peer in self.arrivals:
                self.abort('peer barrier order or duplicate arrival: ' + name)
            self.arrivals.add(peer)
            if len(self.arrivals) == 2 and not self.error:
                if name.startswith('::EPS_ORIGINAL_COMPLETE_') and not self.released:
                    self.held = True
                    self.condition.notify_all()
                else:
                    self._advance()
            deadline = time.monotonic() + 30
            while generation == self.generation and not self.error:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.abort('peer barrier timeout: ' + name)
                    break
                self.condition.wait(remaining)
            if self.error:
                raise RuntimeError(self.error)

    def _advance(self):
        self.history.append(dict(name=self.name, peers=sorted(self.arrivals)))
        self.generation += 1
        self.name = None
        self.arrivals.clear()
        self.condition.notify_all()

    def release(self):
        with self.condition:
            assert self.held and len(self.arrivals) == 2
            self.released = True
            self.held = False
            self._advance()

    def snapshot(self):
        with self.condition:
            return dict(held=self.held, error=self.error, history=list(self.history))


def peer(endpoints, root, run, number, address, failure, mode='train'):
    import tensorflow as tf
    from entropy_source_contract import load, Learner, peer_environment
    from entropy_install import install
    tf.compat.v1.disable_eager_execution()
    BaseManager.register('rendezvous')
    manager = BaseManager(address=('127.0.0.1', address),
                          authkey=bytes.fromhex(os.environ['EPS_TEST_RENDEZVOUS_KEY']))
    manager.connect()
    coordinator = manager.rendezvous()
    train, context, provenance = load(tf)
    original_native_call = train.env.call_local_function
    learners = [Learner(number * 2 + thread, thread) for thread in range(2)]
    def current():
        # The original legacy sparse saver invokes a decorated method on an
        # executor thread. Its native environment is a stand-in in this fixture.
        return getattr(train.MultiThreadingLearner.tls, 'learner', learners[0])
    def native_call(name, *args):
        if name == 'cluster_barrier':
            return coordinator.wait(number, args[1])
        return original_native_call(name, *args)
    barrier_class, provenance['barriers'] = peer_environment(train, current, native_call)
    provenance['legacy_executor_learner_context'] = 'test fallback to local chief; native framework TLS not loaded'
    master = types.SimpleNamespace(_barriers={name: barrier_class(2) for name in ('all', 'worker')})
    for learner in learners:
        learner.master = master
        learner.is_chief = learner.thread_id == 0
    train.env._env.peer_id = number
    train.env.peer_id = lambda: number
    config = dict(schema_version=1, run=run, endpoints=endpoints,
                  learner_id_stride=2, tables=dict(users=dict(id=3, seed=42)))
    errors = []
    with install(train, config) as hooks:
        def execute(thread):
            try:
                learner = learners[thread]
                train.MultiThreadingLearner.tls.learner = learner
                graph = tf.Graph()
                with graph.as_default():
                    optimizer = train.SparseLazyAdam(learning_rate=.01, epsilon=.001)
                    if number == 1:
                        table = train.SparseTable([1, 2], 'users', optimizer=optimizer,
                            initializer=tf.keras.initializers.Constant(.25), pooling_method='sum',
                            location_peers=[0], v2=True, non_unique=True)
                        table.create_table()
                        if thread == 0:
                            (root / 'table.json').write_text(json.dumps(hooks._state(table).definition))
                        output = table.embedding_lookup_and_pooling(
                            [tf.constant([1, 2], tf.int64)], [tf.constant([0, 2], tf.int32)],
                            None, shapes=[[1, 2]])
                        builder = train.InputBuilder()
                        builder.input_spec = types.SimpleNamespace(sparse_tables=dict(users=table))
                        features = builder.sparse_vars()
                        gradients = tf.gradients(tf.reduce_sum(output[0]), features)
                        step = tf.group(*builder.apply_sparse_grads(list(zip(gradients, features)), []))
                    else:
                        # The chief has original tables but no selected external
                        # table, so its global save must use the control backend.
                        if mode == 'train':
                            train.SparseTable([4, 2], 'legacy', optimizer=optimizer)
                        step = tf.no_op()
                        if thread == 0:
                            variable = tf.Variable([.5 if mode == 'train' else -.5], dtype=tf.float32)
                            dense = train.DenseTable(variable)
                            if failure:
                                def fail(*args):
                                    raise RuntimeError('injected dense checkpoint failure')
                                dense.checkpoint = fail
                            learner.var_collections[variable] = dense
                    initialize = tf.compat.v1.global_variables_initializer()
                options = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                                   inter_op_parallelism_threads=1,
                                                   device_count={'GPU': 0})
                with tf.compat.v1.Session(graph=graph, config=options) as session:
                    runner = train._MultiDatasetRunner()
                    runner.learner = learner
                    runner.is_worker = True
                    runner.session = session
                    session.run(initialize)
                    if mode == 'restore':
                        runner.restore(None, str(root / 'checkpoints'), 'two')
                        if number == 0 and thread == 0:
                            assert session.run(variable).tolist() == [.5]
                    for iteration in range(1 if failure or mode == 'restore' else 2):
                        session.run(step)
                        train.env.barrier('TEST_STEP_' + str(iteration), cluster='all')
                        if mode == 'restore':
                            if number == 0 and thread == 0:
                                backend = hooks._control_backend()
                                backend.save('restored-final')
                                backend.export_checkpoint('restored-final', str(root / 'restored-final'), tf.io.gfile)
                        else:
                            runner.checkpoint(str(root / 'checkpoints'), ('one', 'two')[iteration])
                        train.env.barrier('TEST_AFTER_' + str(iteration), cluster='all')
            except BaseException as error:
                errors.append(str(error))
                coordinator.abort(str(error))
                traceback.print_exc()
        threads = [threading.Thread(target=execute, args=(i,), daemon=True) for i in range(2)]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + 45
        while any(thread.is_alive() for thread in threads) and not errors and time.monotonic() < deadline:
            time.sleep(.02)
        if errors or any(thread.is_alive() for thread in threads):
            # Original local barriers have no abort API. Let the parent kill the
            # owned process group after checking the durable failure state.
            coordinator.abort(errors[0] if errors else 'learner thread timeout')
            while True:
                time.sleep(1)
        record = dict(peer=number, learner_threads=2, backend_ids=sorted(hooks.backends()),
                      administrative_backend=hooks._administrative_backend is not None,
                      dense_restored='dense_restored' in context.events,
                      source=provenance)
        prefix = 'restored-' if mode == 'restore' else ''
        (root / f'{prefix}peer-{number}.json').write_text(json.dumps(record, indent=2) + '\n')


def run(server, root, failure):
    from eps_backend import Backend
    from checkpoint_restore import load_committed
    coordinator = Rendezvous()
    BaseManager.register('rendezvous', callable=lambda: coordinator)
    key = os.urandom(32)
    manager = BaseManager(address=('127.0.0.1', 0), authkey=key)
    manager_server = manager.get_server()
    threading.Thread(target=manager_server.serve_forever, daemon=True).start()
    processes, peers, logs, endpoints = [], [], [], []
    run_id = 'peer-failure' if failure else 'peer-success'
    try:
        for shard in range(2):
            config = root / f'shard-{shard}.json'
            config.write_text(json.dumps(dict(schema_version=1, run_id=run_id, listen='127.0.0.1:0',
                partition=dict(shard_id=shard, shard_count=2, epoch=1), tables=[], cache=dict(enabled=False),
                snapshot_directory=str(root / 'snapshots'))))
            log = open(root / f'shard-{shard}.log', 'w+')
            logs.append(log)
            process = subprocess.Popen([server, '--config', str(config)], stdout=subprocess.PIPE, stderr=log, text=True)
            processes.append(process)
            endpoints.append(ready(process))
        env = dict(os.environ, EPS_TEST_RENDEZVOUS_KEY=key.hex())
        for number in range(2):
            log = open(root / f'peer-{number}.log', 'w+')
            logs.append(log)
            process = subprocess.Popen([sys.executable, __file__, '--peer', *endpoints, str(root), run_id,
                str(number), str(manager_server.address[1]), str(int(failure))], stdout=log, stderr=log,
                env=env, start_new_session=True)
            peers.append(process)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            state = coordinator.snapshot()
            if state['held'] or state['error']:
                break
            if any(process.poll() is not None for process in peers):
                raise AssertionError('peer exited before checkpoint observation')
            time.sleep(.02)
        else:
            raise AssertionError('checkpoint observation timeout')
        checkpoint = root / 'checkpoints/one'
        assert (checkpoint / 'external_parameters/parameter-service.json').is_file()
        assert not (checkpoint / 'external_parameters/committed.json').exists()
        probe = Backend(endpoints, run_id, 1000)
        try:
            probe.create_table((root / 'table.json').read_text())
            stats = probe.stats()
            assert all(item['updates'] == 2 and item['live_tickets'] == 0 for item in stats), stats
            import numpy as np
            try:
                probe.pull([dict(table=3, feature=0, pooling='sum', keys=np.array([1], np.uint64),
                                 offsets=np.array([0, 1], np.int64))], read_only=True)
            except RuntimeError as error:
                assert 'shard is draining' in str(error), str(error)
            else:
                raise AssertionError('checkpoint admitted a lookup before global commit')
        finally:
            probe.close()
        if failure:
            assert 'injected dense checkpoint failure' in state['error'], state
            assert not (checkpoint / 'native-committed.json').exists()
            return dict(test='peer_checkpoint_dense_failure', peers=2, learner_threads=4,
                        joint_marker_absent=True, admission_frozen=True, live_tickets=0,
                        recovery='owned test process groups terminated; original local barriers have no abort API')
        assert (checkpoint / 'native-committed.json').is_file()
        coordinator.release()
        for process in peers:
            assert process.wait(timeout=30) == 0
        for name in ('one', 'two'):
            descriptor = load_committed(root / 'checkpoints' / name, checkpoint_id=name)
            for item in descriptor['manifest']['files']:
                raw = pathlib.Path(descriptor['directory'], item['file']).read_bytes()
                assert len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256']
            assert (root / 'checkpoints' / name / 'dense/model.index').is_file()
        first, second = [json.loads((root / f'peer-{number}.json').read_text()) for number in range(2)]
        assert first['backend_ids'] == [] and first['administrative_backend'], first
        assert second['backend_ids'] == [4, 7] and not second['administrative_backend'], second
        probe = Backend(endpoints, run_id, 1001)
        try:
            probe.create_table((root / 'table.json').read_text())
            stats = probe.stats()
            assert all(item['updates'] == 4 and item['live_tickets'] == 0 for item in stats), stats
            handle, _ = probe.pull([dict(table=3, feature=0, pooling='sum', keys=np.array([1], np.uint64),
                                        offsets=np.array([0, 1], np.int64))], read_only=True)
            probe.release(handle)
        finally:
            probe.close()
        history = coordinator.snapshot()['history']
        assert all(item['peers'] == [0, 1] for item in history)
        return dict(test='peer_checkpoint_repeated_commit', peers=2, learner_threads=4,
                    checkpoints=2, updates_per_shard=4, admission_resumed=True,
                    precommit_marker_absent=True, administrative_only_chief=True,
                    exported_shard_hashes_verified=True,
                    global_barrier_rounds=len(history), peer_records=[first, second])
    except BaseException:
        for log in logs:
            log.flush()
            log.seek(0)
            print(log.read()[-7000:], file=sys.stderr)
        raise
    finally:
        for process in peers:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            stop(process)
        for process in processes:
            stop(process)
        for log in logs:
            log.close()


def restore_followup(server, comparator, root):
    import numpy as np
    from eps_backend import Backend
    from checkpoint_store import LocalFileSystem
    from checkpoint_restore import stage_checkpoint
    coordinator = Rendezvous()
    BaseManager.register('rendezvous', callable=lambda: coordinator)
    key = os.urandom(32)
    manager = BaseManager(address=('127.0.0.1', 0), authkey=key)
    manager_server = manager.get_server()
    threading.Thread(target=manager_server.serve_forever, daemon=True).start()
    processes, peers, logs = [], [], []
    def start_shards(run_id, restored):
        endpoints = []
        for shard in range(2):
            config = root / f'{run_id}-{shard}.json'
            config.write_text(json.dumps(dict(schema_version=1, run_id=run_id, listen='127.0.0.1:0',
                partition=dict(shard_id=shard, shard_count=2, epoch=1), tables=[], cache=dict(enabled=False),
                snapshot_directory=str(root / (run_id + '-snapshots')))))
            command = [server, '--config', str(config)]
            if restored:
                staged = stage_checkpoint(root / 'checkpoints/two', root / f'staged-{shard}', shard,
                                          checkpoint_id='two')
                command += ['--restore', staged['file']]
            log = open(root / f'{run_id}-{shard}.log', 'w+')
            logs.append(log)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True)
            processes.append(process)
            endpoints.append(ready(process))
        return endpoints
    try:
        endpoints = start_shards('peer-restored', True)
        env = dict(os.environ, EPS_TEST_RENDEZVOUS_KEY=key.hex())
        for number in range(2):
            log = open(root / f'restored-peer-{number}.log', 'w+')
            logs.append(log)
            process = subprocess.Popen([sys.executable, __file__, '--peer', *endpoints, str(root),
                'peer-restored', str(number), str(manager_server.address[1]), '0', 'restore'],
                stdout=log, stderr=log, env=env, start_new_session=True)
            peers.append(process)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            state = coordinator.snapshot()
            if state['error']:
                raise AssertionError(state['error'])
            if all(process.poll() is not None for process in peers):
                break
            time.sleep(.02)
        else:
            raise AssertionError('restored peers did not finish')
        assert all(process.returncode == 0 for process in peers)
        records = [json.loads((root / f'restored-peer-{i}.json').read_text()) for i in range(2)]
        assert records[0]['backend_ids'] == [] and records[0]['administrative_backend']
        assert records[0]['dense_restored'] and not records[1]['dense_restored']
        probe = Backend(endpoints, 'peer-restored', 1000)
        try:
            stats = probe.stats()
            # RPC counters start with the fresh run; persisted row/optimizer
            # counters include the four saved updates and are compared below.
            assert all(item['updates'] == 2 and item['live_tickets'] == 0 for item in stats), stats
        finally:
            probe.close()
        reference_endpoints = start_shards('peer-uninterrupted', False)
        reference = Backend(reference_endpoints, 'peer-uninterrupted', 1001)
        try:
            reference.create_table((root / 'table.json').read_text())
            features = [dict(table=3, feature=0, pooling='sum', keys=np.array([1, 2], np.uint64),
                             offsets=np.array([0, 2], np.int64))]
            for _ in range(6):
                handle, _ = reference.pull(features)
                reference.push(handle, [0], [np.ones((1, 2), np.float32)])
            reference.save('uninterrupted')
            from checkpoint_store import export_checkpoint
            export_checkpoint(reference, 'uninterrupted', root / 'uninterrupted-final',
                              run='peer-uninterrupted', epoch=1, shard_count=2, filesystem=LocalFileSystem)
        finally:
            reference.close()
        rows = 0
        maximum = 0.
        for shard in range(2):
            result = subprocess.run([comparator,
                str(root / 'uninterrupted-final' / f'parameter-shard-{shard}.bin'),
                str(root / 'restored-final' / f'parameter-shard-{shard}.bin'), str(shard), '2', '1', '3'],
                capture_output=True, text=True, check=True, timeout=20)
            record = json.loads(result.stdout)
            rows += record['checked_rows']
            maximum = max(maximum, record['max_state_absolute_error'])
        assert rows == 2 and maximum == 0., (rows, maximum)
        history = coordinator.snapshot()['history']
        assert all(item['peers'] == [0, 1] for item in history)
        return dict(test='peer_checkpoint_restore_continuation', peer_processes=2, learner_threads=4,
                    parameter_shards=2, saved_updates_per_shard=4, final_updates_per_shard=6,
                    resumed_run_acknowledged_updates_per_shard=2,
                    checked_rows=rows, max_state_absolute_error=maximum,
                    compared=['weights', 'first_moment', 'second_moment', 'row_counters',
                              'optimizer_sequence', 'beta_powers'],
                    dense_value_before_restore=-.5, dense_value_after_restore=.5,
                    administrative_only_chief=True, global_barrier_rounds=len(history),
                    peer_records=records)
    except BaseException:
        for log in logs:
            log.flush()
            log.seek(0)
            print(log.read()[-7000:], file=sys.stderr)
        raise
    finally:
        for process in peers:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            stop(process)
        for process in processes:
            stop(process)
        for log in logs:
            log.close()


def main():
    try:
        assert (source_root() / 'tensorswitch/python/tensorswitch/env.py').is_file()
    except (FileNotFoundError, AssertionError):
        print('SKIP: preserved train.py and env.py are required')
        return 77
    if sys.argv[1] == '--peer':
        peer(sys.argv[2:4], pathlib.Path(sys.argv[4]), sys.argv[5], int(sys.argv[6]), int(sys.argv[7]),
             bool(int(sys.argv[8])), sys.argv[9] if len(sys.argv)>9 else 'train')
    else:
        with tempfile.TemporaryDirectory(prefix='eps-peer-checkpoint-') as directory:
            for failure in (False, True):
                root = pathlib.Path(directory) / ('failure' if failure else 'success')
                root.mkdir()
                print(json.dumps(run(sys.argv[1], root, failure)), flush=True)
                if not failure and len(sys.argv)>2:
                    print(json.dumps(restore_followup(sys.argv[1], sys.argv[2], root)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
