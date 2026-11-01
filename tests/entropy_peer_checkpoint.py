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


def peer(endpoints, root, run, number, address, failure):
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
                        train.SparseTable([4, 2], 'legacy', optimizer=optimizer)
                        step = tf.no_op()
                        if thread == 0:
                            variable = tf.Variable([.5], dtype=tf.float32)
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
                    for iteration in range(1 if failure else 2):
                        session.run(step)
                        train.env.barrier('TEST_STEP_' + str(iteration), cluster='all')
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
                      source=provenance)
        (root / f'peer-{number}.json').write_text(json.dumps(record, indent=2) + '\n')


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


def main():
    try:
        assert (source_root() / 'tensorswitch/python/tensorswitch/env.py').is_file()
    except (FileNotFoundError, AssertionError):
        print('SKIP: preserved train.py and env.py are required')
        return 77
    if sys.argv[1] == '--peer':
        peer(sys.argv[2:4], pathlib.Path(sys.argv[4]), sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), bool(int(sys.argv[8])))
    else:
        with tempfile.TemporaryDirectory(prefix='eps-peer-checkpoint-') as directory:
            for failure in (False, True):
                root = pathlib.Path(directory) / ('failure' if failure else 'success')
                root.mkdir()
                print(json.dumps(run(sys.argv[1], root, failure)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
