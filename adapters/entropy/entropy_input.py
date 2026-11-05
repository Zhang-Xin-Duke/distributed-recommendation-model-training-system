"""Original Runner input capture and fresh-run restore coordination."""
import ctypes
import functools
import hashlib
import importlib
import json
import pathlib
import posixpath
import shutil
import subprocess
import tempfile
import threading
import time

from input_checkpoint_store import (_read_json, publish_worker, publish_group,
                                    stage_file)


class IteratorState:
    def __init__(self, manager, dataset, mode):
        import tensorflow as tf
        self.manager, self.dataset, self.mode = manager, dataset, mode
        self.scope = getattr(dataset, '_eps_input_scope', None)
        if not self.scope:
            raise ValueError('input checkpoints require the original FileFeeder scope overlay')
        config = manager.train.ts.config
        if config.process_global_shuffle or config.partition_label_names:
            raise ValueError('serialize the custom shuffle/label-partition buffers before enabling this input checkpoint mode')
        resource = dataset.iterator._iterator_resource
        with resource.graph.as_default(), tf.device('/CPU:0'):
            self.prefix = tf.compat.v1.placeholder(tf.string, shape=[], name='eps_input_prefix')
            serialized = tf.raw_ops.SerializeIterator(resource_handle=resource)
            self.save_op = tf.raw_ops.SaveV2(prefix=self.prefix, tensor_names=['iterator'],
                shape_and_slices=[''], tensors=[serialized])
            saved = tf.raw_ops.RestoreV2(prefix=self.prefix, tensor_names=['iterator'],
                shape_and_slices=[''], dtypes=[tf.variant])[0]
            self.restore_op = tf.raw_ops.DeserializeIterator(resource_handle=resource, serialized=saved)

    def identity(self):
        raw = json.dumps(self.dataset.inputs, sort_keys=True, separators=(',', ':')).encode()
        return dict(scope=self.scope, mode=self.mode,
                    source_handle=self.dataset.file_feeder.input_handle,
                    input_spec_sha256=hashlib.sha256(raw).hexdigest(),
                    shot_count=len(self.dataset.input_shots))

    def metadata(self, runner):
        value = self.identity()
        index = self.dataset.next_index
        if not 1 <= index <= value['shot_count']:
            raise ValueError('input iterator has no initialized input shot')
        value.update(next_index=index, end=self.dataset.end,
            ident=str(self.dataset.input_shots[index - 1].ident), passed_ident=runner.passed_ident)
        return value

    def capture(self, session, directory):
        prefix = directory / 'iterator'
        capture = directory / 'capture.bin'
        with self.manager.capture_lock:
            for attempt in range(self.manager.retries):
                self.manager.call('eps_input_begin_scoped_capture', self.scope)
                try:
                    session.run(self.save_op, feed_dict={self.prefix: str(prefix)})
                except BaseException:
                    self.manager.call('eps_input_cancel_capture')
                    raise
                result = self.manager.train.env.call_local_function('eps_input_finish_capture', str(capture))
                if result == 0:
                    return {name: directory / name for name in (
                        'iterator.index', 'iterator.data-00000-of-00001', 'capture.bin')}
                self.manager.call('eps_input_cancel_capture')
                if attempt + 1 < self.manager.retries:
                    time.sleep(.05)
            raise RuntimeError('parallel input did not reach a complete serialization cut')

    def validate_position(self, metadata):
        expected = self.identity()
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError('restored input scope, source or specification differs from the Runner')
        index = metadata['next_index']
        if str(self.dataset.input_shots[index - 1].ident) != metadata['ident']:
            raise ValueError('restored input shot identity differs from the Runner')


class InputManager:
    def __init__(self, installation):
        self.installation = installation
        self.train = installation.train
        self.options = installation.config.input_checkpoint
        self.planner = self.options['planner']
        if shutil.which(self.planner) is None:
            raise ValueError('input checkpoint planner is not executable: ' + self.planner)
        self.retries = self.options.get('max_retries', 20)
        self.restore_pending = self.options.get('restore', False)
        self.capture_lock = threading.Lock()
        self.states = {}
        self.restored_runners = set()
        capability = self.train.CLIB.eps_entropy_input_checkpoint_version
        capability.restype = ctypes.c_int
        if capability() < 3:
            raise ValueError('original input checkpoint requires native capability version 3')
        self._install()

    def call(self, name, *values):
        if self.train.env.call_local_function(name, *values) != 0:
            raise RuntimeError('native input checkpoint operation failed: ' + name)

    def _install(self):
        owner = self.train.FileFeeder
        original = owner.__init__

        @functools.wraps(original)
        def initialize(feeder, *args, **kwargs):
            original(feeder, *args, **kwargs)
            feeder._eps_input_scope_factory = self._scope
            if self.restore_pending:
                self.call('eps_input_hold_source', feeder.input_handle, 1)
        self.installation._patch_controller(owner, '__init__', initialize)
        pipeline = importlib.import_module(self.train.__package__ + '.dataset_pipeline')
        owner = pipeline.FileFeederNextableDataset
        original_next = owner.next_dataset

        @functools.wraps(original_next)
        def next_dataset(dataset, session):
            scope = getattr(dataset, '_eps_input_scope', None)
            if scope and dataset.next_index < len(dataset.input_shots):
                self.call('eps_input_begin_generation', scope)
            return original_next(dataset, session)
        self.installation._patch_controller(owner, 'next_dataset', next_dataset)
        owner = self.train._MultiDatasetRunner
        original_runner = owner.__init__

        @functools.wraps(original_runner)
        def runner(instance, learner, session_creator):
            states = {}
            if learner.cluster_role == 'worker':
                for mode, dataset in [('train', learner.train_dataset), ('eval', learner.eval_dataset)]:
                    if dataset is not None and getattr(dataset, '_eps_input_scope', None):
                        states[id(dataset)] = IteratorState(self, dataset, mode)
                active = learner.train_dataset or learner.eval_dataset
                if active is not None and id(active) not in states:
                    raise ValueError('active Runner input requires the original FileFeeder scope overlay')
            self.states[id(instance)] = states
            return original_runner(instance, learner, session_creator)
        self.installation._patch_controller(owner, '__init__', runner)

    def _scope(self, feeder, is_train):
        _, worker = self.installation._context()
        return f'worker-{worker}-source-{feeder.input_handle}-' + ('eval' if is_train is False else 'train')

    def state(self, runner):
        if not runner.is_worker:
            return None
        state = self.states.get(id(runner), {}).get(id(runner.dataset))
        if state is None:
            raise ValueError('Runner has no prepared input checkpoint state')
        return state

    def require_training_ready(self, runner):
        if self.options.get('restore', False) and id(runner) not in self.restored_runners:
            raise RuntimeError('restore the committed input iterator before admitting training')

    def save_local(self, runner, root, fs):
        state = self.state(runner)
        if state is None:
            return
        _, worker = self.installation._context()
        with tempfile.TemporaryDirectory(prefix='eps-original-input-save-') as temporary:
            files = state.capture(runner.session, pathlib.Path(temporary))
            publish_worker(root, worker, state.metadata(runner), files, filesystem=fs)

    def save_group(self, root, progress, fs):
        records = []
        for worker in range(len(progress['completed'])):
            _, record = _read_json(fs, posixpath.join(root, 'input_state', 'worker-' + str(worker), 'manifest.json'))
            records.append(record)
        with tempfile.TemporaryDirectory(prefix='eps-original-input-group-') as temporary:
            directory = pathlib.Path(temporary)
            captures = []
            for record in records:
                item = next(item for item in record['files'] if item['file'] == 'capture.bin')
                captures.append(stage_file(fs, posixpath.join(root, 'input_state', 'worker-' + str(record['worker'])),
                    item, directory / ('capture-' + str(record['worker']) + '.bin')))
            files = {}
            for handle in sorted({record['source_handle'] for record in records}):
                roster, plan = directory / f'source-{handle}.bin', directory / f'plan-{handle}.bin'
                self.call('eps_input_export_source', handle, str(roster))
                command = [self.planner, '--source', str(roster), '--output', str(plan)]
                for capture in captures:
                    command += ['--capture', str(capture)]
                subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
                files[handle] = dict(roster=roster, plan=plan)
            return publish_group(root, run=self.installation.config.run, epoch=self.installation.config.epoch,
                completed=progress['completed'], source_files=files, filesystem=fs)

    def stage_restore(self, runner, group, fs, directory):
        state = self.state(runner)
        if state is None:
            return None
        if not self.options.get('restore', False):
            raise ValueError('fresh input restore must hold source allocation before Runner initialization')
        _, worker = self.installation._context()
        if worker >= len(group['workers']):
            raise ValueError('Runner is absent from the input checkpoint membership')
        record = group['workers'][worker]
        state.validate_position(record)
        for item in record['files']:
            stage_file(fs, posixpath.join(group['directory'], 'worker-' + str(worker)), item, directory / item['file'])
        return dict(state=state, metadata=record, directory=directory)

    def initialize_saved_shot(self, runner, staged):
        if staged is None:
            return
        state, metadata = staged['state'], staged['metadata']
        state.dataset.next_index = metadata['next_index'] - 1
        state.dataset.end = False
        if not state.dataset.next_dataset(runner.session):
            raise ValueError('saved input shot could not be initialized')

    def stage_sources(self, group, fs, directory):
        for source in group['manifest']['sources']:
            for kind in ('roster', 'plan'):
                item = source[kind]
                stage_file(fs, group['directory'], item, directory / item['file'])

    def restore_sources(self, group, directory):
        for source in group['manifest']['sources']:
            self.call('eps_input_restore_source', source['handle'], str(directory / source['plan']['file']))

    def release_sources(self, group):
        for source in group['manifest']['sources']:
            self.call('eps_input_hold_source', source['handle'], 0)
        self.restore_pending = False

    def restore_iterator(self, runner, staged):
        if staged is None:
            return
        state, metadata = staged['state'], staged['metadata']
        self.call('eps_input_begin_generation', state.scope)
        runner.session.run(state.restore_op,
            feed_dict={state.prefix: str(staged['directory'] / 'iterator')})
        state.dataset.next_index = metadata['next_index']
        state.dataset.end = metadata['end']
        runner.passed_ident = metadata.get('passed_ident')
        if metadata['end']:
            # The original Runner raises this saved exception on its next
            # iteration when the final input shot has already been exhausted.
            import tensorflow as tf
            runner.ex = tf.errors.OutOfRangeError(None, None, 'restored input exhausted')
        self.restored_runners.add(id(runner))
