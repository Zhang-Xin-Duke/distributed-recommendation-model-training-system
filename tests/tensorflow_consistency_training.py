"""Compare consistency policies with the same per-Worker Embedding/MLP updates.

Each point is a fresh, finite training run. Loss is evaluated on the complete
fixed training set after all Workers drain; timing excludes model warmup.
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import subprocess
import sys
import tempfile
import time

from rpc_processes import ready, stop
from ssh_training_worker import worker_command


def model_loss(state, samples, dimension, hidden):
    """Evaluate the same sum-pool/tanh MLP directly from quiescent PS rows."""
    import numpy as np
    rows = {r['key']: np.asarray(r['weights'], dtype=np.float64) for r in state['rows']}
    packed = (dimension * hidden + 2 * hidden + 1 + dimension - 1) // dimension
    flat = np.concatenate([rows[k] for k in range(packed)])
    end = dimension * hidden
    w1 = flat[:end].reshape(dimension, hidden)
    b1, w2, b2 = flat[end:end+hidden], flat[end+hidden:end+2*hidden], flat[end+2*hidden]
    loss = 0.
    for sample in samples:
        pooled = sum((rows[k] for k in sample['keys']), np.zeros(dimension))
        logit = float(np.tanh(pooled @ w1 + b1) @ w2 + b2)
        loss += max(logit, 0.) - sample['label'] * logit + math.log1p(math.exp(-abs(logit)))
    return loss / len(samples)


def trial(build, root, policy, iterations, repeat, workers, delay_ms, device='/CPU:0',
          *, workload=None, update_mode='per_worker', max_delay=2, delay_every=1, remote=(), slow_worker=None, order_book=None):
    slow_worker = workers - 1 if slow_worker is None else slow_worker
    dimension, hidden, count = 8, 4, 16
    packed = (dimension * hidden + 2 * hidden + 1 + dimension - 1) // dimension
    rng = random.Random(20261104)
    samples = [[{'keys': [packed + 32 * (s % 2) + rng.randrange(32) for _ in range(8)],
                 'label': float(s % 2)} for s in range(count)] for _ in range(workers)]
    batches = [[group] for group in samples]
    validation = [example for group in samples for example in group]
    if workload is not None:
        batches = workload['batches']
        validation = workload['validation']
    if order_book is not None:
        from order_book_strategy_workloads import validate_order_book
        dimension, packed = 256, (142435 + 255) // 256
        evaluate = lambda state: validate_order_book(state, order_book['path'], order_book)
    else:
        evaluate = lambda state: model_loss(state, validation, dimension, hidden)
    training = [example for sequence in batches for batch in sequence for example in batch]
    keys = sorted(set(range(packed)) | {k for s in training + validation for k in s['keys']})
    synchronous = update_mode == 'synchronous'
    admission_delay = max_delay if policy == 'bounded_delay' else 0
    run = f'mlp-{update_mode}-{policy}-{iterations}-{repeat}'
    home = root / run
    home.mkdir()
    env = os.environ.copy()
    source = pathlib.Path(__file__).resolve().parents[1]
    env['PYTHONPATH'] = os.pathsep.join([str(build/'python'), str(source/'adapters/entropy'),
                                        env.get('PYTHONPATH', '')])
    env.update(CUDA_VISIBLE_DEVICES='0' if device=='/GPU:0' else '', TF_CPP_MIN_LOG_LEVEL='2')
    table = {'id': 1, 'dimension': dimension, 'dtype': 'float32',
             'optimizer': {'kind': 'sparse_lazy_adam',
                           'learning_rate': .02 if order_book is None else order_book.get('learning_rate', .02),
                           'beta1': .9, 'beta2': .999, 'epsilon': 1e-8,
                           'initial_beta1_power': .9, 'initial_beta2_power': .999},
             'initialization': {'min': -.1, 'max': .1, 'seed': 42}}
    if order_book is not None and order_book.get('initial_weights'):
        # Store trainable deltas; all Workers add the identical immutable initial weights.
        table['initialization'] = {'min': 0., 'max': 0., 'seed': 42}
    processes, logs = [], []

    def launch(command, name, config):
        path = home / (name+'.json')
        path.write_text(json.dumps(config))
        log = open(home/(name+'.log'), 'w+')
        logs.append(log)
        process = subprocess.Popen([*map(str, command), '--config', str(path)],
                                   stdout=subprocess.PIPE, stderr=log, text=True, env=env)
        processes.append(process)
        return process

    try:
        shards = []
        for shard in range(2):
            shard_config = {
                'schema_version': 1, 'run_id': run, 'listen': '127.0.0.1:0',
                'partition': {'shard_id': shard, 'shard_count': 2, 'epoch': 1},
                'tables': [table], 'cache': {'enabled': False},
                'snapshot_directory': str(home/'snapshots')}
            if synchronous:
                shard_config['synchronous_training'] = {
                    'workers': workers, 'reduction': 'sample_mean', 'round_timeout_ms': 20000}
            shards.append(ready(launch([build/'eps_server'], f'shard-{shard}', shard_config)))
        coordinator = ready(launch([build/'eps_training_coordinator'], 'coordinator', {
            'schema_version': 1, 'run': run, 'workers': workers, 'epoch': 1,
            'consistency': policy, 'max_delay': admission_delay}))
        settings = {'run': run, 'shards': shards, 'coordinator': coordinator, 'epoch': 1, 'keys': keys,
                    'synchronous': synchronous}
        if workload is not None or order_book is not None:
            prepared = launch([build/'eps_training_prepare'], 'prepare', {
                **settings, 'rows': max(keys) + 1, 'dimension': dimension,
                'resident': False, 'workers': workers})
            prepared.communicate(timeout=60)
            if prepared.returncode:
                raise RuntimeError('parameter initialization failed')
        admin = home/'admin.json'
        admin.write_text(json.dumps(settings))

        def inspect():
            result = subprocess.run([str(build/'eps_training_control'), '--config', str(admin),
                                     '--action', 'inspect'], env=env, capture_output=True,
                                    text=True, timeout=20, check=True)
            return json.loads(result.stdout)

        gate = home/'start'
        configs = [{**settings, 'table': table, 'worker': w, 'dimension': dimension,
                    'hidden': hidden, 'iterations': iterations, 'batches': batches[w],
                    'update_mode': update_mode, 'start_gate': str(gate),
                    'delay_ms': delay_ms if w == slow_worker else 0, 'delay_every': delay_every,
                    'record_stage_times': workload is not None, 'device': device}
                   for w in range(workers)]
        if order_book is not None:
            for config in configs:
                config['order_book'] = order_book
        entry = 'order_book_training_worker.py' if order_book is not None else 'tensorflow_training_worker.py'
        children = []
        for w, config in enumerate(configs):
            if 1 <= w <= len(remote):
                log = open(home/f'worker-{w}.log', 'w+')
                logs.append(log)
                child = subprocess.Popen(worker_command(*remote[w - 1], [*shards, coordinator]),
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=log, text=True)
                processes.append(child)
                child.stdin.write(json.dumps(config) + '\n')
                child.stdin.flush()
            else:
                child = launch([sys.executable, source/'examples'/entry],
                               f'worker-{w}', config)
            children.append(child)
        deadline = time.monotonic()+60
        for w, child in enumerate(children):
            if 1 <= w <= len(remote):
                assert json.loads(ready(child, timeout=60)) == dict(run=run, worker=w, warmed_model=True)
                continue
            marker = pathlib.Path(str(gate)+f'.worker-{w}.ready')
            while not marker.exists():
                if child.poll() is not None:
                    raise RuntimeError(f'Worker {w} exited before warmup: {child.returncode}')
                if time.monotonic() > deadline:
                    raise TimeoutError('model warmup timed out')
                time.sleep(.01)
            assert json.loads(marker.read_text()) == dict(run=run, worker=w, warmed_model=True)
        initial = inspect()
        assert initial['completed'] == [0]*workers and not initial['active'], initial
        initial_evaluation = evaluate(initial)
        initial_loss = initial_evaluation['loss'] if order_book is not None else initial_evaluation
        started = time.monotonic()
        temporary = gate.with_suffix('.tmp')
        temporary.write_text('start')
        temporary.replace(gate)
        for child in children[1:1 + len(remote)]:
            child.stdin.write('start\n')
            child.stdin.close()
            child.stdin = None
        outputs = []
        for child in children:
            budget = 120 + 2 * iterations * delay_ms / (1000 * delay_every)
            if order_book is not None:
                budget = max(budget, min(1500, iterations*3))
            stdout, _ = child.communicate(timeout=budget)
            if child.returncode:
                raise RuntimeError(f'model Worker failed: {child.returncode}')
            outputs.append(json.loads(stdout))
        process_elapsed = time.monotonic()-started
        elapsed = (process_elapsed if remote
                   else max(w['training_finished_monotonic'] for w in outputs)-started)
        assert 0 < elapsed <= process_elapsed
        final = inspect()
        assert final['completed'] == [iterations]*workers, final
        assert final['completed_tasks'] == final['granted_tasks'] == iterations*workers, final
        assert not final['active'] and not final['waiting'] and not final['failed'], final
        assert all(s['live_tickets'] == 0 for s in final['shard_stats']), final
        optimizer_steps = iterations if synchronous else iterations * workers
        if synchronous:
            for shard in final['shard_stats']:
                progress = shard['synchronous']
                assert progress['next_round'] == iterations and not progress['failed'], progress
                assert progress['received_workers'] == progress['retained_bytes'] == 0, progress
        for row in final['rows']:
            if row['key'] < packed:
                assert row['row_updates'] == optimizer_steps, row
            assert all(math.isfinite(v) for f in ('weights', 'first_moment', 'second_moment') for v in row[f])
        for output in outputs:
            assert output['update_mode'] == update_mode and output['completed'] == iterations
            if device=='/GPU:0':
                assert output['devices'] and all('GPU:' in value.upper() for value in output['devices'].values()),output
            if policy != 'eventual':
                assert output['max_admission_lead'] <= admission_delay, output
        evaluation = evaluate(final)
        loss = evaluation['loss'] if order_book is not None else evaluation
        assert math.isfinite(loss), loss
        if workload is None and order_book is None:
            assert loss < initial_loss, (loss, initial_loss)
        committed_samples = sum(sum(w['batch_sizes']) for w in outputs)
        return {'policy': policy, 'iterations_per_worker': iterations, 'repeat': repeat,
                'workers': workers, 'parameter_shards': 2, 'model': 'embedding_tanh_mlp',
                'dimension': dimension, 'hidden': hidden, 'table': table,
                'worker_inputs': samples if workload is None else batches,
                'optimizer_step': ('one_sample_weighted_update_per_round' if synchronous
                                   else 'one_local_batch_mean_update_per_worker'),
                'optimizer_steps': optimizer_steps, 'committed_samples': committed_samples,
                'samples_per_second': committed_samples/elapsed,
                'update_mode': update_mode, 'max_delay': admission_delay,
                'delay_every': delay_every, 'loss_split': 'training' if workload is None and order_book is None else 'validation',
                'initial_loss': initial_loss, 'final_loss': loss,
                'warmup_excluded_seconds': elapsed,
                **({'initial_training_loss': initial_loss, 'final_training_loss': loss}
                   if workload is None and order_book is None else {}),
                'process_exit_seconds': process_elapsed,
                'controlled_slow_worker_delay_ms': delay_ms,
                'admission_wait_worker_seconds': sum(sum(w['admission_wait_seconds']) for w in outputs),
                'max_admission_lead': max(w['max_admission_lead'] for w in outputs),
                'updates_per_second': optimizer_steps/elapsed,
                'workers_report': outputs, 'initial_state': initial, 'final_state': final,
                **({'initial_validation': initial_evaluation, 'validation': evaluation}
                   if order_book is not None else {})}
    except Exception:
        for log in logs:
            log.flush(); log.seek(0)
            print(log.read()[-5000:], file=sys.stderr)
        raise
    finally:
        for process in reversed(processes):
            stop(process)
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', required=True, type=pathlib.Path)
    p.add_argument('--out', required=True, type=pathlib.Path)
    p.add_argument('--iterations', default='8', help='comma-separated fresh-run endpoints')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--delay-ms', type=int, default=20)
    p.add_argument('--target-loss', type=float, default=.5)
    p.add_argument('--device', choices=['/CPU:0','/GPU:0'], default='/CPU:0')
    a = p.parse_args()
    endpoints = sorted(set(map(int, a.iterations.split(','))))
    if (not endpoints or min(endpoints)<1 or max(endpoints)>10000 or not 1<=a.repeats<=20
            or not 2<=a.workers<=8 or not 0<=a.delay_ms<=1000 or not 0<a.target_loss<1):
        p.error('invalid model policy experiment bounds')
    result = {'test': 'tensorflow_consistency_model_training', 'execution_device': 'CUDA' if a.device=='/GPU:0' else 'CPU',
              'complete': False, 'platform': platform.platform(), 'machine': platform.machine(),
              'host_logical_cpus': os.cpu_count(), 'tensorflow_threads_per_worker': {'inter': 2, 'intra': 2},
              'source_sha256': {name: hashlib.sha256((pathlib.Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                  for name in ('examples/tensorflow_training_worker.py', 'tests/tensorflow_consistency_training.py')},
              'target_training_loss': a.target_loss,
              'measurement': 'quiescent endpoints of independent fresh runs; not exact first crossing',
              'trials': [], 'first_tested_endpoints_meeting_target': []}
    if a.device=='/GPU:0':
        result['gpu']=subprocess.run(['nvidia-smi','--id=0',
            '--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],
            text=True,capture_output=True,check=True,timeout=15).stdout.strip()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='eps-tf-policies-') as tmp:
        for repeat in range(a.repeats):
            for point, iterations in enumerate(endpoints):
                policies = ['sequential', 'bounded_delay', 'eventual']
                shift = (repeat+point) % len(policies)
                for policy in policies[shift:]+policies[:shift]:
                    record = trial(a.build.resolve(), pathlib.Path(tmp), policy, iterations,
                                   repeat, a.workers, a.delay_ms,a.device)
                    result['trials'].append(record)
                    if record['final_training_loss'] <= a.target_loss and not any(
                            r['policy']==policy and r['repeat']==repeat
                            for r in result['first_tested_endpoints_meeting_target']):
                        result['first_tested_endpoints_meeting_target'].append({
                            k: record[k] for k in ('policy', 'repeat', 'iterations_per_worker',
                                                   'warmup_excluded_seconds', 'final_training_loss')})
                    a.out.write_text(json.dumps(result, indent=2)+'\n')
                    print(json.dumps({k: record[k] for k in ('policy', 'iterations_per_worker',
                        'repeat', 'final_training_loss', 'warmup_excluded_seconds',
                        'admission_wait_worker_seconds', 'max_admission_lead')}), flush=True)
    result['complete'] = True
    a.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
