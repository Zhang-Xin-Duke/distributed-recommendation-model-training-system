"""Compare synchronous and asynchronous training on paired workload endpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile

from tensorflow_consistency_training import trial

STRATEGIES = {
    'sync': ('sequential', 'synchronous'),
    'async': ('eventual', 'per_worker'),
    'bounded': ('bounded_delay', 'per_worker'),
    'per-worker-sequential': ('sequential', 'per_worker'),
}
DELAY_PATTERNS = ('balanced', 'persistent', 'bursty')


def synthetic_workload(seed, overlap, batch_size, bag_size, batch_count, key_pool):
    """Use identical local frequency distributions for shared and disjoint IDs."""
    def batch(rng, worker, size):
        offset = 11 + (2 * key_pool * worker if overlap == 'disjoint' else 0)
        examples = []
        for index in range(size):
            label = index % 2
            keys = [offset + label * key_pool + int(rng.random() ** 2 * key_pool)
                    for _ in range(bag_size)]
            examples.append({'keys': keys, 'label': float(label)})
        rng.shuffle(examples)
        return examples

    streams = []
    validation = []
    for worker in range(2):
        rng = random.Random(seed * 4 + worker)
        streams.append([batch(rng, worker, batch_size) for _ in range(batch_count)])
        validation.extend(batch(random.Random(seed * 4 + worker + 2), worker, 128))
    return {'batches': streams, 'validation': validation}


def validate_workload(data):
    batches, validation = data['batches'], data['validation']
    all_samples = [s for worker in batches for batch in worker for s in batch] + validation
    if any(not s['keys'] or any(type(k) is not int or not 11 <= k < 32768 for k in s['keys'])
           or not isinstance(s['label'], (int, float)) or not 0 <= s['label'] <= 1
           for s in all_samples):
        raise ValueError('examples require remapped integer keys in [11, 32768) and labels in [0, 1]')
    if len(batches) != 2 or not validation or any(not worker for worker in batches):
        raise ValueError('expected two nonempty Worker streams and a nonempty validation set')
    if any(not batch for worker in batches for batch in worker):
        raise ValueError('Worker batches cannot be empty')
    sizes = {len(batch) for worker in batches for batch in worker}
    if len(sizes) != 1:
        raise ValueError('use the same local batch size in both Worker streams')
    training_keys = []
    training_examples = set()
    for worker in batches:
        keys = set()
        for batch in worker:
            for sample in batch:
                keys.update(sample['keys'])
                training_examples.add((tuple(sample['keys']), sample['label']))
        training_keys.append(keys)
    matching_examples = sum((tuple(sample['keys']), sample['label']) in training_examples
                            for sample in validation)
    union = training_keys[0] | training_keys[1]
    if not union:
        raise ValueError('expected embedding keys')
    return {'embedding_key_jaccard': len(training_keys[0] & training_keys[1]) / len(union),
            'local_batch_size': sizes.pop(), 'validation_samples': len(validation),
            'validation_examples_matching_training': matching_examples}


def delay_for(pattern, delay_ms, burst_every):
    if pattern == 'balanced':
        return 0, 1
    if pattern == 'persistent':
        return delay_ms, 1
    return delay_ms * burst_every, burst_every


def write_report(path, report):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def summarize(trials, repeats, target, minimum_gain):
    workloads = {}
    for record in trials:
        key = (record['workload'], record['strategy'], record['repeat'])
        workloads.setdefault(key, []).append(record)
    result = []
    for name in sorted({key[0] for key in workloads}):
        endpoints = {}
        rows = {}
        for strategy in STRATEGIES:
            successes = {}
            final_losses = []
            for repeat in range(repeats):
                records = sorted(workloads[name, strategy, repeat],
                                 key=lambda record: record['committed_samples'])
                final_losses.append(records[-1]['final_loss'])
                passing = next((x for x in records if x['final_loss'] <= target), None)
                if passing:
                    successes[repeat] = passing
            endpoints[strategy] = successes
            times = [x['warmup_excluded_seconds'] for x in successes.values()]
            rows[strategy] = {'passed': len(times), 'attempted': repeats,
                              'median_passing_seconds': statistics.median(times) if times else None,
                              'median_final_budget_loss': statistics.median(final_losses),
                              'passing_endpoints': [
                                  {k: x[k] for k in ('repeat', 'committed_samples', 'final_loss',
                                                     'warmup_excluded_seconds')}
                                  for x in successes.values()]}
        comparisons = {}
        pairs = [('sync', 'async'), ('sync', 'bounded'), ('sync', 'per-worker-sequential'),
                 ('async', 'bounded'), ('per-worker-sequential', 'async'),
                 ('per-worker-sequential', 'bounded')]
        for baseline, strategy in pairs:
            reference = endpoints[baseline]
            contender = endpoints[strategy]
            gains = [(reference[i]['warmup_excluded_seconds'] - contender[i]['warmup_excluded_seconds'])
                     / reference[i]['warmup_excluded_seconds']
                     for i in sorted(reference.keys() & contender.keys())]
            decision = 'no_clear_advantage'
            if len(gains) == repeats and repeats >= 5:
                if all(gain >= minimum_gain for gain in gains):
                    decision = strategy
                elif all(gain <= -minimum_gain for gain in gains):
                    decision = baseline
            comparisons[f'{strategy}_vs_{baseline}'] = {
                'decision': decision, 'paired_relative_time_gains': gains,
                'both_passed_pairs': len(gains)}
        result.append({'workload': name, 'strategies': rows, 'paired_comparisons': comparisons})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('/CPU:0', '/GPU:0'), default='/GPU:0')
    parser.add_argument('--iterations', type=int, nargs='+', default=[8, 32, 128])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--patterns', nargs='+', choices=DELAY_PATTERNS, default=DELAY_PATTERNS)
    parser.add_argument('--overlap', nargs='+', choices=('disjoint', 'shared'),
                        default=('disjoint', 'shared'))
    parser.add_argument('--delay-ms', type=int, default=20)
    parser.add_argument('--burst-every', type=int, default=4)
    parser.add_argument('--max-delay', type=int, default=2)
    parser.add_argument('--target-loss', type=float, default=.5)
    parser.add_argument('--minimum-gain', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--bag-size', type=int, default=8)
    parser.add_argument('--key-pool', type=int, default=64)
    parser.add_argument('--batch-count', type=int, default=16)
    parser.add_argument('--dataset', type=Path, help='JSON with batches[worker][batch][example] and validation')
    parser.add_argument('--remote-worker', nargs=4, metavar=('SSH_HOST', 'CHECKOUT', 'PYTHON', 'BUILD'),
                        help='run Worker 1 on a second host via reverse SSH port forwarding')
    args = parser.parse_args()
    if args.remote_worker and any(not Path(p).is_absolute() for p in args.remote_worker[1:]):
        parser.error('remote checkout, Python and build paths must be absolute')
    if (not 2 <= args.batch_size <= 1024 or args.batch_size % 2
            or not 1 <= args.bag_size <= 128 or not 2 <= args.key_pool <= 2048
            or not 1 <= args.batch_count <= 128 or not 1 <= args.repeats <= 20
            or not 1 <= args.max_delay <= 32 or not 2 <= args.burst_every <= 32
            or not 0 <= args.delay_ms <= 1000 or not 0 < args.minimum_gain < 1
            or not math.isfinite(args.target_loss) or args.target_loss <= 0
            or any(not 1 <= n <= 10000 for n in args.iterations)):
        parser.error('invalid experiment bounds')
    endpoints = sorted(set(args.iterations))
    patterns = tuple(dict.fromkeys(args.patterns))
    overlaps = tuple(dict.fromkeys(args.overlap))
    build = args.build.resolve()
    for binary in ('eps_server', 'eps_training_coordinator', 'eps_training_control', 'eps_training_prepare'):
        if not (build / binary).is_file():
            parser.error(f'missing {build / binary}')
    external = json.loads(args.dataset.read_text()) if args.dataset else None
    if external is not None:
        validate_workload(external)
        overlaps = ('provided',)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    settings = vars(args).copy()
    for field in ('build', 'out', 'dataset'):
        settings[field] = str(settings[field]) if settings[field] is not None else None
    source = Path(__file__).resolve().parents[1]
    report = {'schema': 1, 'complete': False, 'config': settings, 'platform': platform.platform(),
              'git_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                                    text=True).strip(),
              'measurement': 'time to first passing tested endpoint after warmup on held-out validation',
              'timing': ('controller release through process exit (includes SSH and teardown)'
                         if args.remote_worker else 'gate release through last Worker training completion'),
              'trials': [], 'summary': []}
    if args.device == '/GPU:0':
        report['gpu'] = subprocess.check_output(
            ['nvidia-smi', '--id=0', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'], text=True, timeout=15).strip()
    write_report(args.out, report)
    with tempfile.TemporaryDirectory(prefix='eps-strategy-workloads-') as temp:
        for repeat in range(args.repeats):
            for overlap in overlaps:
                workload = external if external is not None else synthetic_workload(
                    args.seed + repeat, overlap, args.batch_size, args.bag_size,
                    args.batch_count, args.key_pool)
                metadata = validate_workload(workload)
                metadata['dataset_sha256'] = hashlib.sha256(
                    json.dumps(workload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                initial_state = None
                for pattern in patterns:
                    delay, every = delay_for(pattern, args.delay_ms, args.burst_every)
                    for point, iterations in enumerate(endpoints):
                        strategies = list(STRATEGIES)
                        shift = (repeat + point) % len(strategies)
                        for strategy in strategies[shift:] + strategies[:shift]:
                            policy, mode = STRATEGIES[strategy]
                            # A separate directory prevents a failed trial's artifacts from being reused.
                            root = Path(temp) / str(len(report['trials']))
                            root.mkdir()
                            record = trial(build, root, policy, iterations, repeat, 2, delay, args.device,
                                           workload=workload, update_mode=mode, max_delay=args.max_delay,
                                           delay_every=every, remote=args.remote_worker)
                            if record['committed_samples'] != iterations * 2 * metadata['local_batch_size']:
                                raise RuntimeError('trial did not reach the requested sample budget')
                            initial = record.pop('initial_state')
                            fingerprint = hashlib.sha256(
                                json.dumps(initial['rows'], sort_keys=True).encode()).hexdigest()
                            if initial_state is not None and fingerprint != initial_state:
                                raise RuntimeError('paired strategies did not start from identical parameter state')
                            initial_state = fingerprint
                            record.pop('worker_inputs')
                            for worker in record['workers_report']:
                                worker['input_state'].pop('dataset_json', None)
                            record.update(metadata, workload=f'{pattern}-{overlap}', strategy=strategy,
                                          initial_parameter_sha256=fingerprint)
                            report['trials'].append(record)
                            write_report(args.out, report)
                            print(json.dumps({k: record[k] for k in ('workload', 'strategy', 'repeat',
                                'committed_samples', 'final_loss', 'warmup_excluded_seconds')}),
                                flush=True)
    report['summary'] = summarize(report['trials'], args.repeats, args.target_loss, args.minimum_gain)
    report['complete'] = True
    write_report(args.out, report)


if __name__ == '__main__':
    main()
