"""Select a parameter-server policy before training and measure its full cost."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time

from tensorflow_consistency_training import trial
from ssh_training_worker import probe_remote
from tensorflow_strategy_workloads import DELAY_PATTERNS, delay_for, synthetic_workload, validate_workload, write_report


def select(pilots, target):
    passing = [(name, row) for name, row in pilots.items() if row['loss'] <= target]
    if not passing:
        return 'sync', 'no_pilot_passed'
    name, _ = min(passing, key=lambda item: (item[1]['wall_seconds'], item[0] != 'sync', item[0]))
    return name, 'fastest_passing_pilot'


def paired_results(records, candidates):
    results = {}
    for pattern in dict.fromkeys(row['pattern'] for row in records):
        rows = [row for row in records if row['pattern'] == pattern]
        comparisons = {}
        for candidate in candidates:
            pairs = [(row['runs']['auto'], row['runs'][candidate]) for row in rows]
            deltas = [auto['wall_seconds'] - fixed['wall_seconds'] for auto, fixed in pairs
                      if auto['passed'] and fixed['passed']]
            comparisons[candidate] = dict(both_passed=len(deltas),
                                          auto_passed=sum(a['passed'] for a, _ in pairs),
                                          fixed_passed=sum(b['passed'] for _, b in pairs),
                                          auto_faster=sum(d < 0 for d in deltas),
                                          median_auto_minus_fixed_seconds=statistics.median(deltas) if deltas else None)
        results[pattern] = comparisons
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--mode', default='auto', help='auto, compare, or one fixed candidate')
    parser.add_argument('--device', choices=('/CPU:0', '/GPU:0'), default='/CPU:0')
    parser.add_argument('--bounds', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--patterns', nargs='+', choices=DELAY_PATTERNS)
    parser.add_argument('--pilot-steps', type=int, default=8)
    parser.add_argument('--endpoints', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--target-loss', type=float, default=.60)
    parser.add_argument('--delay-ms', type=int, default=20)
    parser.add_argument('--burst-every', type=int, default=4)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--dataset', type=Path, help='optional remapped training and validation JSON')
    parser.add_argument('--remote-worker', nargs=4, metavar=('SSH_HOST', 'CHECKOUT', 'PYTHON', 'BUILD'),
                        help='run the second Worker on an existing SSH host')
    parser.add_argument('--model-out', type=Path, help='export inspected model state for one successful run')
    args = parser.parse_args()
    if args.patterns is None:
        args.patterns = list(DELAY_PATTERNS) if args.mode == 'compare' else ['balanced']
    if args.repeats is None:
        args.repeats = 3 if args.mode == 'compare' else 1
    candidates = {'sync': ('sequential', 'synchronous', 0),
                  'async': ('eventual', 'per_worker', 0)}
    candidates.update({f'bounded-{n}': ('bounded_delay', 'per_worker', n) for n in sorted(set(args.bounds))})
    if (args.mode not in ('auto', 'compare', *candidates)
            or any(not 1 <= n <= 32 for n in args.bounds)
            or min(args.pilot_steps, *args.endpoints, args.repeats) < 1
            or not 0 <= args.delay_ms <= 1000 or not 2 <= args.burst_every <= 32
            or not math.isfinite(args.target_loss) or args.target_loss <= 0):
        parser.error('invalid candidate, step budget, delay or validation target')
    if args.remote_worker and any(not Path(p).is_absolute() for p in args.remote_worker[1:]):
        parser.error('remote checkout, Python and build paths must be absolute')
    if args.model_out and (args.mode == 'compare' or args.repeats != 1 or len(set(args.patterns)) != 1
                           or args.model_out.resolve() == args.out.resolve() or args.model_out.exists()):
        parser.error('model export requires a new output path and one non-comparison run')
    build = args.build.resolve()
    for binary in ('eps_server', 'eps_training_coordinator', 'eps_training_control', 'eps_training_prepare'):
        if not (build / binary).is_file():
            parser.error(f'missing {build / binary}')
    dataset = json.loads(args.dataset.read_text()) if args.dataset else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(complete=False, config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  topology='two local Worker processes, two local parameter shards', records=[])
    if args.remote_worker:
        source = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        remote = probe_remote(*args.remote_worker)
        if remote['git_commit'] != commit:
            raise RuntimeError('remote Worker checkout differs from the controller')
        report.update(topology='two SSH-connected Worker hosts, two controller-local parameter shards',
                      remote=remote)
    write_report(args.out, report)
    endpoints = sorted(set(args.endpoints))
    with tempfile.TemporaryDirectory(prefix='eps-auto-policy-') as tmp:
        run_id = 0
        for repeat in range(args.repeats):
            workload = dataset if dataset is not None else synthetic_workload(args.seed + repeat, 'shared', 32, 8, 16, 64)
            metadata = validate_workload(workload)
            workload_hash = hashlib.sha256(json.dumps(workload, sort_keys=True).encode()).hexdigest()
            for pattern in dict.fromkeys(args.patterns):
                delay, every = delay_for(pattern, args.delay_ms, args.burst_every)
                fingerprint = None

                def measure(name, steps, export=False):
                    nonlocal fingerprint, run_id
                    started = time.monotonic()
                    root = Path(tmp) / str(run_id)
                    run_id += 1
                    root.mkdir()
                    policy, mode, bound = candidates[name]
                    result = trial(build, root, policy, steps, repeat, 2, delay, args.device,
                                   workload=workload, update_mode=mode, max_delay=bound,
                                   delay_every=every, slow_worker=repeat % 2,
                                   remote=[args.remote_worker] if args.remote_worker else ())
                    digest = hashlib.sha256(json.dumps(result['initial_state']['rows'], sort_keys=True).encode()).hexdigest()
                    if fingerprint is not None and digest != fingerprint:
                        raise RuntimeError('candidate started from a different parameter or optimizer state')
                    fingerprint = digest
                    if result['committed_samples'] != steps * 2 * metadata['local_batch_size']:
                        raise RuntimeError('candidate used a different sample budget')
                    if args.remote_worker and args.device == '/GPU:0':
                        uuids = {worker['runtime']['gpu']['uuid'] for worker in result['workers_report']}
                        if len(uuids) != 2:
                            raise RuntimeError('expected distinct GPUs on the two Worker hosts')
                    if export and result['final_loss'] <= args.target_loss:
                        args.model_out.parent.mkdir(parents=True, exist_ok=True)
                        write_report(args.model_out, dict(strategy=name, table=result['table'],
                                                          state=result['final_state']))
                    return dict(steps=steps, loss=result['final_loss'],
                                wall_seconds=time.monotonic()-started,
                                training_seconds=result['warmup_excluded_seconds'],
                                optimizer_steps=result['optimizer_steps'],
                                max_admission_lead=result['max_admission_lead'])

                arms = [*candidates, 'auto'] if args.mode == 'compare' else [args.mode]
                rng = random.Random(args.seed + repeat)
                rng.shuffle(arms)
                row = dict(pattern=pattern, repeat=repeat, dataset_sha256=workload_hash,
                           delay_ms=delay, delay_every=every, slow_worker=repeat % 2, runs={})
                for arm in arms:
                    started = time.monotonic()
                    selected, reason, pilots = arm, 'fixed', {}
                    if arm == 'auto':
                        order = list(candidates)
                        rng.shuffle(order)
                        pilots = {name: measure(name, args.pilot_steps) for name in order}
                        selected, reason = select(pilots, args.target_loss)
                    pilot_seconds = time.monotonic() - started if arm == 'auto' else 0.
                    attempts = []
                    for steps in endpoints:
                        attempts.append(measure(selected, steps, export=args.model_out is not None))
                        if attempts[-1]['loss'] <= args.target_loss:
                            break
                    row['runs'][arm] = dict(selected=selected, reason=reason, pilots=pilots,
                                            pilot_seconds=pilot_seconds, attempts=attempts,
                                            passed=attempts[-1]['loss'] <= args.target_loss,
                                            wall_seconds=time.monotonic()-started)
                    row['initial_parameter_sha256'] = fingerprint
                    print(pattern, repeat, arm, selected, row['runs'][arm]['passed'],
                          round(row['runs'][arm]['wall_seconds'], 3), flush=True)
                report['records'].append(row)
                write_report(args.out, report)
    if args.mode == 'compare':
        report['paired_results'] = paired_results(report['records'], candidates)
    report['complete'] = True
    write_report(args.out, report)


if __name__ == '__main__':
    main()
