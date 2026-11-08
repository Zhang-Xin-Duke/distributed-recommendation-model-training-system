"""Compare parameter-server policies on a fixed FI-2010 order-book workload."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np

from ssh_training_worker import probe_remote
from tensorflow_consistency_training import trial
from tensorflow_strategy_workloads import STRATEGIES, summarize, write_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from order_book_training_worker import batch
from financial_order_book_model import build_order_book_model, flat_order_book_step


def prepare(archive, output, train_rows, validation_rows):
    def read(prefix, rows):
        with zipfile.ZipFile(archive) as source, source.open(
                f'{prefix}_Dst_NoAuction_DecPre_CF_7.txt') as raw:
            stream = io.TextIOWrapper(raw, encoding='ascii')
            features, last = [], []
            for index, line in enumerate(stream):
                if index < 40:
                    features.append(np.fromstring(line, sep=' ', dtype=np.float32)[:rows])
                else:
                    last.append(line)
                    if len(last) > 5:
                        last.pop(0)
        labels = np.fromstring(last[0], sep=' ', dtype=np.float32)[:rows] - 1
        result = np.asarray(features).T.copy()
        if result.shape != (rows, 40) or labels.shape != (rows,) or not np.isin(labels, (0, 1, 2)).all():
            raise ValueError('invalid or insufficient FI-2010 rows')
        return result, labels.astype(np.int32)

    train_x, train_y = read('Train', train_rows)
    val_x, val_y = read('Test', validation_rows)
    np.savez(output, train_x=train_x, train_y=train_y, val_x=val_x, val_y=val_y)


def weights(state, name='weights'):
    return np.asarray([row[name] for row in sorted(state['rows'], key=lambda row: row['key'])],
                      dtype=np.float32).reshape(-1)


def validate_order_book(state, path):
    import tensorflow as tf

    with np.load(path) as data:
        features, labels = data['val_x'], data['val_y']
    with tf.device('/CPU:0'):
        model = build_order_book_model()
        sizes = [int(np.prod(v.shape)) for v in model.trainable_variables]
        values = weights(state)[:sum(sizes)]
        for variable, value in zip(model.trainable_variables, np.split(values, np.cumsum(sizes)[:-1])):
            variable.assign(value.reshape(variable.shape))
        losses, predictions = [], []
        for start in range(99, len(labels), 32):
            x, y = batch(features, labels, start, min(32, len(labels)-start))
            logits = model(x, training=False)
            losses.extend(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits).numpy())
            predictions.extend(tf.argmax(logits, axis=1).numpy())
    counts = np.zeros((3, 3), dtype=int)
    np.add.at(counts, (labels[99:], predictions), 1)
    denom = counts.sum(0) + counts.sum(1)
    f1 = np.divide(2 * counts.diagonal(), denom, out=np.zeros(3), where=denom != 0)
    return dict(loss=float(np.mean(losses)), macro_f1=float(np.mean(f1)), confusion=counts.tolist())


def reference(record, path, size, workers, device):
    import tensorflow as tf

    with np.load(path) as data:
        x, y = batch(data['train_x'], data['train_y'], 99, workers * size)
    with tf.device(device):
        initial, step = flat_order_book_step(build_order_book_model())
        count = int(initial.shape[0])
        before = weights(record['initial_state'])
        weights_tensor = tf.constant(before[:count])
        gradients = []
        for worker in range(workers):
            start = worker * size
            _, _, gradient = step(weights_tensor, tf.constant(x[start:start+size]),
                                  tf.constant(y[start:start+size]))
            gradients.append(gradient)
        gradient = np.zeros_like(before)
        gradient[:count] = (tf.add_n(gradients) / workers).numpy()
    # SparseLazyAdam puts epsilon outside sqrt of the uncorrected second moment.
    m, v = .1 * gradient, .001 * gradient**2
    expected = before - (.02 * np.sqrt(.001) / .1) * m / (np.sqrt(v) + 1e-8)
    errors = {}
    for name, target in (('weights', expected), ('first_moment', m), ('second_moment', v)):
        actual = weights(record['final_state'], name)
        errors[name] = float(np.max(np.abs(actual - target)))
        np.testing.assert_allclose(actual, target, rtol=.005, atol=3e-5, err_msg=name)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path, help='author FI-2010 data.zip')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--train-rows', type=int, default=100000)
    parser.add_argument('--validation-rows', type=int, default=355)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--device', choices=('/CPU:0', '/GPU:0'), default='/GPU:0')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--iterations', type=int, nargs='+', default=[8, 24])
    parser.add_argument('--target-loss', type=float, default=.95)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--delay-ms', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=.02)
    parser.add_argument('--shuffle-seed', type=int,
                        help='shuffle training windows within each epoch; permits repeated epochs')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--sync-pilot', action='store_true',
                        help='run only synchronous trials to calibrate a quality target')
    parser.add_argument('--remote-worker', nargs=4, action='append', default=[])
    args = parser.parse_args()
    if args.prepare:
        args.data.parent.mkdir(parents=True, exist_ok=True)
        prepare(args.prepare, args.data, args.train_rows, args.validation_rows)
        return
    if not args.build or not args.out:
        parser.error('--build and --out are required for training')
    if args.reference and args.iterations != [1]:
        parser.error('--reference requires --iterations 1')
    if args.reference and (args.shuffle_seed is not None or args.learning_rate != .02):
        parser.error('--reference requires the default chronological batch and learning rate')
    if (not np.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            (args.shuffle_seed is not None and args.shuffle_seed < 0)):
        parser.error('learning rate must be positive and shuffle seed must be non-negative')
    hosts = [spec[0] for spec in args.remote_worker]
    if len(set(hosts)) != len(hosts) or len(hosts) > 5:
        parser.error('use at most five distinct remote Worker hosts')
    workers = 1 + len(hosts) if hosts else 2
    if min(args.batch_size, *args.iterations, args.repeats) < 1 or not np.isfinite(args.target_loss) or args.target_loss <= 0:
        parser.error('batch size, iterations and repeats must be positive')
    if args.device == '/GPU:0':
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
    path = args.data.resolve()
    with np.load(path) as data:
        rows = len(data['train_y'])
        if rows < 100 or (args.shuffle_seed is None and
                          99 + max(args.iterations) * workers * args.batch_size > rows):
            parser.error('training data does not cover the requested sample budget')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    report = dict(complete=False, model='deeplob_style', workers=workers,
                  batch_size=args.batch_size, iterations=args.iterations, repeats=args.repeats,
                  shuffle_seed=args.shuffle_seed, learning_rate=args.learning_rate,
                  data_sha256=digest, target_loss=args.target_loss,
                  timing='start gate to worker process exit for remote runs',
                  remote_hosts={spec[0]: probe_remote(*spec) for spec in args.remote_worker}, trials=[])
    import subprocess
    source = Path(__file__).resolve().parents[1]
    report['git_commit'] = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if any(remote['git_commit'] != report['git_commit'] for remote in report['remote_hosts'].values()):
        raise RuntimeError('a remote checkout differs from the controller')
    write_report(args.out, report)
    spec = dict(path=str(path), batch_size=args.batch_size, workers=workers, sha256=digest,
                shuffle_seed=args.shuffle_seed, learning_rate=args.learning_rate)
    with tempfile.TemporaryDirectory(prefix='eps-deeplob-') as temp:
        for repeat in range(args.repeats):
            names = ['sync'] if args.reference or args.sync_pilot else list(STRATEGIES)
            shift = repeat % len(names)
            for iterations in sorted(set(args.iterations)):
                for name in names[shift:] + names[:shift]:
                    policy, mode = STRATEGIES[name]
                    root = Path(temp) / str(len(report['trials']))
                    root.mkdir()
                    result = trial(args.build.resolve(), root, policy, iterations, repeat, workers,
                                   0 if args.reference else args.delay_ms, args.device, update_mode=mode,
                                   remote=args.remote_worker, slow_worker=(workers-1+repeat)%workers, order_book=spec)
                    if args.reference:
                        result['reference_max_absolute_error'] = reference(result, path, args.batch_size, workers, args.device)
                    if hosts and args.device == '/GPU:0':
                        uuids = {w['gpu_identity'].split(',')[0] for w in result['workers_report']}
                        if len(uuids) != workers:
                            raise RuntimeError('expected distinct GPU identities')
                    for key in ('initial_state', 'final_state', 'worker_inputs'):
                        result.pop(key)
                    result['model'] = 'deeplob_style'
                    result['strategy'] = name
                    result['workload'] = 'fi2010'
                    report['trials'].append(result)
                    write_report(args.out, report)
                    print(json.dumps(dict(strategy=name, repeat=repeat, loss=result['final_loss'],
                                          elapsed=result['warmup_excluded_seconds'],
                                          reference=result.get('reference_max_absolute_error'))), flush=True)
    if not args.reference and not args.sync_pilot:
        report['summary'] = summarize(report['trials'], args.repeats, args.target_loss, .05)
    report['complete'] = True
    write_report(args.out, report)


if __name__ == '__main__':
    main()
