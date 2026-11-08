"""Prepare a chronological FI-2010 split and train a single-device TensorFlow baseline."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))


def prepare(source, output, features):
    with zipfile.ZipFile(source) as archive, archive.open('Train_Dst_NoAuction_DecPre_CF_7.txt') as stream:
        lines = io.TextIOWrapper(stream, encoding='ascii').readlines()
    x = np.stack([np.fromstring(row, sep=' ', dtype=np.float32) for row in lines[:40]], axis=1)
    y = np.fromstring(lines[-5], sep=' ', dtype=np.float32).astype(np.int32) - 1
    split = int(len(y)*.8)
    train_end = split-100
    if features == 'relative':
        mid = (x[:, 0:1] + x[:, 2:3])*.5
        if np.any(mid <= 0):
            raise ValueError('nonpositive midpoint in FI-2010')
        x[:, ::2] = (x[:, ::2]-mid)/mid
        x[:, 1::2] = np.log1p(x[:, 1::2])
        x = (x-x[:train_end].mean(0))/np.maximum(x[:train_end].std(0), 1e-8)
    np.savez(output, train_x=x[:train_end], train_y=y[:train_end],
             val_x=x[split:], val_y=y[split:])
    print(json.dumps(dict(features=features, split=split, train_rows=train_end, validation_rows=len(y)-split,
                          train_class_counts=np.bincount(y[99:train_end]).tolist(),
                          validation_class_counts=np.bincount(y[split+99:]).tolist())))


def metrics(labels, logits):
    import tensorflow as tf
    counts = np.zeros((3, 3), dtype=np.int64)
    np.add.at(counts, (labels, logits.argmax(1)), 1)
    totals = counts.sum(0)+counts.sum(1)
    scores = np.divide(2*counts.diagonal(), totals, out=np.zeros(3), where=totals>0)
    recall = counts.diagonal()/np.maximum(counts.sum(1), 1)
    return dict(loss=float(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits).numpy().mean()),
                macro_f1=float(scores.mean()), recall=recall.tolist(), confusion=counts.tolist())


def train(args):
    import tensorflow as tf
    from financial_order_book_model import build_order_book_model
    from order_book_training_worker import batch_at

    tf.config.set_soft_device_placement(sys.platform == "darwin")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    gpus = tf.config.list_physical_devices('GPU')
    if args.device == '/GPU:0' and not gpus:
        raise RuntimeError('GPU required; no GPU detected')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.data) as d:
        tx, ty, vx, vy = (d[k] for k in ('train_x','train_y','val_x','val_y'))
    ends = np.linspace(99, len(vy)-1, min(args.val_windows,len(vy)-99), dtype=np.int32)
    counts = np.bincount(ty[99:], minlength=3)
    class_weights = (counts.sum()/(3*counts) if args.balance else np.ones(3)).astype(np.float32)
    with tf.device(args.device):
        model = build_order_book_model()
        initial = np.concatenate([v.numpy().reshape(-1) for v in model.trainable_variables])
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, epsilon=1e-8)
        optimizer.build(model.trainable_variables)

        @tf.function(reduce_retracing=True)
        def update(x,y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
                loss = tf.reduce_mean(losses*tf.gather(class_weights,y))
            optimizer.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables))
            return loss

        @tf.function(reduce_retracing=True)
        def predict(x):
            return model(x, training=False)

        report = dict(device=args.device, tensorflow=tf.__version__, seed=args.seed, batch=args.batch_size,
                      learning_rate=args.learning_rate, evaluation_windows=len(ends),
                      target_loss=args.target_loss, target_macro_f1=args.target_macro_f1,
                      class_weights=class_weights.tolist(), points=[],
                      data_sha256=hashlib.file_digest(args.data.open('rb'),'sha256').hexdigest())
        if args.initial_weights:
            np.save(args.initial_weights, initial)
        training_seconds = 0.
        streak = 0
        for step in range(args.steps+1):
            if step:
                epoch_step = (step-1) % (len(ty[99:])//args.batch_size)
                if epoch_step == 0:
                    order = rng.permutation(len(ty)-99)+99
                start = time.monotonic()
                x,y = batch_at(tx,ty,order[epoch_step*args.batch_size:(epoch_step+1)*args.batch_size])
                loss = float(update(x,y).numpy())
                training_seconds += time.monotonic()-start
                if not np.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
            if step % args.eval_every and step != args.steps:
                continue
            outputs = np.concatenate([predict(batch_at(vx,vy,ends[i:i+args.batch_size])[0]).numpy()
                                      for i in range(0,len(ends),args.batch_size)])
            point = metrics(vy[ends],outputs)
            passed = (point['loss']<=args.target_loss and point['macro_f1']>=args.target_macro_f1
                      and min(point['recall'])>=.1)
            streak = streak+1 if passed else 0
            point.update(step=step, samples=step*args.batch_size, training_seconds=training_seconds,
                         passed=passed, consecutive_passes=streak)
            report['points'].append(point)
            report['stable_pass'] = streak>=2
            pending=args.out.with_suffix('.tmp')
            pending.write_text(json.dumps(report,indent=2))
            pending.replace(args.out)
            print(json.dumps(point),flush=True)
            if streak>=2:
                break


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--prepare',type=Path)
    p.add_argument('--features',choices=['decpre','relative'],default='relative')
    p.add_argument('--device',choices=['/CPU:0','/GPU:0'],default='/GPU:0')
    p.add_argument('--out',type=Path)
    p.add_argument('--initial-weights',type=Path)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--val-windows',type=int,default=2048)
    p.add_argument('--seed',type=int,default=20261108)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--target-loss',type=float,default=.95)
    p.add_argument('--target-macro-f1',type=float,default=.45)
    p.add_argument('--balance',action='store_true')
    a=p.parse_args()
    if a.prepare:
        prepare(a.prepare,a.data,a.features)
    elif not a.out or min(a.batch_size,a.steps,a.eval_every,a.val_windows)<1:
        p.error('training requires --out and positive batch, step and evaluation sizes')
    else:
        train(a)


if __name__=='__main__':
    main()
