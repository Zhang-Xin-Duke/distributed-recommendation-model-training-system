"""Train DeepLOB on chronological FI-2010 windows through the parameter service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import tensorflow as tf

from financial_order_book_model import build_order_book_model, flat_order_book_step


def batch(features, labels, start, size):
    ends = np.arange(start, start + size)
    windows = ends[:, None] + np.arange(-99, 1)
    return features[windows, :, None], labels[ends]


def main():
    from eps_backend import Backend
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    cfg = json.loads(parser.parse_args().config.read_text())
    spec = cfg['order_book']
    worker, size = cfg['worker'], spec['batch_size']
    with open(spec['path'], 'rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != spec['sha256']:
            raise RuntimeError('FI-2010 input hash differs from the controller')
    with np.load(spec['path']) as data:
        features, labels = data['train_x'], data['train_y']
    gpu = cfg['device'].startswith('/GPU')
    tf.config.set_soft_device_placement(False)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    width = cfg['dimension']
    with tf.device(cfg['device']):
        initial, step = flat_order_book_step(build_order_book_model())
    count = int(initial.shape[0])
    keys = np.arange((count + width - 1) // width, dtype=np.uint64)
    feature = dict(table=1, feature=0, pooling='sequence', keys=keys,
                   offsets=np.array([0, len(keys)], dtype=np.uint64))
    backend = Backend(cfg['shards'], cfg['run'], worker, cfg['epoch'])
    stage = {name: [] for name in ('admission', 'parameter_pull', 'model_and_staging', 'update', 'completion')}
    max_lead = 0
    sampler = None
    with tempfile.TemporaryFile(mode='w+') as gpu_samples:
        try:
            backend.create_table(json.dumps(cfg['table']))
            backend.configure_training(cfg['coordinator'], 0)
            backend.lookup_layout_cache(True)
            backend.gradient_layout_cache(True)

            def compute(iteration, values):
                start = 99 + (iteration * spec['workers'] + worker) * size
                x, y = batch(features, labels, start, size)
                with tf.device(cfg['device']):
                    weights = tf.convert_to_tensor(values.reshape(-1)[:count])
                    return step(weights, tf.convert_to_tensor(x), tf.convert_to_tensor(y))

            token, values = backend.pull([feature], True)
            backend.release(token)
            for value in compute(0, values[0]):
                value.numpy()
            gate = Path(cfg['start_gate'])
            marker = Path(f'{gate}.worker-{worker}.ready')
            pending = marker.with_suffix(f'.pending-{os.getpid()}')
            pending.write_text(json.dumps(dict(run=cfg['run'], worker=worker, warmed_model=True)))
            pending.replace(marker)
            deadline = time.monotonic() + 180
            while not gate.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('DeepLOB start gate did not open')
                time.sleep(.005)
            if gpu:
                sampler = subprocess.Popen(
                    ['nvidia-smi', '--id=0', '--query-gpu=utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits', '-lms', '200'], stdout=gpu_samples)
            sync = cfg['update_mode'] == 'synchronous'
            for iteration in range(cfg['iterations']):
                t0 = time.monotonic()
                floor = backend.begin_training(iteration)
                t1 = time.monotonic()
                max_lead = max(max_lead, iteration - floor)
                if cfg['delay_ms'] and iteration % cfg['delay_every'] == 0:
                    time.sleep(cfg['delay_ms'] / 1000)
                t2 = time.monotonic()
                token, values = backend.pull([feature], sync)
                if sync:
                    backend.release(token)
                t3 = time.monotonic()
                loss, logits, gradient = compute(iteration, values[0])
                grad = np.zeros((len(keys), width), dtype=np.float32)
                grad.flat[:count] = gradient.numpy()
                if not np.isfinite(loss.numpy()) or not np.isfinite(grad).all():
                    raise RuntimeError('non-finite DeepLOB loss or gradient')
                t4 = time.monotonic()
                if sync:
                    backend.submit_synchronous(iteration, size, [dict(table=1, keys=keys, values=grad)])
                else:
                    backend.push(token, [0], [grad])
                t5 = time.monotonic()
                backend.complete_training(iteration)
                t6 = time.monotonic()
                for times, elapsed in zip(stage.values(), (t1-t0, t3-t2, t4-t3, t5-t4, t6-t5)):
                    times.append(elapsed)
            backend.join()
            finished = time.monotonic()
            if sampler is not None:
                sampler.terminate()
                sampler.wait(timeout=10)
                sampler = None
            gpu_samples.seek(0)
            samples = [[int(v.strip()) for v in line.split(',')] for line in gpu_samples if ',' in line]
            devices = dict(logits=logits.device, gradient=gradient.device)
            gpu_info = subprocess.check_output(
                ['nvidia-smi', '--id=0', '--query-gpu=uuid,name,driver_version',
                 '--format=csv,noheader'], text=True, timeout=10).strip() if gpu else None
            print(json.dumps(dict(worker=worker, completed=cfg['iterations'],
                                  update_mode=cfg['update_mode'], batch_sizes=[size]*cfg['iterations'],
                                  admission_wait_seconds=stage['admission'], max_admission_lead=max_lead,
                                  devices=devices, training_finished_monotonic=finished,
                                  stage_seconds=stage, gpu_samples=samples, gpu_identity=gpu_info)))
        except Exception as error:
            try:
                backend.abort_training(str(error))
            except Exception:
                pass
            raise
        finally:
            if sampler is not None:
                sampler.kill()
                sampler.wait()
            backend.close()


if __name__ == '__main__':
    main()
