"""Embedding + two-layer MLP training through distributed parameter shards.

All trainable parameters, including packed MLP weights/biases, live in one
row-sharded table. Host staging bridges remote parameter RPCs to TensorFlow;
the configured device executes model forward/backward. Updates use either a
sample-weighted synchronous round or one local-batch-mean update per Worker.
"""
import argparse
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    import numpy as np
    import tensorflow as tf
    from eps_backend import Backend

    tf.config.set_soft_device_placement(False)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    device = config.get("device", "/CPU:0")
    gpu = device.upper().startswith("/GPU:")
    if gpu and (not devices or int(device.split(":")[-1]) >= len(devices)):
        raise RuntimeError("requested model GPU is unavailable")
    if gpu:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    if config.get('deterministic_model',False):
        tf.config.experimental.enable_op_determinism()
    dimension = int(config["dimension"])
    hidden = int(config.get("hidden", 8))
    iterations = int(config["iterations"])
    update_mode = config.get('update_mode', 'synchronous')
    if update_mode not in ('synchronous', 'per_worker'):
        raise ValueError('unknown model update mode')
    synchronous = update_mode == 'synchronous'
    if not synchronous and (config.get('journal_directory') or config.get('journal_mirror')):
        raise ValueError('retained-gradient recovery requires synchronous updates')
    delay_ms, delay_every = int(config.get('delay_ms', 0)), int(config.get('delay_every', 1))
    if not 0 <= delay_ms <= 60000 or delay_every < 1:
        raise ValueError('invalid controlled Worker delay')
    if not 1 <= dimension <= 4096 or not 1 <= hidden <= 1024 or not 1 <= iterations <= 1000000:
        raise ValueError("invalid model dimensions or iteration count")
    packed_size = dimension * hidden + hidden + hidden + 1
    dense_rows = (packed_size + dimension - 1) // dimension
    dense_keys = np.arange(dense_rows, dtype=np.uint64)
    start_iteration = int(config.get("start_iteration", 0))
    if not 0 <= start_iteration < (1 << 63) - iterations:
        raise ValueError("invalid input start iteration")
    batches = config.get("batches", [config.get("samples", [])])
    if not isinstance(batches, list) or not 1 <= len(batches) <= 65536:
        raise ValueError("invalid batch sequence")
    dataset_json = json.dumps(batches, sort_keys=True, separators=(",", ":"), allow_nan=False)
    dataset_hash = hashlib.sha256(dataset_json.encode()).hexdigest()
    model = dict(kind="embedding_tanh_mlp", dimension=dimension, hidden=hidden)

    def input_state(iteration):
        return dict(kind="cyclic-json-batches-v1", dataset_json=dataset_json,
                    dataset_sha256=dataset_hash, next_iteration=iteration,
                    next_batch=iteration % len(batches), model=model)

    if start_iteration or "input_state" in config:
        if config.get("input_state") != input_state(start_iteration):
            raise ValueError("restored dataset, model or input cursor mismatch")
    prepared = []
    for samples in batches:
        if not samples or len(samples) > 65536:
            raise ValueError("invalid local batch size")
        keys, segments = [], []
        for sample, example in enumerate(samples):
            if not example["keys"] or not 0 <= example["label"] <= 1:
                raise ValueError("invalid training example")
            for key in example["keys"]:
                if not isinstance(key, int) or key < dense_rows or key >= (1 << 63):
                    raise ValueError("embedding IDs overlap packed MLP rows or exceed supported range")
                keys.append(key)
                segments.append(sample)
        keys = np.asarray(keys, dtype=np.uint64)
        features = [dict(table=1, feature=0, pooling="sequence", keys=keys,
                         offsets=np.array([0, len(keys)], dtype=np.uint64)),
                    dict(table=1, feature=1, pooling="sequence", keys=dense_keys,
                         offsets=np.array([0, len(dense_keys)], dtype=np.uint64))]
        unique_keys, inverse = np.unique(np.concatenate((keys, dense_keys)), return_inverse=True)
        with tf.device(device):
            segment_ids = tf.constant(segments, dtype=tf.int32)
            labels = tf.constant([s["label"] for s in samples], dtype=tf.float32)
        prepared.append((samples, features, unique_keys, inverse, segment_ids, labels))

    @tf.function(autograph=False)
    def forward_backward(embedding, dense, segment_ids, labels):
        with tf.device(device), tf.GradientTape() as tape:
            tape.watch((embedding, dense))
            flat = tf.reshape(dense, [-1])
            stop = dimension * hidden
            w1 = tf.reshape(flat[:stop], [dimension, hidden])
            b1 = flat[stop:stop + hidden]
            w2 = flat[stop + hidden:stop + 2 * hidden]
            b2 = flat[stop + 2 * hidden]
            pooled = tf.math.unsorted_segment_sum(embedding, segment_ids, tf.shape(labels)[0])
            activations = tf.math.tanh(tf.linalg.matmul(pooled, w1) + b1)
            logits = tf.linalg.matvec(activations, w2) + b2
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        gradients = tape.gradient(loss, (embedding, dense))
        return loss, logits, gradients[0], gradients[1]

    backend = Backend(config["shards"], config["run"], config["worker"], config.get("epoch", 1))
    losses, observed_devices, first_gradients, times = [], {}, None, []
    journal, mirror, replayed = None, None, []
    stage_times={name:[] for name in ('admission','controlled_delay','parameter_pull',
        'model_and_gradient_staging','journal_io','gradient_replication','synchronous_update','task_completion')} if config.get('record_stage_times',False) else None
    read_retries=0
    admission_wait, max_lead = [], 0
    read_retry_ms=int(config.get('read_retry_ms',2000))
    if not 0<=read_retry_ms<=10000:raise ValueError('read retry budget must be between 0 and 10000 milliseconds')
    def test_gate(stage, iteration):
        gate=config.get("test_pause",{})
        if gate.get("stage")==stage and gate.get("iteration")==iteration:
            path=pathlib.Path(gate["path"])
            pathlib.Path(str(path)+'.waiting').write_text(json.dumps(dict(
                stage=stage,iteration=iteration,pid=os.getpid(),native_pid=backend._process.pid)))
            deadline=time.monotonic()+30
            while not path.exists():
                if time.monotonic()>deadline:raise RuntimeError('test pause expired')
                time.sleep(.02)
    try:
        if config.get("journal_directory"):
            from training_journal import WorkerJournal, identity
            journal=WorkerJournal(config['journal_directory'],identity(config),
                                  config.get('journal_record_bytes',32<<20),config.get('journal_disk_bytes',1<<30))
        if config.get('journal_mirror'):
            if journal is None:raise ValueError('gradient replication requires a local worker journal')
            from gradient_replica import ReplicaClient
            mirror=ReplicaClient(config['journal_mirror'],journal)
        backend.lookup_layout_cache(config.get("reuse_lookup_layouts", True))
        backend.create_table(json.dumps(config["table"]))
        if config.get("agents"):
            backend.cache_endpoints([(a["id"], a["endpoint"], a.get("owner_run",config["run"])) for a in config["agents"]], True)
        backend.configure_training(config["coordinator"],config.get("gradient_retry_ms",0))
        backend.gradient_layout_cache(config.get("reuse_gradient_layouts", True))
        if 'read_sequence' in config:
            backend.restore_read_frontier(config['read_sequence'])
        batch_sizes = []
        runtime = None
        if config.get('record_stage_times'):
            runtime = {'hostname': socket.gethostname(), 'tensorflow': tf.__version__}
            if gpu:
                info = subprocess.check_output(
                    ['nvidia-smi', '--id=0', '--query-gpu=uuid,name,driver_version,memory.total',
                     '--format=csv,noheader,nounits'], text=True, timeout=10)
                uuid, name, driver, memory = (value.strip() for value in info.strip().split(','))
                runtime['gpu'] = {'uuid': uuid, 'name': name, 'driver': driver, 'memory_mib': int(memory)}
        model_warmup_seconds=None
        if config.get('start_gate'):
            warmup_started=time.monotonic()
            gate = pathlib.Path(config['start_gate'])
            warm = prepared[start_iteration % len(prepared)]
            token, arrays = backend.pull(warm[1], True)
            backend.release(token)
            with tf.device(device):
                result = forward_backward(tf.convert_to_tensor(arrays[0], dtype=tf.float32),
                    tf.convert_to_tensor(arrays[1], dtype=tf.float32), warm[4], warm[5])
            for tensor in result:
                if not np.isfinite(tensor.numpy()).all():
                    raise RuntimeError('non-finite model warmup')
            model_warmup_seconds=time.monotonic()-warmup_started
            ready_path=pathlib.Path(str(gate)+f'.worker-{config["worker"]}.ready')
            pending=ready_path.with_suffix('.pending-'+str(os.getpid()))
            pending.write_text(json.dumps(dict(run=config['run'],worker=config['worker'],warmed_model=True)))
            pending.replace(ready_path)
            deadline = time.monotonic()+60
            while not gate.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError('model start gate was not released')
                time.sleep(.002)
        for iteration in range(start_iteration, start_iteration + iterations):
            samples, features, unique_keys, inverse, segment_ids, labels = prepared[iteration % len(prepared)]
            started = time.monotonic()
            stages={name:0. for name in stage_times} if stage_times is not None else None
            floor = backend.begin_training(iteration)
            admission_wait.append(time.monotonic()-started)
            if stages is not None:stages['admission']=admission_wait[-1]
            if floor > iteration:
                raise RuntimeError("training permit floor exceeds current iteration")
            max_lead = max(max_lead, iteration-floor)
            if delay_ms and iteration % delay_every == 0:
                stamp=time.monotonic()
                time.sleep(delay_ms/1000)
                if stages is not None:stages['controlled_delay']=time.monotonic()-stamp
            stamp=time.monotonic()
            saved=journal.get(iteration) if journal else None
            if stages is not None:stages['journal_io']+=time.monotonic()-stamp
            if saved:
                if saved['sample_count']!=len(samples) or saved['completed_reads']!=backend.completed_read_frontier():
                    raise ValueError('retained gradient sample/read cursor mismatch')
                value=saved['loss'];tables=saved['tables'];replayed.append(iteration)
                unique_keys,values=tables[0]['keys'],tables[0]['values']
            else:
                stamp=time.monotonic()
                retry_deadline=None
                while True:
                    try:
                        handle, arrays = backend.pull(features, synchronous)
                        break
                    except RuntimeError:
                        if retry_deadline is None:retry_deadline=time.monotonic()+read_retry_ms/1000
                        if not read_retry_ms or not backend.read_lifecycle_healthy() or time.monotonic()>=retry_deadline:
                            raise
                        read_retries+=1
                        time.sleep(.01)
                test_gate('after_pull',iteration)
                if synchronous:
                    backend.release(handle)
                if stages is not None:stages['parameter_pull']=time.monotonic()-stamp
                stamp=time.monotonic()
                with tf.device(device):
                    embedding = tf.convert_to_tensor(arrays[0], dtype=tf.float32)
                    dense = tf.convert_to_tensor(arrays[1], dtype=tf.float32)
                    loss, logits, g_embedding, g_dense = forward_backward(embedding, dense, segment_ids, labels)
                observed_devices = {"embedding": embedding.device, "packed_mlp": dense.device,
                                "loss": loss.device, "embedding_gradient": g_embedding.device,
                                "mlp_gradient": g_dense.device}
                if gpu and not all("GPU:" in d.upper() for d in observed_devices.values()):
                    raise RuntimeError("model computation fell back from the requested GPU")
                value = float(loss.numpy())
                gradients = np.concatenate((g_embedding.numpy(), g_dense.numpy()), axis=0)
                values = np.zeros((len(unique_keys), dimension), dtype=np.float32)
                np.add.at(values, inverse, gradients)
                if not np.isfinite(value) or not np.isfinite(values).all() or not np.isfinite(logits.numpy()).all():
                    raise RuntimeError("non-finite model output or gradient")
                tables=[dict(table=1,keys=unique_keys,values=values)]
                if stages is not None:stages['model_and_gradient_staging']=time.monotonic()-stamp
                if journal:
                    stamp=time.monotonic()
                    journal.put(iteration,len(samples),backend.completed_read_frontier(),value,tables)
                    if stages is not None:stages['journal_io']+=time.monotonic()-stamp
            if mirror:
                stamp=time.monotonic();mirror.ensure(iteration)
                if stages is not None:stages['gradient_replication']=time.monotonic()-stamp
            test_gate('after_journal',iteration)
            if iteration == start_iteration and config.get("record_gradients", False):
                first_gradients = {"keys": unique_keys.tolist(), "values": values.tolist()}
            stamp=time.monotonic()
            if synchronous:
                backend.submit_synchronous(iteration, len(samples),tables)
            else:
                backend.push(handle, [0, 1], [g_embedding.numpy(), g_dense.numpy()])
            if stages is not None:stages['synchronous_update']=time.monotonic()-stamp
            test_gate('after_update',iteration)
            stamp=time.monotonic()
            backend.complete_training(iteration)
            if stages is not None:
                stages['task_completion']=time.monotonic()-stamp
                for name in stage_times:stage_times[name].append(stages[name])
            batch_sizes.append(len(samples))
            losses.append(value)
            times.append(time.monotonic() - started)
        backend.join()
        training_finished = time.monotonic()
        memory = tf.config.experimental.get_memory_info(device.lstrip("/")) if gpu else None
        print(json.dumps({"worker": config["worker"], "completed": iterations,
                          "model": "embedding_tanh_mlp", "dimension": dimension, "hidden": hidden,
                          "start_iteration": start_iteration, "input_state": input_state(start_iteration + iterations),
                          "batch_sizes": batch_sizes,
                          "update_mode": update_mode,
                          "admission_wait_seconds": admission_wait,
                          "max_admission_lead": max_lead,
                          "training_finished_monotonic": training_finished if config.get('start_gate') else None,
                          "model_warmup_seconds": model_warmup_seconds,
                          "deterministic_model": bool(config.get('deterministic_model',False)),
                          "stage_seconds": stage_times,
                          **({"runtime": runtime} if runtime is not None else {}),
                          "local_samples": len(samples), "packed_mlp_rows": dense_rows,
                          "losses": losses, "step_seconds": times, "devices": observed_devices,
                          "first_row_gradients": first_gradients,
                          "tensorflow_device_memory": memory,
                          "lookup_transport": backend.lookup_transport_stats(),
                          "gradient_transport": backend.gradient_transport_stats(),
                          "gradient_retry_count": backend.training_retry_count(),
                          "read_retry_count": read_retries,
                          "replayed_iterations": replayed,
                          "gradient_replica": mirror.stats() if mirror else None,
                          "parameter_stats": backend.stats()}, sort_keys=True))
    except Exception as error:
        try:
            backend.abort_training(str(error))
        except Exception:
            pass
        raise
    finally:
        if mirror:mirror.close()
        backend.close()
        if journal:journal.close()


if __name__ == "__main__":
    main()
