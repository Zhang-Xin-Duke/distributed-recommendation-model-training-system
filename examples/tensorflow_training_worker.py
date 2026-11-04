"""Embedding + two-layer MLP training through synchronous parameter shards.

All trainable parameters, including packed MLP weights/biases, live in one
row-sharded table. Host staging bridges remote parameter RPCs to TensorFlow;
the configured device executes model forward/backward.
"""
import argparse
import hashlib
import json
import os
import pathlib
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
    dimension = int(config["dimension"])
    hidden = int(config.get("hidden", 8))
    iterations = int(config["iterations"])
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
    read_retries=0
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
        if 'read_sequence' in config:
            backend.restore_read_frontier(config['read_sequence'])
        batch_sizes = []
        for iteration in range(start_iteration, start_iteration + iterations):
            samples, features, unique_keys, inverse, segment_ids, labels = prepared[iteration % len(prepared)]
            started = time.monotonic()
            floor = backend.begin_training(iteration)
            if floor > iteration:
                raise RuntimeError("training permit floor exceeds current iteration")
            saved=journal.get(iteration) if journal else None
            if saved:
                if saved['sample_count']!=len(samples) or saved['completed_reads']!=backend.completed_read_frontier():
                    raise ValueError('retained gradient sample/read cursor mismatch')
                value=saved['loss'];tables=saved['tables'];replayed.append(iteration)
                unique_keys,values=tables[0]['keys'],tables[0]['values']
            else:
                retry_deadline=None
                while True:
                    try:
                        handle, arrays = backend.pull(features, True)
                        break
                    except RuntimeError:
                        if retry_deadline is None:retry_deadline=time.monotonic()+read_retry_ms/1000
                        if not read_retry_ms or not backend.read_lifecycle_healthy() or time.monotonic()>=retry_deadline:
                            raise
                        read_retries+=1
                        time.sleep(.01)
                test_gate('after_pull',iteration)
                backend.release(handle)
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
                if journal:
                    journal.put(iteration,len(samples),backend.completed_read_frontier(),value,tables)
            if mirror:mirror.ensure(iteration)
            test_gate('after_journal',iteration)
            if iteration == start_iteration and config.get("record_gradients", False):
                first_gradients = {"keys": unique_keys.tolist(), "values": values.tolist()}
            backend.submit_synchronous(iteration, len(samples),tables)
            test_gate('after_update',iteration)
            backend.complete_training(iteration)
            batch_sizes.append(len(samples))
            losses.append(value)
            times.append(time.monotonic() - started)
        backend.join()
        memory = tf.config.experimental.get_memory_info(device.lstrip("/")) if gpu else None
        print(json.dumps({"worker": config["worker"], "completed": iterations,
                          "model": "embedding_tanh_mlp", "dimension": dimension, "hidden": hidden,
                          "start_iteration": start_iteration, "input_state": input_state(start_iteration + iterations),
                          "batch_sizes": batch_sizes,
                          "local_samples": len(samples), "packed_mlp_rows": dense_rows,
                          "losses": losses, "step_seconds": times, "devices": observed_devices,
                          "first_row_gradients": first_gradients,
                          "tensorflow_device_memory": memory,
                          "lookup_transport": backend.lookup_transport_stats(),
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
