"""Embedding + two-layer MLP training through synchronous parameter shards.

All trainable parameters, including packed MLP weights/biases, live in one
row-sharded table. Host staging bridges remote parameter RPCs to TensorFlow;
the configured device executes model forward/backward.
"""
import argparse
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
    samples = config["samples"]
    if not samples or len(samples) > 65536:
        raise ValueError("invalid local batch size")
    keys, segments = [], []
    for sample, example in enumerate(samples):
        if not example["keys"] or not 0 <= example["label"] <= 1:
            raise ValueError("invalid training example")
        for key in example["keys"]:
            if key < dense_rows or key >= (1 << 63):
                raise ValueError("embedding IDs overlap packed MLP rows or exceed supported range")
            keys.append(key)
            segments.append(sample)
    keys = np.asarray(keys, dtype=np.uint64)
    features = [dict(table=1, feature=0, pooling="sequence", keys=keys,
                     offsets=np.array([0, len(keys)], dtype=np.uint64)),
                dict(table=1, feature=1, pooling="sequence", keys=dense_keys,
                     offsets=np.array([0, len(dense_keys)], dtype=np.uint64))]
    all_keys = np.concatenate((keys, dense_keys))
    unique_keys, inverse = np.unique(all_keys, return_inverse=True)
    with tf.device(device):
        segment_ids = tf.constant(segments, dtype=tf.int32)
        labels = tf.constant([s["label"] for s in samples], dtype=tf.float32)

    @tf.function(autograph=False)
    def forward_backward(embedding, dense):
        with tf.device(device), tf.GradientTape() as tape:
            tape.watch((embedding, dense))
            flat = tf.reshape(dense, [-1])
            stop = dimension * hidden
            w1 = tf.reshape(flat[:stop], [dimension, hidden])
            b1 = flat[stop:stop + hidden]
            w2 = flat[stop + hidden:stop + 2 * hidden]
            b2 = flat[stop + 2 * hidden]
            pooled = tf.math.unsorted_segment_sum(embedding, segment_ids, len(samples))
            activations = tf.math.tanh(tf.linalg.matmul(pooled, w1) + b1)
            logits = tf.linalg.matvec(activations, w2) + b2
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        gradients = tape.gradient(loss, (embedding, dense))
        return loss, logits, gradients[0], gradients[1]

    backend = Backend(config["shards"], config["run"], config["worker"], config.get("epoch", 1))
    losses, observed_devices, first_gradients, times = [], {}, None, []
    try:
        backend.create_table(json.dumps(config["table"]))
        if config.get("agents"):
            backend.cache_endpoints([(a["id"], a["endpoint"]) for a in config["agents"]], True)
        backend.configure_training(config["coordinator"])
        for iteration in range(iterations):
            started = time.monotonic()
            floor = backend.begin_training(iteration)
            if floor > iteration:
                raise RuntimeError("training permit floor exceeds current iteration")
            handle, arrays = backend.pull(features, True)
            backend.release(handle)
            with tf.device(device):
                embedding = tf.convert_to_tensor(arrays[0], dtype=tf.float32)
                dense = tf.convert_to_tensor(arrays[1], dtype=tf.float32)
                loss, logits, g_embedding, g_dense = forward_backward(embedding, dense)
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
            if iteration == 0 and config.get("record_gradients", False):
                first_gradients = {"keys": unique_keys.tolist(), "values": values.tolist()}
            backend.submit_synchronous(iteration, len(samples),
                                       [dict(table=1, keys=unique_keys, values=values)])
            backend.complete_training(iteration)
            losses.append(value)
            times.append(time.monotonic() - started)
        backend.join()
        memory = tf.config.experimental.get_memory_info(device.lstrip("/")) if gpu else None
        print(json.dumps({"worker": config["worker"], "completed": iterations,
                          "model": "embedding_tanh_mlp", "dimension": dimension, "hidden": hidden,
                          "local_samples": len(samples), "packed_mlp_rows": dense_rows,
                          "losses": losses, "step_seconds": times, "devices": observed_devices,
                          "first_row_gradients": first_gradients,
                          "tensorflow_device_memory": memory,
                          "parameter_stats": backend.stats()}, sort_keys=True))
    except Exception as error:
        try:
            backend.abort_training(str(error))
        except Exception:
            pass
        raise
    finally:
        backend.close()


if __name__ == "__main__":
    main()
