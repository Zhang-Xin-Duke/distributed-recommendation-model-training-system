"""Existing TensorFlow graph + two external parameter shards, no replacement trainer."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

from rpc_processes import ready, stop


def graph_probe(endpoints, snapshot_root, agent_endpoint=None):
    import numpy as np
    import tensorflow as tf
    from eps_tensorflow import Feature, ParameterServer

    backend = ParameterServer(endpoints, "tensorflow-check")
    if agent_endpoint:
        backend.cache_endpoints([(0, agent_endpoint)], coalesce=os.environ.get("EPS_TEST_COALESCE") == "1")
    backend.create_table(dict(id=1, dimension=2, dtype="float32",
                             optimizer=dict(kind="sparse_lazy_adam", learning_rate=0.01,
                                            beta1=0.9, beta2=0.999, epsilon=1e-8,
                                            initial_beta1_power=0.9, initial_beta2_power=0.999),
                             initialization=dict(min=0.25, max=0.25, seed=42)))
    dense = tf.Variable([[0.5], [0.25]], dtype=tf.float32)
    dense_optimizer = tf.keras.optimizers.SGD(0.01)
    dense_optimizer.build([dense])

    @tf.function(input_signature=[tf.TensorSpec([None], tf.int64),
                                 tf.TensorSpec([None], tf.int64)])
    def train(ids, splits):
        handle, embeddings = backend.pull([
            Feature(1, 0, ids, splits, 2, pooling="mean", padding_key=0)])
        with tf.GradientTape() as tape:
            tape.watch(embeddings)
            prediction = embeddings[0] @ dense
            loss = tf.reduce_mean(tf.square(prediction-1.0))
        sparse_gradient, dense_gradient = tape.gradient(loss, [embeddings[0], dense])
        sparse_ack = backend.push(handle, [0], [sparse_gradient])
        dense_optimizer.apply_gradients([(dense_gradient, dense)])
        done = backend.join([sparse_ack])
        with tf.control_dependencies([done]):
            return tf.identity(loss), tf.identity(prediction)

    losses = []
    for _ in range(5):
        loss, prediction = train(tf.constant([1, 0, 1, 2], tf.int64),
                                 tf.constant([0, 3, 4], tf.int64))
        losses.append(float(loss))
        assert np.isfinite(prediction.numpy()).all()
    assert losses[-1] < losses[0], losses
    assert train.experimental_get_tracing_count() == 1

    @tf.function
    def evaluate():
        h, values = backend.pull([
            Feature(1, 1, tf.constant([0, 1, 2, 0], tf.int64),
                    tf.constant([0, 2, 4], tf.int64), 2, "sequence", 0)], read_only=True)
        ack = backend.release(h)
        with tf.control_dependencies([ack]):
            return tf.identity(values[0])

    values = evaluate().numpy()
    np.testing.assert_array_equal(values[[0, 3]], np.zeros((2, 2)))
    assert (values[[1, 2]] > 0.25).all()
    files = backend.save("tf-checkpoint")
    assert len(files) == 2 and all(pathlib.Path(f).is_file() for f in files)
    print(json.dumps(dict(test="tensorflow_graph_pull_push_join", tensorflow=tf.__version__,
                          cache_route="transfer_unique_host_reference" if agent_endpoint else "cpu_pool",
                          train_steps=5, loss_before=losses[0], loss_after=losses[-1],
                          checkpoint_shards=len(files), retraces=0)), flush=True)
    backend.close()


def main():
    if sys.argv[1] == "--probe":
        graph_probe(sys.argv[2:4], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
        return
    server = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="eps-tensorflow-") as temporary:
        root = pathlib.Path(temporary)
        processes, logs, endpoints = [], [], []
        try:
            agent_endpoint = None
            if len(sys.argv) == 3:
                log = open(root / "agent.log", "w+")
                logs.append(log)
                agent = subprocess.Popen([sys.argv[2], "--run", "tensorflow-check",
                    "--backend", "host", "--listen", "127.0.0.1:0"],
                    stdout=subprocess.PIPE, stderr=log, text=True)
                processes.append(agent)
                agent_endpoint = ready(agent)
            for shard in range(2):
                config = root / f"shard-{shard}.json"
                config.write_text(json.dumps(dict(schema_version=1, run_id="tensorflow-check",
                    listen="127.0.0.1:0", partition=dict(shard_id=shard, shard_count=2, epoch=1),
                    tables=[], cache=dict(enabled=bool(agent_endpoint)),
                    access=dict(policy="transfer_unique" if agent_endpoint else "cpu_only", agent_id=0),
                    agents=[dict(id=0, endpoint=agent_endpoint)] if agent_endpoint else [],
                    snapshot_directory=str(root / "snapshots"))))
                log = open(root / f"shard-{shard}.log", "w+")
                logs.append(log)
                process = subprocess.Popen([server, "--config", str(config)],
                    stdout=subprocess.PIPE, stderr=log, text=True)
                processes.append(process)
                endpoints.append(ready(process))
            subprocess.run([sys.executable, __file__, "--probe", *endpoints, str(root),
                            *([agent_endpoint] if agent_endpoint else [])],
                           check=True, timeout=90, env=os.environ.copy())
        except Exception:
            for log in logs:
                log.flush(); log.seek(0)
                print(log.read()[-6000:], file=sys.stderr)
            raise
        finally:
            for process in reversed(processes):
                stop(process)
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
