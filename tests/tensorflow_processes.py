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
                                            initial_beta1_power=0.9, initial_beta2_power=0.999, weight_bounds=[-10, 10]),
                             initialization=dict(min=0.25, max=0.25, seed=42)))
    dense = tf.Variable([[0.5], [0.25]], dtype=tf.float32)
    dense_optimizer = tf.keras.optimizers.SGD(0.01)
    dense_optimizer.build([dense])

    prepared_mode = os.environ.get("EPS_TEST_PREPARE") == "1"
    periodic_mode = os.environ.get("EPS_TEST_PERIODIC") == "1"

    def model_step(handle, embeddings):
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

    if prepared_mode:
        @tf.function(input_signature=[tf.TensorSpec([], tf.uint64)])
        def train(token):
            handle, embeddings = backend.pull_prepared(token, (2,))
            return model_step(handle, embeddings)
    else:
        @tf.function(input_signature=[tf.TensorSpec([None], tf.int64),
                                     tf.TensorSpec([None], tf.int64)])
        def train(ids, splits):
            handle, embeddings = backend.pull([
                Feature(1, 0, ids, splits, 2, pooling="mean", padding_key=0)])
            return model_step(handle, embeddings)

    inputs = [Feature(1, 0, np.array([1, 0, 1, 2], np.int64),
                      np.array([0, 3, 4], np.int64), 2, "mean", 0)]
    tokens = [backend.prepare(inputs) for _ in range(2)] if prepared_mode else []
    if prepared_mode:
        assert all(m["lookups"] == 0 and m["live_tickets"] == 0 for m in backend.stats())
        assert backend.preparation_stats()["accepted"] == 2
    losses = []
    for step in range(5):
        if prepared_mode:
            loss, prediction = train(tf.constant(tokens.pop(0), tf.uint64))
            tokens.append(backend.prepare(inputs))
        else:
            loss, prediction = train(tf.constant([1, 0, 1, 2], tf.int64),
                                     tf.constant([0, 3, 4], tf.int64))
        losses.append(float(loss))
        assert np.isfinite(prediction.numpy()).all()
        if periodic_mode and step == 2:
            for token in tokens:
                backend.cancel_prepared(token)
            tokens=[]
            state_files=backend.save("periodic-mid")
            # Fault injection into a disposable server file: export must detect
            # its checksum error before publishing a checkpoint-store manifest.
            corrupt=str(pathlib.Path(snapshot_root)/"corrupt-export")
            with open(state_files[0],"r+b") as stream:
                stream.seek(-1,2);original=stream.read(1);stream.seek(-1,2);stream.write(bytes([original[0]^1]))
            try:
                try:
                    backend.export_checkpoint("periodic-mid",corrupt,tf.io.gfile,chunk_bytes=7)
                except RuntimeError as error:
                    assert "checksum mismatch" in str(error),str(error)
                else:
                    raise AssertionError("corrupt snapshot was exported")
                assert not tf.io.gfile.exists(corrupt+"/parameter-service.json")
                assert not tf.io.gfile.glob(corrupt+"/*.pending-*")
            finally:
                with open(state_files[0],"r+b") as stream:
                    stream.seek(-1,2);stream.write(original)
            dense_path=str(pathlib.Path(snapshot_root)/"dense-mid")
            tf.train.Checkpoint(dense=dense,optimizer=dense_optimizer).write(dense_path)
            with tf.io.gfile.GFile(str(pathlib.Path(snapshot_root)/"input-step.json"),"w") as stream:
                json.dump(dict(next_step=step+1),stream)
            destination=str(pathlib.Path(snapshot_root)/"exported-mid")
            manifest=backend.export_checkpoint("periodic-mid",destination,tf.io.gfile,chunk_bytes=7)
            assert len(manifest["files"])==2
            assert backend.export_checkpoint("periodic-mid",destination,tf.io.gfile,chunk_bytes=13)==manifest
            try:
                backend.prepare(inputs)
            except RuntimeError as error:
                assert "closed" in str(error),str(error)
            else:
                raise AssertionError("export reopened training before joint commit")
            try:
                backend.resume("stale-checkpoint")
            except RuntimeError:
                pass
            else:
                raise AssertionError("stale checkpoint resumed the adapter")
            with tf.io.gfile.GFile(str(pathlib.Path(snapshot_root)/"joint-commit.json"),"w") as stream:
                json.dump(dict(dense=dense_path,parameters=manifest,next_step=step+1),stream)
            backend.resume("periodic-mid")
            backend.resume("periodic-mid")
            if prepared_mode:
                tokens=[backend.prepare(inputs) for _ in range(2)]
    if prepared_mode:
        try:
            backend.save("must-wait-for-input-cancel")
        except RuntimeError as error:
            assert "prepared inputs" in str(error), str(error)
        else:
            raise AssertionError("checkpoint passed pending prepared inputs")
        for token in tokens:
            backend.cancel_prepared(token)
        stats = backend.preparation_stats()
        assert stats["accepted"] == (9 if periodic_mode else 7) and stats["consumed"] == 5 and stats["cancelled"] == (4 if periodic_mode else 2) and stats["pending"] == 0, stats
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
    metrics = backend.stats()
    assert len(metrics) == 2 and all(m["updates"] == 5 and m["live_tickets"] == 0 for m in metrics), metrics
    assert all(m["waiting_operations"] == 0 and m["active_payload_bytes"] == 0 and
               m["active_operations"] >= 1 and m["dependency_payload_bytes"] == 0 for m in metrics), metrics
    assert all(m["dependency_prepare_nanoseconds"] > 0 and m["dependency_rebuilds"] > 0 and
               m["dependency_cache_hits"] >= 0 for m in metrics), metrics
    assert all(m["lookup_prepare_nanoseconds"] > 0 and m["gradient_prepare_nanoseconds"] > 0 and m["cpu_optimizer_nanoseconds"] > 0 for m in metrics), metrics
    automatic = os.environ.get("EPS_TEST_AUTO_CACHE") == "1"
    if automatic:
        assert all(m["gpu_packet_prepare_nanoseconds"] > 0 and m["update_dispatch_nanoseconds"] > 0 and m["agent_wait_nanoseconds"] > 0 and m["cache_maintenance_nanoseconds"] > 0 for m in metrics), metrics
        assert sum(m["policy_admitted_rows"] for m in metrics) == (4 if periodic_mode else 2), metrics
        assert sum(m["cached_rows"] for m in metrics) == 2, metrics
        assert all(m["policy_windows"] == 6 and m["policy_tracked_rows"] == 1 and m["policy_failures"] == 0 for m in metrics), metrics
    if prepared_mode:
        # A schema error after the native Pull must release its unpublished ticket.
        token = backend.prepare(inputs)
        try:
            backend.pull_prepared(tf.constant(token, tf.uint64), (3,), read_only=True)
        except tf.errors.OpError as error:
            assert "layout differs" in str(error), str(error)
        else:
            raise AssertionError("prepared output schema mismatch passed")
        assert all(m["live_tickets"] == 0 for m in backend.stats())
        token = backend.prepare(inputs)
        handle, checked = backend.pull_prepared(tf.constant(token, tf.uint64), (2,), read_only=True)
        backend.release(handle).numpy()
        assert checked[0].shape == (2, 2)
        constrained = ParameterServer(endpoints, "tensorflow-check", learner=99,
                                      preparation=dict(max_pending_batches=1, payload_bytes=64))
        try:
            constrained.prepare(inputs)
        except RuntimeError as error:
            assert "queue is full" in str(error), str(error)
        else:
            raise AssertionError("configured preparation byte budget was ignored")
        assert constrained.preparation_stats()["rejected"] == 1
        constrained.close()
    # Zero-valued bounds must retain presence across JSON/native/protobuf.
    zero_table=dict(id=2,dimension=2,dtype="float32",
                    optimizer=dict(kind="sparse_lazy_adam",learning_rate=.01,beta1=.9,beta2=.999,
                                   epsilon=1e-8,initial_beta1_power=.9,initial_beta2_power=.999,
                                   weight_bounds=[0,0]),initialization=dict(min=.5,max=.5,seed=42))
    backend.create_table(zero_table)
    zero_handle, zero_values=backend.pull([Feature(2,7,tf.constant([1,2],tf.int64),
                                                   tf.constant([0,2],tf.int64),2)],read_only=True)
    np.testing.assert_array_equal(zero_values[0].numpy(),np.zeros((1,2),np.float32))
    backend.release(zero_handle).numpy()
    files = backend.save("tf-checkpoint")
    assert len(files) == 2 and all(pathlib.Path(f).is_file() for f in files)
    assert all(m["cached_rows"] == 0 for m in backend.stats())
    print(json.dumps(dict(test="tensorflow_graph_pull_push_join", tensorflow=tf.__version__,
                          cache_route=("automatic_resident_" if automatic else "transfer_unique_")+
                                      os.environ.get("EPS_CACHE_BACKEND", "host") if agent_endpoint else "cpu_pool",
                          train_steps=5, prepared_inputs=prepared_mode, loss_before=losses[0], loss_after=losses[-1],
                          periodic_checkpoint=periodic_mode,
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
                    "--backend", os.environ.get("EPS_CACHE_BACKEND", "host"), "--listen", "127.0.0.1:0"],
                    stdout=subprocess.PIPE, stderr=log, text=True)
                processes.append(agent)
                agent_endpoint = ready(agent)
            for shard in range(2):
                config = root / f"shard-{shard}.json"
                cache = dict(enabled=bool(agent_endpoint))
                if os.environ.get("EPS_TEST_AUTO_CACHE") == "1":
                    assert agent_endpoint, "automatic graph check requires an agent"
                    cache.update(admission="reuse_cost", decision_window_requests=1,
                        max_tracked_rows=16, minimum_accesses=1,
                        saved_microseconds_per_access=100,
                        load_microseconds_per_byte=0, writeback_microseconds_per_byte=0,
                        learn_migration_costs=False, reserve_slots=1)
                config.write_text(json.dumps(dict(schema_version=1, run_id="tensorflow-check",
                    listen="127.0.0.1:0", partition=dict(shard_id=shard, shard_count=2, epoch=1),
                    tables=[], cache=cache,
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
