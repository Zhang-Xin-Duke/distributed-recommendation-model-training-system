"""Train all Embedding/MLP parameters across worker processes and compare a reference.

Use --devices 0,1 for real CUDA models and CUDA parameter owners. CPU mode uses
HostCacheBackend owners to exercise the same routing/migration/update protocol.
"""
import argparse
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import tempfile

from rpc_processes import ready, stop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=pathlib.Path, required=True)
    p.add_argument("--devices", default="")
    p.add_argument("--rows", type=int, default=64)
    p.add_argument("--dimension", type=int, default=8)
    p.add_argument("--hidden", type=int, default=4)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--samples-per-worker", type=int, default=0)
    p.add_argument("--bag-size", type=int, default=16)
    p.add_argument("--out", type=pathlib.Path)
    p.add_argument("--disable-lookup-cache", action="store_true")
    p.add_argument("--disable-gradient-cache", action="store_true")
    p.add_argument("--owner-incarnation", default="", help="Owner identity suffix independent of the stable training run")
    a = p.parse_args()
    build = a.build.resolve()
    source = pathlib.Path(__file__).resolve().parents[1]
    devices = [int(v) for v in a.devices.split(",") if v]
    if len(devices) != len(set(devices)) or any(v < 0 for v in devices):
        p.error("GPU device ordinals must be distinct nonnegative integers")
    gpu = bool(devices)
    owners = len(devices) if gpu else 2
    packed = (a.dimension * a.hidden + 2 * a.hidden + 1 + a.dimension - 1) // a.dimension
    if a.rows <= packed + 5 or not 1 <= a.dimension <= 4096 or not 1 <= a.hidden <= 1024:
        p.error("invalid table/model size")
    if packed * a.dimension * 4 > (8 << 20):
        p.error("packed MLP parameters exceed the bounded feature payload")
    samples = [[{"keys": [packed + 1, packed + 2, packed + 1], "label": 0.},
                {"keys": [packed + 3], "label": 1.}],
               [{"keys": [packed + 3, packed + 4], "label": 1.}]]
    if a.samples_per_worker:
        if not 1 <= a.samples_per_worker <= 4096 or not 1 <= a.bag_size <= 256:
            p.error("invalid training batch size")
        if (2 * a.samples_per_worker * a.bag_size + packed) * a.dimension * 4 > (12 << 20):
            p.error("global reference batch exceeds the bounded transport payload")
        rng = random.Random(20261104)
        samples = [[{"keys": [rng.randrange(packed, a.rows) for _ in range(a.bag_size)],
                     "label": float(sample % 2)} for sample in range(a.samples_per_worker)] for _ in range(2)]
    all_samples = [s for part in samples for s in part]
    all_keys = sorted(set(range(packed)) | {k for s in all_samples for k in s["keys"]})
    inspect_limit = min(512, (8 << 20) // (a.dimension * 12 + 256))
    keys = all_keys if len(all_keys) <= inspect_limit else [
        all_keys[i * len(all_keys) // inspect_limit] for i in range(inspect_limit)]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(build / "python"), str(source / "adapters/entropy"),
                                         env.get("PYTHONPATH", "")])
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    table = {"id": 1, "dimension": a.dimension, "dtype": "float32",
             "optimizer": {"kind": "sparse_lazy_adam", "learning_rate": .02, "beta1": .9,
                           "beta2": .999, "epsilon": 1e-8, "initial_beta1_power": .9,
                           "initial_beta2_power": .999},
             "initialization": {"min": -.1, "max": .1, "seed": 42}}
    reports = []
    with tempfile.TemporaryDirectory(prefix="eps-tf-sync-") as temporary:
        root = pathlib.Path(temporary)
        for reference in (True, False):
            run = "tf-sync-reference" if reference else "tf-sync-actual"
            worker_samples = [all_samples] if reference else samples
            processes, logs = [], []

            def launch(command, name, config=None, environment=None):
                if config is not None:
                    path = root / f"{run}-{name}.json"
                    path.write_text(json.dumps(config))
                    command = [*command, "--config", str(path)]
                log = open(root / f"{run}-{name}.log", "w+")
                logs.append(log)
                child = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE, stderr=log,
                                         text=True, env=environment or env)
                processes.append(child)
                return child

            try:
                agents = []
                if not reference:
                    for home in range(owners):
                        slots = (a.rows + owners - 1) // owners
                        owner_run=run+f"-owner-{home}-"+a.owner_incarnation if a.owner_incarnation else run
                        cmd = [build / "eps_cache_agent", "--run", owner_run, "--training-run", run, "--listen", "127.0.0.1:0",
                               "--agent", str(home), "--backend", "cuda" if gpu else "host",
                               "--rows-per-table", str(slots), "--state-bytes",
                               str(slots * (a.dimension * 12 + 8) + 4096)]
                        if gpu:
                            cmd += ["--device", str(devices[home])]
                        agents.append({"id": home, "endpoint": ready(launch(cmd, f"agent-{home}")), "owner_run":owner_run})
                endpoints = []
                for home in range(1 if reference else owners):
                    config = {"schema_version": 1, "run_id": run, "listen": "127.0.0.1:0",
                              "partition": {"shard_id": home, "shard_count": 1 if reference else owners, "epoch": 1},
                              "tables": [table], "cache": {"enabled": bool(agents)},
                              "synchronous_training": {"workers": len(worker_samples), "reduction": "sample_mean",
                                                       "round_timeout_ms": 20000},
                              "snapshot_directory": str(root / run / "snapshots"),
                              "limits": {"cpu_state_bytes": max(1 << 28, a.rows * (a.dimension * 12 + 128))}}
                    if agents:
                        config["agents"] = [agents[home]]
                    endpoints.append(ready(launch([build / "eps_server"], f"ps-{home}", config)))
                coordinator = ready(launch([build / "eps_training_coordinator"], "coordinator", {
                    "schema_version": 1, "run": run, "workers": len(worker_samples), "epoch": 1,
                    "consistency": "sequential", "max_delay": 0,
                }))
                settings = {"run": run, "shards": endpoints, "coordinator": coordinator, "epoch": 1,
                            "keys": keys, "synchronous": True}
                preparation = None
                if agents:
                    prepared = launch([build / "eps_training_prepare"], "prepare", {
                        **settings, "rows": a.rows, "dimension": a.dimension, "agents": agents})
                    stdout, _ = prepared.communicate(timeout=1800)
                    if prepared.returncode:
                        raise RuntimeError("parameter initialization/migration failed")
                    preparation = json.loads(stdout)
                    assert sum(s["resident_rows"] for s in preparation["shards"]) == a.rows
                children = []
                for worker, examples in enumerate(worker_samples):
                    worker_env = env.copy()
                    if gpu and not reference:
                        worker_env["CUDA_VISIBLE_DEVICES"] = str(devices[worker % owners])
                    else:
                        worker_env["CUDA_VISIBLE_DEVICES"] = ""
                    children.append(launch([sys.executable, source / "examples/tensorflow_training_worker.py"],
                        f"worker-{worker}", {**settings, "table": table, "worker": worker,
                            "dimension": a.dimension, "hidden": a.hidden, "iterations": a.iterations,
                            "samples": examples, "agents": agents, "record_gradients": True,
                            "reuse_lookup_layouts": not a.disable_lookup_cache,
                            "reuse_gradient_layouts": not a.disable_gradient_cache,
                            "device": "/GPU:0" if gpu and not reference else "/CPU:0"}, worker_env))
                workers = []
                for child in children:
                    stdout, _ = child.communicate(timeout=120)
                    if child.returncode:
                        raise RuntimeError("model worker failed")
                    workers.append(json.loads(stdout))
                path = root / f"{run}-inspect.json"
                path.write_text(json.dumps(settings))
                output = subprocess.run([str(build / "eps_training_control"), "--config", str(path),
                                         "--action", "inspect"], env=env, check=True,
                                        capture_output=True, text=True, timeout=20)
                state = json.loads(output.stdout)
                assert state["completed"] == [a.iterations] * len(worker_samples), state
                assert not state["active"] and not state["waiting"] and not state["failed"], state
                for row in state["rows"]:
                    assert row["row_updates"] == a.iterations, row
                    assert all(math.isfinite(v) for field in ("weights", "first_moment", "second_moment") for v in row[field])
                assert all(s["live_tickets"] == 0 and s["synchronous"]["next_round"] == a.iterations
                           for s in state["shard_stats"]), state
                reports.append({"reference": reference, "gpu": gpu and not reference,
                                "workers": workers, "state": state, "preparation": preparation})
            except Exception:
                for log in logs:
                    log.flush(); log.seek(0); print(log.read()[-6000:], file=sys.stderr)
                raise
            finally:
                for process in reversed(processes):
                    stop(process)
                for log in logs:
                    log.close()
    expected = {r["key"]: r for r in reports[0]["state"]["rows"]}
    maximum = 0.
    for row in reports[1]["state"]["rows"]:
        for field in ("weights", "first_moment", "second_moment"):
            for got, wanted in zip(row[field], expected[row["key"]][field]):
                maximum = max(maximum, abs(got - wanted))
    tolerance = 5e-5 if gpu else 3e-6
    assert maximum <= tolerance, (maximum, tolerance)
    reference_gradients = reports[0]["workers"][0]["first_row_gradients"]
    expected_g = dict(zip(reference_gradients["keys"], reference_gradients["values"]))
    merged = {key: [0.] * a.dimension for key in all_keys}
    for worker in reports[1]["workers"]:
        gradient = worker["first_row_gradients"]
        for key, values in zip(gradient["keys"], gradient["values"]):
            for d, value in enumerate(values):
                merged[key][d] += value * worker["local_samples"] / len(all_samples)
    gradient_error = max(abs(merged[k][d] - expected_g[k][d]) for k in all_keys for d in range(a.dimension))
    assert gradient_error < 2e-6, gradient_error
    initial = sum(w["losses"][0] * w["local_samples"] / len(all_samples) for w in reports[1]["workers"])
    final = sum(w["losses"][-1] * w["local_samples"] / len(all_samples) for w in reports[1]["workers"])
    assert final < initial and math.isfinite(final), (initial, final)
    result = {"test": "tensorflow_synchronous_model_training", "gpu": gpu, "rows": a.rows,
              "dimension": a.dimension, "hidden": a.hidden, "iterations": a.iterations,
              "model": "trainable_embedding_and_tanh_mlp", "model_reference_max_absolute_error": maximum,
              "first_global_gradient_max_absolute_error": gradient_error,
              "initial_weighted_loss": initial, "final_weighted_loss": final, "reports": reports}
    result["reuse_lookup_layouts"] = not a.disable_lookup_cache
    result["owner_incarnation"] = a.owner_incarnation
    result["touched_rows"] = len(all_keys)
    result["inspected_rows"] = len(keys)
    result["reference_state_scope"] = "all_touched_rows" if len(keys) == len(all_keys) else "sampled_touched_rows"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "reports"}, sort_keys=True))


if __name__ == "__main__":
    main()
