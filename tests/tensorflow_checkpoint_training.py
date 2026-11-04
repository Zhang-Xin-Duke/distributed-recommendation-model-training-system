"""Joint process restart matches uninterrupted training on changing input batches."""
import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import resource
import signal
import subprocess
import sys
import tempfile
from rpc_processes import ready, stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path)
    a = parser.parse_args()
    build = a.build.resolve()
    source = pathlib.Path(__file__).resolve().parents[1]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL="2")
    env["PYTHONPATH"] = os.pathsep.join([str(build / "python"), str(source / "adapters/entropy"), env.get("PYTHONPATH", "")])
    dimension, hidden, rounds, split = 8, 4, 8, 4
    packed = (dimension * hidden + 2 * hidden + 1 + dimension - 1) // dimension
    table = dict(id=1, dimension=dimension, dtype="float32",
                 optimizer=dict(kind="sparse_lazy_adam", learning_rate=.02, beta1=.9, beta2=.999,
                                epsilon=1e-8, initial_beta1_power=.9, initial_beta2_power=.999),
                 initialization=dict(min=-.1, max=.1, seed=42))
    # Split at iteration four, not a multiple of this three-batch input cycle.
    batches = [[
        [dict(keys=[packed+1, packed+2, packed+1], label=0.), dict(keys=[packed+3], label=1.)],
        [dict(keys=[packed+4], label=1.)],
        [dict(keys=[packed+5, packed+6], label=0.), dict(keys=[packed+3], label=1.)],
    ], [
        [dict(keys=[packed+3, packed+4], label=1.)],
        [dict(keys=[packed+5], label=0.), dict(keys=[packed+6, packed+3], label=1.)],
        [dict(keys=[packed+1, packed+6], label=0.)],
    ]]
    rows = 32
    keys = list(range(rows))
    model = dict(kind="embedding_tanh_mlp", dimension=dimension, hidden=hidden)
    failures = []
    with tempfile.TemporaryDirectory(prefix="eps-training-resume-") as temporary:
        root = pathlib.Path(temporary)
        counter = 0

        def control(settings, action, ok=True, file_limit=None):
            nonlocal counter
            counter += 1
            path = root / f"control-{counter}.json"
            path.write_text(json.dumps(settings))
            def limit_output():
                signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
            p = subprocess.run([str(build / "eps_training_control"), "--config", str(path), "--action", action],
                               env=env, capture_output=True, text=True, timeout=90,
                               preexec_fn=limit_output if file_limit is not None else None)
            if ok:
                if p.returncode:
                    raise RuntimeError(p.stderr)
                return json.loads(p.stdout)
            assert p.returncode != 0, (action, p.stdout)
            failures.append(dict(action=action, reason=p.stderr.strip()))

        def phase(run, iterations, templates=None, manifest=None, save=False):
            children, logs = [], []

            def command(name, arguments):
                log = open(root / f"{run}-{name}.log", "w+")
                logs.append(log)
                child = subprocess.Popen(list(map(str, arguments)), env=env,
                                         stdout=subprocess.PIPE, stderr=log, text=True)
                children.append(child)
                return child

            def launch(name, binary, config, extra=()):
                path = root / f"{run}-{name}.json"
                path.write_text(json.dumps(config))
                return command(name, [binary, "--config", path, *extra])

            try:
                addresses = []
                agents = []
                for home in range(2):
                    address = ready(command(f"owner-{home}", [build / "eps_cache_agent", "--run", run,
                        "--listen", "127.0.0.1:0", "--agent", home, "--backend", "host",
                        "--rows-per-table", rows//2, "--state-bytes", 4096]))
                    agents.append(dict(id=home, endpoint=address))
                for home in range(2):
                    config = copy.deepcopy(templates["shards"][home]) if templates else dict(
                        schema_version=1, run_id=run, listen="127.0.0.1:0",
                        partition=dict(shard_id=home, shard_count=2, epoch=1), tables=[table],
                        synchronous_training=dict(workers=2, reduction="sample_mean", round_timeout_ms=20000))
                    config["snapshot_directory"] = str(root / f"{run}-snapshots")
                    if agents:
                        config.update(cache=dict(enabled=True), agents=[agents[home]])
                    extra = ["--restore", config.pop("restore_file")] if templates else []
                    addresses.append(ready(launch(f"ps-{home}", build / "eps_server", config, extra)))
                cconfig = templates["coordinator"] if templates else dict(
                    schema_version=1, run=run, workers=2, epoch=1, consistency="sequential", max_delay=0)
                coordinator = ready(launch("coordinator", build / "eps_training_coordinator", cconfig))
                settings = dict(run=run, epoch=1, shards=addresses, coordinator=coordinator, keys=keys, synchronous=True)
                if agents:
                    preparation = launch("prepare", build / "eps_training_prepare", dict(settings,
                        rows=rows, dimension=dimension, agents=agents, restore_existing=bool(templates)))
                    output, _ = preparation.communicate(timeout=60)
                    assert preparation.returncode == 0
                    prepared = json.loads(output)
                    assert sum(s["resident_rows"] for s in prepared["shards"]) == rows
                    if templates:
                        # Repeating rehydration must preserve existing state and ownership.
                        retry = launch("prepare-retry", build / "eps_training_prepare", dict(settings,
                            rows=rows, dimension=dimension, agents=agents, restore_existing=True))
                        retried, _ = retry.communicate(timeout=60)
                        assert retry.returncode == 0 and sum(s["resident_rows"] for s in json.loads(retried)["shards"]) == rows
                if templates:
                    frozen = control(settings, "stats")
                    assert frozen["paused"] and frozen["completed"] == [split, split], frozen
                    # Valid files and metadata, but wrong live shard ordering must not unfreeze admission.
                    control(dict(settings, shards=list(reversed(addresses)), manifest=str(manifest)), "resume-job", False)
                    assert control(settings, "stats")["paused"]
                    control(dict(settings, manifest=str(manifest)), "resume-job")
                workers = []
                for worker in range(2):
                    config = dict(settings, worker=worker, table=table, dimension=dimension, hidden=hidden,
                                  iterations=iterations, batches=batches[worker], device="/CPU:0", agents=agents)
                    if templates:
                        state = next(w["input_state"] for w in templates["workers"] if w["worker"] == worker)
                        config.update(start_iteration=split, input_state=state)
                        # Bad input position is rejected before opening the parameter client.
                        bad = dict(config, input_state=dict(state, next_batch=0))
                        path = root / f"bad-input-{worker}.json"
                        path.write_text(json.dumps(bad))
                        bad_result = subprocess.run([sys.executable, str(source / "examples/tensorflow_training_worker.py"),
                                                     "--config", str(path)], env=env, capture_output=True, text=True, timeout=45)
                        assert bad_result.returncode and "input cursor mismatch" in bad_result.stderr
                        failures.append(dict(action="worker-input", reason="incorrect restored input cursor rejected"))
                    path = root / f"{run}-worker-{worker}.json"
                    path.write_text(json.dumps(config))
                    child = command(f"worker-{worker}", [sys.executable,
                        source / "examples/tensorflow_training_worker.py", "--config", path])
                    workers.append(child)
                reports = []
                for child in workers:
                    output, _ = child.communicate(timeout=120)
                    assert child.returncode == 0, (run, child.returncode)
                    reports.append(json.loads(output))
                state = control(settings, "inspect")
                end = split + iterations if templates else iterations
                assert state["completed"] == [end, end] and not state["active"] and not state["waiting"]
                assert all(s["live_tickets"] == 0 and s["synchronous"]["next_round"] == end for s in state["shard_stats"])
                result = dict(workers=reports, state=state)
                if save:
                    inputs = [dict(worker=r["worker"], input_state=r["input_state"]) for r in reports]
                    # No commit marker is published for an inconsistent input cursor.
                    bad_dir = root / "bad-save"
                    bad_inputs = copy.deepcopy(inputs)
                    bad_inputs[0]["input_state"]["next_iteration"] -= 1
                    base = dict(settings, tables=[table], model=model, workers=inputs)
                    control(dict(base, workers=bad_inputs, checkpoint_directory=str(bad_dir)), "save-job", False)
                    assert not (bad_dir / "manifest.json").exists()
                    assert control(settings, "stats")["paused"]
                    # Force an actual controller-side export failure after the server saved.
                    partial_dir = root / "partial-export"
                    control(dict(base, checkpoint_directory=str(partial_dir)), "save-job", False, file_limit=256)
                    assert (partial_dir / "shard-0.eps").stat().st_size == 256
                    assert not (partial_dir / "manifest.json").exists()
                    assert control(settings, "stats")["paused"]
                    saved = control(dict(base, checkpoint_directory=str(root / "saved")), "save-job")
                    result["checkpoint"] = saved
                    # Directory publication is immutable.
                    control(dict(base, checkpoint_directory=str(root / "saved")), "save-job", False)
                return result
            except Exception:
                for log in logs:
                    log.flush(); log.seek(0); print(log.read()[-6000:], file=sys.stderr)
                raise
            finally:
                for child in reversed(children):
                    stop(child)
                for log in logs:
                    log.close()

        baseline = phase("continuous", rounds)
        prefix = phase("before-restart", split, save=True)
        manifest_path = pathlib.Path(prefix["checkpoint"]["manifest"])
        manifest = json.loads(manifest_path.read_text())
        templates = control(dict(run="after-restart", manifest=str(manifest_path)), "restore-config")
        # Corrupt, missing and schema/clock mismatched checkpoint data are rejected.
        shard_file = manifest_path.parent / "shard-0.eps"
        original = shard_file.read_bytes()
        shard_file.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        control(dict(run="after-restart", manifest=str(manifest_path)), "restore-config", False)
        shard_file.write_bytes(original)
        shard_file.rename(shard_file.with_suffix(".held"))
        control(dict(run="after-restart", manifest=str(manifest_path)), "restore-config", False)
        shard_file.with_suffix(".held").rename(shard_file)
        for field in ("clock", "schema", "dataset", "sha256"):
            bad = copy.deepcopy(manifest)
            if field == "clock": bad["groups"][0]["next_round"] += 1
            elif field == "schema": bad["tables"][0]["dimension"] += 1
            elif field == "dataset": bad["workers"][0]["input_state"]["dataset_sha256"] = "0" * 64
            else: bad["parameters"][0]["sha256"] = "0" * 64
            path = manifest_path.parent / f"bad-{field}.json"
            path.write_text(json.dumps(bad))
            control(dict(run="after-restart", manifest=str(path)), "restore-config", False)
        resumed = phase("after-restart", rounds-split, templates, manifest_path)
        expected = {row["key"]: row for row in baseline["state"]["rows"]}
        maximum = 0.
        for row in resumed["state"]["rows"]:
            assert row["row_updates"] == expected[row["key"]]["row_updates"]
            for field in ("weights", "first_moment", "second_moment"):
                for got, wanted in zip(row[field], expected[row["key"]][field]):
                    assert math.isfinite(got)
                    maximum = max(maximum, abs(got-wanted))
        assert maximum <= 3e-6, maximum
        loss_error = max(abs(w["losses"][i] - baseline["workers"][worker]["losses"][split+i])
                         for worker, w in enumerate(resumed["workers"]) for i in range(rounds-split))
        assert loss_error <= 3e-6, loss_error
        for w in resumed["workers"]:
            assert w["input_state"] == baseline["workers"][w["worker"]]["input_state"]
            assert w["input_state"]["next_iteration"] == rounds and w["input_state"]["next_batch"] == rounds % 3
        result = dict(test="joint_training_checkpoint_restart", device="CPU", shards=2, workers=2,
                      iterations=rounds, checkpoint_round=split, input_cycle_batches=3,
                      max_state_absolute_error=maximum, max_continuation_loss_error=loss_error,
                      compared_rows=len(expected), negative_cases=failures,
                      baseline=baseline, prefix=prefix, resumed=resumed,
                      checkpoint_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k not in ("baseline", "prefix", "resumed")}))


if __name__ == "__main__":
    main()
