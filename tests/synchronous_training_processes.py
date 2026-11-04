"""Real sparse synchronous training versus an independent float32 Adam reference."""
import json
import math
import pathlib
import struct
import subprocess
import sys
import tempfile
import ctypes
import ctypes.util

from rpc_processes import ready, stop


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


_fmaf = ctypes.CDLL(ctypes.util.find_library("m") or None).fmaf
_fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
_fmaf.restype = ctypes.c_float


def reference(examples, rounds, reduction, fused=False):
    rows = {key: {"weights": [0.], "first_moment": [0.], "second_moment": [0.]}
            for samples in examples for sample in samples for key in sample["keys"]}
    b1, b2, eps, lr = map(f32, (.9, .999, 1e-8, .02))
    p1, p2 = b1, b2
    for _ in range(rounds):
        gradients = []
        for samples in examples:
            local = {}
            for sample in samples:
                score = 0.
                for key in sample["keys"]:
                    score = f32(score + rows[key]["weights"][0])
                gradient = f32((1 / (1 + math.exp(-score)) - sample["label"]) / len(samples))
                for key in sample["keys"]:
                    local[key] = local.get(key, 0.) + gradient
            gradients.append({key: f32(value) for key, value in local.items()})
        normalizer = sum(map(len, examples)) if reduction == "sample_mean" else len(examples)
        alpha = f32(f32(lr * f32(math.sqrt(f32(1 - p2)))) / f32(1 - p1))
        for key, row in rows.items():
            gradient = f32(sum(local.get(key, 0.) * (len(examples[w]) if reduction == "sample_mean" else 1)
                               / normalizer for w, local in enumerate(gradients)))
            mg = f32(f32(1 - b1) * gradient)
            vg = f32(f32(f32(1 - b2) * gradient) * gradient)
            m = _fmaf(b1, row["first_moment"][0], mg) if fused else f32(f32(b1 * row["first_moment"][0]) + mg)
            v = _fmaf(b2, row["second_moment"][0], vg) if fused else f32(f32(b2 * row["second_moment"][0]) + vg)
            weight = f32(row["weights"][0] - f32(f32(alpha * m) / f32(f32(math.sqrt(v)) + eps)))
            row.update(weights=[weight], first_moment=[m], second_moment=[v])
        p1, p2 = f32(p1 * b1), f32(p2 * b2)
    return rows


def main():
    server, coordinator, worker, control = map(str, map(pathlib.Path, sys.argv[1:]))
    rounds = 40
    # Unequal local batch sizes and different sparse working sets. Home shard 2
    # has no keys but must receive every empty contribution and advance its step.
    examples = [[{"keys": keys, "label": w % 2}] * (w + 1)
                for w, keys in enumerate(([1, 3], [1, 4], [3], [4]))]
    reports = []
    for reduction in ("worker_mean", "sample_mean"):
        with tempfile.TemporaryDirectory(prefix="eps-sync-") as temporary:
            root = pathlib.Path(temporary)
            processes, logs = [], []

            def launch(binary, name, config):
                path = root / f"{name}.json"
                path.write_text(json.dumps(config))
                log = open(root / f"{name}.log", "w+")
                logs.append(log)
                p = subprocess.Popen([binary, "--config", str(path)], stdout=subprocess.PIPE,
                                     stderr=log, text=True)
                processes.append(p)
                return p

            try:
                run = f"synchronous-{reduction}"
                endpoints = []
                table = {"id": 1, "dimension": 1, "dtype": "float32",
                         "optimizer": {"kind": "sparse_lazy_adam", "learning_rate": .02,
                                       "beta1": .9, "beta2": .999, "epsilon": 1e-8,
                                       "initial_beta1_power": .9, "initial_beta2_power": .999},
                         "initialization": {"min": 0., "max": 0., "seed": 42}}
                for home in range(3):
                    endpoints.append(ready(launch(server, f"shard-{home}", {
                        "schema_version": 1, "run_id": run, "listen": "127.0.0.1:0",
                        "partition": {"shard_id": home, "shard_count": 3, "epoch": 1},
                        "tables": [table], "cache": {"enabled": False},
                        "synchronous_training": {"workers": 4, "reduction": reduction,
                                                 "round_timeout_ms": 10000},
                    })))
                coordinator_address = ready(launch(coordinator, "coordinator", {
                    "schema_version": 1, "run": run, "epoch": 1, "workers": 4,
                    "consistency": "sequential", "max_delay": 0,
                }))
                settings = {"schema_version": 1, "run": run, "coordinator": coordinator_address,
                            "shards": endpoints, "epoch": 1, "keys": [1, 3, 4], "synchronous": True}
                children = [launch(worker, f"worker-{w}", {
                    **settings, "worker": w, "iterations": rounds, "samples": samples,
                    "update_mode": "synchronous", "retry_completed_submit": True,
                    "delay_ms": 2 if w == 3 else 0, "delay_every": 3,
                }) for w, samples in enumerate(examples)]
                workers = []
                for p in children:
                    stdout, _ = p.communicate(timeout=30)
                    assert p.returncode == 0, f"worker failed: {p.returncode}"
                    workers.append(json.loads(stdout))
                admin = root / "inspect.json"
                admin.write_text(json.dumps(settings))
                output = subprocess.run([control, "--config", str(admin), "--action", "inspect"],
                                        check=True, capture_output=True, text=True, timeout=10)
                state = json.loads(output.stdout)
                assert state["completed"] == [rounds] * 4 and state["completed_tasks"] == 4 * rounds, state
                assert not state["active"] and not state["waiting"] and not state["failed"], state
                # Compilers can fuse the moment multiply/add. Compare against
                # both complete FP32 references, selecting one arithmetic for
                # the entire model rather than mixing per-field outcomes.
                references = {"separate": reference(examples, rounds, reduction),
                              "fma": reference(examples, rounds, reduction, fused=True)}
                errors = {arithmetic: 0. for arithmetic in references}
                for row in state["rows"]:
                    assert row["row_updates"] == rounds, row
                    for field in ("weights", "first_moment", "second_moment"):
                        value = row[field][0]
                        assert math.isfinite(value), row
                        for arithmetic, expected in references.items():
                            errors[arithmetic] = max(errors[arithmetic], abs(value - expected[row["key"]][field][0]))
                arithmetic = min(errors, key=errors.get)
                error = errors[arithmetic]
                assert error < 2e-6, (reduction, state["rows"], errors)
                assert len(state["rows"]) == len(references[arithmetic])
                for shard in state["shard_stats"]:
                    assert shard["live_tickets"] == 0 and shard["updates"] == rounds, shard
                    assert shard["synchronous"]["next_round"] == rounds, shard
                    assert not shard["synchronous"]["received_workers"] and not shard["synchronous"]["retained_bytes"], shard
                    assert shard["synchronous"]["duplicate_submissions"] == rounds * 4, shard
                    assert not shard["synchronous"]["failed"], shard
                assert state["shard_stats"][2]["rows"] == 0, state["shard_stats"]
                assert all(w["optimizer_step"] == "per_group_round" and w["completed"] == rounds for w in workers)
                reports.append({"reduction": reduction, "workers": 4, "parameter_shards": 3,
                                "worker_batches": rounds * 4, "optimizer_steps_per_shard": rounds,
                                "empty_shard_participated": True, "duplicate_submits_per_shard": rounds * 4,
                                "reference_arithmetic": arithmetic, "reference_errors": errors,
                                "reference_max_absolute_error": error, "rows": state["rows"]})
            except Exception:
                for log in logs:
                    log.flush()
                    log.seek(0)
                    print(log.read()[-5000:], file=sys.stderr)
                raise
            finally:
                for p in reversed(processes):
                    stop(p)
                for log in logs:
                    log.close()
    print(json.dumps({"test": "synchronous_training_processes", "reports": reports}, sort_keys=True))


if __name__ == "__main__":
    main()
