"""Train through real worker/coordinator/shard OS processes and BRPC sockets."""
import json
import math
import pathlib
import subprocess
import sys
import tempfile

from rpc_processes import ready, stop


def main():
    server, coordinator, worker, control = map(str, map(pathlib.Path, sys.argv[1:]))
    reports = []
    with tempfile.TemporaryDirectory(prefix="eps-training-") as tmp:
        root = pathlib.Path(tmp)
        for policy, delay in (("sequential", 0), ("bounded_delay", 2), ("eventual", 0)):
            processes, logs = [], []

            def launch(binary, name, config):
                path = root / f"{policy}-{name}.json"
                path.write_text(json.dumps(config))
                log = open(root / f"{policy}-{name}.log", "w+")
                logs.append(log)
                p = subprocess.Popen([binary, "--config", str(path)],
                                     stdout=subprocess.PIPE, stderr=log, text=True)
                processes.append(p)
                return p

            try:
                run = f"training-{policy}"
                addresses = []
                for shard in range(2):
                    p = launch(server, f"shard-{shard}", {
                        "schema_version": 1, "run_id": run, "listen": "127.0.0.1:0",
                        "partition": {"shard_id": shard, "shard_count": 2, "epoch": 1},
                        "tables": [], "cache": {"enabled": False},
                        "snapshot_directory": str(root / policy / "snapshots"),
                    })
                    addresses.append(ready(p))
                endpoint = ready(launch(coordinator, "coordinator", {
                    "schema_version": 1, "run": run, "workers": 4, "epoch": 1,
                    "consistency": policy, "max_delay": delay,
                }))
                settings = {"schema_version": 1, "run": run, "epoch": 1,
                            "coordinator": endpoint, "shards": addresses,
                            "keys": list(range(1, 6))}
                admin = root / f"{policy}-admin.json"
                admin.write_text(json.dumps(settings))

                def command(action, config=admin):
                    r = subprocess.run([control, "--config", str(config), "--action", action],
                                       capture_output=True, text=True, check=True, timeout=10)
                    return json.loads(r.stdout)

                children = []
                for w in range(4):
                    children.append(launch(worker, f"worker-{w}", {
                        **settings, "worker": w, "iterations": 40,
                        "delay_ms": 2 if w == 3 else 0, "delay_every": 3,
                        # Two examples verify mean-loss scaling within a batch.
                        "samples": [{"keys": [1, w + 2], "label": w % 2}] * 2,
                    }))
                outputs = []
                for p in children:
                    stdout, _ = p.communicate(timeout=30)
                    if p.returncode:
                        raise RuntimeError(f"training worker failed: {p.returncode}")
                    outputs.append(json.loads(stdout))
                state = command("inspect")
                assert state["completed"] == [40] * 4, state
                assert state["completed_tasks"] == state["granted_tasks"] == 160, state
                assert not state["active"] and not state["waiting"] and not state["failed"], state
                rows = {r["key"]: r for r in state["rows"]}
                assert set(rows) == set(range(1, 6)), rows
                for key, row in rows.items():
                    assert row["row_updates"] == (160 if key == 1 else 40), row
                    assert all(math.isfinite(v) for field in ("weights", "first_moment", "second_moment")
                               for v in row[field]), row
                for s in state["shard_stats"]:
                    assert s["live_tickets"] == 0, s
                loss = 0.
                for w in range(4):
                    score = rows[1]["weights"][0] + rows[w + 2]["weights"][0]
                    loss += (max(score, 0) - (w % 2) * score + math.log1p(math.exp(-abs(score)))) / 4
                assert loss < math.log(2) and math.isfinite(loss), loss
                for output in outputs:
                    assert output["completed"] == output["input_batches_committed"] == 40, output
                    assert output["gradient_reduction"] == "local_batch_mean", output
                    assert output["optimizer_step"] == "per_worker_update", output
                    if policy != "eventual":
                        assert output["max_admission_lead"] <= delay, output
                command("pause")
                checkpoint = command("checkpoint")
                assert checkpoint["completed"] == [40] * 4, checkpoint
                restored = ready(launch(coordinator, "restored-progress", {
                    "schema_version": 1, "run": run + "-resumed", "workers": 4, "epoch": 1,
                    "consistency": policy, "max_delay": delay, "restore": checkpoint,
                }))
                restored_config = root / f"{policy}-restored-admin.json"
                restored_config.write_text(json.dumps({**settings, "coordinator": restored,
                                                       "run": run + "-resumed"}))
                restored_stats = command("stats", restored_config)
                assert restored_stats["paused"] and restored_stats["completed"] == [40] * 4
                command("resume", restored_config)
                assert not command("stats", restored_config)["paused"]
                reports.append({"policy": policy, "workers": 4, "parameter_shards": 2,
                                "worker_updates": 160, "initial_loss": math.log(2),
                                "final_loss": loss, "tickets_drained": True,
                                "progress_checkpoint_restored": True, "workers_report": outputs})
            except Exception:
                for log in logs:
                    log.flush()
                    log.seek(0)
                    print(log.read()[-4000:], file=sys.stderr)
                raise
            finally:
                for p in reversed(processes):
                    stop(p)
                for log in logs:
                    log.close()
    print(json.dumps({"test": "distributed_training_processes", "reports": reports}, sort_keys=True))


if __name__ == "__main__":
    main()
