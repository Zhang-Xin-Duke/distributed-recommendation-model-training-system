"""Automatic admission/replacement against two external PS shards and one agent."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from rpc_processes import ready, stop


def main():
    server, client, agent = sys.argv[1:]
    backend = os.environ.get("EPS_CACHE_BACKEND", "host")
    with tempfile.TemporaryDirectory(prefix="eps-policy-rpc-") as temporary:
        root = pathlib.Path(temporary)
        for cache, expected in [
            ({"enabled": False, "admission": "unknown"}, "cache.admission must be"),
            ({"enabled": False, "admission": "reuse_cost"}, "automatic cache admission requires"),
        ]:
            invalid = root / "invalid.json"
            invalid.write_text(json.dumps({"schema_version": 1, "cache": cache}))
            rejected = subprocess.run([server, "--config", str(invalid)],
                                      capture_output=True, text=True, timeout=10)
            assert rejected.returncode and expected in rejected.stderr, rejected.stderr
        for resumed in (False, True):
            processes, logs, endpoints = [], [], []
            run = "auto-resumed" if resumed else "auto-fresh"
            try:
                log = open(root / f"agent-{resumed}.log", "w+")
                logs.append(log)
                worker = subprocess.Popen([agent, "--run", run, "--backend", backend,
                    "--state-bytes", "192", "--rows-per-table", "6", "--listen", "127.0.0.1:0"],
                    stdout=subprocess.PIPE, stderr=log, text=True)
                processes.append(worker)
                agent_endpoint = ready(worker)
                for shard in range(2):
                    config = root / f"shard-{shard}.json"
                    settings = {
                        "schema_version": 1, "run_id": run, "listen": "127.0.0.1:0",
                        "partition": {"shard_id": shard, "shard_count": 2, "epoch": 1},
                        "tables": [], "cache": {
                            "enabled": True, "admission": "reuse_cost", "policy_agents": [0],
                            "decision_window_requests": 2, "max_tracked_rows": 16,
                            "minimum_accesses": 1, "window_decay": 0.5,
                            "minimum_residence_windows": 1, "replacement_margin": 0.1,
                            "saved_microseconds_per_access": 100,
                            "load_microseconds_per_byte": 0,
                            "writeback_microseconds_per_byte": 0,
                            "learn_migration_costs": False,
                            "network_bytes_per_window": 1024, "staging_bytes_limit": 1024,
                            "max_inflight_batches": 4, "reserve_slots": 1,
                        },
                        "agents": [{"id": 0, "endpoint": agent_endpoint}],
                        "snapshot_directory": str(root / "snapshots"),
                    }
                    config.write_text(json.dumps(settings))
                    if not resumed and shard == 0:
                        invalid = root / "invalid-bounds.json"
                        bad = json.loads(json.dumps(settings))
                        bad["cache"]["decision_window_requests"] = 0
                        invalid.write_text(json.dumps(bad))
                        rejected = subprocess.run([server, "--config", str(invalid)],
                                                  capture_output=True, text=True, timeout=10)
                        assert rejected.returncode and "invalid cache policy bounds" in rejected.stderr, rejected.stderr
                    command = [server, "--config", str(config)]
                    if resumed:
                        command += ["--restore", str(root / "snapshots" / "checkpoint" / f"shard-{shard}.bin")]
                    log = open(root / f"server-{shard}-{resumed}.log", "w+")
                    logs.append(log)
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True)
                    processes.append(process)
                    endpoints.append(ready(process))
                subprocess.run([client, *endpoints, agent_endpoint, "resumed" if resumed else "fresh"],
                               check=True, timeout=45)
            except Exception:
                for log in logs:
                    log.flush()
                    log.seek(0)
                    print(log.read()[-6000:], file=sys.stderr)
                raise
            finally:
                for process in reversed(processes):
                    stop(process)
                for log in logs:
                    log.close()


if __name__ == "__main__":
    main()
