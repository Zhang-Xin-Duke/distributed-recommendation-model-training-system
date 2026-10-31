"""Two parameter shards and a separate worker-hosted reference cache agent."""
import json
import pathlib
import os
import subprocess
import sys
import tempfile

from rpc_processes import ready, stop


def main():
    server, client, agent = sys.argv[1:]
    backend = os.environ.get("EPS_CACHE_BACKEND", "host")
    print(f"CACHE_BACKEND={backend}", flush=True)
    with tempfile.TemporaryDirectory(prefix="eps-cache-rpc-") as temporary:
        root = pathlib.Path(temporary)
        for resumed in (False, True):
            processes, logs, endpoints = [], [], []
            run = "wire-resumed" if resumed else "wire-test"
            try:
                log = open(root / f"agent-{resumed}.log", "w+")
                logs.append(log)
                worker = subprocess.Popen([agent, "--run", run, "--backend", backend,
                                           "--listen", "127.0.0.1:0"],
                                          stdout=subprocess.PIPE, stderr=log, text=True)
                processes.append(worker)
                agent_endpoint = ready(worker)
                for shard in range(2):
                    config = root / f"shard-{shard}.json"
                    config.write_text(json.dumps({
                        "schema_version": 1, "run_id": run, "listen": "127.0.0.1:0",
                        "partition": {"shard_id": shard, "shard_count": 2, "epoch": 1},
                        "tables": [], "cache": {"enabled": True},
                        "access": {"policy": "transfer_unique", "agent_id": 0},
                        "agents": [{"id": 0, "endpoint": agent_endpoint}],
                        "snapshot_directory": str(root / "snapshots"),
                    }))
                    command = [server, "--config", str(config)]
                    if resumed:
                        command += ["--restore", str(root / "snapshots" / "checkpoint" / f"shard-{shard}.bin")]
                    log = open(root / f"server-{shard}-{resumed}.log", "w+")
                    logs.append(log)
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log, text=True)
                    processes.append(process)
                    endpoints.append(ready(process))
                subprocess.run([client, *endpoints, "resumed" if resumed else "fresh", agent_endpoint],
                               check=True, timeout=30)
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
