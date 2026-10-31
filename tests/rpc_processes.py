"""Launch real BRPC server processes and verify a two-shard restart lifecycle."""
import json
import pathlib
import selectors
import subprocess
import sys
import tempfile
import time


def ready(process, timeout=15):
    deadline = time.monotonic() + timeout
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited with {process.returncode}")
            for _key, _event in selector.select(timeout=0.1):
                line = process.stdout.readline().strip()
                if line.startswith("READY "):
                    return line.removeprefix("READY ")
        raise RuntimeError("parameter server did not become ready")
    finally:
        selector.close()


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def main():
    server, client = map(str, map(pathlib.Path, sys.argv[1:]))
    with tempfile.TemporaryDirectory(prefix="eps-rpc-") as temporary:
        root = pathlib.Path(temporary)
        for resumed in (False, True):
            processes, logs, endpoints = [], [], []
            try:
                for shard in range(2):
                    config = root / f"shard-{shard}.json"
                    config.write_text(json.dumps({
                        "schema_version": 1,
                        "run_id": "wire-resumed" if resumed else "wire-test",
                        "listen": "127.0.0.1:0",
                        "partition": {"shard_id": shard, "shard_count": 2, "epoch": 1},
                        "tables": [],
                        "cache": {"enabled": False},
                        "snapshot_directory": str(root / "snapshots"),
                    }))
                    command = [server, "--config", str(config)]
                    if resumed:
                        command += ["--restore", str(root / "snapshots" / "checkpoint" / f"shard-{shard}.bin")]
                    log = open(root / f"server-{shard}-{resumed}.log", "w+")
                    logs.append(log)
                    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                               stderr=log, text=True)
                    processes.append(process)
                    endpoints.append(ready(process))
                subprocess.run([client, *endpoints, "resumed" if resumed else "fresh"],
                               check=True, timeout=30)
            except Exception:
                for log in logs:
                    log.flush()
                    log.seek(0)
                    print(log.read()[-6000:], file=sys.stderr)
                raise
            finally:
                for process in processes:
                    stop(process)
                for log in logs:
                    log.close()


if __name__ == "__main__":
    main()
