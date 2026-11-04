"""Node-side lifecycle for sharded model training; requests arrive through SSH/stdin."""
import json
import os
import pathlib
import signal
import subprocess
import sys
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.chmod(0o600)
    temporary.replace(path)


def live(pid):
    if not pid:
        return False
    if pathlib.Path("/proc").exists():
        p = pathlib.Path("/proc") / str(pid)
        try:
            return bool((p / "cmdline").read_bytes()) and (p / "stat").read_text().split()[2] != "Z"
        except FileNotFoundError:
            return False
    r = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip().startswith("Z")


def hardware(gpu):
    result = {"gpu": None, "host_available_bytes": None, "cgroup_limit_bytes": None}
    if gpu:
        values = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free",
                                          "--format=csv,noheader,nounits"], text=True).splitlines()
        if len(values) != 1:
            raise RuntimeError("node mode requires exactly one visible physical GPU")
        uuid, name, total, free = map(str.strip, values[0].split(","))
        result["gpu"] = {"uuid": uuid, "name": name, "total_bytes": int(total) << 20,
                         "free_bytes": int(free) << 20}
    p = pathlib.Path("/proc/meminfo")
    if p.exists():
        result["host_available_bytes"] = int(next(s.split()[1] for s in p.read_text().splitlines()
                                                   if s.startswith("MemAvailable:"))) * 1024
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        p = pathlib.Path(name)
        if p.exists() and p.read_text().strip().isdigit():
            result["cgroup_limit_bytes"] = int(p.read_text())
            used = p.with_name("memory.current" if p.name == "memory.max" else "memory.usage_in_bytes")
            if used.exists():
                available = max(0, result["cgroup_limit_bytes"] - int(used.read_text()))
                result["host_available_bytes"] = min(result["host_available_bytes"] or available, available)
            break
    return result


def paths(o):
    root = pathlib.Path(o["root"])
    run = o["run"]
    if not root.is_absolute() or not run or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run):
        raise ValueError("invalid node root or run")
    work = root / "results" / run
    work.mkdir(parents=True, exist_ok=True)
    return root, work


def environment(o):
    root, _ = paths(o)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([str(root / "portable-runtime/lib"), str(root / "dependencies/lib"),
                                        env.get("LD_LIBRARY_PATH", "")])
    env["PYTHONPATH"] = ":".join([o["module_dir"], str(pathlib.Path(o["source"]) / "adapters/entropy")])
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    return env


def launch(o, name, command, config=None, env=None):
    _, work = paths(o)
    result = work / (name + ".result.json")
    job = work / (name + ".job.json")
    if job.exists():
        raise RuntimeError("job already exists; inspect its handle instead of restarting")
    if config is not None:
        path = work / (name + ".config.json")
        write(path, config)
        command = [*command, "--config", str(path)]
    write(job, {"command": list(map(str, command)), "env": env or environment(o), "result": str(result)})
    with (work / (name + ".wrapper.log")).open("w") as log:
        child = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve()), "--execute", str(job)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    write(work / (name + ".handle.json"), {"pid": child.pid, "job": str(job)})
    return {"pid": child.pid, "name": name}


def execute(job):
    info = json.loads(job.read_text())
    prefix = job.with_name(job.name.removesuffix(".job.json"))
    with pathlib.Path(str(prefix) + ".stdout").open("w") as out, pathlib.Path(str(prefix) + ".stderr").open("w") as err:
        child = subprocess.Popen(info["command"], env=info["env"], stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        write(pathlib.Path(str(prefix) + ".child.json"), {"pid": child.pid})
        code = child.wait()
    write(pathlib.Path(info["result"]), {"exit_code": code, "finished": time.time()})


def ready(o, name, timeout=40):
    _, work = paths(o)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        path = work / (name + ".stdout")
        if path.exists() and any(s.startswith("READY ") for s in path.read_text().splitlines()):
            return
        if (work / (name + ".result.json")).exists():
            raise RuntimeError(name + " exited: " + (work / (name + ".stderr")).read_text()[-2500:])
        time.sleep(.1)
    raise TimeoutError(name + " did not become ready; inspect existing handle")


def status(o):
    _, work = paths(o)
    result = {}
    for path in work.glob("*.handle.json"):
        name = path.name.removesuffix(".handle.json")
        handle = json.loads(path.read_text())
        terminal = work / (name + ".result.json")
        result[name] = json.loads(terminal.read_text()) if terminal.exists() else {"live": live(handle["pid"]), "pid": handle["pid"]}
    return result


def monitor(o):
    _, work = paths(o)
    with (work / "resources.jsonl").open("a") as out:
        while True:
            record = {"time": time.time(), "rss_bytes": {}, "gpu": None}
            if o["gpu"]:
                raw = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,utilization.memory",
                                                "--format=csv,noheader,nounits"], text=True).split(",")
                record["gpu"] = {"used_bytes": int(raw[0]) << 20, "compute_percent": int(raw[1]),
                                 "memory_percent": int(raw[2])}
            for path in work.glob("*.child.json"):
                pid = json.loads(path.read_text())["pid"]
                try:
                    if pathlib.Path("/proc").exists():
                        fields = pathlib.Path("/proc", str(pid), "stat").read_text().split()
                        record["rss_bytes"][path.name.removesuffix(".child.json")] = int(fields[23]) * os.sysconf("SC_PAGE_SIZE")
                    else:
                        raw = subprocess.check_output(["ps", "-p", str(pid), "-o", "rss="], text=True).strip()
                        record["rss_bytes"][path.name.removesuffix(".child.json")] = int(raw) * 1024
                except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
                    pass
            for name in ("memory.current", "memory.peak"):
                path = pathlib.Path("/sys/fs/cgroup") / name
                if path.exists():
                    record[name] = int(path.read_text())
            out.write(json.dumps(record) + "\n"); out.flush(); time.sleep(1)


def dispatch(request):
    op, o = request["op"], request["options"]
    root, work = paths(o)
    binary = pathlib.Path(o["bin_dir"])
    if op == "hardware":
        return hardware(o["gpu"])
    if op == "start":
        rows = (o["rows"] + o["count"] - 1 - o["index"]) // o["count"]
        agent = [str(binary / "eps_cache_agent"), "--run", o["run"], "--listen", o["agent_listen"],
                 "--agent", str(o["index"]), "--backend", "cuda" if o["gpu"] else "host",
                 "--rows-per-table", str(rows), "--state-bytes", str(rows * (o["dimension"] * 12 + 8) + 4096)]
        launch(o, "agent", agent); ready(o, "agent")
        ps = {"schema_version": 1, "run_id": o["run"], "listen": o["ps_listen"],
              "partition": {"shard_id": o["index"], "shard_count": o["count"], "epoch": 1},
              "tables": [o["table"]], "cache": {"enabled": True},
              "agents": [{"id": o["index"], "endpoint": o["agent_listen"]}],
              "synchronous_training": {"workers": o["count"], "reduction": "sample_mean", "round_timeout_ms": 20000},
              "limits": {"cpu_state_bytes": max(1 << 26, rows * (o["dimension"] * 12 + 256))},
              "snapshot_directory": str(work / "snapshots")}
        launch(o, "ps", [str(binary / "eps_server")], ps); ready(o, "ps")
        if not o["index"]:
            launch(o, "coordinator", [str(binary / "eps_training_coordinator")], {
                "schema_version": 1, "run": o["run"], "listen": o["coordinator_listen"],
                "workers": o["count"], "consistency": "sequential", "max_delay": 0})
            ready(o, "coordinator")
        launch(o, "monitor", [sys.executable, str(pathlib.Path(__file__).resolve()), "--monitor", json.dumps(o)])
        return status(o)
    if op == "preflight":
        script = "from eps_backend import Backend\nimport json,numpy as np\n" + "o=" + repr(o) + "\n" + r'''
b=Backend(o['shards'],o['run'],1021)
try:
 b.create_table(json.dumps(o['table']))
 ids=np.arange(o['count'],dtype=np.uint64)
 handle,arrays=b.pull([dict(table=1,feature=0,pooling='sequence',keys=ids,offsets=np.array([0,len(ids)],np.uint64))],True)
 b.release(handle)
 assert arrays[0].shape==(o['count'],o['dimension'])
 print(json.dumps({'verified_dimension':int(arrays[0].shape[1]),'shards':o['count']}))
finally:b.close()
'''
        r = subprocess.run([o["python"], "-c", script], env=environment(o), capture_output=True, text=True, timeout=35)
        if r.returncode: raise RuntimeError(r.stderr[-2000:])
        return json.loads(r.stdout)
    if op == "prepare":
        return launch(o, "prepare", [str(binary / "eps_training_prepare")], {
            "run": o["run"], "shards": o["shards"], "rows": o["rows"], "dimension": o["dimension"],
            "only_shard": o["index"], "workers": o["count"], "agents": o["agents"]})
    if op == "train":
        env = environment(o); env["CUDA_VISIBLE_DEVICES"] = "0" if o["gpu"] else ""
        return launch(o, "worker", [o["python"], str(pathlib.Path(o["source"]) / "examples/tensorflow_training_worker.py")],
                      {"run": o["run"], "shards": o["shards"], "coordinator": o["coordinator"],
                       "worker": o["index"], "dimension": o["dimension"], "hidden": o["hidden"],
                       "iterations": o["iterations"], "samples": o["samples"], "table": o["table"],
                       "agents": o["agents"], "device": "/GPU:0" if o["gpu"] else "/CPU:0"}, env)
    if op == "status": return status(o)
    if op == "inspect":
        path = work / "inspect.json"
        write(path, {"run": o["run"], "coordinator": o["coordinator"], "shards": o["shards"],
                     "keys": o["inspect_keys"], "synchronous": True})
        r = subprocess.run([str(binary / "eps_training_control"), "--config", str(path), "--action", "inspect"],
                           env=environment(o), capture_output=True, text=True, timeout=30)
        if r.returncode: raise RuntimeError(r.stderr[-2000:])
        return json.loads(r.stdout)
    if op == "collect":
        result = {"jobs": status(o), "hardware": hardware(o["gpu"])}
        for name in ("prepare", "worker"):
            path = work / (name + ".stdout")
            if path.exists() and path.read_text().strip():
                result[name] = json.loads(path.read_text())
        resources = []
        if (work / "resources.jsonl").exists():
            for line in (work / "resources.jsonl").read_text().splitlines():
                try: resources.append(json.loads(line))
                except ValueError: pass
        result["resources"] = resources
        return result
    if op == "stop":
        groups = []
        for name in ("worker", "prepare", "coordinator", "ps", "agent", "monitor"):
            path = work / (name + ".handle.json")
            if not path.exists(): continue
            handle = json.loads(path.read_text())
            child_file = work / (name + ".child.json")
            child = json.loads(child_file.read_text())["pid"] if child_file.exists() else None
            groups.append((handle["pid"], child))
            if live(handle["pid"]):
                # Every wrapper executes only its uniquely named job file.
                command = subprocess.check_output(["ps", "-p", str(handle["pid"]), "-o", "command="], text=True)
                if handle["job"] not in command: raise RuntimeError("process identity changed")
                try: os.killpg(handle["pid"], signal.SIGTERM)
                except ProcessLookupError: pass
            elif live(child):
                command = subprocess.check_output(["ps", "-ww", "-p", str(child), "-o", "command="], text=True)
                if o["run"] not in command: raise RuntimeError("child process identity changed")
                try: os.killpg(handle["pid"], signal.SIGTERM)
                except ProcessLookupError: pass
        deadline = time.monotonic() + 8
        while any(live(parent) or live(child) for parent, child in groups) and time.monotonic() < deadline:
            time.sleep(.1)
        for parent, child in groups:
            candidate = child if live(child) else parent if live(parent) else None
            if candidate:
                command = subprocess.check_output(["ps", "-ww", "-p", str(candidate), "-o", "command="], text=True)
                if o["run"] not in command: raise RuntimeError("remaining process identity changed")
                try: os.killpg(parent, signal.SIGKILL)
                except ProcessLookupError: pass
        return {"stop_requested": True, "owned_processes_gone": all(not live(p) and not live(c) for p, c in groups)}
    raise ValueError("unknown node operation")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--execute"]: execute(pathlib.Path(sys.argv[2]))
    elif sys.argv[1:2] == ["--monitor"]: monitor(json.loads(sys.argv[2]))
    else: print(json.dumps(dispatch(json.load(sys.stdin))))
