"""Train one shared model with one parameter shard/owner/model worker per node.

Fleet credentials stay in an external JSON file. Each node supplies source,
bin_dir, module_dir, python and already connected shard/agent/coordinator
endpoints. --local exercises the same lifecycle with Host owners.
"""
import argparse
import concurrent.futures
import json
import math
import pathlib
import random
import subprocess
import sys
import time
import uuid

from fleet_capacity_sweep import remote


class TerminalJobFailure(RuntimeError):
    pass


def invoke(node, op, options):
    request = {"op": op, "options": options}
    if node.get("local"):
        result = subprocess.run([sys.executable, str(pathlib.Path(options["source"]) / "scripts/training_node.py")],
                                input=json.dumps(request), text=True, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:])
        return json.loads(result.stdout)
    script = "import sys,json\nsys.path.insert(0," + repr(str(pathlib.PurePosixPath(options["source"]) / "scripts")) + ")\n"
    script += "from training_node import dispatch\nprint(json.dumps(dispatch(" + repr(request) + ")))\n"
    return json.loads(remote(node, script, timeout=180))


def parallel(nodes, options, op):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
        return list(pool.map(lambda p: invoke(p[0], op, p[1]), zip(nodes, options)))


def wait_jobs(nodes, options, name):
    next_update = 0
    while True:
        try:
            states = parallel(nodes, options, "status")
        except (subprocess.TimeoutExpired, ConnectionError, RuntimeError) as error:
            print("Observation delayed; retaining existing job handles: " + str(type(error).__name__), flush=True)
            time.sleep(3)
            continue
        finished = True
        for i, state in enumerate(states):
            job = state.get(name)
            if not job:
                raise RuntimeError("missing launched job handle on node " + str(i))
            if "exit_code" in job:
                if job["exit_code"]:
                    raise TerminalJobFailure(name + " failed on node " + str(i))
            elif not job.get("live"):
                raise TerminalJobFailure(name + " handle is no longer live on node " + str(i))
            else:
                finished = False
        if finished:
            return states
        if time.monotonic() >= next_update:
            print(json.dumps({"phase": name, "jobs": [s[name] for s in states]}), flush=True)
            next_update = time.monotonic() + 15
        time.sleep(3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fleet", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--run", default="model-" + uuid.uuid4().hex[:12])
    p.add_argument("--rows", type=int, default=16384)
    p.add_argument("--dimension", type=int, default=64)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--bag", type=int, default=16)
    p.add_argument("--require-capacity", action="store_true")
    p.add_argument("--host", action="store_true")
    a = p.parse_args()
    fleet = json.loads(a.fleet.read_text())
    nodes = fleet["nodes"]
    if not 2 <= len(nodes) <= 8 or not 1 <= a.dimension <= 4096 or not 1 <= a.hidden <= 256:
        p.error("invalid node/model layout")
    packed = (a.dimension * a.hidden + 2 * a.hidden + 1 + a.dimension - 1) // a.dimension
    if a.rows <= packed or not 1 <= a.iterations <= 10000 or not 1 <= a.samples <= 1024 or not 1 <= a.bag <= 128:
        p.error("invalid training workload")
    table = {"id": 1, "dimension": a.dimension, "dtype": "float32",
             "optimizer": {"kind": "sparse_lazy_adam", "learning_rate": .02, "beta1": .9,
                           "beta2": .999, "epsilon": 1e-8, "initial_beta1_power": .9, "initial_beta2_power": .999},
             "initialization": {"min": -.1, "max": .1, "seed": 42}}
    rng = random.Random(20261104)
    batches = [[{"keys": [rng.randrange(packed, a.rows) for _ in range(a.bag)], "label": float(s % 2)}
                for s in range(a.samples)] for _ in nodes]
    touched = sorted(set(range(packed)) | {k for batch in batches for sample in batch for k in sample["keys"]})
    limit = min(64, (4 << 20) // (a.dimension * 12 + 256))
    inspect = touched if len(touched) <= limit else [touched[i * len(touched) // limit] for i in range(limit)]
    options = []
    for i, node in enumerate(nodes):
        options.append({**{k: node[k] for k in ("root", "source", "bin_dir", "module_dir", "python",
                                                "ps_listen", "agent_listen", "coordinator_listen",
                                                "shards", "agents", "coordinator")},
                        "index": i, "count": len(nodes), "run": a.run, "gpu": not a.host,
                        "rows": a.rows, "dimension": a.dimension, "hidden": a.hidden,
                        "iterations": a.iterations, "table": table, "samples": batches[i], "inspect_keys": inspect})
    a.out.mkdir(parents=True, exist_ok=True)
    result = {"run": a.run, "workload": {"rows": a.rows, "dimension": a.dimension, "iterations": a.iterations,
                                         "workers": len(nodes), "local_samples": a.samples, "bag": a.bag},
              "global_parameter_optimizer_bytes": a.rows * a.dimension * 12, "require_capacity": a.require_capacity}
    result["transport"] = fleet.get("transport", "fleet-provided endpoints")
    terminal = False
    try:
        hardware = parallel(nodes, options, "hardware")
        result["hardware_before"] = hardware
        if not a.host:
            if len({h["gpu"]["uuid"] for h in hardware}) != len(nodes):
                raise RuntimeError("nodes do not expose distinct physical GPU UUIDs")
            largest = max(h["gpu"]["total_bytes"] for h in hardware)
            result["largest_single_gpu_bytes"] = largest
            if a.require_capacity and result["global_parameter_optimizer_bytes"] <= largest:
                raise RuntimeError("global parameter state does not exceed physical single-GPU memory")
            for i, h in enumerate(hardware):
                count = (a.rows + len(nodes) - 1 - i) // len(nodes)
                state = count * a.dimension * 12
                if state + (3 << 30) >= h["gpu"]["free_bytes"]:
                    raise RuntimeError("insufficient actual GPU memory for the assigned state and workspace")
                if h["host_available_bytes"] is not None and state + count * 256 + (4 << 30) >= h["host_available_bytes"]:
                    raise RuntimeError("insufficient host memory for local backing and model process")
        elif a.require_capacity:
            raise RuntimeError("Host owners cannot establish GPU capacity")
        print(json.dumps({"phase": "hardware_checked", "global_state_bytes": result["global_parameter_optimizer_bytes"]}), flush=True)
        parallel(nodes, options, "start")
        result["schema_preflight"] = invoke(nodes[0], "preflight", options[0])
        parallel(nodes, options, "prepare")
        wait_jobs(nodes, options, "prepare")
        prepared = parallel(nodes, options, "collect")
        resident = sum(n["prepare"]["shards"][0]["resident_rows"] for n in prepared)
        if resident != a.rows:
            raise RuntimeError("global actual residency differs from requested table")
        result["initialized_resident_rows"] = resident
        print(json.dumps({"phase": "all_rows_resident", "rows": resident}), flush=True)
        parallel(nodes, options, "train")
        wait_jobs(nodes, options, "worker")
        state = invoke(nodes[0], "inspect", options[0])
        reports = parallel(nodes, options, "collect")
        if state["completed"] != [a.iterations] * len(nodes) or state["active"] or state["waiting"] or state["failed"]:
            raise RuntimeError("task progress did not finish the distributed training rounds")
        if state["completed_tasks"] != a.iterations * len(nodes):
            raise RuntimeError("completed task count disagrees with worker clocks")
        for row in state["rows"]:
            if row["row_updates"] != a.iterations or any(not math.isfinite(v) for field in
                ("weights", "first_moment", "second_moment") for v in row[field]):
                raise RuntimeError("sampled optimizer state is invalid")
        for home, shard in enumerate(state["shard_stats"]):
            if shard["live_tickets"] or shard["synchronous"]["next_round"] != a.iterations or shard["synchronous"]["failed"]:
                raise RuntimeError("parameter shard round/ticket state did not drain")
            expected_rows = (a.rows + len(nodes) - 1 - home) // len(nodes)
            cached = reports[0]["worker"]["parameter_stats"][home]["cached_rows"]
            if shard["rows"] != expected_rows or cached != expected_rows or shard["updates"] != a.iterations:
                raise RuntimeError("training changed the full resident row counts or optimizer-step unit")
        first = sum(n["worker"]["losses"][0] for n in reports) / len(nodes)
        last = sum(n["worker"]["losses"][-1] for n in reports) / len(nodes)
        if not math.isfinite(last) or not last < first:
            raise RuntimeError("controlled model training did not lower loss")
        if not a.host and any("GPU:" not in value for n in reports for value in n["worker"]["devices"].values()):
            raise RuntimeError("model computation did not run on GPU")
        result.update(success=True, initial_weighted_loss=first, final_weighted_loss=last,
                      sampled_state=state, node_reports=reports, inspected_rows=len(inspect), touched_rows=len(touched))
        terminal = True
    except TerminalJobFailure:
        terminal = True
        result["success"] = False
        try: result["node_reports"] = parallel(nodes, options, "collect")
        except Exception: pass
        raise
    finally:
        if terminal:
            result["process_cleanup"] = parallel(nodes, options, "stop")
        else:
            result["requires_handle_inspection"] = True
        (a.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("run", "success", "initialized_resident_rows",
                      "global_parameter_optimizer_bytes", "initial_weighted_loss", "final_weighted_loss")}), flush=True)


if __name__ == "__main__":
    main()
