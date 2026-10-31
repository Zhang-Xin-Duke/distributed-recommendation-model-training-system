"""Compare fixed sparse-training traces with calibrated cache placement modes."""
import argparse
import hashlib
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from rpc_processes import ready, stop


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def run_trial(args, name, mode, calibration, saved):
    root = args.out / name
    root.mkdir()
    processes, logs, endpoints = [], [], []
    run = "policy-"+uuid.uuid4().hex[:12]
    steps = args.calibration_steps if calibration else args.phase_steps
    try:
        log = (root / "agent.log").open("w+")
        logs.append(log)
        row_bytes = args.dimension*12+8
        # CUDA preallocates the benchmark and temporary warmup tables. The Host
        # backend uses one shared pool; only the benchmark remains resident.
        state_bytes = args.cache_rows*row_bytes*(2 if args.backend == "cuda" else 1)
        worker = subprocess.Popen([str(args.build / "eps_cache_agent"), "--run", run,
            "--backend", args.backend, "--device", str(args.device),
            "--rows-per-table", str(args.cache_rows), "--state-bytes", str(state_bytes),
            "--listen", "127.0.0.1:0"], stdout=subprocess.PIPE, stderr=log, text=True)
        processes.append(worker)
        agent_endpoint = ready(worker, timeout=45)
        for shard in range(2):
            cache = dict(enabled=True, admission=mode if mode in ("frequency", "reuse_cost") else "manual",
                decision_window_requests=args.window_requests, max_tracked_rows=args.max_tracked_rows,
                minimum_accesses=2, window_decay=0.5, minimum_residence_windows=2,
                replacement_margin=0.25, max_rows_per_batch=args.migration_rows,
                saved_microseconds_per_access=saved, load_microseconds_per_byte=0.001,
                writeback_microseconds_per_byte=0.001, cost_average_alpha=0.2,
                learn_migration_costs=True, network_bytes_per_window=args.window_bytes,
                staging_bytes_limit=args.window_bytes, max_inflight_batches=8,
                reserve_slots=args.reserve_slots)
            config = root / f"ps-{shard}.json"
            dump(config, dict(schema_version=1, run_id=run, listen="127.0.0.1:0",
                partition=dict(shard_id=shard, shard_count=2, epoch=1), tables=[], cache=cache,
                agents=[dict(id=0, endpoint=agent_endpoint)], access=dict(policy="cpu_only"),
                limits=dict(cpu_state_bytes=4 << 30, max_live_tickets=64,
                            ticket_payload_bytes=128 << 20, queue_payload_bytes=128 << 20),
                snapshot_directory=str(root / "snapshots")))
            log = (root / f"ps-{shard}.log").open("w+")
            logs.append(log)
            server = subprocess.Popen([str(args.build / "eps_server"), "--config", str(config)],
                                       stdout=subprocess.PIPE, stderr=log, text=True)
            processes.append(server)
            endpoints.append(ready(server, timeout=30))
        config = dict(endpoints=endpoints, agent=agent_endpoint, run=run, mode=mode,
            rows=args.rows, dimension=args.dimension, keys=args.keys, bag=args.bag,
            hot_rows=args.hot_rows, phase_steps=steps, cache_rows=args.cache_rows,
            reserve_slots=args.reserve_slots, calibration=calibration, coalesce=args.coalesce,
            network_bytes_per_window=args.window_bytes, max_inflight_batches=8,
            saved_microseconds_per_access=saved)
        dump(root / "probe.json", config)
        with (root / "raw.jsonl").open("w") as output, (root / "probe.log").open("w+") as log:
            probe = subprocess.Popen([str(args.build / "eps_cache_policy_probe"), str(root / "probe.json")],
                                     stdout=output, stderr=log)
            processes.append(probe)
            code = probe.wait(timeout=args.trial_timeout)
            if code:
                raise RuntimeError(f"probe exited {code}")
        records = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
        expected = ["stable"] if calibration else ["stable", "shift", "scan_mixed"]
        if [r["phase"] for r in records] != expected:
            raise ValueError("missing trace phase")
        for r in records:
            assert r["errors"] == 0 and r["checked_final_rows"] >= args.hot_rows
            assert r["checked_feature_values"] == steps*(args.keys//args.bag)*args.dimension
            assert r["checked_final_state_values"] == r["checked_final_rows"]*args.dimension*3
            assert r["resident_id_occurrences"]+r["transferred_id_occurrences"]+r["cpu_pooled_id_occurrences"] == steps*args.keys
            assert r["final_free_slots"] == args.cache_rows
        shutil.rmtree(root / "snapshots")
        return records
    except Exception:
        for path in root.glob("*.log"):
            print(path.name+":\n"+path.read_text()[-4000:], file=sys.stderr)
        raise
    finally:
        for process in reversed(processes):
            stop(process)
        for log in logs:
            log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--backend", choices=("host", "cuda"), default="cuda")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--rows", type=int, default=131072)
    p.add_argument("--dimension", type=int, default=512)
    p.add_argument("--keys", type=int, default=2048)
    p.add_argument("--bag", type=int, default=64)
    p.add_argument("--hot-rows", type=int, default=2048)
    p.add_argument("--cache-rows", type=int, default=2560)
    p.add_argument("--reserve-slots", type=int, default=64)
    p.add_argument("--phase-steps", type=int, default=96)
    p.add_argument("--calibration-steps", type=int, default=64)
    p.add_argument("--window-requests", type=int, default=16)
    p.add_argument("--max-tracked-rows", type=int, default=16384)
    p.add_argument("--migration-rows", type=int, default=256)
    p.add_argument("--window-bytes", type=int, default=2 << 20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--calibration-repeats", type=int, default=3)
    p.add_argument("--trial-timeout", type=int, default=240)
    p.add_argument("--coalesce", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--source-commit", default="local_worktree")
    a = p.parse_args()
    if a.smoke:
        for k,v in dict(rows=2048,dimension=4,keys=64,bag=8,hot_rows=64,cache_rows=96,
                        reserve_slots=4,phase_steps=8,calibration_steps=8,window_requests=2,
                        max_tracked_rows=512,migration_rows=16,window_bytes=4096).items():
            setattr(a,k,v)
    if not 1 <= a.repeats <= 5 or not 1 <= a.calibration_repeats <= 5:
        p.error("repeat counts must be in 1..5")
    if not (1024 <= a.rows <= 262144 and 1 <= a.dimension <= 512 and
        4 <= a.keys <= 8192 and a.keys % 4 == 0 and a.bag > 0 and a.keys % a.bag == 0 and
        1 <= a.phase_steps <= 512 and 1 <= a.calibration_steps <= 512 and
        0 <= a.reserve_slots < a.cache_rows and 0 < a.hot_rows <= a.cache_rows-a.reserve_slots and
        2*a.hot_rows+max(a.phase_steps,a.calibration_steps)*(a.keys//4)<a.rows and
        a.window_requests > 0 and a.max_tracked_rows > 0 and 1 <= a.migration_rows <= 4096 and
        a.window_bytes > 0 and 10 <= a.trial_timeout <= 600):
        p.error("invalid trace or policy bounds")
    a.build = a.build.resolve();a.out = a.out.resolve();a.out.mkdir(parents=True, exist_ok=False)
    source = pathlib.Path(__file__).resolve().parents[1]
    files = ["tests/cache_policy_probe.cpp", "scripts/cache_policy_sweep.py", "src/server/service.cpp",
             "src/cache/reuse_policy.cpp", "src/cuda/cache_backend.cu", "proto/embedding_service.proto"]
    manifest = dict(source_commit=a.source_commit, backend=a.backend,
        source_sha256={f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files},
        binary_sha256={b:hashlib.sha256((a.build/b).read_bytes()).hexdigest() for b in
                       ["eps_server","eps_cache_agent","eps_cache_policy_probe"]},
        workload={k:getattr(a,k) for k in ["rows","dimension","keys","bag","hot_rows","cache_rows",
            "reserve_slots","phase_steps","calibration_steps","window_requests","max_tracked_rows",
            "migration_rows","window_bytes","repeats","calibration_repeats","coalesce"]},
        experiments=[])
    if a.backend == "cuda":
        hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,compute_cap,driver_version,memory.total", "--format=csv"], text=True)
        (a.out / "hardware.private.csv").write_text(hardware)
        lines = [line.split(",") for line in hardware.splitlines()[1:] if line.strip()]
        device = next((r for r in lines if int(r[0])==a.device),None)
        if device is None:p.error("requested GPU is absent")
        manifest["gpu"] = dict(index=int(device[0]),name=device[1].strip(),compute_cap=device[3].strip(),
                               driver=device[4].strip(),memory=device[5].strip())
    else:
        manifest["gpu"] = None
    all_records = []
    def execute(name,mode,calibration,saved,block):
        print("RUN "+name,flush=True)
        records=run_trial(a,name,mode,calibration,saved)
        raw=a.out/name/"raw.jsonl"
        manifest["experiments"].append(dict(name=name,mode=mode,calibration=calibration,block=block,
            phases=len(records),raw_file=name+"/raw.jsonl",sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
        dump(a.out / "manifest.json", manifest)
        all_records.extend(records)
    for block in range(a.calibration_repeats):
        order=["cpu_only","static"] if block%2==0 else ["static","cpu_only"]
        for mode in order:execute(f"calibration-b{block}-{mode}",mode,True,1.0,block)
    cpu=[r for r in all_records if r["mode"]=="cpu_only"]
    static=[r for r in all_records if r["mode"]=="static"]
    cpu_ms=statistics.median(r["step_mean_ms"] for r in cpu)
    static_ms=statistics.median(r["step_mean_ms"] for r in static)
    hot_unique=statistics.median(r["mean_initial_hot_unique_per_step"] for r in static)
    saved=max(0.0,(cpu_ms-static_ms)*1000/hot_unique)
    calibration=dict(cpu_step_mean_ms_median=cpu_ms,static_step_mean_ms_median=static_ms,
        mean_initial_hot_unique_per_step=hot_unique,saved_microseconds_per_access=saved,
        calculation="max(0, (median CPU step ms - median static step ms) * 1000 / initially hot distinct rows per step)",
        repeats_per_mode=a.calibration_repeats)
    dump(a.out / "calibration.json",calibration);manifest["calibration"]=calibration
    modes=["cpu_only","static","frequency","reuse_cost"]
    for block in range(a.repeats):
        order=modes[block%len(modes):]+modes[:block%len(modes)]
        for mode in order:execute(f"trace-b{block}-{mode}",mode,False,saved,block)
    trace=[r for r in all_records if not r["calibration"]]
    if len({r["trace_fnv1a64"] for r in trace})!=1:raise ValueError("trace identity differs across policies")
    if len({r["checked_final_rows"] for r in trace})!=1:raise ValueError("checked working set differs")
    if a.smoke:
        frequency=[r for r in trace if r["mode"]=="frequency"]
        assert sum(r["policy_admitted_rows"] for r in frequency)>0
        assert sum(r["policy_evicted_rows"] for r in frequency)>0
        assert all(r["resident_id_occurrences"]==0 for r in trace if r["mode"]=="cpu_only")
    summary=[]
    for mode in modes:
        for phase in ["stable","shift","scan_mixed"]:
            group=[r for r in trace if r["mode"]==mode and r["phase"]==phase]
            item=dict(mode=mode,phase=phase,trials=len(group))
            for field in ["steps_per_second","step_p99_ms","resident_hit_fraction","policy_migration_bytes",
                          "policy_admitted_rows","policy_evicted_rows","policy_failures"]:
                values=[r[field] for r in group]
                item[field+"_median"]=statistics.median(values)
                item[field+"_min"]=min(values);item[field+"_max"]=max(values)
            summary.append(item)
    dump(a.out / "summary.json",summary)
    dump(a.out / "manifest.json",manifest)
    print(json.dumps(dict(completed_trials=len(manifest["experiments"]),phase_records=len(all_records),
        completed_steps=sum(r["completed_steps"] for r in all_records),saved_microseconds_per_access=saved,
        errors=sum(r["errors"] for r in all_records))),flush=True)


if __name__ == "__main__":
    main()
