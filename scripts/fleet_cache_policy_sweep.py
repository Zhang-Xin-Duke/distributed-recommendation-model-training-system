"""Replay fixed traces across freshly restarted, independently hosted GPU agents.

Connection details and forwarding keys belong in the private fleet JSON used by
fleet_capacity_sweep.py. Raw records and hashes written here contain no SSH keys.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import statistics
import uuid

from fleet_capacity_sweep import remote, start_worker, stop_worker


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def trial(fleet, args, count, mode, calibration, saved, name):
    workers = fleet["workers"][:count]
    run = "trace-"+uuid.uuid4().hex[:12]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            starts = list(pool.map(lambda pair: start_worker(pair[1], pair[0], run,
                args.rows, args.dimension, cache_slots=args.cache_rows//count), enumerate(workers)))
        hardware = "".join(s["hardware"] for s in starts)
        if len({line.split(",")[0] for line in hardware.splitlines() if line}) != count:
            raise RuntimeError("agents do not have distinct GPU UUIDs")
        node = fleet["coordinator"]
        root = pathlib.PurePosixPath(node["root"])
        options = dict(build=str(root/args.build_name), out=str(root/"results"/args.out_name),
            backend="cuda", devices=list(range(count)), rows=args.rows, dimension=args.dimension,
            keys=args.keys, bag=args.bag, hot_rows=args.hot_rows, cache_rows=args.cache_rows,
            reserve_slots=args.reserve_slots, phase_steps=args.phase_steps,
            calibration_steps=args.calibration_steps, window_requests=16, max_tracked_rows=32768,
            migration_rows=256, window_bytes=2<<20, coalesce=False, trial_timeout=180,
            external_agents=["127.0.0.1:"+str(w["forward_port"]) for w in workers], run_id=run)
        script = "options="+repr(options)+"\nname="+repr(name)+"\nmode="+repr(mode)+"\n"
        script += "calibration="+repr(calibration)+"\nsaved="+repr(saved)+"\n"
        script += "source="+repr(str(root/"source/scripts"))+"\n"
        script += r'''
import sys,pathlib,json
from types import SimpleNamespace
sys.path.insert(0,source)
from cache_policy_sweep import run_trial
options['build']=pathlib.Path(options['build']);options['out']=pathlib.Path(options['out'])
options['out'].mkdir(parents=True,exist_ok=True)
records=run_trial(SimpleNamespace(**options),name,mode,calibration,saved)
print(json.dumps(records))
'''
        records = json.loads(remote(node, script, timeout=300))
        for record in records:
            assert record["errors"] == 0 and record["gpu_agents"] == count
            assert record["final_free_slots"] == args.cache_rows
            assert sum(record["resident_occurrences_by_agent"]) == record["resident_id_occurrences"]
            if mode != "cpu_only" and record["phase"] == "stable":
                assert all(n > 0 for n in record["resident_occurrences_by_agent"])
        return records
    finally:
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            # stop_worker verifies the stored PID command before terminating it.
            list(pool.map(lambda node: stop_worker(node, run), workers))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fleet", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--out-name", required=True)
    p.add_argument("--build-name", default="build-cuda")
    p.add_argument("--gpu-counts", default="1,2,4")
    p.add_argument("--rows", type=int, default=262144)
    p.add_argument("--dimension", type=int, default=512)
    p.add_argument("--keys", type=int, default=8192)
    p.add_argument("--bag", type=int, default=128)
    p.add_argument("--hot-rows", type=int, default=8192)
    p.add_argument("--cache-rows", type=int, default=10240)
    p.add_argument("--reserve-slots", type=int, default=32)
    p.add_argument("--phase-steps", type=int, default=32)
    p.add_argument("--calibration-steps", type=int, default=16)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--calibration-repeats", type=int, default=2)
    p.add_argument("--source-commit", required=True)
    args = p.parse_args()
    fleet = json.loads(args.fleet.read_text())
    counts = [int(x) for x in args.gpu_counts.split(",")]
    if any(c<1 or c>len(fleet["workers"]) or args.cache_rows%c or
           args.hot_rows>args.cache_rows-c*args.reserve_slots for c in counts):
        p.error("invalid fleet/cache capacity")
    if not args.out_name.replace("-", "").replace("_", "").isalnum():
        p.error("out-name must be a single safe directory name")
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(source_commit=args.source_commit, transport="SSH-forwarded BRPC", experiments=[],
        workload={k:v for k,v in vars(args).items() if k not in ("fleet","out")})
    all_records=[]
    for count in counts:
        calibration_records=[]
        def execute(name, mode, calibration, saved, block):
            print("RUN "+name, flush=True)
            records=trial(fleet,args,count,mode,calibration,saved,name)
            path=args.out/(name+".jsonl")
            path.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in records))
            manifest["experiments"].append(dict(name=name,gpu_agents=count,mode=mode,
                calibration=calibration,block=block,raw_file=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            write(args.out/"manifest.json",manifest)
            all_records.extend(records)
            return records
        for block in range(args.calibration_repeats):
            for mode in (["cpu_only","static"] if block%2==0 else ["static","cpu_only"]):
                calibration_records += execute(f"g{count}-cal-b{block}-{mode}",mode,True,1.0,block)
        cpu=statistics.median(r["step_mean_ms"] for r in calibration_records if r["mode"]=="cpu_only")
        gpu=statistics.median(r["step_mean_ms"] for r in calibration_records if r["mode"]=="static")
        hot=statistics.median(r["mean_initial_hot_unique_per_step"] for r in calibration_records if r["mode"]=="static")
        saved=max(0.0,(cpu-gpu)*1000/hot)
        write(args.out/f"calibration-g{count}.json",dict(cpu_step_mean_ms=cpu,static_step_mean_ms=gpu,
            mean_initial_hot_unique_per_step=hot,saved_microseconds_per_access=saved))
        modes=["cpu_only","static","frequency","reuse_cost"]
        for block in range(args.repeats):
            for mode in modes[block%4:]+modes[:block%4]:
                execute(f"g{count}-trace-b{block}-{mode}",mode,False,saved,block)
    trace=[r for r in all_records if not r["calibration"]]
    if len({r["trace_fnv1a64"] for r in trace}) != 1:
        raise RuntimeError("trace identity differs across policies or GPU counts")
    summary=[]
    for count in counts:
        for mode in ["cpu_only","static","frequency","reuse_cost"]:
            for phase in ["stable","shift","scan_mixed"]:
                group=[r for r in trace if r["gpu_agents"]==count and r["mode"]==mode and r["phase"]==phase]
                summary.append(dict(gpu_agents=count,mode=mode,phase=phase,trials=len(group),
                    **{field:statistics.median(r[field] for r in group) for field in
                       ["steps_per_second","step_p99_ms","resident_hit_fraction","policy_admitted_rows",
                        "policy_evicted_rows","policy_migration_bytes","policy_failures"]}))
    write(args.out/"summary.json",summary)
    print(json.dumps(dict(runs=len(manifest["experiments"]),phase_records=len(all_records),
        completed_steps=sum(r["completed_steps"] for r in all_records),errors=sum(r["errors"] for r in all_records))),flush=True)


if __name__ == "__main__":
    main()
