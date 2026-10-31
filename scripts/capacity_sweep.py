"""Run a reproducible process-isolated GPU-agent scaling and saturation sweep."""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
from rpc_processes import ready, stop


def sample_processes(processes, output, stopped):
    ticks = os.sysconf("SC_CLK_TCK")
    page = os.sysconf("SC_PAGE_SIZE")
    with output.open("w") as stream:
        while not stopped.is_set():
            for process in processes:
                try:
                    fields = pathlib.Path(f"/proc/{process.pid}/stat").read_text().split()
                    stream.write(json.dumps(dict(monotonic=time.monotonic(), pid=process.pid,
                        cpu_seconds=(int(fields[13])+int(fields[14]))/ticks,
                        rss_bytes=int(fields[23])*page)) + "\n")
                except (OSError, ValueError):
                    pass
            stream.flush()
            stopped.wait(0.5)


def run_topology(args, count, root):
    run = args.run_id or "capacity-" + uuid.uuid4().hex[:12]
    output = root / f"gpu-{count}-shards-{args.ps_shards}"
    output.mkdir(parents=True, exist_ok=False)
    processes, logs, endpoints, agent_endpoints = [], [], [], []
    telemetry = None
    stopped = threading.Event()
    sampler = None
    try:
        for agent, device in enumerate(args.devices[:count] if not args.external_agents else []):
            log = (output / f"agent-{agent}.log").open("w+")
            logs.append(log)
            slots = (args.rows//2 + count-1)//count + 1024
            process = subprocess.Popen([str(args.build / "eps_cache_agent"), "--run", run,
                "--backend", "host" if args.host_reference else "cuda", "--device", str(device),
                "--agent", str(agent), "--rows-per-table", str(slots),
                "--state-bytes", str(slots*(args.dimension*12+8)+4096), "--listen", "127.0.0.1:0"],
                stdout=subprocess.PIPE, stderr=log, text=True)
            processes.append(process)
            agent_endpoints.append(ready(process, timeout=30))
        if args.external_agents:
            agent_endpoints=args.external_agents[:count]
        for shard in range(args.ps_shards):
            config = output / f"ps-{shard}.json"
            config.write_text(json.dumps(dict(schema_version=1, run_id=run, listen="127.0.0.1:0",
                partition=dict(shard_id=shard, shard_count=args.ps_shards, epoch=1), tables=[],
                cache=dict(enabled=True,concurrent_agent_updates=not args.serial_agent_updates,
                           compact_gradients=not args.expanded_gradients),
                agents=[dict(id=i, endpoint=e) for i,e in enumerate(agent_endpoints)],
                access=dict(policy="cpu_only"),
                limits=dict(cpu_state_bytes=8<<30, max_live_tickets=1024,
                            ticket_payload_bytes=256<<20, queue_payload_bytes=256<<20),
                snapshot_directory=str(output / "snapshots")), indent=2))
            log = (output / f"ps-{shard}.log").open("w+")
            logs.append(log)
            process = subprocess.Popen([str(args.build / "eps_server"), "--config", str(config)],
                                       stdout=subprocess.PIPE, stderr=log, text=True)
            processes.append(process)
            endpoints.append(ready(process))
        if not args.host_reference and not args.external_agents:
            log = (output / "gpu-telemetry.csv").open("w+")
            logs.append(log)
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
                                          "--format=csv", "-l", "1"], stdout=log, stderr=subprocess.DEVNULL)
        sampler = threading.Thread(target=sample_processes,
            args=(processes, output / "process-telemetry.jsonl", stopped), daemon=True)
        sampler.start()
        command = [str(args.build / "eps_capacity_probe"), ",".join(endpoints), ",".join(agent_endpoints), run,
                   str(args.rows), str(args.dimension), str(args.keys), str(args.bag), args.learners,
                   str(args.seconds), str(args.repeats)]
        (output / "command.json").write_text(json.dumps(command, indent=2)+"\n")
        with (output / "raw.jsonl").open("w") as result, (output / "probe.log").open("w+") as log:
            subprocess.run(command, stdout=result, stderr=log, check=True,
                timeout=180 + len(args.learners.split(","))*args.repeats*(args.seconds+30))
        return [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines()]
    except Exception:
        for path in output.glob("*.log"):
            print(path.name + ":\n" + path.read_text()[-3000:], file=sys.stderr)
        raise
    finally:
        stopped.set()
        if sampler:
            sampler.join(timeout=2)
        if telemetry:
            stop(telemetry)
        for process in reversed(processes):
            stop(process)
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--gpu-counts", default="1")
    parser.add_argument("--ps-shards", type=int, default=2)
    parser.add_argument("--rows", type=int, default=262144)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--keys", type=int, default=1024)
    parser.add_argument("--bag", type=int, default=32)
    parser.add_argument("--learners", default="1,2,4,8,16,32,64")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--host-reference", action="store_true")
    parser.add_argument("--serial-agent-updates", action="store_true",
                        help="Ablation: wait for each GPU owner's update before sending the next")
    parser.add_argument("--expanded-gradients", action="store_true",
                        help="Ablation: send expanded per-ID gradients rather than feature-gradient references")
    parser.add_argument("--external-agents", default="",
                        help="Endpoints of independently managed GPU workers; one topology per invocation")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--external-hardware", type=pathlib.Path)
    args = parser.parse_args()
    args.build = args.build.resolve()
    args.devices = [int(x) for x in args.devices.split(",")]
    counts = [int(x) for x in args.gpu_counts.split(",")]
    args.external_agents=[x for x in args.external_agents.split(",") if x]
    if args.external_agents and (len(counts)!=1 or not args.run_id or not args.external_hardware):
        parser.error("external workers require one GPU count, a run-id and their hardware record")
    if args.external_agents and counts[0]>len(args.external_agents):
        parser.error("not enough external agents")
    if len(set(args.devices)) != len(args.devices) or any(c<1 or c>len(args.devices) for c in counts):
        parser.error("provide one distinct device per requested GPU agent")
    if not 1 <= args.ps_shards <= 16:
        parser.error("ps-shards must be in 1..16")
    if args.external_agents:
        inventory=args.external_hardware.read_text()
    elif not args.host_reference:
        inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                                             "--format=csv"], text=True)
        available = {int(line.split(",")[0]) for line in inventory.splitlines()[1:] if line.strip()}
        if not set(args.devices).issubset(available):
            parser.error("requested devices are absent from nvidia-smi")
    else:
        inventory = "Explicit host-reference smoke check; no CUDA scaling claim.\n"
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "hardware.csv").write_text(inventory)
    config = vars(args).copy()
    config = {k:str(v) if isinstance(v,pathlib.Path) else v for k,v in config.items()}
    (args.out / "configuration.json").write_text(json.dumps(config, indent=2)+"\n")
    raw = []
    for count in counts:
        raw += run_topology(args, count, args.out)
        print(f"Completed {count} agent(s), {args.ps_shards} parameter shard(s)", flush=True)
    grouped = {}
    for row in raw:
        grouped.setdefault((row["gpu_agents"],row["learners"]), []).append(row)
    summary = []
    for (gpus, learners), values in grouped.items():
        summary.append(dict(gpu_agents=gpus, learners=learners, ps_shards=args.ps_shards,
            repetitions=len(values), steps_per_second_median=statistics.median(x["steps_per_second"] for x in values),
            steps_per_second_min=min(x["steps_per_second"] for x in values),
            steps_per_second_max=max(x["steps_per_second"] for x in values),
            step_p99_ms_median=statistics.median(x["step_p99_ms"] for x in values),
            lookup_p99_ms_median=statistics.median(x["lookup_p99_ms"] for x in values),
            push_p99_ms_median=statistics.median(x["push_p99_ms"] for x in values),
            sequencer_busy_fraction_median=statistics.median(x["sequencer_busy_fraction"] for x in values),
            queue_wait_per_operation_ms_median=statistics.median(x["queue_wait_per_operation_ms"] for x in values),
            total_completed_steps=sum(x["completed_steps"] for x in values), errors=sum(x["errors"] for x in values)))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
