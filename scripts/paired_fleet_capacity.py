"""Interleave two prepared client builds on the same CUDA fleet.

Only the client/probe binary changes between build directories. Both must point
to the same server binary; agents use the same installed bundle. Connection
configuration stays in the private fleet file. Each run starts fresh parameter
servers and agents, while the cloud instances stay fixed across the comparison.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fleet", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--prefix", default="gather-pair")
    p.add_argument("--baseline-build", default="paired-baseline")
    p.add_argument("--candidate-build", default="paired-candidate")
    p.add_argument("--baseline-commit", required=True)
    p.add_argument("--candidate-commit", required=True)
    p.add_argument("--candidate-coalescing", choices=("0", "1"), default=None)
    p.add_argument("--baseline-label", default="Per-shard gathers")
    p.add_argument("--candidate-label", default="Coalesced gathers")
    p.add_argument("--gpu-counts", default="1,2,3")
    p.add_argument("--learners", default="4,8,16,32")
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--ps-shards", type=int, default=2)
    p.add_argument("--rows", type=int, default=262144)
    p.add_argument("--dimension", type=int, default=128)
    p.add_argument("--keys", type=int, default=1024)
    p.add_argument("--bag", type=int, default=32)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--require-candidate-preflight", action="store_true",
                   help="Require candidate trial records to contain checked preflight values")
    a = p.parse_args()
    if not a.prefix.replace("-", "").replace("_", "").isalnum():
        p.error("invalid run prefix")
    for build in (a.baseline_build, a.candidate_build):
        if not build.replace("-", "").replace("_", "").isalnum():
            p.error("build paths must be one prepared directory name")
    if not 1 <= a.blocks <= 10 or not 0 < a.seconds <= 60:
        p.error("invalid block count or duration")
    counts = [int(x) for x in a.gpu_counts.split(",")]
    if len(set(counts)) != len(counts) or any(x < 1 for x in counts):
        p.error("GPU counts must be distinct positive integers")
    variants = dict(baseline=(a.baseline_build, a.baseline_commit),
                    candidate=(a.candidate_build, a.candidate_commit))
    runs = []
    for block in range(a.blocks):
        for index, count in enumerate(counts):
            order = ("baseline", "candidate") if (block+index) % 2 == 0 else ("candidate", "baseline")
            for variant in order:
                build, commit = variants[variant]
                runs.append(dict(name=f"{a.prefix}-b{block}-{count}gpu-{variant}",
                                 block=block, gpu_count=count, variant=variant,
                                 client_commit=commit, build_name=build, state="planned"))
    manifest = dict(workload={k: getattr(a, k) for k in
        ("rows", "dimension", "keys", "bag", "learners", "seconds", "ps_shards")},
        comparison="same server/agent builds; alternating client order; fresh state per run",
        candidate_read_preflight=a.require_candidate_preflight,
        candidate_coalescing=a.candidate_coalescing,
        labels=dict(baseline=a.baseline_label,candidate=a.candidate_label), runs=runs)
    if a.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    from fleet_capacity_sweep import remote
    fleet = json.loads(a.fleet.read_text())
    options = dict(root=fleet["coordinator"]["root"], builds={k: v[0] for k, v in variants.items()})
    script = "options="+repr(options)+"\n"+r'''
import hashlib,json,pathlib
root=pathlib.Path(options['root'])
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
out={name:{binary:digest(root/build/binary) for binary in ['eps_server','eps_capacity_probe']}
     for name,build in options['builds'].items()}
assert out['baseline']['eps_server']==out['candidate']['eps_server'],'server binaries differ'
assert out['baseline']['eps_capacity_probe']!=out['candidate']['eps_capacity_probe'],'client binaries are identical'
out['cache_agent']=digest(root/'agent-bundle/bin/eps_cache_agent')
print(json.dumps(out))
'''
    manifest["binary_sha256"] = json.loads(remote(fleet["coordinator"], script))
    a.out.mkdir(parents=True, exist_ok=False)
    record = a.out / "manifest.json"
    def save():
        record.write_text(json.dumps(manifest, indent=2)+"\n")
    save()
    for run in runs:
        command = [sys.executable, str(pathlib.Path(__file__).with_name("fleet_capacity_sweep.py")),
            "--fleet", str(a.fleet), "--out-name", run["name"], "--build-name", run["build_name"],
            "--gpu-counts", str(run["gpu_count"]), "--learners", a.learners,
            "--seconds", str(a.seconds), "--repeats", "1"]
        for field in ("rows", "dimension", "keys", "bag", "ps_shards"):
            command += ["--"+field.replace("_", "-"), str(getattr(a, field))]
        if run["variant"] == "candidate" and a.candidate_coalescing is not None:
            command += ["--coalesce-gathers", a.candidate_coalescing]
        run.update(state="running", started_epoch=time.time());save()
        print(f"START {run['name']}", flush=True)
        with (a.out / (run["name"]+".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=300+len(a.learners.split(","))*(a.seconds+30))
        run.update(state="complete" if result.returncode == 0 else "failed",
                   finished_epoch=time.time(), exit_code=result.returncode);save()
        if result.returncode:
            raise SystemExit(f"Failed {run['name']}; inspect its local log before resuming")
        print(f"DONE {run['name']}", flush=True)


if __name__ == "__main__":
    main()
