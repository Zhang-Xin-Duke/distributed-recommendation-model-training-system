"""Publish selected raw records without host addresses or connection files.

The private specification supplies experiments with source_file, name, suite,
variant and source_commit. Optional telemetry_file records are reduced into CPU
and sampled-RSS summaries over each native probe's timed interval.
"""
import argparse
import bisect
import hashlib
import json
import math
import pathlib
import shutil
import statistics


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resources(rows,telemetry):
    processes={}
    for sample in telemetry:
        processes.setdefault(sample["pid"],[]).append(sample)
    output=[]
    for row in rows:
        if "start_monotonic_seconds" not in row:
            continue
        start,end=row["start_monotonic_seconds"],row["end_monotonic_seconds"]
        record=dict(learners=row["learners"],repeat=row["repeat"],ps_cpu_cores=0,load_generator_cpu_cores=0,ps_sampled_rss_bytes=0)
        found=False
        for samples in processes.values():
            role=samples[0].get("role")
            if role not in ("eps_server","eps_capacity_probe"):
                continue
            times=[s["monotonic"] for s in samples]
            if start<times[0] or end>times[-1]:
                continue
            def cpu_at(t):
                i=max(1,min(len(times)-1,bisect.bisect_right(times,t)))
                fraction=(t-times[i-1])/(times[i]-times[i-1])
                return samples[i-1]["cpu_seconds"]+fraction*(samples[i]["cpu_seconds"]-samples[i-1]["cpu_seconds"])
            cpu=(cpu_at(end)-cpu_at(start))/(end-start)
            if cpu<0:
                raise ValueError("CPU accounting went backwards")
            found=True
            if role=="eps_server":
                record["ps_cpu_cores"]+=cpu
                window=[s["rss_bytes"] for s in samples if start<=s["monotonic"]<=end]
                record["ps_sampled_rss_bytes"]+=max(window,default=0)
            else:
                record["load_generator_cpu_cores"]+=cpu
        if found:
            output.append(record)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec",type=pathlib.Path,required=True)
    p.add_argument("--out",type=pathlib.Path,required=True)
    a=p.parse_args();spec=json.loads(a.spec.read_text());a.out.mkdir(parents=True,exist_ok=True)
    manifest={k:v for k,v in spec.items() if k not in ("experiments","transport_source")}
    manifest["experiments"]=[]
    summary=[]
    for experiment in spec["experiments"]:
        source=pathlib.Path(experiment["source_file"]);rows=read_jsonl(source)
        if not rows or any(r["errors"] or r.get("rejected_admissions",0) or r["completed_steps"]<=0 for r in rows):
            raise ValueError("failed or incomplete capacity data: "+str(source))
        name=experiment["name"]
        if not name.replace("-","").replace("_","").isalnum():
            raise ValueError("invalid experiment name")
        destination=a.out/(name+".jsonl");shutil.copyfile(source,destination)
        entry={k:v for k,v in experiment.items() if k not in ("source_file","telemetry_file")}
        entry.update(raw_file=destination.name,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            gpu_agents=rows[0]["gpu_agents"],ps_shards=rows[0]["ps_shards"],rows=rows[0]["rows"],
            dimension=rows[0]["dimension"],keys_per_step=rows[0]["keys_per_step"],bag_length=rows[0]["bag_length"])
        for field in ("gpu_agents","ps_shards","rows","dimension","keys_per_step","bag_length"):
            if any(r[field]!=rows[0][field] for r in rows):
                raise ValueError("mixed workload configurations in "+name)
        resource_file=experiment.get("telemetry_file")
        if resource_file:
            derived=resources(rows,read_jsonl(pathlib.Path(resource_file)))
            path=a.out/(name+"-resources.json");path.write_text(json.dumps(derived,indent=2,allow_nan=False)+"\n")
            entry["resources_file"]=path.name
        groups={}
        for row in rows:
            for value in row.values():
                if isinstance(value,float) and not math.isfinite(value):raise ValueError("non-finite metric")
            groups.setdefault(row["learners"],[]).append(row)
        entry["trial_count"]=len(rows);entry["learner_levels"]=sorted(groups)
        manifest["experiments"].append(entry)
        for learners,trials in groups.items():
            rate=[r["steps_per_second"] for r in trials]
            summary.append(dict(experiment=name,variant=entry["variant"],suite=entry["suite"],
                gpu_agents=entry["gpu_agents"],ps_shards=entry["ps_shards"],keys_per_step=entry["keys_per_step"],
                bag_length=entry["bag_length"],learners=learners,trials=len(trials),
                steps_per_second_median=statistics.median(rate),steps_per_second_min=min(rate),steps_per_second_max=max(rate),
                id_occurrences_per_second_median=statistics.median(r["id_occurrences_per_second"] for r in trials),
                step_p99_ms_median=statistics.median(r["step_p99_ms"] for r in trials),
                queue_wait_ms_median=statistics.median(r["queue_wait_per_operation_ms"] for r in trials),
                total_completed_steps=sum(r["completed_steps"] for r in trials),errors=sum(r["errors"] for r in trials)))
    if spec.get("transport_source"):
        destination=a.out/"transport-control.json";shutil.copyfile(spec["transport_source"],destination)
        manifest["transport_control"]=destination.name
    (a.out/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(f"Published {len(summary)} load levels from {len(manifest['experiments'])} experiments")


if __name__=="__main__":main()
