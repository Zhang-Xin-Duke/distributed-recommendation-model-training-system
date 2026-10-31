"""Validate, publish and plot interleaved client-path capacity comparisons."""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
from collect_capacity_report import resources, read_jsonl


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def render(summary, out, labels, prefix, workload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#cbd5e1", "svg.fonttype": "none",
        "svg.hashsalt": "eps-paired-coalesced-gather"})
    counts = sorted({r["gpu_agents"] for r in summary})
    shape=f'{workload["ps_shards"]} PS shards · {workload["rows"]:,} rows × {workload["dimension"]} dimensions · {workload["keys"]:,} IDs/step'
    out.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, suffix in [("steps_per_second", "Completed sparse updates / second", "capacity"),
                                 ("step_p99_ms", "Median trial P99 latency (ms)", "latency")]:
        name=prefix+"-"+suffix
        max_y=max(r[variant+"_"+metric+"_max"] for r in summary for variant in ("baseline","candidate"))*1.12
        fig, axes = plt.subplots(1, len(counts), figsize=(4.2*len(counts), 4),
                                 squeeze=False, sharey=True, layout="constrained")
        for ax, count in zip(axes[0], counts):
            rows = sorted((r for r in summary if r["gpu_agents"] == count), key=lambda r: r["learners"])
            x = [r["learners"] for r in rows]
            for variant, label, color in [("baseline", "Per-shard gathers", "#64748b"),
                                           ("candidate", "Coalesced gathers", "#2563eb")]:
                label=labels.get(variant,label)
                field = variant+"_"+metric
                ax.plot(x, [r[field+"_median"] for r in rows], "o-", color=color, label=label, linewidth=2)
                ax.fill_between(x, [r[field+"_min"] for r in rows],
                                [r[field+"_max"] for r in rows], color=color, alpha=.12)
            ax.set_title(f"{count} GPU"+("s" if count > 1 else ""), fontweight="bold")
            ax.set_xscale("log", base=2);ax.set_xticks(x, labels=x);ax.set_ylim(0,max_y)
            ax.set_xlabel("Concurrent Learners");ax.grid(axis="y", color="#e2e8f0");ax.set_axisbelow(True)
        axes[0][0].set_ylabel(ylabel);axes[0][-1].legend(frameon=False)
        fig.suptitle("Cache gather strategies: paired CUDA comparison", fontsize=15, fontweight="bold")
        fig.supxlabel("Same A6000 fleet and server/agent builds · "+shape+"\n"
            "Interleaved paired blocks; median and observed range. One local agent; others over SSH forwarding.", fontsize=9)
        svg = out/(name+".svg")
        fig.savefig(svg, bbox_inches="tight", facecolor="white", metadata={"Date": None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
        fig.savefig(out/(name+".png"), bbox_inches="tight", facecolor="white", dpi=170)
        plt.close(fig)
    if len(counts)==1:
        fig, axes=plt.subplots(1,2,figsize=(11.4,4.2),layout="constrained")
        rows=sorted(summary,key=lambda r:r["learners"]);x=[r["learners"] for r in rows]
        for ax,metric,ylabel in zip(axes,["steps_per_second","step_p99_ms"],
                ["Completed sparse updates / second","Median trial P99 latency (ms)"]):
            for variant,color in [("baseline","#64748b"),("candidate","#2563eb")]:
                key=variant+"_"+metric
                ax.plot(x,[r[key+"_median"] for r in rows],"o-",color=color,
                        label=labels.get(variant,variant),linewidth=2)
                ax.fill_between(x,[r[key+"_min"] for r in rows],
                                [r[key+"_max"] for r in rows],color=color,alpha=.12)
            ax.set_xscale("log",base=2);ax.set_xticks(x,labels=x);ax.set_xlabel("Concurrent Learners")
            ax.set_ylabel(ylabel);ax.grid(axis="y",color="#e2e8f0");ax.set_axisbelow(True)
            ax.legend(frameon=False)
        axes[0].set_ylim(bottom=0);axes[1].set_yscale("log")
        fig.suptitle(f"{counts[0]}-GPU parameter-service capacity",fontsize=16,fontweight="bold")
        fig.supxlabel("Complete sparse steps · "+shape+"\n"
            "Median and observed range of paired blocks. One local A6000 agent; others over SSH forwarding.",fontsize=9)
        name=prefix+"-overview";svg=out/(name+".svg")
        fig.savefig(svg,bbox_inches="tight",facecolor="white",metadata={"Date":None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
        fig.savefig(out/(name+".png"),bbox_inches="tight",facecolor="white",dpi=170);plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=pathlib.Path, required=True)
    p.add_argument("--data", type=pathlib.Path, required=True,
                   help="Fetched coordinator results directory containing run-name directories")
    p.add_argument("--environment", type=pathlib.Path, required=True,
                   help="Public hardware/toolchain description, excluding connection details")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--figures", type=pathlib.Path)
    p.add_argument("--figure-prefix", default="coalesced")
    a = p.parse_args()
    plan = json.loads(a.plan.read_text());a.out.mkdir(parents=True, exist_ok=True)
    manifest = dict(workload=plan["workload"], comparison=plan["comparison"],
                    binary_sha256=plan["binary_sha256"],
                    candidate_read_preflight=plan.get("candidate_read_preflight", False),
                    candidate_coalescing=plan.get("candidate_coalescing"),
                    labels=plan.get("labels", {}),
                    environment=json.loads(a.environment.read_text()), experiments=[])
    groups = {};trials = 0
    for run in plan["runs"]:
        if run["state"] != "complete" or run["exit_code"]:
            raise ValueError("incomplete run: "+run["name"])
        name = run["name"]
        if not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("invalid run name")
        candidates = list((a.data/name).glob("gpu-*/gpu-*/raw.jsonl"))
        if len(candidates) != 1:
            raise ValueError("expected one raw file for "+name)
        rows = [json.loads(line) for line in candidates[0].read_text().splitlines()]
        expected = {int(x) for x in plan["workload"]["learners"].split(",")}
        if len(rows) != len(expected) or {r["learners"] for r in rows} != expected:
            raise ValueError("incomplete load levels in "+name)
        for row in rows:
            if any(not isinstance(v, (str, int, float, bool)) for v in row.values()):
                raise ValueError("unexpected raw record structure")
            if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
                raise ValueError("non-finite metric")
            if row["errors"] or row["rejected_admissions"] or row["completed_steps"] <= 0 or row["checked_parameter_counters"] != 32:
                raise ValueError("failed or unchecked trial")
            if plan.get("candidate_read_preflight") and run["variant"] == "candidate":
                if row.get("preflight_values_checked", 0) <= 0 or not 0 <= row.get("preflight_max_absolute_error", math.inf) <= 5e-6:
                    raise ValueError("candidate numerical preflight is absent or failed")
            if plan.get("candidate_coalescing") is not None and run["variant"] == "candidate":
                if row.get("coalesced_gathers") != bool(int(plan["candidate_coalescing"])):
                    raise ValueError("candidate gather policy mismatch")
            for field, setting in [("rows", "rows"), ("dimension", "dimension"),
                ("keys_per_step", "keys"), ("bag_length", "bag"), ("ps_shards", "ps_shards")]:
                if row[field] != plan["workload"][setting]:
                    raise ValueError("workload mismatch")
            if row["gpu_agents"] != run["gpu_count"] or not math.isclose(
                row["steps_per_second"], row["completed_steps"]/row["elapsed_seconds"], rel_tol=1e-9):
                raise ValueError("topology/rate mismatch")
            key = run["block"], run["gpu_count"], row["learners"]
            pair = groups.setdefault(key, {})
            if run["variant"] in pair:
                raise ValueError("duplicate comparison member")
            pair[run["variant"]] = row
        raw = a.out/(name+".jsonl")
        raw.write_bytes(candidates[0].read_bytes())
        entry = {**{k: run[k] for k in
            ("name", "block", "gpu_count", "variant", "client_commit")},
            "raw_file": raw.name, "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "trials": len(rows)}
        telemetry = candidates[0].parent/"process-telemetry.jsonl"
        if telemetry.exists():
            resource_file = a.out/(name+"-resources.json")
            dump(resource_file, resources(rows, read_jsonl(telemetry)))
            entry["resources_file"] = resource_file.name
        manifest["experiments"].append(entry)
        trials += len(rows)
    pairs = [];aggregate = {}
    for (block, gpus, learners), members in sorted(groups.items()):
        if set(members) != {"baseline", "candidate"}:
            raise ValueError("unpaired trial")
        x, y = members["baseline"], members["candidate"]
        row = dict(block=block, gpu_agents=gpus, learners=learners,
                   throughput_ratio=y["steps_per_second"]/x["steps_per_second"])
        for variant, raw in members.items():
            for metric in ("steps_per_second", "step_p99_ms", "lookup_p99_ms", "push_p99_ms"):
                row[variant+"_"+metric] = raw[metric]
        pairs.append(row);aggregate.setdefault((gpus, learners), []).append(row)
    summary = []
    for (gpus, learners), rows in sorted(aggregate.items()):
        item = dict(gpu_agents=gpus, learners=learners, paired_blocks=len(rows))
        for field in rows[0]:
            if field in ("block", "gpu_agents", "learners"):
                continue
            values = [r[field] for r in rows]
            item.update({field+"_median": statistics.median(values),
                         field+"_min": min(values), field+"_max": max(values)})
        summary.append(item)
    manifest.update(trials=trials, paired_comparisons=len(pairs), load_levels=len(summary))
    dump(a.out/"manifest.json", manifest);dump(a.out/"pairs.json", pairs);dump(a.out/"summary.json", summary)
    if a.figures:
        render(summary, a.figures, manifest["labels"], a.figure_prefix, manifest["workload"])
    print(f"Validated {trials} trials, {len(pairs)} pairs and {len(summary)} load levels")


if __name__ == "__main__":
    main()
