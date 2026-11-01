"""Verify and publish fixed-trace cache-policy records and adaptation figures."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import statistics

MODES = ["cpu_only", "static", "frequency", "reuse_cost"]
PHASES = ["stable", "shift", "scan_mixed"]
LABELS = dict(cpu_only="CPU only", static="Static cache", frequency="Frequency", reuse_cost="Reuse cost")
COLORS = dict(cpu_only="#64748b", static="#2563eb", frequency="#d97706", reuse_cost="#0f766e")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def render(summary, manifest, directory, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#cbd5e1",
        "svg.fonttype":"none","svg.hashsalt":"eps-cache-policy-trace"})
    directory.mkdir(parents=True, exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.8),layout="constrained")
    fields=[("steps_per_second","Completed sparse updates / second",1),
            ("resident_hit_fraction","Resident input occurrences (%)",100),
            ("policy_migration_bytes","Migrated state per phase (MiB)",1/(1<<20))]
    for mode in MODES:
        rows=[next(r for r in summary if r["mode"]==mode and r["phase"]==phase) for phase in PHASES]
        for ax,(field,label,scale) in zip(axes,fields):
            x=list(range(3));y=[r[field+"_median"]*scale for r in rows]
            ax.plot(x,y,"o-",color=COLORS[mode],label=LABELS[mode],linewidth=2)
            ax.fill_between(x,[r[field+"_min"]*scale for r in rows],
                            [r[field+"_max"]*scale for r in rows],color=COLORS[mode],alpha=.1)
            ax.set_ylabel(label);ax.set_xticks(x,labels=["Stable\nhot set","Shifted\nhot set","Hot +\none-pass scan"])
            ax.grid(axis="y",color="#e2e8f0");ax.set_axisbelow(True)
    for ax in axes:ax.set_ylim(bottom=0)
    axes[1].set_ylim(0,100)
    axes[0].legend(loc="lower left",ncol=2,frameon=False,fontsize=9)
    w=manifest["workload"]
    fig.suptitle("Adaptive sparse-parameter placement",fontsize=17,fontweight="bold")
    gpu=manifest["gpu"]["name"].removeprefix("NVIDIA ")
    fig.supxlabel(f'1 {gpu} · 2 PS shards · {w["dimension"]}D vectors · {w["keys"]:,} IDs/step · {w["hot_rows"]:,} hot / {w["cache_rows"]:,} cache rows\n'
        f'{w["phase_steps"]} updates/phase; median and observed range of {w["repeats"]} fixed-trace trials. Automatic migrations are timed.',fontsize=9)
    for suffix in ("svg","png"):
        p=directory/(prefix+"-adaptation."+suffix)
        fig.savefig(p,bbox_inches="tight",facecolor="white",dpi=170,metadata={"Date":None} if suffix=="svg" else None)
        if suffix=="svg":p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.2,4.4),layout="constrained")
    for index,mode in enumerate(MODES):
        rows=[next(r for r in summary if r["mode"]==mode and r["phase"]==phase) for phase in PHASES]
        x=[p+(index-1.5)*.18 for p in range(3)];y=[r["step_p99_ms_median"] for r in rows]
        errors=[[v-r["step_p99_ms_min"] for v,r in zip(y,rows)],
                [r["step_p99_ms_max"]-v for v,r in zip(y,rows)]]
        ax.bar(x,y,width=.16,color=COLORS[mode],label=LABELS[mode],yerr=errors,capsize=2)
    ax.set_xticks(range(3),labels=["Stable hot set","Shifted hot set","Hot + one-pass scan"])
    ax.set_ylabel("Median trial P99 step latency (ms)")
    ax.set_ylim(0,max(r["step_p99_ms_max"] for r in summary)*1.3)
    ax.grid(axis="y",color="#e2e8f0");ax.set_axisbelow(True);ax.legend(frameon=False,ncol=4,loc="upper center",fontsize=9)
    ax.set_title("Tail latency includes cache migration decisions",fontsize=16,fontweight="bold")
    fig.supxlabel("Identical seeded requests and update order · observed trial ranges · synthetic feature gradients",fontsize=9)
    for suffix in ("svg","png"):
        p=directory/(prefix+"-latency."+suffix)
        fig.savefig(p,bbox_inches="tight",facecolor="white",dpi=170,metadata={"Date":None} if suffix=="svg" else None)
        if suffix=="svg":p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=pathlib.Path,required=True)
    p.add_argument("--out",type=pathlib.Path,required=True)
    p.add_argument("--environment",type=pathlib.Path)
    p.add_argument("--figures",type=pathlib.Path)
    p.add_argument("--figure-prefix",default="cache-policy")
    p.add_argument("--allow-host-reference",action="store_true")
    a=p.parse_args();source=json.loads((a.data/"manifest.json").read_text())
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.figure_prefix):raise ValueError("invalid figure prefix")
    if source["backend"]!="cuda" and not a.allow_host_reference:raise ValueError("CUDA records required")
    if source["backend"]!="cuda" and a.figures:raise ValueError("GPU figures require CUDA measurements")
    w=source["workload"];a.out.mkdir(parents=True,exist_ok=True)
    manifest={k:source[k] for k in ["source_commit","backend","source_sha256","binary_sha256","workload","gpu","calibration"]}
    if a.environment:manifest["environment"]=json.loads(a.environment.read_text())
    manifest["experiments"]=[];records=[];seen=set()
    for entry in source["experiments"]:
        name=entry["name"]
        if not re.fullmatch(r'(calibration|trace)-b[0-9]+-(cpu_only|static|frequency|reuse_cost)',name) or name in seen:
            raise ValueError("unexpected experiment name")
        seen.add(name);raw=(a.data/entry["raw_file"]).resolve()
        if not raw.is_relative_to(a.data.resolve()):raise ValueError("raw path escapes dataset")
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=entry["sha256"]:raise ValueError("raw hash mismatch")
        rows=[json.loads(line) for line in raw.read_text().splitlines()]
        phases=["stable"] if entry["calibration"] else PHASES
        if [r["phase"] for r in rows]!=phases:raise ValueError("missing or reordered phases")
        for r in rows:
            if any(not isinstance(v,(str,int,float,bool)) for v in r.values()):raise ValueError("unexpected nested record")
            if any(not math.isfinite(v) for v in r.values() if isinstance(v,float)):raise ValueError("nonfinite metric")
            steps=w["calibration_steps"] if entry["calibration"] else w["phase_steps"]
            if r["mode"]!=entry["mode"] or r["calibration"]!=entry["calibration"]:raise ValueError("mode mismatch")
            if r["completed_steps"]!=steps or r["errors"] or r["elapsed_seconds"]<=0:raise ValueError("incomplete trial")
            if not math.isclose(r["steps_per_second"],steps/r["elapsed_seconds"],rel_tol=1e-9):raise ValueError("rate mismatch")
            for field,setting in [("rows","rows"),("dimension","dimension"),("keys_per_step","keys"),
                ("bag_length","bag"),("hot_rows","hot_rows"),("cache_rows","cache_rows")]:
                if r[field]!=w[setting]:raise ValueError("workload mismatch")
            if r["ps_shards"]!=2 or r["gpu_agents"]!=1 or r["learners"]!=1:raise ValueError("topology mismatch")
            if r["input_id_occurrences"]!=steps*w["keys"] or sum(r[f] for f in
                ["resident_id_occurrences","transferred_id_occurrences","cpu_pooled_id_occurrences"])!=r["input_id_occurrences"]:
                raise ValueError("access accounting mismatch")
            if r["checked_feature_values"]!=steps*(w["keys"]//w["bag"])*w["dimension"] or \
               r["checked_final_state_values"]!=r["checked_final_rows"]*w["dimension"]*3 or r["checked_final_rows"]<w["hot_rows"]:
                raise ValueError("insufficient correctness checks")
            if not 0<=r["feature_max_absolute_error"]<=5e-6 or not 0<=r["state_max_absolute_error"]<=5e-6:
                raise ValueError("numerical mismatch")
            if r["cached_rows_end"]>w["cache_rows"]-w["reserve_slots"] or r["final_free_slots"]!=w["cache_rows"]:
                raise ValueError("capacity/reserve mismatch")
            if r["policy_migration_bytes"]>r["policy_windows"]*w["window_bytes"] or r["policy_migration_batches"]>r["policy_windows"]*8:
                raise ValueError("window budget mismatch")
            expected_saved=1 if r["calibration"] else source["calibration"]["saved_microseconds_per_access"]
            if r["saved_microseconds_per_access"]!=expected_saved:raise ValueError("cost prior differs")
        destination=a.out/(name+".jsonl");destination.write_bytes(raw.read_bytes())
        manifest["experiments"].append({**entry,"raw_file":destination.name})
        records.extend(rows)
    expected_trials=2*w["calibration_repeats"]+4*w["repeats"]
    if len(seen)!=expected_trials:raise ValueError("missing experiment")
    trace=[r for r in records if not r["calibration"]]
    if len({r["trace_fnv1a64"] for r in trace})!=1 or len({r["checked_final_rows"] for r in trace})!=1:
        raise ValueError("main traces differ")
    summary=[]
    for mode in MODES:
        for phase in PHASES:
            group=[r for r in trace if r["mode"]==mode and r["phase"]==phase]
            if len(group)!=w["repeats"]:raise ValueError("unbalanced trials")
            row=dict(mode=mode,phase=phase,trials=len(group))
            for field in ["steps_per_second","step_p99_ms","resident_hit_fraction","policy_migration_bytes",
                          "policy_admitted_rows","policy_evicted_rows","policy_failures"]:
                values=[r[field] for r in group]
                for suffix,value in [("median",statistics.median(values)),("min",min(values)),("max",max(values))]:row[field+"_"+suffix]=value
            summary.append(row)
    # Recompute calibration from the actual alternating CPU/static records.
    calibration_rows=[r for r in records if r["calibration"]]
    cpu=statistics.median(r["step_mean_ms"] for r in calibration_rows if r["mode"]=="cpu_only")
    static=[r for r in calibration_rows if r["mode"]=="static"]
    saved=max(0,(cpu-statistics.median(r["step_mean_ms"] for r in static))*1000/
              statistics.median(r["mean_initial_hot_unique_per_step"] for r in static))
    if not math.isclose(saved,source["calibration"]["saved_microseconds_per_access"],rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError("calibration differs from records")
    manifest["counts"]=dict(independent_runs=len(seen),phase_records=len(records),
        completed_updates=sum(r["completed_steps"] for r in records),errors=sum(r["errors"] for r in records))
    dump(a.out/"manifest.json",manifest);dump(a.out/"summary.json",summary);dump(a.out/"calibration.json",source["calibration"])
    if a.figures:render(summary,manifest,a.figures,a.figure_prefix)
    print(json.dumps(manifest["counts"]))


if __name__=="__main__":main()
