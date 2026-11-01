"""Render fixed-workload scaling curves from hash-checked published records."""
import argparse
import hashlib
import json
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=pathlib.Path,required=True)
    p.add_argument("--out",type=pathlib.Path,required=True)
    args=p.parse_args()
    manifest=json.loads((args.data/"manifest.json").read_text())
    rows=[]
    for e in manifest["experiments"]:
        if e["suite"]!="limits-fixed":continue
        path=args.data/e["raw_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=e["sha256"]:
            raise ValueError("raw record hash mismatch")
        rows.extend(json.loads(s) for s in path.read_text().splitlines())
    if not rows or any(r["errors"] or r["ps_shards"]!=4 or r["dimension"]!=128 or
                       r["rows"]!=262144 or r["keys_per_step"]!=1024 for r in rows):
        raise ValueError("expected successful, matching fixed-workload records")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none",
        "svg.hashsalt":"eps-four-agent-limits"})
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout="constrained")
    for count,color in [(1,"#2563eb"),(2,"#0f766e"),(4,"#b45309")]:
        group={}
        for r in rows:
            if r["gpu_agents"]==count:group.setdefault(r["learners"],[]).append(r)
        xs=sorted(group)
        for ax,field in zip(axes,["steps_per_second","step_p99_ms","queue_wait_per_operation_ms"]):
            values=[[r[field] for r in group[x]] for x in xs]
            ax.plot(xs,[statistics.median(v) for v in values],"o-",color=color,label=f"{count} GPU agent"+("s" if count!=1 else ""))
            ax.fill_between(xs,[min(v) for v in values],[max(v) for v in values],color=color,alpha=.12)
    for ax,title,ylabel in zip(axes,["Throughput plateau","Tail latency","PS queue wait"],
        ["Completed sparse updates / second","Trial P99 (ms)","Mean wait per operation (ms)"]):
        ax.set_title(title,fontweight="bold");ax.set_ylabel(ylabel)
        ax.set_xscale("log",base=2);ax.set_xticks([1,2,4,8,16,32,64],labels=[1,2,4,8,16,32,64])
        ax.set_xlabel("Concurrent Learners");ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    fig.suptitle("Fixed-workload multi-GPU parameter-service limits",fontsize=16,fontweight="bold")
    fig.supxlabel("262,144 rows × 128 FP32 dimensions · 1,024 IDs/step · 4 parameter shards · 131,072 total resident rows\n"
        "Three 4-second windows + full drain per point; median and observed range. One local agent; additional agents use SSH-forwarded BRPC.",fontsize=9)
    args.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out/"multigpu-limits.svg",bbox_inches="tight",facecolor="white",metadata={"Date":None})
    svg=args.out/"multigpu-limits.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    fig.savefig(args.out/"multigpu-limits.png",bbox_inches="tight",facecolor="white",dpi=160)


if __name__=="__main__":main()
