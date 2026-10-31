"""Render capacity figures from the published raw experiment manifest."""
import argparse
import json
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {1:"#2563eb",2:"#0f766e",3:"#d97706"}


def load(root, experiment):
    rows=[json.loads(line) for line in (root/experiment["raw_file"]).read_text().splitlines()]
    if not rows or any(r["errors"] or r.get("rejected_admissions",0) for r in rows):
        raise ValueError("incomplete or failed experiment: "+experiment["raw_file"])
    return rows


def group(rows,field):
    result={}
    for row in rows:
        result.setdefault(row["learners"],[]).append(row[field])
    x=sorted(result)
    return x,[statistics.median(result[i]) for i in x],[min(result[i]) for i in x],[max(result[i]) for i in x]


def save(fig,root,name):
    svg=root/(name+".svg")
    fig.savefig(svg,bbox_inches="tight",facecolor="white",metadata={"Date":None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    fig.savefig(root/(name+".png"),bbox_inches="tight",facecolor="white",dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=pathlib.Path,required=True)
    p.add_argument("--out",type=pathlib.Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data/"manifest.json").read_text())
    experiments=manifest["experiments"]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"axes.edgecolor":"#cbd5e1","axes.labelcolor":"#334155",
        "text.color":"#0f172a","xtick.color":"#475569","ytick.color":"#475569","svg.fonttype":"none",
        "svg.hashsalt":"embedding-parameter-service-capacity"})
    curves=[e for e in experiments if e["variant"]=="concurrent_compact" and e["suite"]=="scaling"]
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.4),layout="constrained")
    for e in curves:
        rows=load(args.data,e);color=COLORS[e["gpu_agents"]]
        for ax,field in zip(axes,["steps_per_second","step_p99_ms"]):
            x,y,low,high=group(rows,field)
            ax.plot(x,y,"o-",color=color,label=f'{e["gpu_agents"]} GPU'+("s" if e["gpu_agents"]>1 else ""),linewidth=2,markersize=4)
            ax.fill_between(x,low,high,color=color,alpha=.12)
            ax.set_xscale("log",base=2);ax.set_xticks(x,labels=x);ax.set_xlabel("Concurrent Learners")
            ax.grid(axis="y",color="#e2e8f0",linewidth=.7);ax.set_axisbelow(True)
    axes[0].set_ylabel("Completed sparse updates / second");axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("Median trial P99 latency (ms)");axes[1].set_yscale("log")
    for ax in axes:ax.legend(frameon=False)
    fig.suptitle("Distributed parameter-service capacity",fontsize=16,fontweight="bold")
    fig.supxlabel("RTX A6000 workers · 2 PS shards · 262,144 rows × 128 dimensions · 1,024 IDs/step\nMedian of three trials; shading shows observed range. BRPC over SSH forwarding; sparse subsystem only.",fontsize=9)
    save(fig,args.out,"capacity")

    fig,ax=plt.subplots(figsize=(8.4,3.8),layout="constrained")
    for offset,(field,label,color) in enumerate([("expanded_gradient_bytes","Expanded representation of the same updates","#94a3b8"),
                                                ("transferred_gradient_bytes","Transmitted feature gradients + references","#2563eb")]):
        values=[]
        for count in (1,2,3):
            candidates=[e for e in curves if e["gpu_agents"]==count]
            if not candidates:values.append(0);continue
            rows=[r for r in load(args.data,candidates[0]) if r["learners"]==8]
            values.append(statistics.median(r[field]/r["completed_steps"]/1024 for r in rows))
        locations=[x+(-.18 if offset==0 else .18) for x in range(3)]
        bars=ax.bar(locations,values,width=.34,color=color,label=label)
        ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=9)
    ax.set_xticks(range(3),labels=["1 GPU","2 GPUs","3 GPUs"])
    ax.set_ylabel("Numeric gradient payload per step (KiB)");ax.set_ylim(0,700)
    ax.set_title("Avoid repeated pooled-gradient transfers",fontweight="bold",pad=14)
    ax.legend(frameon=False,loc="upper right");ax.grid(axis="y",color="#e2e8f0");ax.set_axisbelow(True)
    fig.supxlabel("8 Learners · 1,024 IDs/step · measured GPU-update attachments; protocol row identities excluded",fontsize=9)
    save(fig,args.out,"gradient-payload")

    control=manifest.get("transport_control")
    if control:
        rows=json.loads((args.data/control).read_text());fig,ax=plt.subplots(figsize=(8.4,3.5),layout="constrained")
        x=[r["payload_bytes"]/1024 for r in rows];y=[r["median_ms"] for r in rows]
        ax.plot(x,y,"o-",color="#d97706",linewidth=2);ax.set_xscale("log",base=2)
        ax.set_xticks(x,labels=[f"{v:g}" for v in x]);ax.set_xlabel("Payload (KiB)");ax.set_ylabel("Median acknowledged transfer (ms)")
        ax.set_ylim(bottom=0);ax.grid(axis="y",color="#e2e8f0");ax.set_title("Transport control without GPU execution",fontweight="bold",pad=12)
        fig.supxlabel("Same SSH-forwarded link · framed CPU transfer + acknowledgement · 20 samples per size",fontsize=9)
        save(fig,args.out,"transport-control")


if __name__=="__main__":main()
