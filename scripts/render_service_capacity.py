"""Plot the default three-agent service across two and four parameter shards."""
import argparse
import hashlib
import json
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def records(root, entries):
    output=[]
    for entry in entries:
        path=root/entry["raw_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError("raw-record hash mismatch")
        output.extend(json.loads(line) for line in path.read_text().splitlines())
    for row in output:
        if row["errors"] or row.get("coalesced_gathers") is not False:
            raise ValueError("expected successful default-policy records")
        for field,value in [("gpu_agents",3),("rows",262144),("dimension",128),
                            ("keys_per_step",1024),("bag_length",32)]:
            if row[field]!=value:raise ValueError("mixed figure workload")
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paired",type=pathlib.Path,required=True)
    p.add_argument("--controls",type=pathlib.Path,required=True)
    p.add_argument("--out",type=pathlib.Path,required=True)
    a=p.parse_args()
    paired=json.loads((a.paired/"manifest.json").read_text())
    controls=json.loads((a.controls/"manifest.json").read_text())
    two=records(a.paired,[e for e in paired["experiments"] if e["variant"]=="candidate"])
    four=records(a.controls,[e for e in controls["experiments"] if e["name"]=="streamed-four-shards"])
    if not two or not four or any(r["ps_shards"]!=2 for r in two) or any(r["ps_shards"]!=4 for r in four):
        raise ValueError("expected two- and four-shard data")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#cbd5e1",
        "svg.fonttype":"none","svg.hashsalt":"eps-service-capacity"})
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.3),layout="constrained")
    for label,rows,color in [("2 parameter shards",two,"#2563eb"),("4 parameter shards",four,"#0f766e")]:
        groups={}
        for row in rows:groups.setdefault(row["learners"],[]).append(row)
        x=sorted(groups)
        for ax,metric in zip(axes,["steps_per_second","step_p99_ms"]):
            values=[[r[metric] for r in groups[n]] for n in x]
            ax.plot(x,[statistics.median(v) for v in values],"o-",label=label,color=color,linewidth=2)
            ax.fill_between(x,[min(v) for v in values],[max(v) for v in values],color=color,alpha=.12)
    for ax in axes:
        ax.set_xscale("log",base=2);ax.set_xticks([4,8,16,32,64,128],labels=[4,8,16,32,64,128])
        ax.set_xlabel("Concurrent Learners");ax.grid(axis="y",color="#e2e8f0");ax.set_axisbelow(True)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Completed sparse updates / second");axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("Median trial P99 latency (ms)");axes[1].set_yscale("log")
    fig.suptitle("Three-GPU parameter-service capacity",fontsize=16,fontweight="bold")
    fig.supxlabel("RTX A6000 agents · 262,144 rows × 128 dimensions · 1,024 IDs/step · default per-shard projected gathers\n"
        "5-second windows + drain; median and observed range (3 trials for 2 shards, 2 for 4). One local agent; two forwarded links.",fontsize=9)
    a.out.mkdir(parents=True,exist_ok=True);svg=a.out/"service-capacity.svg"
    fig.savefig(svg,bbox_inches="tight",facecolor="white",metadata={"Date":None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    fig.savefig(a.out/"service-capacity.png",bbox_inches="tight",facecolor="white",dpi=170)
    plt.close(fig)


if __name__=="__main__":main()
