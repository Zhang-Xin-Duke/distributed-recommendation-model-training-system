"""Render the recorded two-GPU training capacity and loss measurements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
record = json.loads((ROOT / "docs/validation/2026-11-04-multi-gpu-training-capacity.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
fig, (capacity, loss) = plt.subplots(1, 2, figsize=(12, 4.1), gridspec_kw={"width_ratios": [1.2, 1]})
fig.patch.set_facecolor("white")
names = ["L40 physical capacity", "A6000 physical capacity", "Sharded training state"]
values = [n["gpu"]["total_bytes"] / 2**30 for n in record["nodes"]] + [record["global_parameter_optimizer_bytes"] / 2**30]
capacity.barh(names[:2], values[:2], color="#a6b0be", height=.55)
capacity.barh(names[2], 30, color="#2463b4", height=.55, label="L40 owner")
capacity.barh(names[2], 30, left=30, color="#28a3a1", height=.55, label="A6000 owner")
for i, value in enumerate(values):
    capacity.text(value + .7, i, f"{value:.2f} GiB", va="center", fontsize=10)
capacity.text(15, 2, "L40 · 30 GiB", color="white", ha="center", va="center", weight="bold")
capacity.text(45, 2, "A6000 · 30 GiB", color="white", ha="center", va="center", weight="bold")
capacity.set(xlim=(0, 72), xlabel="GPU memory (GiB)", title="60 GiB state exceeds either single GPU")
capacity.invert_yaxis()
losses = [sum(n["losses"][i] for n in record["nodes"]) / len(record["nodes"]) for i in range(8)]
loss.plot(range(1, 9), losses, color="#2463b4", marker="o", linewidth=2)
loss.set(xlabel="Synchronous training round", ylabel="Mean training loss", title="Both GPU workers complete eight rounds", xticks=range(1, 9), ylim=(.15, .76))
loss.annotate(f"{losses[-1]:.3f}", (8, losses[-1]), xytext=(-30, 13), textcoords="offset points", weight="bold", color="#2463b4")
for ax in [capacity, loss]:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x" if ax is capacity else "y", color="#e9edf2", linewidth=.7)
    ax.set_axisbelow(True)
fig.text(.5, .015, "5,242,880 rows × 1,024 FP32 dimensions · weights + two Adam moments · one parameter owner and GPU model worker per node", ha="center", fontsize=9, color="#475569")
fig.tight_layout(rect=(0, .05, 1, 1))
fig.savefig(ROOT / "docs/assets/multi-gpu-training-capacity.svg", metadata={"Date": None}, bbox_inches="tight")
fig.savefig(ROOT / "build/read-cleanup-gpu/capacity-preview.png", dpi=130, bbox_inches="tight")
