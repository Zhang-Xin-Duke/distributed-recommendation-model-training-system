"""Validate the standalone job schema and bind it to prepared input."""
from pathlib import Path
import copy
import json
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters" / "entropy"))
from recommendation_input import canonical, digest, IndexedBatches


def normalize(value):
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("expected recommendation job schema_version 1")
    allowed = {"schema_version", "id", "priority", "groups", "device", "data", "model", "training", "resources"}
    if set(value) - allowed:
        raise ValueError("unknown recommendation job option")
    data, model, training = (value.get(key, {}) for key in ("data", "model", "training"))
    if (set(data) != {"root", "version"} or set(model) - {"embedding_dimension", "hidden"}
            or set(training) - {"rounds", "batch_size", "learning_rate", "seed", "checkpoint_root"}):
        raise ValueError("invalid data, model or training configuration")
    root = Path(data["root"])
    checkpoint = Path(training["checkpoint_root"])
    if not root.is_absolute() or not checkpoint.is_absolute():
        raise ValueError("dataset and checkpoint roots must be absolute")
    with (root / "manifest.json").open("rb") as stream:
        raw = stream.read((64 << 10) + 1)
    if len(raw) > 64 << 10:
        raise ValueError("dataset manifest is too large")
    manifest = json.loads(raw)
    version = manifest.pop("version")
    groups = value.get("groups", 2)
    if (manifest["schema_version"] != 1 or version != digest(canonical(manifest))
            or version != data["version"] or manifest["workers"] != groups
            or manifest["batch_size"] != training.get("batch_size", manifest["batch_size"])):
        raise ValueError("prepared dataset version, worker count or batch size differs")
    dim, hidden = model.get("embedding_dimension", 32), model.get("hidden", 32)
    lr, seed = training.get("learning_rate", .002), training.get("seed", 42)
    if (type(dim) is not int or not 1 <= dim <= 4096 or type(hidden) is not int or not 1 <= hidden <= 256
            or type(lr) not in (int, float) or not math.isfinite(lr) or not 0 < lr <= 1
            or type(seed) is not int or not 0 <= seed < 2**32):
        raise ValueError("invalid recommendation model or optimizer settings")
    packed = (dim * hidden + 2 * hidden + dim) // dim
    datasets = []
    for worker in range(groups):
        spec = dict(root=str(root), version=version, worker=worker)
        for split in ("train", "validation"):
            record = manifest[split][worker]
            spec.update({split + "_batches": record["batches"], split + "_sha256": record["sha256"]})
            IndexedBatches(spec, split)
        datasets.append(spec)
    spec = {key: copy.deepcopy(value[key]) for key in ("id", "priority", "resources") if key in value}
    spec.update(groups=groups, device=value.get("device", "gpu"))
    spec.setdefault("resources", {}).setdefault("max_groups_per_node", 1 if spec["device"] == "gpu" else groups)
    spec["workload"] = dict(rows=packed + max(7, manifest["feature_rows"]),
                            hidden=hidden, rounds=training.get("rounds", datasets[0]["train_batches"]),
                            datasets=datasets, checkpoint_root=str(checkpoint),
                            table=dict(id=1, dimension=dim, dtype="float32",
                                       optimizer=dict(kind="sparse_lazy_adam", learning_rate=lr, beta1=.9,
                                                      beta2=.999, epsilon=1e-8, initial_beta1_power=.9,
                                                      initial_beta2_power=.999),
                                       initialization=dict(min=-.1, max=.1, seed=seed)))
    return spec
