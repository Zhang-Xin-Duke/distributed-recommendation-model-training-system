"""Indexed, checksummed recommendation batches and restart identity."""
import hashlib
import json
from pathlib import Path

KIND = "indexed-recommendation-v1"
MAX_BATCH_BYTES = 8 << 20
MAX_INDEX_BYTES = 16 << 20


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def input_identity(config):
    source = config.get("dataset")
    if source is None:
        return config.get("batches", [config.get("samples", [])])
    return {"kind": KIND, **{key: source[key] for key in (
        "version", "worker", "train_batches", "train_sha256", "validation_batches", "validation_sha256"
    )}}


def input_state(descriptor, model, iteration):
    encoded = canonical(descriptor)
    indexed = isinstance(descriptor, dict)
    count = descriptor["train_batches"] if indexed else len(descriptor)
    return dict(kind=KIND if indexed else "cyclic-json-batches-v1",
                dataset_json=encoded.decode(), dataset_sha256=digest(encoded),
                next_iteration=iteration, next_batch=iteration % count, model=model)


class IndexedBatches:
    def __init__(self, source, split):
        worker = source["worker"]
        if type(worker) is not int or not 0 <= worker < 8 or split not in ("train", "validation"):
            raise ValueError("invalid input shard")
        base = Path(source["root"]) / f"{split}-{worker}"
        with base.with_suffix(".index").open("rb") as stream:
            raw = stream.read(MAX_INDEX_BYTES + 1)
        if len(raw) > MAX_INDEX_BYTES or digest(raw) != source[split + "_sha256"]:
            raise ValueError("input index checksum mismatch")
        self.entries = json.loads(raw)
        if not self.entries or len(self.entries) != source[split + "_batches"]:
            raise ValueError("input batch count mismatch")
        self.path = base.with_suffix(".jsonl")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        offset, size, checksum = self.entries[index]
        if not 0 < size <= MAX_BATCH_BYTES or offset < 0:
            raise ValueError("invalid input batch extent")
        with self.path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(size)
        if len(raw) != size or digest(raw) != checksum:
            raise ValueError(f"input batch checksum mismatch: {self.path.name}, batch {index}")
        return json.loads(raw)
