#!/usr/bin/env python3
"""Prepare immutable, indexed train and validation batches for GPU jobs."""
import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters" / "entropy"))
from recommendation_input import MAX_BATCH_BYTES, MAX_INDEX_BYTES, canonical, digest


def read_rows(path, movielens):
    with path.open(newline="", encoding="utf-8") as stream:
        if movielens:
            records = (dict(zip(("user_id", "item_id", "rating", "timestamp"), row, strict=True))
                       for row in csv.reader(stream, delimiter="\t"))
        else:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not {"user_id", "item_id", "label", "timestamp"} <= set(reader.fieldnames):
                raise ValueError("CSV requires user_id,item_id,label,timestamp")
            records = reader
        rows = []
        for row in records:
            user, item = row["user_id"], row["item_id"]
            rating = float(row["rating"]) if movielens else float(row["label"])
            if not user or not item or not (1 <= rating <= 5 if movielens else 0 <= rating <= 1):
                raise ValueError("invalid user, item, or label")
            rows.append((int(row["timestamp"]), user, item, float(rating >= 4) if movielens else rating))
    return sorted(rows)


def write_batches(root, split, rows, workers, batch_size, users, items):
    # Give each worker the same number of batches, retaining every input row.
    rounds = (len(rows) + workers * batch_size - 1) // (workers * batch_size)
    if not rounds or len(rows) < rounds * workers:
        raise ValueError(f"{split} split is too small for {workers} workers")
    metadata = []
    for worker in range(workers):
        local = rows[worker::workers]
        base = root / f"{split}-{worker}"
        entries = []
        with base.with_suffix(".jsonl").open("wb") as stream:
            for i in range(rounds):
                segment = local[i * len(local) // rounds:(i + 1) * len(local) // rounds]
                raw = canonical([dict(keys=[users[u], items[v]], label=y) for _, u, v, y in segment])
                if len(raw) > MAX_BATCH_BYTES:
                    raise ValueError("batch exceeds the reader limit")
                entries.append((stream.tell(), len(raw), digest(raw)))
                stream.write(raw + b"\n")
        index = canonical(entries)
        if len(index) > MAX_INDEX_BYTES:
            raise ValueError("batch index exceeds the reader limit")
        base.with_suffix(".index").write_bytes(index)
        metadata.append(dict(batches=rounds, sha256=digest(index)))
    return metadata


def prepare(path, output, workers, batch_size, movielens=False):
    if not 2 <= workers <= 8 or not 1 <= batch_size <= 4096:
        raise ValueError("workers must be 2–8 and batch size 1–4096")
    rows = read_rows(path, movielens)
    if len(rows) < 2 * workers:
        raise ValueError("dataset too small for a train and validation split")
    boundary = rows[int(len(rows) * .8)][0]
    split = next(i for i, row in enumerate(rows) if row[0] == boundary)
    users = {key: i for i, key in enumerate(sorted({row[1] for row in rows}))}
    items = {key: len(users) + i for i, key in enumerate(sorted({row[2] for row in rows}))}
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".recommendation-", dir=output.parent))
    try:
        train = write_batches(staging, "train", rows[:split], workers, batch_size, users, items)
        validation = write_batches(staging, "validation", rows[split:], workers, batch_size, users, items)
        manifest = dict(schema_version=1, workers=workers, batch_size=batch_size,
                        feature_rows=len(users) + len(items), validation_start=boundary,
                        train_rows=split, validation_rows=len(rows)-split, train=train, validation=validation)
        manifest["version"] = digest(canonical(manifest))
        (staging / "manifest.json").write_bytes(canonical(manifest) + b"\n")
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--movielens", action="store_true", help="read GroupLens ml-100k/u.data")
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.workers, args.batch_size, args.movielens), indent=2))
