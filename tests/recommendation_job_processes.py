"""Exercise real indexed input, evaluation and a checkpoint restart via the CLI store."""
import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

from training_job_controller_tests import await_jobs, inventory, spec as local_spec
from prepare_recommendation_data import prepare
from training_job_controller import Controller, TERMINAL


def run(build, root):
    root.mkdir(parents=True)
    csv_path = root / "input.csv"
    with csv_path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "item_id", "label", "timestamp"])
        for i in range(120):
            writer.writerow([i % 6, i % 10, i % 2, i])
    data = root / "data"
    manifest = prepare(csv_path, data, 2, 8)
    inv = inventory(build, root, 2)
    controller = Controller(inv, root / "controller")
    spec = dict(schema_version=1, id="recommendation", device="cpu",
                resources=local_spec("recommendation", 2)["resources"],
                data=dict(root=str(data), version=manifest["version"]),
                model=dict(embedding_dimension=8, hidden=4),
                training=dict(rounds=3, checkpoint_root=str(root / "checkpoints")))
    try:
        controller.store.submit(spec)
        await_jobs(controller, [spec["id"]])
        status = controller.store.status(spec["id"])["training"]
        assert sum(w["validation"]["samples"] for w in status["workers"]) == manifest["validation_rows"]
        assert all(w["input_state"]["next_iteration"] == 3 for w in status["workers"])
        assert Path(status["checkpoint"]["manifest"]).is_file()
        bad_spec = dict(spec, id="bad-dataset", data=dict(spec["data"], version="0" * 64))
        try:
            controller.store.submit(bad_spec)
        except ValueError as error:
            assert "dataset version" in str(error)
        else:
            raise AssertionError("dataset version mismatch was accepted")
        manifest_path = Path(status["checkpoint"]["manifest"])
        shard = manifest_path.parent / "shard-0.eps"
        original = shard.read_bytes()
        try:
            shard.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            probe = root / "restore-probe.json"
            probe.write_text(json.dumps(dict(run="corrupt", manifest=str(manifest_path))))
            result = subprocess.run([build / "eps_training_control", "--config", probe,
                                     "--action", "restore-config"], capture_output=True, timeout=30)
            assert result.returncode != 0
        finally:
            shard.write_bytes(original)
        controller.store.resume(spec["id"], "resumed", 3)
        controller.store.submit(dict(spec, id="continuous", training=dict(spec["training"], rounds=6)))
        await_jobs(controller, ["resumed", "continuous"])
        summaries = []
        for name in ("resumed", "continuous"):
            report = controller._runtime(name) / "training/summary.json"
            summaries.append(json.loads(report.read_text())["model"])
        resumed, continuous = summaries
        assert all(w["start_iteration"] == 3 for w in resumed["workers"])
        assert resumed["state"]["completed"] == continuous["state"]["completed"] == [6, 6]
        for w, reference in zip(resumed["loss_history"], continuous["loss_history"]):
            assert len(w) == 3 and max(abs(x - y) for x, y in zip(w, reference[3:])) < 3e-6
        for row, reference in zip(resumed["state"]["rows"], continuous["state"]["rows"], strict=True):
            assert row["key"] == reference["key"]
            assert row["row_updates"] == reference["row_updates"]
            for key in ("weights", "first_moment", "second_moment"):
                assert max(abs(x - y) for x, y in zip(row[key], reference[key])) < 3e-6
    finally:
        for job in controller.store.jobs():
            if job["state"] not in TERMINAL:
                controller.store.cancel(job["id"])
        controller.tick()
        controller.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.out:
        run(args.build.resolve(), args.out.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="eps-recommendation-") as tmp:
            run(args.build.resolve(), Path(tmp) / "job")
