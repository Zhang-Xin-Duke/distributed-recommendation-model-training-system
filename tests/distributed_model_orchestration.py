"""Exercise the node lifecycle with real local PS/owner/model processes."""
import json
import pathlib
import socket
import subprocess
import sys
import tempfile


def main():
    build = pathlib.Path(sys.argv[1]).resolve()
    source = pathlib.Path(__file__).resolve().parents[1]
    sockets = []
    for _ in range(5):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); sockets.append(s)
    ports = [s.getsockname()[1] for s in sockets]
    for s in sockets: s.close()
    with tempfile.TemporaryDirectory(prefix="eps-node-lifecycle-") as temporary:
        root = pathlib.Path(temporary)
        nodes = []
        for i in range(2):
            nodes.append({"local": True, "root": str(root / f"node-{i}"), "source": str(source),
                          "bin_dir": str(build), "module_dir": str(build / "python"), "python": sys.executable,
                          "ps_listen": f"127.0.0.1:{ports[i]}", "agent_listen": f"127.0.0.1:{ports[i+2]}",
                          "coordinator_listen": f"127.0.0.1:{ports[4]}", "coordinator": f"127.0.0.1:{ports[4]}",
                          "shards": [f"127.0.0.1:{v}" for v in ports[:2]],
                          "agents": [{"id": j, "endpoint": f"127.0.0.1:{ports[j+2]}"} for j in range(2)]})
        fleet = root / "fleet.json"
        fleet.write_text(json.dumps({"nodes": nodes, "transport": "local loopback; Host owners"}))
        output = root / "results"
        try:
            result = subprocess.run([sys.executable, str(source / "scripts/distributed_model_training.py"),
                                     "--fleet", str(fleet), "--host", "--rows", "1024", "--dimension", "8",
                                     "--hidden", "4", "--samples", "4", "--bag", "4", "--iterations", "8",
                                     "--out", str(output)], capture_output=True, text=True, timeout=100)
            if result.returncode:
                raise RuntimeError(result.stdout[-4000:] + result.stderr[-4000:])
            report = json.loads((output / "summary.json").read_text())
            assert report["success"] and report["initialized_resident_rows"] == 1024
            assert report["sampled_state"]["completed"] == [8, 8]
            assert report["sampled_state"]["completed_tasks"] == 16
            assert all(s["updates"] == 8 and s["rows"] == 512 for s in report["sampled_state"]["shard_stats"])
            assert all(s["row_updates"] == 8 for s in report["sampled_state"]["rows"])
            assert all(n["resources"] for n in report["node_reports"])
            assert all(n["owned_processes_gone"] for n in report["process_cleanup"])
            assert report["final_weighted_loss"] < report["initial_weighted_loss"]
            print(json.dumps({"test": "distributed_model_orchestration", "workers": 2, "parameter_shards": 2,
                              "resident_rows": 1024, "rounds": 8, "completed_tasks": 16,
                              "initial_loss": report["initial_weighted_loss"],
                              "final_loss": report["final_weighted_loss"], "cleanup_verified": True}))
        finally:
            summary = output / "summary.json"
            if summary.exists():
                run = json.loads(summary.read_text())["run"]
                sys.path.insert(0, str(source / "scripts"))
                from training_node import dispatch
                for node in nodes:
                    dispatch({"op": "stop", "options": {**node, "run": run, "gpu": False}})


if __name__ == "__main__":
    main()
