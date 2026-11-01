"""Verify failed capacity points retain post-failure diagnostics and exit nonzero."""
import json
import pathlib
import subprocess
import sys
import tempfile

from rpc_processes import ready, stop


def main():
    build = pathlib.Path(sys.argv[1]).resolve()
    processes = []
    with tempfile.TemporaryDirectory(prefix="eps-capacity-failure-") as directory:
        root = pathlib.Path(directory)
        logs = []
        try:
            log = (root / "agent.log").open("w+")
            logs.append(log)
            agent = subprocess.Popen([str(build / "eps_cache_agent"), "--run", "failure-test",
                "--backend", "host", "--agent", "0", "--rows-per-table", "1024",
                "--state-bytes", str(8 << 20), "--listen", "127.0.0.1:0"],
                stdout=subprocess.PIPE, stderr=log, text=True)
            processes.append(agent)
            endpoint = ready(agent)
            config = root / "server.json"
            config.write_text(json.dumps(dict(schema_version=1, run_id="failure-test",
                listen="127.0.0.1:0", partition=dict(shard_id=0, shard_count=1, epoch=1),
                tables=[], cache=dict(enabled=True), agents=[dict(id=0, endpoint=endpoint)],
                access=dict(policy="cpu_only"),
                limits=dict(cpu_state_bytes=8 << 20, max_live_tickets=1,
                    ticket_payload_bytes=8 << 20, queue_payload_bytes=8 << 20,
                    max_active_operations=16), snapshot_directory=str(root / "snapshots"))))
            log = (root / "server.log").open("w+")
            logs.append(log)
            server = subprocess.Popen([str(build / "eps_server"), "--config", str(config)],
                stdout=subprocess.PIPE, stderr=log, text=True)
            processes.append(server)
            server_endpoint = ready(server)
            result = subprocess.run([str(build / "eps_capacity_probe"), server_endpoint,
                endpoint, "failure-test", "1024", "16", "128", "32", "16", "1", "1"],
                capture_output=True, text=True, timeout=60)
            assert result.returncode != 0, "expected ticket admission failure"
            records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
            assert len(records) == 1, result.stdout + result.stderr
            row = records[0]
            assert row["test"] == "distributed_sparse_training_overload" and row["errors"] > 0
            assert row["optimizer_trajectory_verified"] is False
            assert "steps_per_second" not in row
            assert len(row["post_failure_shards"]) == 1
            assert row["post_failure_shards"][0]["responsive"] is True
            assert "completed" in row["fresh_read_only_probe"]
            assert "checked_rows" in row["sampled_state"]
            assert len(row["after_health_probe_shards"]) == 1
            print(json.dumps(row))
        finally:
            for process in reversed(processes):
                stop(process)
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
