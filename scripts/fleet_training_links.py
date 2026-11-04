"""Create scoped SSH relays between owned training nodes without copying keys.

Run this on the controller with the fleet file. Nodes access one another through
controller-local forwards and node-local reverse forwards. This transport must
be identified explicitly in latency/throughput results.
"""
import argparse
import json
import os
import pathlib
import signal
import socket
import subprocess
import time


def ssh_options(node):
    return ["ssh", "-NT", "-o", "BatchMode=yes", "-o", "IdentityAgent=none",
            "-o", "IdentitiesOnly=yes", "-o", "ExitOnForwardFailure=yes",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=3", "-o", "UserKnownHostsFile=" + node["known_hosts"],
            "-i", node["identity_file"], "-p", str(node["port"])]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def stop(path):
    state = json.loads(path.read_text())
    for item in state["processes"]:
        result = subprocess.run(["ps", "-ww", "-p", str(item["pid"]), "-o", "command="],
                                text=True, capture_output=True)
        if result.returncode:
            continue
        if "ssh" not in result.stdout or item["forward_marker"] not in result.stdout:
            raise RuntimeError("relay process identity changed; refusing to signal another process")
        try:
            os.killpg(item["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    state["stop_requested"] = True
    write(path, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fleet", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path)
    p.add_argument("--state", type=pathlib.Path, required=True)
    p.add_argument("--base-port", type=int, default=33000)
    p.add_argument("--stop", action="store_true")
    a = p.parse_args()
    if a.stop:
        stop(a.state)
        return
    if not a.fleet or not a.out or a.state.exists():
        p.error("provide fleet/out and a fresh state path; inspect existing relays before restarting")
    fleet = json.loads(a.fleet.read_text())
    nodes = fleet["nodes"]
    count = len(nodes)
    if not 2 <= count <= 8 or not 1024 <= a.base_port <= 64000:
        p.error("invalid relay layout")
    ps = [a.base_port + i for i in range(count)]
    agents = [a.base_port + 100 + i for i in range(count)]
    coordinator = a.base_port + 200
    # Check controller ports without stealing an existing listener.
    for port in [*ps, *agents, coordinator]:
        with socket.socket() as check:
            check.bind(("127.0.0.1", port))
    state = {"transport": "SSH relay through controller", "processes": [], "stop_requested": False}
    write(a.state, state)
    children = []
    try:
        for i, node in enumerate(nodes):
            own_ps = node.get("ps_listen", "127.0.0.1:19100")
            own_agent = node.get("agent_listen", "127.0.0.1:19200")
            own_coordinator = node.get("coordinator_listen", "127.0.0.1:19300")
            command = ssh_options(node)
            marker = str(ps[i]) + ":" + own_ps
            command += ["-L", marker, "-L", str(agents[i]) + ":" + own_agent]
            if i == 0:
                command += ["-L", str(coordinator) + ":" + own_coordinator]
            for other in range(count):
                if other != i:
                    command += ["-R", f"{a.base_port+1000+other}:127.0.0.1:{ps[other]}",
                                "-R", f"{a.base_port+1100+other}:127.0.0.1:{agents[other]}"]
            if i:
                command += ["-R", f"{a.base_port+1200}:127.0.0.1:{coordinator}"]
            command.append(node.get("user", "ubuntu") + "@" + node["host"])
            log = a.state.parent / (a.state.stem + f"-node-{i}.log")
            with log.open("w") as stream:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                         stderr=stream, start_new_session=True)
            children.append(child)
            state["processes"].append({"pid": child.pid, "forward_marker": marker, "log": str(log)})
            write(a.state, state)
            node["ps_listen"], node["agent_listen"], node["coordinator_listen"] = own_ps, own_agent, own_coordinator
            node["shards"] = [own_ps if j == i else f"127.0.0.1:{a.base_port+1000+j}" for j in range(count)]
            node["agents"] = [{"id": j, "endpoint": own_agent if j == i else f"127.0.0.1:{a.base_port+1100+j}"}
                              for j in range(count)]
            node["coordinator"] = own_coordinator if i == 0 else f"127.0.0.1:{a.base_port+1200}"
        deadline = time.monotonic() + 40
        while True:
            for child, item in zip(children, state["processes"]):
                if child.poll() is not None:
                    raise RuntimeError("SSH relay exited: " + pathlib.Path(item["log"]).read_text()[-2000:])
            connected = True
            for port in [*ps, *agents, coordinator]:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        pass
                except OSError:
                    connected = False
            if connected:
                break
            if time.monotonic() >= deadline:
                # Preserve authoritative process handles for inspection.
                raise TimeoutError("relay observation timed out; inspect the existing state file")
            time.sleep(.2)
        fleet["transport"] = state["transport"]
        write(a.out, fleet)
        state["connected_fleet"] = str(a.out)
        write(a.state, state)
        print(json.dumps({"relays": len(children), "connected_fleet": str(a.out)}))
    except TimeoutError:
        raise
    except Exception:
        stop(a.state)
        raise


if __name__ == "__main__":
    main()
