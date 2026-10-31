"""Verify bounded admission and recovery with an external cache process."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from rpc_processes import ready, stop


def main():
    server,agent,probe=sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="eps-overload-") as temporary:
        root=pathlib.Path(temporary)
        processes,logs=[],[]
        try:
            log=(root/"agent.log").open("w+");logs.append(log)
            worker=subprocess.Popen([agent,"--run","overload-test","--backend",os.environ.get("EPS_CACHE_BACKEND","host"),
                                     "--listen","127.0.0.1:0"],stdout=subprocess.PIPE,stderr=log,text=True)
            processes.append(worker);endpoint=ready(worker,timeout=30)
            config=dict(schema_version=1,run_id="overload-test",listen="127.0.0.1:0",
                        partition=dict(shard_id=0,shard_count=1,epoch=1),tables=[],cache=dict(enabled=True),
                        agents=[dict(id=0,endpoint=endpoint)],limits=dict(max_live_tickets=32),
                        snapshot_directory=str(root/"snapshots"))
            path=root/"server.json";path.write_text(json.dumps(config))
            log=(root/"server.log").open("w+");logs.append(log)
            ps=subprocess.Popen([server,"--config",str(path)],stdout=subprocess.PIPE,stderr=log,text=True)
            processes.append(ps)
            subprocess.run([probe,ready(ps),endpoint],check=True,timeout=30)
        except Exception:
            for log in logs:
                log.flush();log.seek(0);print(log.read()[-4000:],file=sys.stderr)
            raise
        finally:
            for process in reversed(processes):stop(process)
            for log in logs:log.close()


if __name__=="__main__":main()
