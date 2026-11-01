"""Exercise configured Agent budgets through the documented HTTP statistics API."""
import json
import pathlib
import subprocess
import sys
import tempfile
import urllib.request
from rpc_processes import ready, stop


def main():
    binary=str(pathlib.Path(sys.argv[1]).resolve())
    with tempfile.TemporaryFile(mode="w+") as log:
        process=subprocess.Popen([binary,"--run","admission-cli-check","--backend","host",
            "--listen","127.0.0.1:0","--max-active-rpcs","2","--max-pending-rpcs","64",
            "--request-bytes","65536"],stdout=subprocess.PIPE,stderr=log,text=True)
        try:
            endpoint=ready(process)
            def stats(body):
                request=urllib.request.Request("http://"+endpoint+"/eps.v1.CacheAgentService/AdmissionStats",
                    data=body,headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(request,timeout=5) as response:
                    return json.load(response)
            observed=stats(b"{}")
            assert int(observed["active_limit"])==2,observed
            assert int(observed["pending_limit"])==64,observed
            assert int(observed["request_bytes_limit"])==65536,observed
            oversized=stats(b" "*2048+b"{}")
            assert oversized["status"]["code"]=="INVALID_ARGUMENT",oversized
            assert int(stats(b"{}")["active_limit"])==2
        finally:
            stop(process)
    for value in ("0","129"):
        result=subprocess.run([binary,"--run","bad-limits","--backend","host",
            "--max-active-rpcs",value],capture_output=True,text=True,timeout=5)
        assert result.returncode and "invalid Agent admission limits" in result.stderr
    print(json.dumps(dict(test="agent_admission_cli",active_limit=2,pending_limit=64,
        request_bytes_limit=65536,http_statistics_verified=True,
        oversized_statistics_payload_rejected=True,invalid_active_limits_rejected=[0,129])))


if __name__=="__main__":
    main()
