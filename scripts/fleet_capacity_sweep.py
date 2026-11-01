"""Coordinate GPU workers on separate SSH hosts and run the capacity sweep.

The fleet JSON contains a coordinator and workers, each with host, port,
identity_file, user and root. These connection details stay outside the repo.
Existing SSH port forwards expose workers to the coordinator as loopback ports.
"""
import argparse
import concurrent.futures
import json
import pathlib
import shlex
import subprocess
import uuid


def remote(node, script, timeout=120):
    command = ["ssh", "-o", "IdentityAgent=none", "-o", "IdentitiesOnly=yes",
               "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "UserKnownHostsFile="+node["known_hosts"],
               "-i", node["identity_file"], "-p", str(node["port"]),
               node.get("user", "ubuntu")+"@"+node["host"], "python3 -"]
    result = subprocess.run(command, input=script, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:]+result.stdout[-1000:])
    return result.stdout


def start_worker(node, agent, run, rows, dimension, host_reference=False, cache_slots=None):
    options = dict(root=node["root"], agent=agent, run=run, rows=rows, dimension=dimension,host_reference=host_reference,cache_slots=cache_slots)
    script = "options="+repr(options)+"\n"+r'''
import os,pathlib,subprocess,json,time
o=options;root=pathlib.Path(o['root']);out=root/'results'/o['run'];out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['LD_LIBRARY_PATH']=str(root/'agent-bundle/lib')+':'+env.get('LD_LIBRARY_PATH','')
stdout=open(out/'agent.stdout','w');stderr=open(out/'agent.stderr','w')
slots=o['cache_slots'] or o['rows']//2+1024
p=subprocess.Popen([str(root/'agent-bundle/bin/eps_cache_agent'),'--run',o['run'],'--backend','host' if o['host_reference'] else 'cuda',
 '--device','0','--agent',str(o['agent']),'--rows-per-table',str(slots),
 '--state-bytes',str(slots*(o['dimension']*12+8)*(2 if o['cache_slots'] else 1)+4096),'--listen','127.0.0.1:19200'],
 stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
record=dict(agent=p.pid,monitor=None,cpu_monitor=None,run=o['run'])
(out/'processes.json').write_text(json.dumps(record))
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,compute_cap,driver_version,memory.total','--format=csv,noheader'],text=True)
(out/'hardware.csv').write_text(hardware)
for i in range(300):
 if p.poll() is not None:raise RuntimeError((out/'agent.stderr').read_text())
 if 'READY ' in (out/'agent.stdout').read_text():break
 time.sleep(.1)
else:raise RuntimeError('GPU agent did not become ready')
gpu_log=open(out/'gpu-telemetry.csv','w')
monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw',
 '--format=csv','-l','1'],stdin=subprocess.DEVNULL,stdout=gpu_log,stderr=subprocess.DEVNULL,start_new_session=True)
sampler="import pathlib,time,os,json,sys\npid=int(sys.argv[1]);out=open(sys.argv[2],'w');ticks=os.sysconf('SC_CLK_TCK');page=os.sysconf('SC_PAGE_SIZE')\nwhile pathlib.Path('/proc/'+str(pid)+'/stat').exists():\n try:\n  f=pathlib.Path('/proc/'+str(pid)+'/stat').read_text().split();out.write(json.dumps(dict(monotonic=time.monotonic(),cpu_seconds=(int(f[13])+int(f[14]))/ticks,rss_bytes=int(f[23])*page))+'\\n');out.flush()\n except FileNotFoundError:break\n time.sleep(.5)\n"
cpu_monitor=subprocess.Popen(['python3','-c',sampler,str(p.pid),str(out/'agent-cpu.jsonl')],
 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=stderr,start_new_session=True)
record.update(monitor=monitor.pid,cpu_monitor=cpu_monitor.pid,hardware=hardware)
(out/'processes.json').write_text(json.dumps(record))
print(json.dumps(record))
'''
    return json.loads(remote(node, script))


def stop_worker(node, run):
    script = "root="+repr(node["root"])+"\nrun="+repr(run)+"\n"+r'''
import pathlib,json,os,signal
out=pathlib.Path(root)/'results'/run
record=out/'processes.json'
if record.exists():
 p=json.loads(record.read_text())
 for field in ('agent','monitor','cpu_monitor'):
  pid=p.get(field)
  if pid is None:continue
  try:
   cmd=pathlib.Path('/proc/'+str(pid)+'/cmdline').read_bytes().decode(errors='replace')
   if (field=='agent' and run in cmd and 'eps_cache_agent' in cmd) or (field=='monitor' and 'nvidia-smi' in cmd) or (field=='cpu_monitor' and str(out/'agent-cpu.jsonl') in cmd):
    os.killpg(pid,signal.SIGTERM)
  except (FileNotFoundError,ProcessLookupError):pass
print('stopped '+run)
'''
    return remote(node, script)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fleet", type=pathlib.Path, required=True)
    p.add_argument("--out-name", required=True)
    p.add_argument("--build-name", default="build-cuda",
                   help="Prepared coordinator build directory, relative to its project root")
    p.add_argument("--coalesce-gathers", choices=("0", "1"), default=None,
                   help="Set EPS_COALESCE_GATHERS for a probe that supports selectable gather policies")
    p.add_argument("--gpu-counts", default="1,2,3")
    p.add_argument("--ps-shards", type=int, default=2)
    p.add_argument("--rows", type=int, default=262144)
    p.add_argument("--dimension", type=int, default=128)
    p.add_argument("--keys", type=int, default=1024)
    p.add_argument("--bag", type=int, default=32)
    p.add_argument("--learners", default="1,2,4,8,16,32,64")
    p.add_argument("--seconds", type=float, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--serial-agent-updates", action="store_true")
    p.add_argument("--expanded-gradients", action="store_true")
    p.add_argument("--host-reference", action="store_true",help="CPU cache backend control on the same transport; no GPU-throughput claim")
    a = p.parse_args()
    if not a.out_name.replace("-", "").replace("_", "").isalnum():
        p.error("out-name must contain letters, digits, underscores or dashes")
    if not a.build_name.replace("-", "").replace("_", "").isalnum():
        p.error("build-name must be one directory name")
    fleet = json.loads(a.fleet.read_text());coordinator=fleet["coordinator"]
    for count in map(int,a.gpu_counts.split(",")):
        if not 1<=count<=len(fleet["workers"]):
            p.error("GPU count exceeds the prepared fleet")
        workers=fleet["workers"][:count]
        run="fleet-"+uuid.uuid4().hex[:12]
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                records=list(pool.map(lambda pair:start_worker(pair[1],pair[0],run,a.rows,a.dimension,a.host_reference),enumerate(workers)))
            hardware="".join(record["hardware"] for record in records)
            identities=[line.split(",")[0] for line in hardware.splitlines() if line]
            if len(set(identities))!=count:
                raise RuntimeError("workers do not expose distinct GPU UUIDs")
            root=pathlib.PurePosixPath(coordinator["root"])
            output=root/"results"/a.out_name/f"gpu-{count}"
            command=[str(root/"environment/bin/python"),str(root/"source/scripts/capacity_sweep.py"),
                "--build",str(root/a.build_name),"--out",str(output),
                "--external-agents",",".join("127.0.0.1:"+str(w["forward_port"]) for w in workers),
                "--external-hardware",str(root/"results"/(run+"-hardware.csv")),"--run-id",run,
                "--devices",",".join(map(str,range(count))),"--gpu-counts",str(count),
                "--ps-shards",str(a.ps_shards),"--rows",str(a.rows),"--dimension",str(a.dimension),
                "--keys",str(a.keys),"--bag",str(a.bag),"--learners",a.learners,
                "--seconds",str(a.seconds),"--repeats",str(a.repeats)]
            if a.serial_agent_updates:
                command.append("--serial-agent-updates")
            if a.expanded_gradients:
                command.append("--expanded-gradients")
            if a.host_reference:
                command.append("--host-reference")
            script="hardware="+repr(hardware)+"\ncommand="+repr(command)+"\ncoalesce="+repr(a.coalesce_gathers)+"\n"+r'''
import pathlib,subprocess,os
pathlib.Path(command[command.index('--external-hardware')+1]).write_text(hardware)
environment=None if coalesce is None else dict(os.environ,EPS_COALESCE_GATHERS=coalesce)
subprocess.run(command,check=True,env=environment)
'''
            result=remote(coordinator,script,timeout=300+len(a.learners.split(","))*a.repeats*(a.seconds+30))
            print(result,flush=True)
        finally:
            with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
                for result in pool.map(lambda node:stop_worker(node,run),workers):
                    print(result.strip(),flush=True)


if __name__ == "__main__":
    main()
