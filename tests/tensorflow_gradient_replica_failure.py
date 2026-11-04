"""A failed gradient replica must prevent all optimizer submissions."""
import argparse
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile

from rpc_processes import ready,stop


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();build=a.build.resolve();source=pathlib.Path(__file__).resolve().parents[1]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='2',PYTHONPATH=str(build/'python')+':'+str(source/'adapters/entropy'))
    table=dict(id=1,dimension=8,dtype='float32',optimizer=dict(kind='sparse_lazy_adam',learning_rate=.02,beta1=.9,beta2=.999,epsilon=1e-8,initial_beta1_power=.9,initial_beta2_power=.999),initialization=dict(min=-.1,max=.1,seed=42))
    children=[];logs=[];worker=None
    with tempfile.TemporaryDirectory(prefix='eps-mirror-failure-') as tmp, socket.socket() as unavailable:
        root=pathlib.Path(tmp);unavailable.bind(('127.0.0.1',0));run='replica-before-submit'
        def launch(name,binary,cfg,python=False):
            path=root/(name+'.json');path.write_text(json.dumps(cfg));log=open(root/(name+'.stderr'),'w+');logs.append(log)
            cmd=[sys.executable,str(binary)] if python else [str(binary)]
            child=subprocess.Popen([*cmd,'--config',str(path)],env=env,stdout=subprocess.PIPE,stderr=log,text=True,start_new_session=python);children.append(child);return child
        try:
            shards=[]
            for h in range(2):
                cfg=dict(schema_version=1,run_id=run,listen='127.0.0.1:0',partition=dict(shard_id=h,shard_count=2,epoch=1),tables=[table],cache=dict(enabled=False),synchronous_training=dict(workers=1,reduction='sample_mean'))
                shards.append(ready(launch('primary-'+str(h),build/'eps_server',cfg)))
            coordinator=ready(launch('coordinator',build/'eps_training_coordinator',dict(schema_version=1,run=run,workers=1,listen='127.0.0.1:0',consistency='sequential',max_delay=0)))
            cfg=dict(run=run,shards=shards,coordinator=coordinator,worker=0,dimension=8,hidden=4,iterations=1,samples=[dict(keys=[7,8],label=0.),dict(keys=[9],label=1.)],table=table,device='/CPU:0',journal_directory=str(root/'journal'),journal_workers=1,journal_mirror=dict(endpoint='127.0.0.1:'+str(unavailable.getsockname()[1]),generation=1,writer='worker-0-g0',timeout_ms=100))
            worker=launch('worker',source/'examples/tensorflow_training_worker.py',cfg,True);worker.communicate(timeout=60)
            assert worker.returncode!=0,'worker submitted without a durable gradient replica'
            control=root/'inspect.json';control.write_text(json.dumps(dict(run=run,shards=shards,coordinator=coordinator,keys=[0,1,2,3,4,5,7,8,9],synchronous=True)))
            inspected=subprocess.run([str(build/'eps_training_control'),'--config',str(control),'--action','inspect'],env=env,capture_output=True,text=True,timeout=20)
            assert inspected.returncode==0,inspected.stderr
            state=json.loads(inspected.stdout)
            assert state['completed']==[0] and state['failed']
            assert all(s['updates']==0 and s['synchronous']['next_round']==0 and s['live_tickets']==0 for s in state['shard_stats'])
            assert all(r['row_updates']==0 for r in state['rows'])
            sys.path.insert(0,str(source/'adapters/entropy'))
            from training_journal import WorkerJournal,identity
            with WorkerJournal(cfg['journal_directory'],identity(cfg)) as journal:assert journal.rounds()==[0] and journal.get(0)['completed_reads']>0
            result=dict(test='gradient_replica_failure_before_submit',worker_exit_code=worker.returncode,local_gradient_retained=True,completed_worker_clocks=state['completed'],parameter_shards=2,optimizer_updates=[s['updates'] for s in state['shard_stats']],zero_live_tickets=True,state=state,cleanup_verified=True)
        except Exception:
            for log in logs:log.flush();log.seek(0);print(log.read()[-3000:],file=sys.stderr)
            raise
        finally:
            if worker is not None and worker.poll() is None:os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=5)
            for child in reversed(children):stop(child)
            for log in logs:log.close()
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='state'}))


if __name__=='__main__':main()
