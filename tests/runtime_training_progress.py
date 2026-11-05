"""Real native C contexts, parameter shards and training-task coordinator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes as c
import json
from pathlib import Path
import subprocess
import tempfile
import time

from rpc_processes import ready,stop
from runtime_c_processes import Api,Status,Text,Batch,Feature,Buffer,Release,Wait


TABLE=dict(id=1,dimension=3,dtype='float32',optimizer=dict(kind='sparse_lazy_adam',
    learning_rate=.01,beta1=.9,beta2=.999,epsilon=1e-8,initial_beta1_power=.9,initial_beta2_power=.999),
    initialization=dict(min=.25,max=.25,seed=42))


@contextmanager
def services(build,root,run,workers=2):
    processes=[];logs=[]
    def launch(name,binary,config):
        path=root/(name+'.json');path.write_text(json.dumps(config))
        log=(root/(name+'.log')).open('w+');logs.append(log)
        child=subprocess.Popen([str(build/binary),'--config',str(path)],stdout=subprocess.PIPE,stderr=log,text=True)
        processes.append(child);return ready(child)
    try:
        endpoints=[]
        for shard in range(2):
            endpoints.append(launch('shard-'+str(shard),'eps_server',dict(schema_version=1,run_id=run,
                listen='127.0.0.1:0',partition=dict(shard_id=shard,shard_count=2,epoch=1),
                tables=[],cache=dict(enabled=False),snapshot_directory=str(root/'snapshots'))))
        coordinator=launch('coordinator','eps_training_coordinator',dict(schema_version=1,run=run,
            listen='127.0.0.1:0',workers=workers,epoch=1,consistency='sequential',max_delay=0))
        yield endpoints,coordinator
    except BaseException:
        for log in logs:log.flush();log.seek(0);print(log.read()[-2000:])
        raise
    finally:
        for child in reversed(processes):stop(child)
        for log in logs:log.close()


class Client:
    def __init__(self,api,run,endpoints,coordinator,worker):
        self.api=api;self.pointer=c.c_void_p();self.status=Status()
        config=dict(run=run,endpoints=endpoints,learner=worker+17,tables=[TABLE],release_read_only=True,
                    training=dict(coordinator=coordinator,worker=worker))
        code=api.create(json.dumps(config).encode(),c.byref(self.pointer),c.byref(self.status))
        assert code==0,self.status.message
        self.keys=(c.c_uint64*2)(0,1);self.offsets=(c.c_uint64*2)(0,2)
        self.features=(Feature*1)(Feature(1,0,1,self.keys,2,self.offsets,2,0,0))

    def command(self,name,args=None,error=False):
        output=Text()
        code=self.api.command(self.pointer,name.encode(),json.dumps(args or {}).encode(),c.byref(output),c.byref(self.status))
        try:
            if error:
                assert code!=0,'unexpected command success: '+name
                return self.status.message.decode()
            assert code==0,(name,self.status.message)
            return json.loads(c.string_at(output.data,output.size))
        finally:
            if output.owner:output.release(output.owner)

    def pull(self,prepared=False,read_only=False,error=False):
        batch=Batch()
        if prepared:
            token=c.c_uint64()
            code=self.api.prepare(self.pointer,self.features,1,c.byref(token),c.byref(self.status))
            assert code==0,self.status.message
            code=self.api.pull_prepared(self.pointer,token.value,int(read_only),0,c.byref(batch),c.byref(self.status))
        else:
            code=self.api.pull(self.pointer,self.features,1,int(read_only),0,c.byref(batch),c.byref(self.status))
        if error:
            assert code!=0,'unarmed parameter read was accepted'
            return
        assert code==0,self.status.message
        return batch

    def push(self,batch):
        gradient=(c.c_float*3)(.1,.1,.1)
        buffers=(Buffer*1)(Buffer(0,1,3,c.cast(gradient,c.c_void_p),-1,None,Release(),Wait()))
        code=self.api.push(self.pointer,batch.handle,buffers,1,None,c.byref(self.status))
        assert code==0,self.status.message

    def close(self):
        if self.pointer.value:
            try:self.command('abort')
            finally:self.api.destroy(self.pointer);self.pointer=c.c_void_p()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--out',type=Path)
    args=parser.parse_args();build=args.build.resolve()
    library=c.CDLL(str(args.runtime.resolve()));library.eps_runtime_api_v1.restype=c.POINTER(Api)
    api=library.eps_runtime_api_v1().contents
    with tempfile.TemporaryDirectory(prefix='eps-native-progress-') as temporary:
        root=Path(temporary);normal=root/'normal';normal.mkdir()
        with services(build,normal,'native-progress') as (endpoints,coordinator):
            clients=[Client(api,'native-progress',endpoints,coordinator,w) for w in range(2)]
            try:
                for client in clients:
                    client.command('training_arm');client.command('training_cancel',dict(reason='input EOF before Pull'))
                    state=client.command('training_status')
                    assert state['completed']==[0,0] and state['granted_tasks']==0 and not state['failed']
                    client.pull(error=True)
                    batch=client.pull(read_only=True)
                    assert batch.handle==2**64-1;batch.release(batch.owner)
                    assert client.command('training_status')['granted_tasks']==0
                def worker(w):
                    client=clients[w];floors=[]
                    for iteration in range(3):
                        if w:time.sleep(.03)
                        client.command('training_arm')
                        client.command('training_arm',error=True)
                        batch=client.pull(prepared=iteration==1)
                        try:
                            client.command('training_finish',error=True)
                            state=client.command('training_status')
                            assert state['completed'][w]==iteration
                            floors.append(state['observed_completion_floor'])
                            client.push(batch);client.command('training_finish')
                        finally:batch.release(batch.owner)
                    return floors
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results=list(pool.map(worker,range(2)))
                state=clients[0].command('training_status')
                assert state['completed']==[3,3] and state['granted_tasks']==state['completed_tasks']==6
                assert not state['armed'] and not state['active'] and not state['coordinator_active'] and not state['waiting']
                assert results==[[0,1,2],[0,1,2]],results
                shard_stats=clients[0].command('stats')
                assert all(s['updates']==6 and s['live_tickets']==0 for s in shard_stats)
                clients[0].command('training_checkpoint',error=True)
                clients[0].command('training_pause')
                saved_progress=clients[0].command('training_checkpoint')
                assert saved_progress['source_run']=='native-progress' and saved_progress['completed']==[3,3]
                assert clients[0].command('training_status')['paused']
                clients[0].command('training_resume')
                assert not clients[0].command('training_status')['paused']
            finally:
                for client in clients:client.close()
        failed=root/'failed';failed.mkdir()
        with services(build,failed,'native-progress-failure',workers=1) as (endpoints,coordinator):
            client=Client(api,'native-progress-failure',endpoints,coordinator,0)
            batch=None
            try:
                client.command('training_arm');batch=client.pull()
                client.command('training_cancel',dict(reason='model failed after parameter read'))
                aborted=client.command('training_status')
                assert aborted['failed'] and aborted['completed']==[0]
                client.command('training_arm',error=True)
            finally:
                client.close()
                if batch is not None:batch.release(batch.owner)
    report=dict(test='native_runtime_training_progress',workers=2,shards=2,iterations=3,
        final_progress=state,shard_stats=shard_stats,prepared_pull_admission=True,
        eof_before_pull_consumes_no_task=True,prediction_consumes_no_task=True,
        finish_before_update_rejected=True,model_failure_aborts_group=aborted['failed'],
        progress_worker_ids_independent_of_parameter_learner_ids=True)
    report['training_checkpoint']=saved_progress
    if args.out:args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
