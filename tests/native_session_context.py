"""Native context identity across controller/model Sessions and learner owners."""
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
from rpc_processes import ready,stop


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return s.getsockname()[1]


def probe(endpoints,runtime,operations,mode,agent,directory):
    import numpy as np
    import tensorflow as tf
    from eps_tensorflow import Feature
    from eps_tensorflow_native import NativeParameterServer
    from native_session_backend import NativeSessionBackend
    from native_worker_owner import NativeWorkerOwner
    tf.compat.v1.disable_eager_execution()
    gpu=mode=='cuda';device='/GPU:0' if gpu else '/CPU:0'
    if gpu:assert tf.config.list_physical_devices('GPU')
    definition=dict(id=1,dimension=3,dtype='float32',optimizer=dict(kind='sparse_lazy_adam',
        learning_rate=.01,beta1=.9,beta2=.999,epsilon=1e-8,initial_beta1_power=.9,initial_beta2_power=.999),
        initialization=dict(min=.25,max=.25,seed=42))
    cfg=dict(run='native-sessions',endpoints=endpoints,learner=0,tables=[],release_read_only=True)
    if gpu:cfg['cuda']=dict(agent=0,device=0,listen=agent,state_bytes=1 << 20,rows_per_table=64,owner_key='session-worker')
    options=dict(device=device,op_library=operations,runtime_library=runtime)
    warm=None
    if gpu:
        warm=NativeWorkerOwner(cfg,runtime_library=runtime)
        for physical in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(physical,True)
    owner=NativeSessionBackend(cfg,**options);second=None
    assert all(d.device_type!='GPU' for d in owner._session.list_devices())
    try:
        owner.create_table(definition)
        feature=Feature(1,3,np.array([0,1,1,2],np.uint64),np.array([0,3,4],np.uint64),3,'mean')
        def graph(backend):
            g=tf.Graph()
            with g.as_default():
                h,values=backend.pull_graph([feature]);gradients=tf.gradients(tf.reduce_sum(values[0]),values)
                step=backend.push(h,[3],gradients)
                kinds={op.type for op in g.get_operations()}
                assert {'EpsRuntime','EpsPull','EpsPush'}<=kinds and not any('PyFunc' in kind for kind in kinds)
            return g,h,values,step
        previous=None
        session_options=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=4)
        session_options.gpu_options.allow_growth=True
        for iteration in range(3):
            g,h,values,step=graph(owner)
            with tf.compat.v1.Session(graph=g,config=session_options) as session:
                before,_=session.run([values[0],step])
                if previous is not None:assert (before<previous).all(),(before,previous)
                previous=before
            assert all(s['updates']==iteration+1 and s['live_tickets']==0 for s in owner.stats())
        owner.set_prediction(1,True)
        with tf.compat.v1.Session(graph=g,config=session_options) as session:
            for _ in range(4):
                token,output=session.run([h,values[0]]);assert token==2**64-1 and np.isfinite(output).all()
            owner.set_prediction(1,False);session.run(step)
        assert all(s['updates']==4 and s['live_tickets']==0 for s in owner.stats())
        other=dict(cfg,learner=1);second=NativeSessionBackend(other,**options)
        if warm:warm.close()
        second.create_table(definition);second.set_prediction(1,True)
        g2,h2,v2,_=graph(second)
        with tf.compat.v1.Session(graph=g2,config=session_options) as session:
            assert session.run(h2)==2**64-1
        mismatch=tf.Graph()
        with mismatch.as_default():
            wrong=NativeParameterServer(dict(cfg,learner=17),context_key=owner.context_key,
                op_library=operations,runtime_library=runtime).stats()
        with tf.compat.v1.Session(graph=mismatch,config=session_options) as session:
            try:session.run(wrong)
            except tf.errors.InvalidArgumentError as error:assert 'configuration mismatch' in str(error)
            else:raise AssertionError('conflicting native context accepted')
        if gpu:
            wrong=dict(cfg,cuda=dict(cfg['cuda'],rows_per_table=65))
            try:NativeSessionBackend(wrong,**options)
            except tf.errors.InvalidArgumentError as error:assert 'owner configuration mismatch' in str(error)
            else:raise AssertionError('conflicting GPU owner accepted')
            assert sum(s['cached_rows'] for s in owner.stats())>0
        owner.save('four');manifest=owner.export_checkpoint('four',directory,chunk_bytes=17)
        assert len(manifest['files'])==2
        owner.resume('four');owner.close()
        assert all(s['updates']==4 and s['live_tickets']==0 for s in second.stats())
        if gpu:
            import urllib.request
            req=urllib.request.Request('http://'+agent+'/eps.v1.CacheAgentService/AdmissionStats',data=b'{}',headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=3) as response:stats=json.load(response)
            assert int(stats['accepted_requests'])==int(stats['completed_requests'])>0
        print(json.dumps(dict(test='native_session_context',mode=mode,model_graphs=3,
            updates_per_shard=4,shared_context_mismatch_rejected=True,prediction_auto_release=True,
            checkpoint_export_shards=2,second_learner_survives_first_close=True,
            shared_gpu_owner=gpu,controller_has_gpu=False,graph_python_callbacks=False)),flush=True)
    finally:
        owner.close()
        if second:second.close()
        if warm:warm.close()


def main():
    if sys.argv[1]=='--probe':
        probe(sys.argv[2:4],*sys.argv[4:]);return
    server,runtime,operations,mode=sys.argv[1:]
    agent='127.0.0.1:'+str(free_port());gpu=mode=='cuda'
    with tempfile.TemporaryDirectory(prefix='eps-native-sessions-') as temporary:
        root=pathlib.Path(temporary);processes=[];logs=[];endpoints=[]
        try:
            for shard in range(2):
                config=root/f'ps-{shard}.json'
                cache=dict(enabled=gpu)
                if gpu:cache.update(admission='frequency',decision_window_requests=1,minimum_accesses=1,reserve_slots=1)
                config.write_text(json.dumps(dict(schema_version=1,run_id='native-sessions',listen='127.0.0.1:0',
                    partition=dict(shard_id=shard,shard_count=2,epoch=1),tables=[],cache=cache,
                    agents=[dict(id=0,endpoint=agent)] if gpu else [],snapshot_directory=str(root/'snapshots'))))
                log=open(root/f'ps-{shard}.log','w+');logs.append(log)
                p=subprocess.Popen([server,'--config',str(config)],stdout=subprocess.PIPE,stderr=log,text=True)
                processes.append(p);endpoints.append(ready(p))
            subprocess.run([sys.executable,__file__,'--probe',*endpoints,runtime,operations,mode,agent,str(root/'export')],check=True,timeout=120)
        except BaseException:
            for log in logs:log.flush();log.seek(0);print(log.read()[-3000:],file=sys.stderr)
            raise
        finally:
            for p in processes:stop(p)
            for log in logs:log.close()


if __name__=='__main__':main()
