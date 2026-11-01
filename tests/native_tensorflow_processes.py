"""Native TensorFlow CPU/CUDA graphs, automatic gradients and external shards."""
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import dataclasses
from rpc_processes import ready,stop

def port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]

def main():
    server,runtime,operations,compare,mode=sys.argv[1:];gpu=mode=='cuda'
    import numpy as np
    import tensorflow as tf
    from eps_tensorflow import Feature
    from eps_tensorflow_native import NativeParameterServer
    tf.config.set_soft_device_placement(False)
    for device in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(device,True)
    if gpu:assert tf.config.list_physical_devices('GPU'),'native CUDA test requires a visible GPU'
    tf.config.threading.set_inter_op_parallelism_threads(4);tf.config.threading.set_intra_op_parallelism_threads(2)
    processes=[];logs=[];backends=[]
    with tempfile.TemporaryDirectory(prefix='eps-native-tf-') as temporary:
        root=pathlib.Path(temporary);agent_endpoint='127.0.0.1:'+str(port());endpoints=[]
        try:
            for group in ['actual','reference']:
                group_endpoints=[]
                for shard in range(2):
                    cache=dict(enabled=False);agents=[]
                    if gpu and group=='actual':
                        cache=dict(enabled=True,admission='reuse_cost',policy_agents=[0],decision_window_requests=1,max_tracked_rows=32,minimum_accesses=1,window_decay=.5,minimum_residence_windows=1,replacement_margin=.1,saved_microseconds_per_access=100,load_microseconds_per_byte=0,writeback_microseconds_per_byte=0,learn_migration_costs=False,network_bytes_per_window=1<<20,staging_bytes_limit=1<<20,max_inflight_batches=8,reserve_slots=1)
                        agents=[dict(id=0,endpoint=agent_endpoint)]
                    config=root/f'{group}-{shard}.json';config.write_text(json.dumps(dict(schema_version=1,run_id='native-'+group,listen='127.0.0.1:0',partition=dict(shard_id=shard,shard_count=2,epoch=1),tables=[],cache=cache,agents=agents,snapshot_directory=str(root/group))))
                    log=open(root/f'{group}-{shard}.log','w+');logs.append(log)
                    p=subprocess.Popen([server,'--config',str(config)],stdout=subprocess.PIPE,stderr=log,text=True);processes.append(p);group_endpoints.append(ready(p))
                endpoints.append(group_endpoints)
            tables=[dict(id=i,dimension=d,dtype='float32',optimizer=dict(kind='sparse_lazy_adam',learning_rate=.01,beta1=.9,beta2=.999,epsilon=1e-8,initial_beta1_power=.9,initial_beta2_power=.999),initialization=dict(min=-.25,max=.25,seed=42)) for i,d in [(1,3),(2,5)]]
            for i,name in enumerate(['actual','reference']):
                config=dict(run='native-'+name,endpoints=endpoints[i],tables=tables)
                if gpu and not i:config['cuda']=dict(agent=0,device=0,listen=agent_endpoint,state_bytes=1<<20,rows_per_table=64,transfer_bytes=1<<20,outstanding_output_bytes=1<<20)
                backends.append(NativeParameterServer(config,op_library=operations,runtime_library=runtime,device='/GPU:0' if gpu and not i else '/CPU:0'))
            features=[Feature(1,10,np.array([1,2,1,0,3,4,0],np.uint64),np.array([0,4,4,7],np.uint64),3,'mean',0),
                Feature(1,11,np.array([0,4,1,0,5,2,0],np.uint64),np.array([0,2,7],np.uint64),3,'sequence',0),
                Feature(2,12,np.array([2,1,2,3],np.uint64),np.array([0,2,4],np.uint64),5,'sum',0),
                Feature(2,13,np.array([0,0],np.uint64),np.array([0,0,2],np.uint64),5,'sequence',0),
                Feature(2,14,np.array([],np.uint64),np.array([0,0],np.uint64),5,'mean',0),
                Feature(2,15,np.array([],np.uint64),np.array([0,0],np.uint64),5,'sequence',0),
                Feature(2,16,np.array([0,0],np.uint64),np.array([0,2],np.uint64),5,'mean',0)]
            dimensions=[f.dimension for f in features];feature_ids=[f.feature for f in features]
            def model(backend,device):
                with tf.device(device):
                    dense=tf.Variable([[.2],[-.1],[.3]],dtype=tf.float32);optimizer=tf.keras.optimizers.SGD(.01,momentum=.9);optimizer.build([dense]);step=tf.Variable(0,dtype=tf.int64,trainable=False)
                def execute(handle,embeddings):
                    with tf.device(device),tf.GradientTape() as tape:
                        tape.watch(embeddings);prediction=embeddings[0]@dense;loss=tf.reduce_mean(tf.square(prediction-1))
                        for e in embeddings[1:]:loss=loss+.1*tf.reduce_sum(tf.square(e+.1))
                    gradients=tape.gradient(loss,list(embeddings)+[dense]);ack=backend.push(handle,feature_ids,gradients[:-1]);optimizer.apply_gradients([(gradients[-1],dense)]);advanced=step.assign_add(1)
                    done=backend.join([ack,advanced])
                    with tf.control_dependencies([done]):return tf.identity(loss),tuple(tf.identity(e) for e in embeddings),tf.identity(dense)
                @tf.function(autograph=False)
                def train():return execute(*backend.pull(features))
                @tf.function(input_signature=[tf.TensorSpec([],tf.uint64)],autograph=False)
                def prepared(token):return execute(*backend.pull_prepared(token,dimensions))
                return train,prepared,dense,optimizer,step
            models=[model(backends[i],'/GPU:0' if gpu and i==0 else '/CPU:0') for i in range(2)]
            retained=None;maximum=0.;cached_seen=False
            for iteration in range(5):
                outputs=[]
                for backend,m in zip(backends,models):
                    if iteration%2:outputs.append(m[1](backend.prepare(features)))
                    else:outputs.append(m[0]())
                a,b=outputs
                for x,y in [(a[0],b[0]),(a[2],b[2]),*zip(a[1],b[1])]:
                    delta=float(np.max(np.abs(x.numpy()-y.numpy()),initial=0));maximum=max(maximum,delta);assert delta<1e-5,(iteration,delta)
                if gpu:assert all('/device:GPU:0' in x.device for x in a[1])
                if iteration==0:retained=(a[1],tuple(x.numpy().copy() for x in a[1]))
                states=[json.loads(x.stats().numpy()) for x in backends];cached_seen|=any(x['cached_rows'] for x in states[0]);assert all(x['live_tickets']==0 for state in states for x in state)
                if iteration==1:
                    for i,(backend,m) in enumerate(zip(backends,models)):
                        backend.save('middle').numpy();checkpoint=tf.train.Checkpoint(dense=m[2],optimizer=m[3],input_step=m[4]);path=checkpoint.write(str(root/f'dense-{i}'))
                        m[2].assign(tf.zeros_like(m[2]));m[4].assign(99);checkpoint.restore(path).expect_partial();assert int(m[4].numpy())==2;backend.resume('middle').numpy()
            if gpu:assert cached_seen,'native GPU graph never loaded resident parameters'
            graph_types=set()
            for m in models:
                for function in [m[0].get_concrete_function(),m[1].get_concrete_function()]:
                    types={op.type for op in function.graph.get_operations()};graph_types|=types;assert not any('PyFunc' in kind for kind in types)
            assert {'EpsPull','EpsPullPrepared','EpsPush','EpsCommand'}<=graph_types
            bad=list(features);bad[0]=dataclasses.replace(bad[0],dimension=4)
            try:backends[0].pull(bad,read_only=True)
            except tf.errors.InvalidArgumentError as error:assert 'output shape mismatch' in str(error)
            else:raise AssertionError('native output schema mismatch was accepted')
            assert all(x['live_tickets']==0 for x in json.loads(backends[0].stats().numpy()))
            for backend in backends:backend.save('final').numpy()
            state_records=[]
            for shard in range(2):
                result=subprocess.run([compare,str(root/'actual'/'final'/f'shard-{shard}.bin'),str(root/'reference'/'final'/f'shard-{shard}.bin'),str(shard),'2','1','1','2'],capture_output=True,text=True,check=True)
                state_records.append(json.loads(result.stdout))
            for backend in backends:backend.close().numpy()
            for tensor,values in zip(*retained):np.testing.assert_array_equal(tensor.numpy(),values)
            print(json.dumps(dict(test='native_tensorflow_graph',mode=mode,tensorflow=tf.__version__,external_ps_processes=4,training_steps=5,features=7,prepared_steps=2,
                checked_parameter_rows=sum(x['checked_rows'] for x in state_records),max_state_absolute_error=max(x['max_state_absolute_error'] for x in state_records),
                max_model_absolute_error=maximum,resident_parameters_observed=bool(cached_seen),native_op_types=sorted(graph_types & {'EpsPull','EpsPullPrepared','EpsPush','EpsCommand'}),
                python_callbacks=False,output_valid_after_runtime_close=True,mid_run_dense_and_sparse_save_resume=True,output_schema_failure_tickets_released=True)))
        except Exception:
            for log in logs:log.flush();log.seek(0);print(log.read()[-5000:],file=sys.stderr)
            raise
        finally:
            for backend in backends:
                try:backend.abort().numpy()
                except Exception:pass
            for p in reversed(processes):stop(p)
            for log in logs:log.close()

if __name__=='__main__':main()
