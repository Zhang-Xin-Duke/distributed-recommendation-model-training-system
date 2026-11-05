"""Two original-method Workers: joint progress/model save and fresh-run restore."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
from multiprocessing.managers import BaseManager

from entropy_peer_checkpoint import Rendezvous
from rpc_processes import ready,stop


class Gate(Rendezvous):
    def __init__(self,start):
        super().__init__();self.released=True;self.turn=start*2

    def wait_turn(self,turn):
        with self.condition:
            deadline=time.monotonic()+60
            while self.turn!=turn and not self.error:
                if self.turn>turn or time.monotonic()>deadline:
                    self.abort('model turn order or timeout');break
                self.condition.wait(.05)
            if self.error:raise RuntimeError(self.error)

    def advance_turn(self,turn):
        with self.condition:
            if self.turn!=turn:raise ValueError('model turn changed')
            self.turn+=1;self.condition.notify_all()


def peer(a):
    import numpy as np
    import tensorflow as tf
    from entropy_source_contract import load,Learner,peer_environment
    from entropy_install import install
    tf.compat.v1.disable_eager_execution()
    for physical in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(physical,True)
    if a.device=='cuda':assert tf.config.list_physical_devices('GPU')
    BaseManager.register('gate')
    manager=BaseManager(address=('127.0.0.1',a.gate_port),authkey=bytes.fromhex(os.environ['EPS_CHECKPOINT_TEST_KEY']))
    manager.connect();gate=manager.gate()
    train,context,provenance=load(tf);learner=Learner(a.worker)
    native=train.env.call_local_function
    def current():return getattr(train.MultiThreadingLearner.tls,'learner',learner)
    def native_call(name,*args):
        if name=='cluster_barrier':return gate.wait(a.worker,args[1])
        return native(name,*args)
    barrier,provenance['barriers']=peer_environment(train,current,native_call)
    learner.master=types.SimpleNamespace(_barriers={name:barrier(1) for name in ('all','worker')})
    learner.is_chief=True;train.MultiThreadingLearner.tls.learner=learner;context.learner=learner
    train.env._env.peer_id=a.worker;train.env._env.shard_count=2;train.env.peer_id=lambda:a.worker
    start=0
    if a.restore:
        start=int((a.restore/'icb_feeder_ckpt').read_text())
        assert start==3
    config=dict(schema_version=1,run=a.run,endpoints=a.endpoints,backend='native',learner_id_stride=8,
        native=dict(device='/CPU:0',runtime_library=str(a.runtime),op_library=str(a.operations)),
        training=dict(coordinator=a.coordinator,start_iteration=start+int(a.mode=='bad-start')),
        tables=dict(entities=dict(id=1,seed=42),weights=dict(id=2,seed=43)))
    options=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=4,allow_soft_placement=False)
    options.gpu_options.allow_growth=True
    result={}
    with install(train,config) as hooks:
        graph=tf.Graph()
        with graph.as_default():
            offset=tf.compat.v1.placeholder(tf.int64,shape=[])
            dataset=tf.data.Dataset.range(6).skip(offset).take(a.iterations)
            dataset_options=tf.data.Options();dataset_options.threading.private_threadpool_size=1
            iterator=tf.compat.v1.data.make_initializable_iterator(dataset.with_options(dataset_options))
            index=iterator.get_next()
            ids=(index%4)*2+tf.range(2,dtype=tf.int64)
            kwargs=dict(optimizer=train.SparseLazyAdam(learning_rate=.01,epsilon=.001),
                initializer=tf.keras.initializers.Constant(.25),pooling_method='mean',location_peers=[0],v2=True,non_unique=True)
            embeddings=train.SparseTable([1,3],'entities',**kwargs);embeddings.create_table()
            weights=train.SparseTable([1,3],'weights',**kwargs);weights.create_table()
            embed=embeddings.embedding_lookup_and_pooling([ids],[tf.constant([0,2],tf.int32)],None,shapes=[[1,3]])[0]
            dense=weights.embedding_lookup_and_pooling([tf.constant([100],tf.int64)],
                [tf.constant([0,1],tf.int32)],None,shapes=[[1,3]])[0]
            if a.worker==0:
                sentinel=tf.Variable([-.5 if a.restore else .5],trainable=False,dtype=tf.float32)
                dense_table=train.DenseTable(sentinel);learner.var_collections[sentinel]=dense_table
                if a.mode=='bad-save':
                    def fail(*args):raise RuntimeError('injected original dense checkpoint failure')
                    dense_table.checkpoint=fail
            else:sentinel=tf.constant([.5],tf.float32)
            with tf.device('/GPU:0' if a.device=='cuda' else '/CPU:0'):
                prediction=tf.reduce_sum(embed*dense,axis=1)+sentinel
                loss=tf.reduce_mean(tf.square(prediction-1.))
            gradients=tf.gradients(loss,[embed,dense])
            step=tf.group(embeddings.apply_gradients([(gradients[0],embed)],[]),
                          weights.apply_gradients([(gradients[1],dense)],[]))
            initialize=tf.compat.v1.global_variables_initializer()
        runner=train._MultiDatasetRunner();runner.learner=learner;runner.is_worker=True
        runner.prediction=False;runner.dataset=types.SimpleNamespace(end=False)
        def exhausted(error):raise error
        runner.handle_out_of_range=exhausted
        backend=hooks.backends()[a.worker*8]
        with tf.compat.v1.Session(graph=graph,config=options) as session:
            runner.session=session;session.run(initialize)
            if a.restore:
                if a.mode=='bad-start':
                    before=session.run(sentinel)
                    try:runner.restore(None,str(a.restore.parent),a.restore.name)
                    except ValueError as error:assert 'progress differs' in str(error)
                    else:raise AssertionError('wrong local training clock was accepted')
                    np.testing.assert_array_equal(before,session.run(sentinel))
                    assert 'dense_restored' not in context.events
                    result=dict(worker=a.worker,rejected_before_dense_restore=True,progress=backend.training_status())
                else:
                    runner.restore(None,str(a.restore.parent),a.restore.name)
                    assert session.run(sentinel).tolist()==[.5]
                    assert not backend.training_status()['paused']
            if not result:
                session.run(iterator.initializer,{offset:start});losses=[]
                for iteration in range(start,start+a.iterations):
                    gate.wait_turn(iteration*2+a.worker)
                    values=runner.run(dict(run_op=step,loss=loss))
                    losses.append(float(values['loss']));assert np.isfinite(losses[-1])
                    context.input_step=iteration+1
                    gate.advance_turn(iteration*2+a.worker)
                try:runner.run(dict(run_op=step,loss=loss))
                except tf.errors.OutOfRangeError:pass
                else:raise AssertionError('input EOF was not observed')
                train.env.barrier('::TEST_MODEL_STEPS_DONE::',cluster='all')
                if a.mode in ('save','bad-save'):
                    try:runner.checkpoint(str(a.directory/'checkpoints'),'mid')
                    except RuntimeError as error:
                        if a.mode!='bad-save':raise
                        gate.abort(str(error));result=dict(worker=a.worker,save_failed=True,progress=backend.training_status())
                    else:
                        if a.mode=='bad-save':raise AssertionError('injected dense save unexpectedly succeeded')
                if not result:
                    train.env.barrier('::TEST_CHECKPOINT_DONE::',cluster='all')
                    if a.worker==0:
                        backend.save('final')
                        backend.export_checkpoint('final',str(a.directory/'final'),tf.io.gfile)
                    train.env.barrier('::TEST_FINAL_EXPORTED::',cluster='all')
                    memory=tf.config.experimental.get_memory_info('GPU:0') if a.device=='cuda' else None
                    if memory:assert memory['peak']>0 and prediction.device.endswith('GPU:0')
                    result=dict(worker=a.worker,losses=losses,start_iteration=start,
                        final_input_cursor=context.input_step,progress=backend.training_status(),
                        model_device=prediction.device,gpu_memory=memory,source=provenance)
    (a.directory/f'worker-{a.worker}.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--operations',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--out',type=Path)
    parser.add_argument('--worker',type=int);parser.add_argument('--endpoints',nargs=2);parser.add_argument('--coordinator')
    parser.add_argument('--gate-port',type=int);parser.add_argument('--run');parser.add_argument('--directory',type=Path)
    parser.add_argument('--mode',default='train');parser.add_argument('--iterations',type=int);parser.add_argument('--restore',type=Path)
    a=parser.parse_args()
    if a.worker is not None:
        try:peer(a)
        except BaseException:
            raise
        return
    from entropy_source_contract import source_root
    from checkpoint_restore import stage_checkpoint,training_restore_plan,load_committed
    try:source_root()
    except FileNotFoundError as error:print('SKIP: '+str(error));raise SystemExit(77)
    build=a.build.resolve();runtime=a.runtime.resolve();operations=a.operations.resolve()
    with tempfile.TemporaryDirectory(prefix='eps-native-training-checkpoint-') as temporary:
        root=Path(temporary)
        def phase(name,iterations,mode='train',restore=None):
            directory=root/name;directory.mkdir();run='original-progress-'+name
            start=3 if restore else 0;gate=Gate(start);key=secrets.token_bytes(16)
            BaseManager.register('gate',callable=lambda:gate)
            manager=BaseManager(address=('127.0.0.1',0),authkey=key);network=manager.get_server()
            threading.Thread(target=network.serve_forever,daemon=True).start()
            children=[];logs=[];endpoints=[]
            def launch(label,binary,config,extra=()):
                path=directory/(label+'.json');path.write_text(json.dumps(config))
                log=(directory/(label+'.log')).open('w+');logs.append(log)
                child=subprocess.Popen([str(build/binary),'--config',str(path),*extra],stdout=subprocess.PIPE,stderr=log,text=True)
                children.append(child);return ready(child)
            try:
                for shard in range(2):
                    extra=[]
                    if restore:
                        staged=stage_checkpoint(restore,directory/f'staged-{shard}',shard,checkpoint_id='mid')
                        extra=['--restore',staged['file']]
                    endpoints.append(launch(f'parameter-{shard}','eps_server',dict(schema_version=1,run_id=run,
                        listen='127.0.0.1:0',partition=dict(shard_id=shard,shard_count=2,epoch=1),
                        tables=[],cache=dict(enabled=False),snapshot_directory=str(directory/'snapshots')),extra))
                coordinator_config=dict(schema_version=1,run=run,listen='127.0.0.1:0',workers=2,
                                        epoch=1,consistency='sequential',max_delay=0)
                if restore:
                    plan=training_restore_plan(restore,run=run,coordinator_listen='127.0.0.1:0')
                    coordinator_config=plan['coordinator']
                coordinator=launch('coordinator','eps_training_coordinator',coordinator_config)
                peers=[]
                for worker in range(2):
                    log=(directory/f'peer-{worker}.log').open('w+');logs.append(log)
                    command=[sys.executable,__file__,'--build',str(build),'--runtime',str(runtime),'--operations',str(operations),
                        '--device',a.device,'--worker',str(worker),'--endpoints',*endpoints,'--coordinator',coordinator,
                        '--gate-port',str(network.address[1]),'--run',run,'--directory',str(directory),
                        '--mode',mode,'--iterations',str(iterations)]
                    if restore:command+=['--restore',str(restore)]
                    child=subprocess.Popen(command,stdout=log,stderr=log,
                        env=dict(os.environ,EPS_CHECKPOINT_TEST_KEY=key.hex()),start_new_session=True)
                    peers.append(child);children.append(child)
                deadline=time.monotonic()+120
                while not all(p.poll() is not None for p in peers):
                    if any(p.poll() not in (None,0) for p in peers):
                        gate.abort('model Worker failed');raise RuntimeError('model Worker failed')
                    if time.monotonic()>deadline:gate.abort('model timeout');raise TimeoutError('model phase timed out')
                    time.sleep(.05)
                assert all(p.returncode==0 for p in peers)
                records=[json.loads((directory/f'worker-{i}.json').read_text()) for i in range(2)]
                config=directory/'control.json';config.write_text(json.dumps(dict(run=run,coordinator=coordinator)))
                state=json.loads(subprocess.check_output([str(build/'eps_training_control'),'--config',str(config),
                                                         '--action','stats'],text=True,timeout=20))
                if mode=='bad-start':
                    assert state['paused'] and state['completed']==[3,3] and not state['granted_tasks']
                    assert all(r['rejected_before_dense_restore'] for r in records)
                elif mode=='bad-save':
                    assert all(r['save_failed'] for r in records) and state['paused'] and not state['failed']
                    assert not (directory/'checkpoints/mid/external_parameters/committed.json').exists()
                else:
                    assert state['completed']==[start+iterations]*2 and not state['active'] and not state['waiting'] and not state['failed']
                    if mode=='save':
                        saved=load_committed(directory/'checkpoints/mid')
                        assert saved['training_progress']['completed']==[3,3]
                return dict(directory=str(directory),workers=records,progress=state,barriers=gate.snapshot()['history'])
            except BaseException:
                for log in logs:log.flush();log.seek(0);print(log.read()[-5000:],file=sys.stderr)
                raise
            finally:
                for child in reversed(children):stop(child)
                for log in logs:log.close()
                if hasattr(network,'stop_event'):network.stop_event.set()
        baseline=phase('baseline',6)
        prefix=phase('before',3,'save');checkpoint=Path(prefix['directory'])/'checkpoints/mid'
        rejected=phase('wrong-clock',3,'bad-start',checkpoint)
        resumed=phase('restored',3,'train',checkpoint)
        failed=phase('failed-save',3,'bad-save')
        maximum=0.;rows=0
        for shard in range(2):
            record=json.loads(subprocess.check_output([str(build/'eps_snapshot_compare'),
                str(Path(baseline['directory'])/f'final/parameter-shard-{shard}.bin'),
                str(Path(resumed['directory'])/f'final/parameter-shard-{shard}.bin'),str(shard),'2','1','1','2'],text=True,timeout=20))
            maximum=max(maximum,record['max_state_absolute_error']);rows+=record['checked_rows']
        assert rows==9 and maximum<=1e-6
        loss_error=max(abs(value-baseline['workers'][w]['losses'][3+i])
            for w in range(2) for i,value in enumerate(resumed['workers'][w]['losses']))
        assert loss_error<=1e-6,loss_error
        report=dict(test='original_native_training_progress_checkpoint',device=a.device,workers=2,shards=2,
            checkpoint_completed=[3,3],final_completed=resumed['progress']['completed'],input_cycle_batches=4,
            checked_rows=rows,max_state_absolute_error=maximum,max_continuation_loss_error=loss_error,
            wrong_start_rejected_before_dense_restore=True,failed_save_keeps_progress_paused=True,
            baseline=baseline,prefix=prefix,resumed=resumed,rejected=rejected,failed_save=failed,
            scope='preserved original Runner/SparseTable/barrier methods with framework environment stand-ins; real native TensorFlow model, parameter shards and task-progress coordinator')
        if a.out:a.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('baseline','prefix','resumed','rejected','failed_save')}))


if __name__=='__main__':main()
