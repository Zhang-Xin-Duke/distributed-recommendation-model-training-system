"""Original Runner/SparseTable methods with real native TF training and BRPC progress.

The source contract supplies framework environment stand-ins; complete original
main/Executor and reader execution have separate acceptance drivers.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types

from runtime_training_progress import services
from rpc_processes import stop


def probe(args):
    import numpy as np
    import tensorflow as tf
    from entropy_source_contract import load,Learner
    from entropy_install import install
    tf.compat.v1.disable_eager_execution()
    for physical in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(physical,True)
    gpu=args.model_device=='cuda'
    if gpu:assert tf.config.list_physical_devices('GPU')
    train,context,provenance=load(tf);context.learner=Learner(args.worker)
    config=dict(schema_version=1,run='original-runner-progress',endpoints=args.endpoints,
        backend='native',native=dict(device='/CPU:0',runtime_library=str(args.runtime),op_library=str(args.operations)),
        training=dict(coordinator=args.coordinator),learner_id_stride=8,tables=dict(entities=dict(id=1,seed=42)))
    options=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=4,
                                    allow_soft_placement=False)
    options.gpu_options.allow_growth=True
    with install(train,config) as hooks:
        graph=tf.Graph()
        with graph.as_default():
            dataset=tf.data.Dataset.from_tensor_slices(tf.constant([[0,1],[2,3],[4,5]],tf.int64))
            dataset_options=tf.data.Options();dataset_options.threading.private_threadpool_size=1
            iterator=tf.compat.v1.data.make_one_shot_iterator(dataset.with_options(dataset_options))
            ids=iterator.get_next()
            table=train.SparseTable([1,3],'entities',optimizer=train.SparseLazyAdam(learning_rate=.01,epsilon=.001),
                initializer=tf.keras.initializers.Constant(.25),pooling_method='mean',location_peers=[0],v2=True,non_unique=True)
            table.create_table()
            embedding=table.embedding_lookup_and_pooling([ids],[tf.constant([0,2],tf.int32)],None,shapes=[[1,3]])[0]
            with tf.device('/GPU:0' if gpu else '/CPU:0'):
                dense=tf.Variable([[.5],[.25],[.125]],dtype=tf.float32)
                prediction=tf.matmul(embedding,dense)
                loss=tf.reduce_mean(tf.square(prediction-1.))
                dense_update=tf.compat.v1.train.GradientDescentOptimizer(.01).minimize(loss,var_list=[dense])
            sparse_update=table.apply_gradients([(tf.gradients(loss,embedding)[0],embedding)],[])
            step=tf.group(sparse_update,dense_update)
            initialize=tf.compat.v1.global_variables_initializer()
        assert not any('PyFunc' in op.type for op in graph.get_operations())
        runner=train._MultiDatasetRunner();runner.learner=context.learner
        runner.is_worker=True;runner.prediction=False;runner.dataset=types.SimpleNamespace(end=False)
        def exhausted(error):raise error
        runner.handle_out_of_range=exhausted
        with tf.compat.v1.Session(graph=graph,config=options) as session:
            runner.session=session;session.run(initialize);initial=session.run(dense)
            losses=[]
            for iteration in range(3):
                if args.worker:time.sleep(.03)
                result=runner.run(dict(run_op=step,loss=loss))
                assert np.isfinite(result['loss']);losses.append(float(result['loss']))
            backend=hooks.backends()[args.worker*8]
            before=backend.training_status()
            try:runner.run(dict(run_op=step,loss=loss))
            except tf.errors.OutOfRangeError:pass
            else:raise AssertionError('original Runner did not propagate dataset EOF')
            after=backend.training_status()
            assert before['next_iteration']==after['next_iteration']==3
            assert after['completed'][args.worker]==3 and not after['active'] and not after['armed'] and not after['failed']
            assert not np.array_equal(initial,session.run(dense))
            memory=tf.config.experimental.get_memory_info('GPU:0') if gpu else None
            if gpu:assert memory['peak']>0 and prediction.device.endswith('GPU:0')
            report=dict(worker=args.worker,losses=losses,progress=after,model_device=prediction.device,
                gpu_memory=memory,original_runner_run=True,original_sparse_table=True,graph_python_callbacks=False,
                parameter_learner_id=args.worker*8,training_worker_id=args.worker,
                eof_preserves_completed_clock=True,source=provenance)
    print(json.dumps(report))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--operations',type=Path,required=True)
    parser.add_argument('--model-device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--out',type=Path)
    parser.add_argument('--worker',type=int);parser.add_argument('--endpoints',nargs=2);parser.add_argument('--coordinator')
    args=parser.parse_args()
    if args.worker is not None:probe(args);return
    from entropy_source_contract import source_root
    try:source_root()
    except FileNotFoundError as error:print('SKIP: '+str(error));raise SystemExit(77)
    build=args.build.resolve();runtime=args.runtime.resolve();operations=args.operations.resolve()
    with tempfile.TemporaryDirectory(prefix='eps-original-runner-progress-') as temporary:
        root=Path(temporary);children=[];streams=[]
        with services(build,root,'original-runner-progress') as (endpoints,coordinator):
            try:
                for worker in range(2):
                    output=(root/f'worker-{worker}.out').open('w+');error=(root/f'worker-{worker}.err').open('w+')
                    streams.extend([output,error])
                    children.append(subprocess.Popen([sys.executable,__file__,'--build',str(build),'--runtime',str(runtime),
                        '--operations',str(operations),'--worker',str(worker),'--endpoints',*endpoints,
                        '--coordinator',coordinator,'--model-device',args.model_device],stdout=output,stderr=error,env=os.environ.copy()))
                for child in children:assert child.wait(timeout=120)==0,'original Runner model Worker failed'
                workers=[]
                for stream in streams[::2]:stream.flush();stream.seek(0);workers.append(json.loads(stream.read()))
                settings=root/'control.json';settings.write_text(json.dumps(dict(run='original-runner-progress',
                    coordinator=coordinator,shards=endpoints,keys=list(range(6)),table=1)))
                state=json.loads(subprocess.check_output([str(build/'eps_training_control'),'--config',str(settings),
                                                         '--action','inspect'],text=True,timeout=20))
                assert state['completed']==[3,3] and state['completed_tasks']==state['granted_tasks']==6
                assert not state['active'] and not state['waiting'] and not state['failed']
                assert all(s['updates']==6 and s['live_tickets']==0 for s in state['shard_stats'])
                assert len(state['rows'])==6 and all(row['row_updates']==2 for row in state['rows'])
                report=dict(test='original_runner_native_training_progress',model_device=args.model_device,
                    workers=workers,state=state,scope='preserved original Runner.run/SparseTable methods with explicit framework stand-ins; actual TensorFlow native model and two independent parameter shards/coordinator')
                if args.out:args.out.write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(dict(test=report['test'],model_device=args.model_device,completed=state['completed'],rows=len(state['rows']))))
            except BaseException:
                for stream in streams:stream.flush();stream.seek(0);print(stream.read()[-5000:],file=sys.stderr)
                raise
            finally:
                for child in reversed(children):stop(child)
                for stream in streams:stream.close()


if __name__=='__main__':main()
