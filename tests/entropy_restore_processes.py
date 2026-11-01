"""Stage and restart real shards, then execute the original Runner restore flow."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from rpc_processes import ready,stop
from entropy_source_contract import source_root


def probe(endpoints,root,run,mode,agent=None):
    import numpy as np
    import tensorflow as tf
    from entropy_source_contract import load
    from entropy_install import install
    from checkpoint_restore import load_committed,checkpoint_fingerprints
    tf.compat.v1.disable_eager_execution()
    train,context,provenance=load(tf)
    provenance['executed_paths']=['SparseTable construction/registration/decorators',
        'InputBuilder.sparse_vars','InputBuilder.apply_sparse_grads',
        '_MultiDatasetRunner.restore','worker_restore','worker_restore_sparse',
        'worker_restore_sparse_wait','worker_restore_dense']
    config=dict(schema_version=1,run=run,endpoints=endpoints,learner_id_stride=8,
                tables=dict(users=dict(id=3,seed=42),hashed=dict(id=7)))
    if os.environ.get('EPS_ENTROPY_BACKEND')=='native':config.update(backend='native',native=dict(device='/CPU:0'))
    if agent:config['cache_agents']=[dict(id=0,endpoint=agent)]
    with install(train,config) as hooks:
        graph=tf.Graph()
        with graph.as_default():
            optimizer=train.SparseLazyAdam(learning_rate=.01,epsilon=.001)
            kwargs=dict(optimizer=optimizer,initializer=tf.keras.initializers.Constant(.25),
                        pooling_method='mean',location_peers=[0],v2=True,non_unique=True)
            users=train.SparseTable([2,2],'users',**kwargs);users.create_table()
            hashed=train.SparseTable([2,2],'hashed',**dict(kwargs,pooling_method='sum',hash_trick_bucket_size=8));hashed.create_table()
            out=users.embedding_lookup_and_pooling([tf.constant([0,1,-1,1,2],tf.int64),
                tf.constant([[1,-1,2],[0,2,1]],tf.int64)],
                [tf.constant([0,3,5],tf.int32),tf.constant([],tf.int32)],None,shapes=[[2,2],[2,3,2]])
            hout=hashed.embedding_lookup_and_pooling([tf.constant([9,1,-1,0],tf.int64)],
                [tf.constant([0,3,4],tf.int32)],None,shapes=[[2,2]])
            builder=train.InputBuilder();builder.input_spec=type('Spec',(),{'sparse_tables':dict(users=users,hashed=hashed)})()
            features=builder.sparse_vars()
            dense=tf.Variable([[.5],[.25]],dtype=tf.float32)
            context.learner.var_collections[dense]=train.DenseTable(dense)
            prediction=out[0]@dense+tf.reduce_sum(out[1],axis=[1,2])[:,None]*.1+hout[0]@dense
            loss=tf.reduce_mean(tf.square(prediction-1.))
            gradients=tf.gradients(loss,features)
            sparse=builder.apply_sparse_grads(list(zip(gradients,features)),[])
            dense_step=tf.compat.v1.train.GradientDescentOptimizer(.01).minimize(loss,var_list=[dense])
            step=tf.group(*sparse,dense_step)
            initialize=tf.compat.v1.global_variables_initializer()
        backend=next(iter(hooks.backends().values()))
        initial_metadata=backend.restored_from()
        runner=train._MultiDatasetRunner();runner.learner=context.learner;runner.is_worker=True
        checkpoint=root/'framework-checkpoints/mid-run'
        with tf.compat.v1.Session(graph=graph,config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as session:
            runner.session=session;session.run(initialize)
            if mode!='valid':
                if mode=='fresh':assert not any(x['loaded'] for x in initial_metadata)
                else:
                    wanted=checkpoint_fingerprints(load_committed(checkpoint,filesystem=tf.io.gfile),filesystem=tf.io.gfile)
                    assert all(x['loaded'] and x['pristine'] for x in initial_metadata)
                    assert any(a['checksum']!=b['checksum'] for a,b in zip(wanted,initial_metadata))
                before=session.run(dense)
                try:runner.restore(None,str(root/'framework-checkpoints'),'mid-run')
                except ValueError:pass
                else:raise AssertionError('unrestored or wrong snapshot was accepted')
                np.testing.assert_array_equal(before,session.run(dense))
                assert 'dense_restored' not in context.events
                print(json.dumps(dict(test='original_restore_rejection',mode=mode,before_dense_restore=True)),flush=True)
                return
            assert all(x['loaded'] and x['pristine'] for x in initial_metadata)
            runner.restore(None,str(root/'framework-checkpoints'),'mid-run')
            assert 'dense_restored' in context.events
            # The original reader owns its cursor. This fixture uses the state
            # written by the source checkpoint's native-feeder stand-in.
            context.input_step=int((checkpoint/'icb_feeder_ckpt').read_text())
            assert context.input_step==3
            for _ in range(context.input_step,6):
                session.run(step);context.input_step+=1
            expected=np.load(root/'expected-dense-six.npy');actual=session.run(dense)
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
            before=actual.copy()
            try:runner.restore(None,str(root/'framework-checkpoints'),'mid-run')
            except ValueError:pass
            else:raise AssertionError('an active service was rewound without reloading parameters')
            np.testing.assert_array_equal(before,session.run(dense))
            backend.save('restored-final')
            backend.export_checkpoint('restored-final',str(root/'restored-final'),tf.io.gfile)
            print(json.dumps(dict(test='original_runner_restore_continuation',initial_snapshot_verified=True,
                initial_input_cursor=3,final_input_cursor=context.input_step,continued_steps=3,
                input_cursor_kind='state written by the original-checkpoint feeder stand-in',
                dense_max_absolute_error=float(np.max(np.abs(actual-expected))),
                active_service_rewind_rejected=True,parameter_shards=2,
                table_backend=config.get('backend','portable'),
                cache_backend=os.environ.get('EPS_CACHE_BACKEND','host') if agent else 'none',source=provenance)),flush=True)


def main():
    try:source_root()
    except FileNotFoundError as error:print('SKIP: '+str(error));return 77
    if sys.argv[1]=='--probe':
        probe(sys.argv[2:4],pathlib.Path(sys.argv[4]),sys.argv[5],sys.argv[6],sys.argv[7] if len(sys.argv)>7 else None)
        return 0
    server,agent_binary,comparator=sys.argv[1:4]
    with tempfile.TemporaryDirectory(prefix='eps-original-restore-') as directory:
        root=pathlib.Path(directory)
        def phase(run,mode,restore=None):
            processes=[];logs=[];endpoints=[];agent=None
            try:
                if agent_binary!='-':
                    log=open(root/f'{mode}-agent.log','w+');logs.append(log)
                    process=subprocess.Popen([agent_binary,'--run',run,'--agent','0','--listen','127.0.0.1:0',
                        '--backend',os.environ.get('EPS_CACHE_BACKEND','host')],stdout=subprocess.PIPE,stderr=log,text=True)
                    processes.append(process);agent=ready(process,30)
                for shard in range(2):
                    config=root/f'{mode}-shard-{shard}.json';cache=dict(enabled=bool(agent))
                    if agent:cache.update(admission='frequency',decision_window_requests=1,minimum_accesses=1,reserve_slots=1)
                    config.write_text(json.dumps(dict(schema_version=1,run_id=run,listen='127.0.0.1:0',
                        partition=dict(shard_id=shard,shard_count=2,epoch=1),tables=[],cache=cache,
                        access=dict(policy='transfer_unique' if agent else 'cpu_only',agent_id=0),
                        agents=[dict(id=0,endpoint=agent)] if agent else [],snapshot_directory=str(root/f'{mode}-snapshots'))))
                    command=[server,'--config',str(config)]
                    if restore=='committed':
                        command=[sys.executable,'-m','entropy_stage','--checkpoint-root',str(root/'framework-checkpoints/mid-run'),
                            '--checkpoint-id','mid-run','--shard',str(shard),'--output-directory',str(root/f'staged-{shard}'),
                            '--filesystem','tensorflow','--exec-server',server,'--service-config',str(config)]
                    elif restore=='wrong':command+=['--restore',str(root/'expected-six'/f'parameter-shard-{shard}.bin')]
                    log=open(root/f'{mode}-shard-{shard}.log','w+');logs.append(log)
                    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,text=True)
                    processes.append(process);endpoints.append(ready(process,30))
                if mode=='source':
                    command=[sys.executable,str(pathlib.Path(__file__).with_name('entropy_hooks_processes.py')),'--probe',*endpoints,str(root),*([agent] if agent else [])]
                else:command=[sys.executable,__file__,'--probe',*endpoints,str(root),run,mode,*([agent] if agent else [])]
                subprocess.run(command,check=True,timeout=100)
            except Exception:
                for log in logs:log.flush();log.seek(0);print(log.read()[-3500:],file=sys.stderr)
                raise
            finally:
                for process in processes:stop(process)
                for log in logs:log.close()
        phase('entropy-hooks','source')
        phase('restore-not-loaded','fresh')
        phase('restore-wrong-state','wrong',restore='wrong')
        phase('restore-valid','valid',restore='committed')
        rows=0;maximum=0.
        for shard in range(2):
            result=subprocess.run([comparator,str(root/'expected-six'/f'parameter-shard-{shard}.bin'),
                str(root/'restored-final'/f'parameter-shard-{shard}.bin'),str(shard),'2','1','3','7'],
                capture_output=True,text=True,check=True,timeout=20)
            record=json.loads(result.stdout);rows+=record['checked_rows'];maximum=max(maximum,record['max_state_absolute_error'])
        print(json.dumps(dict(test='restored_full_optimizer_state',selected_tables=[3,7],checked_rows=rows,
                              max_state_absolute_error=maximum,compared=['weights','first_moment','second_moment','row_counters','optimizer_sequence','beta_powers'])),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
