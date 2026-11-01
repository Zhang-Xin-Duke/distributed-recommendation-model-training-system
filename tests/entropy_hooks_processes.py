"""Original SparseTable/InputBuilder methods + Session graph + real PS processes."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import socket

from rpc_processes import ready,stop
from entropy_source_contract import source_root


def probe(endpoints,root,agent=None):
    import numpy as np
    import tensorflow as tf
    from entropy_source_contract import load,Learner
    from entropy_install import install,UINT64_MAX
    from entropy_configuration import sparse_table_definition
    from eps_backend import Backend
    tf.compat.v1.disable_eager_execution()
    train,context,provenance=load(tf)
    provenance['executed_paths']=['SparseTable construction/registration/decorators',
        'InputBuilder.sparse_vars','InputBuilder.apply_sparse_grads',
        '_MultiDatasetRunner.checkpoint','_MultiDatasetRunner.prepare_path','checkpoint']
    original_class=train.SparseTable
    original_checkpoint=train.checkpoint
    original_runner_checkpoint=train._MultiDatasetRunner.checkpoint
    config=dict(schema_version=1,run='entropy-hooks',endpoints=endpoints,max_live_batches=8,learner_id_stride=8,
                tables=dict(users=dict(id=3,seed=42),hashed=dict(id=7),unused=dict(id=9)))
    native_backend=os.environ.get('EPS_ENTROPY_BACKEND')=='native'
    if native_backend:
        config.update(backend='native',native=dict(device=os.environ.get('EPS_ENTROPY_DEVICE','/CPU:0')))
        if config['native']['device'].startswith('/GPU:'):
            config['native']['cuda']=dict(agent=0,device=0,listen=agent,state_bytes=1 << 20,rows_per_table=64)
        elif agent:config['cache_agents']=[dict(id=0,endpoint=agent)]
    elif agent:config['cache_agents']=[dict(id=0,endpoint=agent)]
    reference=Backend(endpoints,'entropy-hooks',1023)
    if agent:reference.cache_endpoints([(0,agent)])
    hooks=install(train,config)
    try:
        assert train.SparseTable is original_class
        optimizer=train.SparseLazyAdam(learning_rate=.01,epsilon=.001)
        definitions={name:sparse_table_definition(identifier,2,optimizer,initialization=(.25,.25),seed=42 if name=='users' else 0)
                     for name,identifier in [('users',103),('hashed',107)]}
        graph=tf.Graph()
        with graph.as_default():
            kwargs=dict(optimizer=optimizer,initializer=tf.keras.initializers.Constant(.25),
                        pooling_method='mean',location_peers=[0],v2=True,non_unique=True)
            users=train.SparseTable([2,2],'users',**kwargs);users.create_table()
            hashed=train.SparseTable([2,2],'hashed',**dict(kwargs,pooling_method='sum',hash_trick_bucket_size=8));hashed.create_table()
            # An unselected legacy table retains the original class and vars method.
            native=train.SparseTable([8,2],'native',optimizer=optimizer)
            assert native.vars()==[native._var] and len(hooks.backends())==1
            ids=tf.constant([0,1,-1,1,2],tf.int64);splits=tf.constant([0,3,5],tf.int32)
            seq=tf.constant([[1,-1,2],[0,2,1]],tf.int64)
            out=users.embedding_lookup_and_pooling([ids,seq],[splits,tf.constant([],tf.int32)],None,
                                                 shapes=[[2,2],[2,3,2]])
            hout=hashed.embedding_lookup_and_pooling([tf.constant([9,1,-1,0],tf.int64)],
                        [tf.constant([0,3,4],tf.int32)],None,shapes=[[2,2]])
            assert out[0].shape.as_list()==[2,2] and out[1].shape.as_list()==[2,3,2]
            builder=train.InputBuilder();builder.input_spec=type('Spec',(),{'sparse_tables':dict(users=users,hashed=hashed)})()
            features=builder.sparse_vars();assert all(a is b for a,b in zip(features,[*out,*hout]))
            dense=tf.Variable([[.5],[.25]],dtype=tf.float32)
            dense_table=train.DenseTable(dense)
            context.learner.var_collections[dense]=dense_table
            prediction=out[0]@dense+tf.reduce_sum(out[1],axis=[1,2])[:,None]*.1+hout[0]@dense
            loss=tf.reduce_mean(tf.square(prediction-1.))
            gradients=tf.gradients(loss,features)
            sparse_ops=builder.apply_sparse_grads(list(zip(gradients,features)),[])
            dense_step=tf.compat.v1.train.GradientDescentOptimizer(.01).minimize(loss,var_list=[dense])
            step=tf.group(*sparse_ops,dense_step)
            unused=train.SparseTable([2,2],'unused',**kwargs);unused.create_table()
            uout=unused.embedding_lookup_and_pooling([ids,seq],[splits,tf.constant([],tf.int32)],None,
                                                   shapes=[[2,2],[2,3,2]])
            missing=tf.gradients(tf.reduce_sum(uout[0]),uout)
            assert missing[1] is None
            try:unused.apply_gradients(list(zip(missing,uout)),[])
            except ValueError:pass
            else:raise AssertionError('missing gradient ignored without the original policy')
            train.flags.FLAGS.ignore_none_grads=True
            zero_filled=unused.apply_gradients(list(zip(missing,uout)),[])
            train.flags.FLAGS.ignore_none_grads=False
            all_unused=unused.apply_gradients([(None,v) for v in uout],[])
            assert set([users,hashed,native]).issubset(set(context.learner.var_collections.values()))
            initialize=tf.compat.v1.global_variables_initializer()
        graph_types={op.type for op in graph.get_operations()}
        if native_backend:
            assert {'EpsRuntime','EpsPull','EpsPush','EpsIdCommand'}<=graph_types
            assert not any('PyFunc' in kind for kind in graph_types)
        for definition in definitions.values():reference.create_table(json.dumps(definition))
        user_features=[dict(table=103,feature=0,pooling='mean',padding_key=UINT64_MAX,
                            keys=np.array([0,1,UINT64_MAX,1,2],np.uint64),offsets=np.array([0,3,5],np.int64)),
                       dict(table=103,feature=1,pooling='sequence',padding_key=UINT64_MAX,
                            keys=np.array([1,UINT64_MAX,2,0,2,1],np.uint64),offsets=np.array([0,3,6],np.int64))]
        hash_features=[dict(table=107,feature=0,pooling='sum',padding_key=UINT64_MAX,
                            keys=np.array([1,1,UINT64_MAX,0],np.uint64),offsets=np.array([0,3,4],np.int64))]
        owner=context.learner
        runner=train._MultiDatasetRunner();runner.learner=owner;runner.is_worker=True
        with tf.compat.v1.Session(graph=graph,config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=2)) as session:
            runner.session=session
            session.run(initialize)
            losses=[];maximum_error=0.
            for iteration in range(5):
                h,expected=reference.pull(user_features);hh,hexpected=reference.pull(hash_features)
                # Executor callbacks must use captured state, independent of original TLS.
                context.learner=None
                values,grads,l,_=session.run([features,gradients,loss,step])
                context.learner=owner
                np.testing.assert_allclose(values[0],expected[0],rtol=0,atol=1e-6)
                np.testing.assert_allclose(values[1],expected[1].reshape(2,3,2),rtol=0,atol=1e-6)
                np.testing.assert_allclose(values[2],hexpected[0],rtol=0,atol=1e-6)
                maximum_error=max(maximum_error,*[float(np.max(np.abs(a-b))) for a,b in
                    zip(values,[expected[0],expected[1].reshape(2,3,2),hexpected[0]])])
                reference.push(h,[0,1],[grads[0],grads[1].reshape(-1,2)])
                reference.push(hh,[0],[grads[2]])
                losses.append(float(l))
                if iteration==2:
                    context.input_step=3
                    runner.checkpoint(str(root/'framework-checkpoints'),'mid-run')
                    saved=root/'framework-checkpoints/mid-run'
                    assert (saved/'native-committed.json').is_file()
                    assert int((saved/'icb_feeder_ckpt').read_text())==3
                    manifest=json.loads((saved/'external_parameters/parameter-service.json').read_text())
                    commit=json.loads((saved/'external_parameters/committed.json').read_text())
                    assert len(manifest['files'])==2 and commit['checkpoint_id']=='mid-run'
                    np.testing.assert_allclose(tf.train.load_variable(str(saved/'dense/model'),'weight'),session.run(dense),rtol=0,atol=0)
            assert losses[-1]<losses[0],losses
            users.set_prediction();hashed.set_prediction()
            for _ in range(12):session.run(features)
            assert all(m['live_tickets']==0 for m in next(iter(hooks.backends().values())).stats())
            users.set_training();hashed.set_training()
            session.run(step)
            oracle=next(iter(hooks.backends().values()))
            oracle.save('expected-six')
            oracle.export_checkpoint('expected-six',str(root/'expected-six'),tf.io.gfile)
            oracle.resume('expected-six')
            np.save(root/'expected-dense-six.npy',session.run(dense))
            session.run(zero_filled)
            for _ in range(12):session.run(all_unused)
            assert all(m['live_tickets']==0 for m in next(iter(hooks.backends().values())).stats())
        metrics=reference.stats()
        assert all(m['live_tickets']==0 for m in metrics)
        if agent:assert sum(m['cached_rows'] for m in metrics)>0,metrics
        # Original per_learner_once caches table creation by name, even when a
        # second graph constructs a new Python object for the same logical table.
        for active in (owner,Learner(1,2)):
            context.learner=active
            other_graph=tf.Graph()
            with other_graph.as_default():
                second=train.SparseTable([1,2],'users',**kwargs);second.create_table()
                second_output=second.embedding_lookup_and_pooling(
                    [tf.constant([0],tf.int64)],[tf.constant([0,1],tf.int32)],None,shapes=[[1,2]])
                second.set_prediction()
                init=tf.compat.v1.global_variables_initializer()
            with tf.compat.v1.Session(graph=other_graph) as session:
                session.run(init);context.learner=None
                assert np.isfinite(session.run(second_output)).all()
                context.learner=active
            second.set_training()
        # A different original learner/thread obtains its own request namespace.
        assert set(hooks.backends())=={0,10}
        context.learner=owner
        def fail_dense(*args):raise RuntimeError('injected dense checkpoint failure')
        dense_table.checkpoint=fail_dense
        try:runner.checkpoint(str(root/'framework-checkpoints'),'failed-save')
        except RuntimeError as error:assert 'injected dense' in str(error)
        else:raise AssertionError('failed dense checkpoint unexpectedly committed')
        failed=root/'framework-checkpoints/failed-save'
        assert (failed/'external_parameters/parameter-service.json').is_file()
        assert not (failed/'external_parameters/committed.json').exists()
        try:reference.pull(user_features)
        except RuntimeError:pass
        else:raise AssertionError('parameter admission reopened after failed dense save')
        print(json.dumps(dict(test='entropy_original_table_graph',train_steps=6,
            reference_checked_steps=5,max_reference_absolute_error=maximum_error,
            loss_before=losses[0],loss_after=losses[-1],parameter_shards=2,
            callback_tls_independent=True,prediction_repetitions=12,
            missing_gradient_zero_fill=True,unused_feature_releases=12,
            same_learner_graph_recreation=True,
            original_runner_midrun_checkpoint=True,failed_dense_save_stays_frozen=True,
            table_backend='native' if native_backend else 'portable',graph_python_callbacks=any('PyFunc' in kind for kind in graph_types),
            table_class_preserved=True,backend_learner_ids=sorted(hooks.backends()),
            cache_backend=('cuda' if config.get('native',{}).get('cuda') else os.environ.get('EPS_CACHE_BACKEND','host')) if agent else 'none',
            source=provenance)),flush=True)
    finally:
        context.learner=context.learner or owner
        hooks.close();reference.close()
        assert train.checkpoint is original_checkpoint
        assert train._MultiDatasetRunner.checkpoint is original_runner_checkpoint
        assert not hasattr(train.SparseTable,'_eps_installation')


def main():
    try:source_root()
    except FileNotFoundError as error:
        print('SKIP: '+str(error));return 77
    if sys.argv[1]=='--probe':
        probe(sys.argv[2:4],pathlib.Path(sys.argv[4]),sys.argv[5] if len(sys.argv)>5 else None);return 0
    server=sys.argv[1];agent_binary=sys.argv[2] if len(sys.argv)>2 else None
    with tempfile.TemporaryDirectory(prefix='eps-entropy-hooks-') as directory:
        root=pathlib.Path(directory);processes=[];logs=[];endpoints=[];agent=None
        try:
            native_gpu=os.environ.get('EPS_ENTROPY_BACKEND')=='native' and os.environ.get('EPS_ENTROPY_DEVICE','').startswith('/GPU:')
            if native_gpu:
                with socket.socket() as s:s.bind(('127.0.0.1',0));agent='127.0.0.1:'+str(s.getsockname()[1])
            elif agent_binary:
                log=open(root/'agent.log','w+');logs.append(log)
                process=subprocess.Popen([agent_binary,'--run','entropy-hooks','--agent','0','--listen','127.0.0.1:0',
                                          '--backend',os.environ.get('EPS_CACHE_BACKEND','host')],
                                          stdout=subprocess.PIPE,stderr=log,text=True)
                processes.append(process);agent=ready(process)
            for shard in range(2):
                config=root/f'ps-{shard}.json'
                cache=dict(enabled=bool(agent))
                if agent:cache.update(admission='frequency',decision_window_requests=1,minimum_accesses=1,reserve_slots=1)
                config.write_text(json.dumps(dict(schema_version=1,run_id='entropy-hooks',listen='127.0.0.1:0',
                    partition=dict(shard_id=shard,shard_count=2,epoch=1),tables=[],cache=cache,
                    access=dict(policy='transfer_unique' if agent else 'cpu_only',agent_id=0),
                    agents=[dict(id=0,endpoint=agent)] if agent else [],snapshot_directory=str(root/'snapshots'))))
                log=open(root/f'ps-{shard}.log','w+');logs.append(log)
                process=subprocess.Popen([server,'--config',str(config)],stdout=subprocess.PIPE,stderr=log,text=True)
                processes.append(process);endpoints.append(ready(process))
            subprocess.run([sys.executable,__file__,'--probe',*endpoints,str(root),*([agent] if agent else [])],check=True,timeout=90)
        except Exception:
            for log in logs:log.flush();log.seek(0);print(log.read()[-3000:],file=sys.stderr)
            raise
        finally:
            for process in processes:stop(process)
            for log in logs:log.close()
    return 0


if __name__=='__main__':sys.exit(main())
