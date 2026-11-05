"""Opt-in external parameter tables for the existing Entropy training entry point.

Install on the original class before the job constructs its input/model graph.
Native tables keep their original methods. Graph callbacks capture a learner's
backend directly, so TensorFlow executor threads do not consult framework TLS.
"""
import copy
import functools
import hashlib
import json
import os
import posixpath
import re
import threading
import uuid

from entropy_configuration import sparse_table_definition

UINT64_MAX = 2**64 - 1


def _publish_json(filesystem,target,value):
    from checkpoint_store import _publish
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    digest=hashlib.sha256(raw).hexdigest();pending=target+'.pending-'+uuid.uuid4().hex
    try:
        with filesystem.GFile(pending,'wb') as stream:stream.write(raw)
        _publish(filesystem,pending,target,len(raw),digest)
    finally:
        if filesystem.exists(pending):filesystem.remove(pending)
    return digest


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('invalid ' + name)
    return value


class JobConfiguration:
    def __init__(self, values):
        allowed={'schema_version','run','endpoints','partition_epoch','tables',
                 'learner_id_stride','cache_agents','coalesce_gather','max_live_batches','backend','native','training'}
        if not isinstance(values,dict) or set(values)-allowed:
            raise ValueError('unknown parameter-service job option')
        if type(values.get('schema_version')) is not int or values['schema_version'] != 1:
            raise ValueError('unsupported parameter-service job schema')
        self.run=values.get('run')
        endpoints=values.get('endpoints',())
        if not isinstance(endpoints,(list,tuple)) or not 1<=len(endpoints)<=1024:
            raise ValueError('provide the ordered list of parameter-shard endpoints')
        self.endpoints=tuple(endpoints)
        if not isinstance(self.run,str) or not self.run or not self.endpoints or any(
                not isinstance(e,str) or not e for e in self.endpoints):
            raise ValueError('run and nonempty shard endpoints are required')
        self.epoch=_integer(values.get('partition_epoch',1),'partition epoch',1,UINT64_MAX)
        self.learner_stride=_integer(values.get('learner_id_stride',1),'learner ID stride',1,1024)
        self.max_live=_integer(values.get('max_live_batches',128),'live batch limit',1,1048576)
        self.backend=values.get('backend','portable')
        if self.backend not in ('portable','native'):raise ValueError('unknown table backend')
        self.training=copy.deepcopy(values.get('training'))
        if self.training is not None:
            t=self.training
            if (self.backend!='native' or not isinstance(t,dict) or set(t)-{'coordinator','start_iteration'}
                    or not isinstance(t.get('coordinator'),str) or not t['coordinator']):
                raise ValueError('training progress requires a native backend and coordinator')
            _integer(t.setdefault('start_iteration',0),'initial training iteration',0,UINT64_MAX-1)
        self.native=copy.deepcopy(values.get('native',{}))
        if not isinstance(self.native,dict) or set(self.native)-{'device','cuda','op_library','runtime_library'}:
            raise ValueError('unknown native backend option')
        if self.native and self.backend!='native':raise ValueError('native options require the native backend')
        self.native.setdefault('device','/CPU:0')
        if not isinstance(self.native['device'],str) or not re.fullmatch(r'/(CPU:0|GPU:[0-9]+)',self.native['device']):
            raise ValueError('native device must be /CPU:0 or /GPU:index')
        for name in ('op_library','runtime_library'):
            if name in self.native and (not isinstance(self.native[name],str) or not self.native[name]):
                raise ValueError('native library path must be nonempty')
        if 'cuda' in self.native:
            c=self.native['cuda']
            if not isinstance(c,dict) or set(c)-{'agent','device','listen','state_bytes','rows_per_table','transfer_bytes','outstanding_output_bytes'}:
                raise ValueError('unknown native CUDA owner option')
            _integer(c.get('agent',0),'native Agent ID',0,2**32-1)
            _integer(c.get('device',0),'native CUDA device',0,1023)
            if 'listen' in c and (not isinstance(c['listen'],str) or not c['listen']):raise ValueError('native listener must be nonempty')
            for name in ('state_bytes','rows_per_table','transfer_bytes','outstanding_output_bytes'):
                if name in c:_integer(c[name],'native '+name,1,UINT64_MAX)
        if self.native['device'].startswith('/GPU:') and ('cuda' not in self.native or
                int(self.native['device'].split(':')[1])!=self.native['cuda'].get('device',0)):
            raise ValueError('native graph device must match its CUDA owner')
        tables=values.get('tables')
        if not isinstance(tables,dict) or not tables:
            raise ValueError('select at least one table by its original name')
        self.tables=copy.deepcopy(tables)
        seen=set()
        for name,table in self.tables.items():
            if not isinstance(name,str) or not name or not isinstance(table,dict) or set(table)-{'id','seed'}:
                raise ValueError('invalid external table configuration')
            identifier=_integer(table.get('id'),'table ID',0,2**32-1)
            if identifier in seen:
                raise ValueError('external table IDs must be unique')
            seen.add(identifier)
            _integer(table.setdefault('seed',0),'table seed',0,UINT64_MAX)
        self.cache_agents=[]
        seen=set()
        for agent in values.get('cache_agents',[]):
            if not isinstance(agent,dict) or set(agent)-{'id','endpoint','owner_run'} or not {'id','endpoint'}<=set(agent):
                raise ValueError('invalid cache agent')
            identifier=_integer(agent['id'],'cache agent ID',0,2**32-1)
            if identifier in seen or not isinstance(agent['endpoint'],str) or not agent['endpoint']:
                raise ValueError('duplicate cache agent or empty endpoint')
            scope=agent.get('owner_run',self.run)
            if not isinstance(scope,str) or not scope:raise ValueError('cache owner run must be nonempty')
            seen.add(identifier);self.cache_agents.append((identifier,agent['endpoint'],scope))
        self.coalesce=values.get('coalesce_gather',False)
        if type(self.coalesce) is not bool:
            raise ValueError('coalesce_gather must be boolean')

    def table(self,name):
        return self.tables.get(name)

    def learner_id(self,rank,thread):
        _integer(rank,'learner rank',0,1023)
        _integer(thread,'learner thread',0,self.learner_stride-1)
        return _integer(rank*self.learner_stride+thread,'combined learner ID',0,1023)


class _TableState:
    def __init__(self,backend,definition):
        self.backend=backend
        self.definition=definition
        self.predict=False
        self.created=False
        self.bindings={}


class Installation:
    def __init__(self,train,configuration):
        self.config=configuration if isinstance(configuration,JobConfiguration) else JobConfiguration(configuration)
        self.train=train
        self.cls=train.SparseTable
        if getattr(self.cls,'_eps_installation',None) is not None:
            raise RuntimeError('external table hooks are already installed')
        self._lock=threading.RLock()
        self._backends={}
        self._administrative_backend=None
        self._states={}
        self._originals={}
        self._wrappers={}
        self._controller_originals=[]
        self._checkpoint_users=0
        self._checkpoint_key=None
        self._local=threading.local()
        self._owner_key='eps-worker-'+uuid.uuid4().hex
        self._native_owner=None
        self.closed=False
        replacements={
            '_create_table':self._create_table,
            'embedding_lookup_and_pooling':self._lookup,
            'vars':self._vars,
            'output':self._vars,
            'apply_gradients':self._apply,
            'set_prediction':self._prediction,
            'set_training':self._training,
            'set_upsert':self._upsert,
        }
        if hasattr(self.cls,'checkpoint'):
            replacements['checkpoint']=self._table_checkpoint
        if hasattr(self.cls,'get_reshard'):
            replacements['get_reshard']=self._get_reshard
        if hasattr(self.cls,'restore'):
            replacements['restore']=self._restore_table
        # Validate the whole surface before changing the original class.
        for name in replacements:
            if not callable(getattr(self.cls,name,None)):
                raise ValueError('original SparseTable is missing '+name)
        for name,action in replacements.items():
            original=getattr(self.cls,name)
            @functools.wraps(original)
            def dispatch(table,*args,_original=original,_action=action,**kwargs):
                if self.config.table(table._name) is None:
                    return _original(table,*args,**kwargs)
                if self.closed:
                    raise RuntimeError('external table installation is closed')
                return _action(table,*args,**kwargs)
            self._originals[name]=original
            self._wrappers[name]=dispatch
            setattr(self.cls,name,dispatch)
        self.cls._eps_installation=self
        try:
            if self.config.backend=='native' and 'cuda' in self.config.native:
                from native_worker_owner import NativeWorkerOwner
                self._native_owner=NativeWorkerOwner(self._native_configuration(0),
                    runtime_library=self.config.native.get('runtime_library'))
            self._install_controller()
            self._install_training()
        except BaseException:
            self.close();raise

    def _context(self):
        learner=self.train.get_default_learner()
        identifier=self.config.learner_id(int(learner.learner_rank),int(learner.thread_id))
        return learner,identifier

    def _validate_table(self,table):
        if not table._is_table_v2:
            raise ValueError('external parameter tables require the V2 feature boundary')
        if any(getattr(table,n,False) for n in ('_loc_on_gpu','_use_ps_gpu_embed',
                '_use_gpu_host_embed_cache','_use_vram_cache','_using_array')):
            raise ValueError('select the plain V2 table boundary before external GPU placement')
        if table._sideinfo is not None or table._purge_policy not in ('','NoPurge'):
            raise ValueError('side information and active purge policies require their own integration')
        if table._access_policy is not None:
            active=('_embedding_create_threshold','_embedding_purge_threshold',
                    '_embedding_serving_purge_threshold','_embedding_latest_show_decay_rate',
                    '_embedding_time_decay_threshold','_show_click_plugin_click_label_name',
                    '_show_click_plugin_sync_meta_to_value')
            if any(getattr(table._access_policy,n,0) for n in active):
                raise ValueError('active access-policy metadata requires its own integration')
        if table._pooling_method not in ('sum','mean',None,'none'):
            raise ValueError('unsupported original pooling method')
        if table._hash_trick_bucket_size>1 and not table._non_unique:
            raise ValueError('the original unique Pull path does not support bucket masking')

    def _state(self,table):
        with self._lock:
            _,identifier=self._context()
            self._validate_table(table)
            spec=self.config.table(table._name)
            initialization=None
            if table._initializer_name=='const':
                value=float(table._const_initializer_value)
                initialization=(value,value)
            definition=sparse_table_definition(spec['id'],table._sparse_dim,table._optimizer,
                seed=spec['seed'],initialization=initialization,adam_share_segment=table._adam_share_segment)
            # Original create_table memoizes by logical name per learner, so a
            # new graph can reuse that successful creation with a new object.
            key=(table._name,identifier)
            if key in self._states and self._states[key].definition!=definition:
                raise ValueError('logical table configuration changed between graphs: '+table._name)
            if key not in self._states:
                self._states[key]=_TableState(self._backend(identifier),definition)
            return self._states[key]

    def _backend(self,identifier):
        with self._lock:
            if identifier not in self._backends:
                if self.config.backend=='native':
                    from native_session_backend import NativeSessionBackend
                    backend=NativeSessionBackend(self._native_configuration(identifier),
                        **{k:v for k,v in self.config.native.items() if k!='cuda'})
                else:
                    from eps_tensorflow import ParameterServer
                    backend=ParameterServer(self.config.endpoints,self.config.run,identifier,
                                            self.config.epoch,self.config.max_live)
                try:
                    if self.config.cache_agents and self.config.backend=='portable':
                        backend.cache_endpoints(self.config.cache_agents,self.config.coalesce)
                except Exception:
                    backend.close();raise
                self._backends[identifier]=backend
            return self._backends[identifier]

    def _native_configuration(self,identifier):
        values=dict(run=self.config.run,endpoints=self.config.endpoints,learner=identifier,
            epoch=self.config.epoch,max_live=self.config.max_live,tables=[],release_read_only=True,
            coalesce_gather=self.config.coalesce,
            agents=[dict(id=i,endpoint=e,owner_run=r) for i,e,r in self.config.cache_agents])
        if 'cuda' in self.config.native:
            values['cuda']=dict(self.config.native['cuda'],owner_key=self._owner_key)
        if self.config.training is not None:
            values['training']=dict(self.config.training,worker=identifier//self.config.learner_stride)
        return values

    def _control_backend(self):
        with self._lock:
            if self._administrative_backend is None:
                if self.config.backend=='native':
                    from native_session_backend import NativeSessionBackend
                    values=dict(run=self.config.run,endpoints=self.config.endpoints,learner=0,
                        epoch=self.config.epoch,max_live=1,tables=[])
                    self._administrative_backend=NativeSessionBackend(values,device='/CPU:0',
                        **{k:v for k,v in self.config.native.items() if k in ('op_library','runtime_library')})
                else:
                    from eps_tensorflow import ParameterServer
                    self._administrative_backend=ParameterServer(self.config.endpoints,self.config.run,0,
                                                                 self.config.epoch,1)
            return self._administrative_backend

    def _verify_restore(self,path,checkpoint_id):
        import tensorflow as tf
        from checkpoint_restore import verify_restored
        with self._lock:
            if self.config.training is not None and not self._backends:
                _,identifier=self._context();self._backend(identifier)
            backend=next(iter(self._backends.values())) if self._backends else self._control_backend()
            local_backends=list(self._backends.values())
        descriptor=verify_restored(backend,os.fspath(path),run=self.config.run,epoch=self.config.epoch,
            shard_count=len(self.config.endpoints),filesystem=tf.io.gfile,checkpoint_id=os.fspath(checkpoint_id))
        progress=descriptor['training_progress']
        if (progress is None)!=(self.config.training is None):
            raise ValueError('checkpoint training progress and configured coordinator must match')
        if progress is not None:
            for local in local_backends:
                status=local.training_status();actual=local.checkpoint_training()
                worker=local.configuration['training']['worker']
                if (worker>=len(progress['completed']) or not status['paused'] or status['failed']
                        or status['active'] or status['armed'] or status['coordinator_active'] or status['waiting']
                        or status['next_iteration']!=progress['completed'][worker]
                        or actual['source_run']!=self.config.run
                        or any(actual[k]!=progress[k] for k in ('epoch','consistency','max_delay','completed'))):
                    raise ValueError('restored coordinator or learner progress differs from checkpoint')
        return descriptor

    @staticmethod
    def _restore_options(reshard,migrate,from_serving,delta,delta_id,delta_ckpt_paths=''):
        if reshard or migrate or from_serving or delta or delta_id or delta_ckpt_paths:
            raise ValueError('external restore requires a full training checkpoint with matching partitions')

    def _get_reshard(self,table,path,checkpoint_id,reshard=None,ckpt_cache_path='',migrate=False,
                     migrate_init_policy=None,migrate_table_id=-1,from_serving=False,delta=False,delta_id=''):
        self._restore_options(reshard,migrate,from_serving,delta,delta_id)
        self._verify_restore(path,checkpoint_id)
        return False

    def _restore_table(self,table,path,checkpoint_id,reshard=None,ckpt_cache_path='',migrate=False,
                       migrate_init_policy=None,migrate_table_id=-1,from_serving=False,delta=False,
                       delta_id='',delta_ckpt_paths=''):
        self._restore_options(reshard,migrate,from_serving,delta,delta_id,delta_ckpt_paths)
        # Startup loaded the complete shard. Never pass selected external tables
        # to the original native sparse loader or overwrite an active service.
        self._verify_restore(path,checkpoint_id)

    def _create_table(self,table,force_recreate=False):
        if force_recreate:
            raise ValueError('external table recreation requires a fresh parameter-service run')
        state=self._state(table)
        if state.created:
            return
        state.backend.create_table(state.definition)
        state.created=True

    def _binding(self,table):
        import tensorflow as tf
        state=self._state(table)
        binding=state.bindings.get((table,tf.compat.v1.get_default_graph()))
        if binding is None:
            raise ValueError('build the table feature lookup before requesting its gradients')
        return state,binding

    def _lookup(self,table,ids,row_splits,show_click_plugin_click_label,shapes=None):
        import tensorflow as tf
        from eps_tensorflow import Feature
        if tf.executing_eagerly():
            raise ValueError('Entropy table hooks require the existing Session/graph boundary')
        state=self._state(table)
        if not state.created:
            raise ValueError('create the external table before constructing its lookup')
        ids=ids if isinstance(ids,list) else [ids]
        row_splits=row_splits if isinstance(row_splits,list) else [row_splits]
        shapes=shapes if shapes is not None else [table._sparse_var_shape]
        if not ids or len(ids)!=len(row_splits) or len(ids)!=len(shapes):
            raise ValueError('original IDs, row splits and feature shapes must align')
        tensors=[];descriptors=[];sequence=[]
        for feature,(keys,splits,shape) in enumerate(zip(ids,row_splits,shapes)):
            if len(shape) not in (2,3) or shape[-1]!=table._sparse_dim:
                raise ValueError('original feature shape must be [B,D] or [B,L,D]')
            keys=tf.convert_to_tensor(keys);splits=tf.convert_to_tensor(splits)
            if keys.dtype not in (tf.int64,tf.uint64) or splits.dtype not in (tf.int32,tf.int64):
                raise ValueError('original IDs require 64 bits and row splits require integers')
            is_sequence=len(shape)==3
            if not is_sequence and table._pooling_method not in ('sum','mean'):
                raise ValueError('pooled feature shapes require sum or mean pooling')
            if keys.shape.rank != (2 if is_sequence else 1) or splits.shape.rank!=1:
                raise ValueError('original sequence IDs have rank 2; pooled IDs have rank 1')
            checks=[]
            if is_sequence:
                checks=[tf.debugging.assert_equal(tf.size(splits),0)]
                with tf.control_dependencies(checks):
                    flat=tf.reshape(keys,[-1])
                    width=tf.cast(tf.shape(keys)[1],tf.int64)
                    offsets=tf.range(tf.cast(tf.shape(keys)[0],tf.int64)+1,dtype=tf.int64)*width
            else:
                flat=keys;offsets=tf.cast(splits,tf.int64)
            # The native input operation reads signed int64 tensor bits as uint64.
            # Preserve the padding sentinel before applying an optional key mask.
            unsigned=tf.bitcast(flat,tf.uint64) if flat.dtype==tf.int64 else flat
            bucket=table._hash_trick_bucket_size
            if bucket>1:
                unsigned=tf.where(tf.equal(unsigned,tf.constant(UINT64_MAX,tf.uint64)),unsigned,
                                  tf.bitwise.bitwise_and(unsigned,tf.constant(bucket-1,tf.uint64)))
            tensors.extend((unsigned,offsets))
            descriptors.append(Feature(self.config.table(table._name)['id'],feature,unsigned,offsets,
                table._sparse_dim,'sequence' if is_sequence else table._pooling_method or 'sequence',UINT64_MAX))
            sequence.append((is_sequence,keys))
        table._v2_vars=[tf.compat.v1.get_variable('%s_%d'%(table._name,i),shape)
                        for i,shape in enumerate(shapes)]
        for variable in table._v2_vars:
            self.train.add_var_collections(variable,table)
        build_eval=bool(self.train.ts.cluster.current_registry().build_eval())
        def dispatch(*values):
            features=[dict(table=f.table,feature=f.feature,pooling=f.pooling,padding_key=f.padding_key,
                keys=values[2*i].numpy(),offsets=values[2*i+1].numpy()) for i,f in enumerate(descriptors)]
            read_only=build_eval or state.predict
            handle,outputs=state.backend.backend.pull(features,read_only)
            checked=state.backend._checked_outputs(handle,outputs,[table._sparse_dim]*len(ids))
            if read_only:
                state.backend.backend.release(handle)
            return checked
        if self.config.backend=='native':
            token,flat=state.backend.pull_graph(descriptors,read_only=build_eval)
            values=[token,*flat]
        else:
            values=tf.py_function(dispatch,tensors,[tf.uint64]+[tf.float32]*len(ids),name='external_sparse_pull')
        handle=values[0];handle.set_shape([])
        output=[]
        for i,(is_sequence,keys) in enumerate(sequence):
            value=values[i+1];value.set_shape([None,table._sparse_dim])
            if is_sequence:
                value=tf.reshape(value,tf.concat([tf.shape(keys),[table._sparse_dim]],0))
                value.set_shape(keys.shape.concatenate([table._sparse_dim]))
            else:
                rows=tf.convert_to_tensor(row_splits[i]).shape[0]
                value.set_shape([None if rows is None else rows-1,table._sparse_dim])
            output.append(value)
        table._row_splits=row_splits
        table._pull=output
        state.bindings[(table,tf.compat.v1.get_default_graph())]=(handle,output)
        return output

    def _vars(self,table):
        return self._binding(table)[1][1]

    def _apply(self,table,grads_and_vars,sparse_joinable_ops,name=None,constant_grads=None):
        import tensorflow as tf
        state,(handle,outputs)=self._binding(table)
        pairs=list(grads_and_vars)
        if len(pairs)!=len(outputs) or any(v is not output for (_,v),output in zip(pairs,outputs)):
            raise ValueError('sparse gradients must follow the original table feature order')
        if all(g is None for g,_ in pairs):
            return state.backend.release(handle)
        gradients=[]
        ignore_none=bool(self.train.flags.FLAGS.ignore_none_grads)
        for gradient,output in pairs:
            if gradient is None:
                if not ignore_none:
                    raise ValueError('missing sparse gradient; enable the original ignore_none_grads policy to zero-fill')
                gradient=tf.zeros_like(output)
            if not isinstance(gradient,tf.Tensor) or gradient.dtype!=tf.float32:
                raise ValueError('sparse feature gradients must be FP32 tensors')
            gradients.append(tf.reshape(gradient,[-1,table._sparse_dim]))
        # Original InputBuilder groups every table's returned Push operation.
        # Each ack waits for that update; a per-table global Join would reject
        # other tables' still-live handles in the shared learner backend.
        return state.backend.push(handle,range(len(outputs)),gradients)

    def _prediction(self,table):
        state=self._state(table)
        if self.config.backend=='native':state.backend.set_prediction(state.definition['id'],True)
        state.predict=True

    def _training(self,table):
        state=self._state(table)
        if self.config.backend=='native':state.backend.set_prediction(state.definition['id'],False)
        state.predict=False

    def _upsert(self,table):
        raise ValueError('upsert mode requires explicit external parameter updates')

    def _patch_controller(self,owner,name,replacement):
        original=getattr(owner,name)
        setattr(owner,name,replacement)
        self._controller_originals.append((owner,name,original,replacement))

    def _install_training(self):
        if self.config.training is None:return
        runner=getattr(self.train,'_MultiDatasetRunner',None)
        if runner is None or not callable(getattr(runner,'run',None)):
            raise ValueError('training progress requires the original Runner.run boundary')
        def training_fetch(value):
            return isinstance(value,dict) and ('run_op' in value or any(training_fetch(v) for v in value.values()))
        owners=[runner]
        scene=getattr(self.train,'_MultiSceneRunner',None)
        if scene is not None and 'run' in scene.__dict__:owners.append(scene)
        for owner in owners:
            original=owner.run
            @functools.wraps(original)
            def run(instance,*args,_original=original,**kwargs):
                if (not getattr(instance,'is_worker',False) or not args or not training_fetch(args[0])
                        or instance.prediction_mode()):
                    return _original(instance,*args,**kwargs)
                _,identifier=self._context()
                backend=self._backend(identifier)
                backend.arm_training()
                try:
                    result=_original(instance,*args,**kwargs)
                    backend.finish_training()
                    return result
                except BaseException as error:
                    try:backend.cancel_training(type(error).__name__)
                    except Exception:pass
                    raise
            self._patch_controller(owner,'run',run)

    def _install_controller(self):
        runner=getattr(self.train,'_MultiDatasetRunner',None)
        if runner is None or not hasattr(self.train,'checkpoint'):
            return
        original_checkpoint=runner.checkpoint
        original_prepare=runner.prepare_ckpt
        original_function=self.train.checkpoint
        @functools.wraps(original_prepare)
        def prepare(instance,path):
            transaction=getattr(self._local,'checkpoint',None)
            if transaction is not None and transaction['root']==os.fspath(path):
                return  # The coordinator prepared this path before export.
            return original_prepare(instance,path)
        @functools.wraps(original_function)
        def function(*args,**kwargs):
            if getattr(self._local,'checkpoint',None) is None:
                raise RuntimeError('external tables require the coordinated Runner.checkpoint boundary')
            return original_function(*args,**kwargs)
        @functools.wraps(original_checkpoint)
        def checkpoint(instance,path,checkpoint_id,purge_countdown=None):
            import tensorflow as tf
            path=os.fspath(path);checkpoint_id=os.fspath(checkpoint_id)
            root=posixpath.join(path,checkpoint_id)
            key=(path,checkpoint_id)
            with self._lock:
                if self._checkpoint_key is not None and self._checkpoint_key!=key:
                    raise RuntimeError('a different external checkpoint is already in progress')
                self._checkpoint_key=key;self._checkpoint_users+=1
            registered=True
            transaction=dict(root=root,key=key)
            backend=None
            def barrier(stage):
                tag=hashlib.sha256(root.encode()).hexdigest()[:24]
                self.train.env.barrier('::EPS_'+stage+'_'+tag+'::',cluster='all')
            leader=(self.train.env.peer_id()==0 and self.train.MultiThreadingLearner.is_chief('worker'))
            try:
                # Each learner checks only its own handles before the global
                # barrier. Other learners may still be finishing their steps.
                _,identifier=self._context()
                local_backend=self._backends.get(identifier)
                if local_backend is not None:local_backend.backend.join()
                barrier('DRAINED')
                if leader:
                    backend=local_backend or next(iter(self._backends.values()),None) or self._control_backend()
                    if self.config.training is not None:
                        backend.pause_training()
                        transaction['progress']=backend.checkpoint_training()
                    transaction['snapshot']='entropy-'+uuid.uuid4().hex
                    backend.save(transaction['snapshot'])
                barrier('FROZEN')
                # Original prepare_path removes an existing destination. Run
                # it once before export and suppress its duplicate invocation
                # inside the unchanged Runner.checkpoint body.
                original_prepare(instance,root)
                barrier('PATH_READY')
                if leader:
                    transaction['manifest']=backend.export_checkpoint(transaction['snapshot'],
                        posixpath.join(root,'external_parameters'),tf.io.gfile)
                    if self.config.training is not None:
                        transaction['progress_sha256']=_publish_json(tf.io.gfile,
                            posixpath.join(root,'external_parameters','training-progress.json'),transaction['progress'])
                barrier('EXPORTED')
                self._local.checkpoint=transaction
                original_checkpoint(instance,path,checkpoint_id,purge_countdown)
                barrier('ORIGINAL_COMPLETE')
                if leader:
                    manifest=transaction['manifest']
                    commit=dict(schema_version=1,checkpoint_id=checkpoint_id,
                        run=self.config.run,partition_epoch=self.config.epoch,
                        parameter_manifest_sha256=hashlib.sha256(
                            (json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()).hexdigest())
                    if self.config.training is not None:commit['training_progress_sha256']=transaction['progress_sha256']
                    target=posixpath.join(root,'external_parameters','committed.json')
                    _publish_json(tf.io.gfile,target,commit)
                    backend.resume(transaction['snapshot'])
                    if self.config.training is not None:backend.resume_training()
                # Unregister before the release barrier so a fast learner's
                # next checkpoint cannot race the previous context's teardown.
                with self._lock:
                    self._checkpoint_users-=1;registered=False
                    if not self._checkpoint_users:self._checkpoint_key=None
                barrier('RESUMED')
            finally:
                self._local.checkpoint=None
                if registered:
                    with self._lock:
                        self._checkpoint_users-=1
                        if not self._checkpoint_users:self._checkpoint_key=None
                # A failure leaves admission frozen; never reopen a partial save.
        self._patch_controller(runner,'prepare_ckpt',prepare)
        self._patch_controller(runner,'checkpoint',checkpoint)
        self._patch_controller(self.train,'checkpoint',function)
        if hasattr(runner,'restore'):
            original_restore=runner.restore
            @functools.wraps(original_restore)
            def restore(instance,model,path,checkpoint_id,restore_sparse=True,restore_dense=True,
                        reshard=None,ckpt_cache_path='',from_serving=False,delta=False,delta_id='',delta_ckpt_paths=''):
                self._restore_options(reshard,getattr(self.train.ts.config,'restore_need_migrate',False),
                                      from_serving,delta,delta_id,delta_ckpt_paths)
                if not restore_sparse:raise ValueError('external joint restore requires sparse state verification')
                self._verify_restore(posixpath.join(os.fspath(path),os.fspath(checkpoint_id)),checkpoint_id)
                self.train.env.barrier('::EPS_RESTORE_VERIFIED::',cluster='all')
                result=original_restore(instance,model,path,checkpoint_id,restore_sparse,restore_dense,
                                        reshard,ckpt_cache_path,from_serving,delta,delta_id,delta_ckpt_paths)
                if self.config.training is not None:
                    self.train.env.barrier('::EPS_MODEL_RESTORE_COMPLETE::',cluster='all')
                    if self.train.env.peer_id()==0 and self.train.MultiThreadingLearner.is_chief('worker'):
                        next(iter(self._backends.values())).resume_training()
                    self.train.env.barrier('::EPS_TRAINING_RESTORE_RESUMED::',cluster='all')
                return result
            self._patch_controller(runner,'restore',restore)

    def _table_checkpoint(self,table,table_id,path,checkpoint_id,delta=False):
        import tensorflow as tf
        with self._lock:
            if delta or self._checkpoint_key!=(os.fspath(path),os.fspath(checkpoint_id)):
                raise RuntimeError('external sparse state must be saved by the coordinated Runner.checkpoint')
        manifest=posixpath.join(os.fspath(path),os.fspath(checkpoint_id),
                                'external_parameters','parameter-service.json')
        if not tf.io.gfile.exists(manifest):
            raise RuntimeError('external sparse export is not complete')
        # The shared service snapshot contains every selected table once. The
        # original per-table save thread must not invoke the native sparse saver.

    def backends(self):
        """Controller-owned backend handles; valid while the installation is live."""
        with self._lock:
            return dict(self._backends)

    def close(self):
        with self._lock:
            if self.closed:
                return
            for name in self._originals:
                if getattr(self.cls,name) is not self._wrappers[name]:
                    raise RuntimeError('SparseTable hook changed after installation: '+name)
            for owner,name,original,replacement in self._controller_originals:
                if getattr(owner,name) is not replacement:
                    raise RuntimeError('checkpoint hook changed after installation: '+name)
            for owner,name,original,replacement in reversed(self._controller_originals):
                setattr(owner,name,original)
            for name,original in self._originals.items():
                setattr(self.cls,name,original)
            del self.cls._eps_installation
            self.closed=True
            failure=None
            for backend in self._backends.values():
                try:
                    backend.close()
                except Exception as error:
                    if failure is None:failure=error
            self._backends.clear();self._states.clear()
            if self._administrative_backend is not None:
                try:self._administrative_backend.close()
                except Exception as error:
                    if failure is None:failure=error
                self._administrative_backend=None
            if self._native_owner is not None:
                try:self._native_owner.close()
                except Exception as error:
                    if failure is None:failure=error
                self._native_owner=None
            if failure is not None:raise failure

    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def install(train_module,configuration):
    return Installation(train_module,configuration)
