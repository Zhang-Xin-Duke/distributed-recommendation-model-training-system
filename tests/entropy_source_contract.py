"""Execute preserved framework methods with explicit native-environment stand-ins.

The complete native Entropy module is not imported by this helper. No original
source is redistributed; a missing source snapshot produces exit code 77 in the
process test. Public configuration tests run independently of this helper.
"""
import ast
import concurrent.futures
import functools
import hashlib
import logging
import os
import pathlib
import threading
import time
import types


def source_root():
    default=pathlib.Path(__file__).resolve().parents[1]/'references/private/entropy'
    root=pathlib.Path(os.environ.get('EPS_ENTROPY_SOURCE_ROOT',default))
    if not (root/'tensorswitch/python/tensorswitch/train.py').is_file():
        raise FileNotFoundError('preserved Entropy source is required for the original-method contract')
    return root


class Learner:
    def __init__(self,rank=0,thread=0):
        self.learner_rank=rank;self.thread_id=thread;self.cluster_role='worker'
        self.var_collections={};self.remote_var_collections={};self.storage={}
        self.gpu_sparse_tables=[];self.gpu_host_embed_cache_tables=[]

    def get_or_create(self,name,constructor):
        if name not in self.storage:self.storage[name]=constructor()
        return self.storage[name]


def load(tf):
    root=source_root();path=root/'tensorswitch/python/tensorswitch/train.py'
    source=path.read_text();tree=ast.parse(source)
    context=types.SimpleNamespace(learner=Learner(),events=[],input_step=0)
    def learner():
        if context.learner is None:
            raise RuntimeError('framework learner TLS is unavailable on executor threads')
        return context.learner
    class Config(dict):
        ssd_auto_reserve_ratio=0
        export_parallel=2
        restore_parallel=2
        restore_need_migrate=False
        restore_sparse_when_build=False
        @staticmethod
        def number_of_partitions():return 1
    class Options:
        def __init__(self):self.weight_bounds=[];self.weight_decay=0;self.non_opt_dim=0;self.adam_share_segment=0
    class Feeder:
        count=0
        @classmethod
        def new_handle(cls):cls.count+=1;return cls.count
    def event(name,*args,**kwargs):context.events.append(name)
    def native_save(table_id,path,checkpoint_id,delta):
        context.events.append('native_sparse_'+str(table_id))
        return 0
    def input_save(name,scene,path):
        with tf.io.gfile.GFile(path,'w') as stream:stream.write(str(context.input_step))
    class Dense:
        def __init__(self,variable):self.variable=variable
        def checkpoint(self,session,path,checkpoint_id):
            directory=os.path.join(path,checkpoint_id,'dense');tf.io.gfile.makedirs(directory)
            with session.graph.as_default():
                tf.compat.v1.train.Saver(var_list={'weight':self.variable}).save(session,os.path.join(directory,'model'))
            context.events.append('dense_saved')
        def restore(self,session,path,checkpoint_id):
            with session.graph.as_default():
                tf.compat.v1.train.Saver(var_list={'weight':self.variable}).restore(session,os.path.join(path,'dense/model'))
            context.events.append('dense_restored')
    class MultiLearner:
        @staticmethod
        def is_chief(*args,**kwargs):return True
        @staticmethod
        def barrier(*args,**kwargs):event('local_barrier')
    def done(kind,path,checkpoint_id,delta=False):
        with tf.io.gfile.GFile(os.path.join(path,checkpoint_id,'native-committed.json'),'w') as stream:stream.write('{}')
        context.events.append('native_committed')
    env=types.SimpleNamespace(_env=types.SimpleNamespace(peer_id=0,shard_count=1,gpu_peer_cnt=0),
                              _get_default_learner=learner,peer_id=lambda:0,barrier=event,
                              call_local_function=input_save)
    ts=types.SimpleNamespace(config=Config(learners={}),
         cluster=types.SimpleNamespace(current_registry=lambda:types.SimpleNamespace(build_eval=lambda:False)),
         env=env,chief_gpu=lambda:False)
    module=types.ModuleType('source_executed_entropy')
    namespace=module.__dict__
    namespace.update(tf=tf,threading=threading,functools=functools,logging=logging,os=os,time=time,
        concurrent=concurrent,MultiThreadingLearner=MultiLearner,DenseTable=Dense,BnDenseTable=type('BnDense',(),{}),
        CLIB=types.SimpleNamespace(tensorswitch_sparse_save=native_save,tensorswitch_save_cluster_config=lambda *a:0),
        save_donefile=done,
        get_default_learner=learner,env=env,ts=ts,OptimizerOptions=Options,
        TSException=RuntimeError,FileFeeder=Feeder,MigrateInitPolicy=type('MigrateInitPolicy',(),{}),
        get_dense_ckpt=lambda path,delta_ckpt_paths:path,
        flags=types.SimpleNamespace(FLAGS=types.SimpleNamespace(ignore_none_grads=False)),
        _variable_size=lambda v:v.shape.num_elements())
    names={'cluster_specific','per_learner_once_attr','add_var_collections','checkpoint',
           'restore_sparse_fn','restore_sparse_start','restore_sparse_reshard_fn','restore_sparse_reshard_start',
           'worker_restore_sparse','worker_restore_sparse_wait','worker_restore_dense','worker_restore'}
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='per_learner_once' for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        elif isinstance(node,ast.ClassDef) and node.name in {'Optimizer','SparseLazyAdam','SparseTable'}:
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    original_runner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_MultiDatasetRunner')
    runner_methods=[n for n in original_runner.body if isinstance(n,ast.FunctionDef) and
                    n.name in {'checkpoint','prepare_ckpt','prepare_path','is_chief','restore'}]
    runner=ast.ClassDef(name='_MultiDatasetRunner',bases=[],keywords=[],body=runner_methods,decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[runner],type_ignores=[])),str(path),'exec'),namespace)
    # Original functions retain their own namespace, including original class identity.
    inputs=root/'tensorswitch/python/tensorswitch/inputs.py'
    builder=next(n for n in ast.parse(inputs.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='InputBuilder')
    methods=[n for n in builder.body if isinstance(n,ast.FunctionDef) and n.name in {'sparse_vars','apply_sparse_grads'}]
    extracted=ast.ClassDef(name='InputBuilder',bases=[],keywords=[],body=methods,decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[extracted],type_ignores=[])),str(inputs),'exec'),namespace)
    module.InputBuilder=namespace['InputBuilder']
    provenance=dict(train_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),
        source_definitions=['SparseTable','SparseLazyAdam','per_learner_once_attr','cluster_specific',
                  'add_var_collections','InputBuilder.sparse_vars','InputBuilder.apply_sparse_grads',
                  '_MultiDatasetRunner.checkpoint','_MultiDatasetRunner.prepare_path','checkpoint',
                  '_MultiDatasetRunner.restore','worker_restore','worker_restore_sparse','worker_restore_dense'],
        native_environment='learner/config/options and native save/feeder APIs are stand-ins; dense state uses an actual TensorFlow Saver; complete module not imported')
    return module,context,provenance
