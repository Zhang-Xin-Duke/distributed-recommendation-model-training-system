"""Execute full original main/Executor peers against external parameter shards.

Requires the private matching Linux/NVIDIA environment described in
docs/ORIGINAL_NATIVE_BUILD.md. Every peer imports the complete original module
and native cluster library. The manager retains durable named role handles and
does not substitute source-method fixtures or a replacement training loop.
"""
import argparse
import hashlib
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'adapters/entropy')]


def digest(path):
    result=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):result.update(chunk)
    return result.hexdigest()


def port():
    with socket.socket() as channel:
        channel.bind(('127.0.0.1',0))
        return '127.0.0.1:'+str(channel.getsockname()[1])


def environment(root,build):
    package=root/'original-python/lib/python3.10/site-packages/tensorflow'
    result={}
    result.update(PYTHONPATH=':'.join(map(str,[ROOT/'adapters/entropy',root/'python-feature',
        root/'python-generated-full',root/'native/entropy/tensorswitch/python'])),
        PATH=str(root/'hadoop-client/bin')+':'+os.environ['PATH'],CUDA_VISIBLE_DEVICES='0',
        TF_FORCE_GPU_ALLOW_GROWTH='true',TF_CPP_MIN_LOG_LEVEL='1',HOSTNAME=socket.gethostname(),
        TRINITY_GALAXY_SNAPSHOT_PROTO_FDSET=str(root/'python-generated-full/snapshot_fdset'),
        TS_CLIB_VARIANT='gpu',LD_LIBRARY_PATH=':'.join(map(str,[package,package/'python',
            root/'nccl-2.18/lib',build,root/'eps-native/dependencies/lib',
            pathlib.Path('/usr/local/cuda-12.2/lib64'),pathlib.Path('/usr/lib/x86_64-linux-gnu')])))
    return result


def configuration(work,run,peers,data,spec,local_api,addresses):
    config=dict(cmd='train',uuid=run,user='native-validation',
        user_model_py=str(ROOT/'examples/entropy_executor_model.py'),
        train_meta=dict(batch_size=4,list_size=1,shuffle_files=False,enable_metrics=False,
            purge_interval_offline=1000,batch_save_model_interval=1,
            optimizers=[dict(param_scope='embedding',opt_name='SparseLazyAdam',
                parameters=dict(learning_rate=.01,epsilon=.001,initial_range=.1,weight_bounds=None)),
                dict(param_scope='default',opt_name='DenseAdam',parameters=dict(learning_rate=.01))]),
        learners=dict(worker=dict(locations=list(range(peers)),num_threads=1)),local_shards=1,
        train_data=dict(type='offline',field_name=['sample'],
            data_seq_preparsed=[dict(id='epoch-0',path=str(data))]),
        pyfe_config=dict(feature_spec=str(spec)),number_of_partitions=1,
        batch_unfold_log_date_to_hour_opt=False,batch_unfold_log_date_to_hour=False,
        parse_num_parallel_calls=1,dataset_read_parallel=1,dataset_decode_inter_parallel=1,
        parse_prefetch=1,log_metric_interval=1,cold_layer_interval=1000,save_serving=False,
        save_model_path=str(work/'outputs'),serving_save_model_path=str(work/'outputs'),
        load_model_path='',blackhole_url=local_api,enable_controller_service=True,
        inter_op_parallelism_threads=4,intra_op_parallelism_threads=1,thread_pool_size=4,
        export_parallel=2,scale_sparse_grads=False)
    topology=dict(peers=addresses['native'],**{'ps+worker':list(range(peers))},
        controller_config=dict(worker_learner_services=[dict(address=e,num_threads=1)
            for e in addresses['learner']],gpu_learner_services=[],
            controller_service=dict(address=addresses['controller'])))
    return config,topology


def observation_sink(losses,posts):
    from http.server import BaseHTTPRequestHandler
    from prometheus_client.parser import text_string_to_metric_families
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body=self.rfile.read(int(self.headers.get('Content-Length','0')))
            authenticated=bool(self.headers.get('Authorization'))
            if self.path.startswith('/metrics/'):
                if authenticated:
                    self.send_error(403,'validation metrics must be anonymous');return
                for family in text_string_to_metric_families(body.decode()):
                    for sample in family.samples:
                        if sample.name=='loss' and sample.labels.get('loss_name')=='loss':
                            losses.append(dict(value=float(sample.value),labels=sample.labels))
            # Original state callbacks can supply their existing header. Record
            # only its presence, never the header or callback body contents.
            posts.append(dict(path=self.path,bytes=len(body),authenticated=authenticated))
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(b'{"code":0,"data":{},"message":"ok"}')
        def log_message(self,*_):pass
    return Handler


def execute(args):
    import numpy as np
    import pyarrow as pa
    import pyarrow.orc as orc
    import grpc
    from http.server import ThreadingHTTPServer
    from google.protobuf import text_format
    from tensorflow.core.framework import graph_pb2
    from proto import entropy_instance_list_pb2 as records
    from proto import controller_pb2,controller_pb2_grpc,learner_pb2,learner_pb2_grpc
    from checkpoint_restore import load_committed
    from managed_roles import ManagedRoles
    from role_node import write

    root,work,build=args.native_root,args.work,args.build
    work.mkdir(parents=True,exist_ok=False)
    run='original-executor-'+str(args.peers)+'-peers'
    result=dict(stage='preparing',original_peers=args.peers,physical_gpus=1,
        external_parameter_shards=2,full_framework_training_verified=False)
    def save():write(args.out,result)
    def control(action,cfg):
        path=work/('control-'+action+'.json');write(path,cfg)
        process=subprocess.run([str(build/'eps_training_control'),'--config',str(path),
            '--action',action],capture_output=True,text=True,check=True,timeout=30)
        return json.loads(process.stdout)
    losses=[];posts=[]
    receiver=ThreadingHTTPServer(('127.0.0.1',0),observation_sink(losses,posts))
    thread=threading.Thread(target=receiver.serve_forever,daemon=True);thread.start()
    local_api='http://127.0.0.1:'+str(receiver.server_port)
    (work/'dataset').mkdir()
    for part in range(args.peers):
        binary=[]
        for local in range(16):
            index=part*16+local
            value=records.EntropyInstanceList();row=value.entropy_instance.add()
            row.labels.value.append(index%2);row.importance=1;row.mark=('row-'+str(index)).encode()
            row.feature.add().int_values.value.extend([index%8,(index+1)%8])
            binary.append(value.SerializeToString())
        orc.write_table(pa.table({'sample':pa.array(binary,type=pa.binary())}),work/f'dataset/part-{part}.orc')
    spec=work/'features.json';write(spec,dict(entity=dict(type='tf.int64',size=-1)))
    addresses=dict(native=[port() for _ in range(args.peers)],learner=[port() for _ in range(args.peers)],
        controller=port(),agents=[port() for _ in range(args.peers)])
    config,topology=configuration(work,run,args.peers,work/'dataset/*.orc',spec,local_api,addresses)
    write(work/'original-config.json',config);write(work/'original-peers.json',topology)
    node=dict(local=True,source=str(ROOT),python=str(args.python),root=str(work),
        module_dir=str(build/'python'),bin_dir=str(build))
    role_env=environment(root,build)
    scratch=[];success=False
    try:
        for peer in range(args.peers):
            path=pathlib.Path('/dense_ndarray.'+str(peer)+'.npy')
            if path.exists():raise ValueError('original export scratch already exists: '+str(path))
            subprocess.run(['sudo','-n','install','-o',str(os.getuid()),'-g',str(os.getgid()),
                '-m','0600','/dev/null',str(path)],check=True)
            scratch.append(path)
        with ManagedRoles([node],run) as roles:
            try:
                roles.ensure(0,'progress',[build/'eps_training_coordinator'],dict(schema_version=1,run=run,
                    listen='127.0.0.1:0',workers=args.peers,epoch=1,consistency='eventual',max_delay=0),
                    environment=role_env)
                progress=roles.ready(0,'progress');endpoints=[]
                agents=[dict(id=i,endpoint=e) for i,e in enumerate(addresses['agents'])]
                for shard in range(2):
                    name='parameter-'+str(shard)
                    roles.ensure(0,name,[build/'eps_server'],dict(schema_version=1,run_id=run,
                        listen='127.0.0.1:0',partition=dict(shard_id=shard,shard_count=2,epoch=1),tables=[],
                        agents=agents,cache=dict(enabled=True,admission='frequency',decision_window_requests=1,
                            minimum_accesses=1,reserve_slots=1),access=dict(policy='transfer_unique',agent_id=0),
                        snapshot_directory=str(work/name/'snapshots')),environment=role_env)
                    endpoints.append(roles.ready(0,name))
                children=[]
                for peer in range(args.peers):
                    directory=work/('peer-'+str(peer));directory.mkdir()
                    adapter=directory/'parameter-config.json'
                    write(adapter,dict(schema_version=1,run=run,endpoints=endpoints,
                        tables=dict(entities=dict(id=3,seed=42)),backend='native',training=dict(coordinator=progress),
                        native=dict(device='/GPU:0',op_library=str(args.op_library),
                            runtime_library=str(build/'libeps_runtime.so'),cuda=dict(agent=peer,device=0,
                                listen=addresses['agents'][peer],state_bytes=1<<20,rows_per_table=64))))
                    command=[args.python,'-m','entropy_launch','--parameter-config',adapter,'--metrics-gateway',local_api,
                        '--entry-script',root/'native/entropy/tensorswitch/python/main.py','--','--role=worker',
                        '--real_rank='+str(peer),'--peers='+json.dumps(topology),'--config_str='+json.dumps(config),
                        '--tf_config_for_current_gpu={}','--log_path='+str(directory),'--deterministic=true']
                    name='peer-'+str(peer);children.append(name)
                    roles.ensure(0,name,command,environment=role_env)
                pings={};observation=time.monotonic()+args.observation_window
                result.update(stage='original_executor_running',run=run,addresses=addresses,
                    role_handles={name:roles.status(0,name) for name in roles.roles});save()
                while True:
                    states={name:roles.status(0,name) for name in children}
                    for name,state in states.items():
                        if 'result' in state and state['result']['exit_code']:
                            raise RuntimeError(name+' failed: '+roles.output(0,name,'stderr')[-4000:])
                        if state['state']=='exited' and 'result' not in state:
                            raise RuntimeError(name+' exited without a completion record')
                    services=[('controller',addresses['controller'],controller_pb2_grpc.EntropyControllerStub,
                        controller_pb2.PingRequest),*[(f'learner-{i}',endpoint,learner_pb2_grpc.LearnerStub,
                            learner_pb2.PingRequest) for i,endpoint in enumerate(addresses['learner'])]]
                    for name,endpoint,stub,request in services:
                        if pings.get(name):continue
                        try:
                            with grpc.insecure_channel(endpoint) as channel:
                                assert stub(channel).Ping(request(message='executor-validation'),timeout=.2).message=='executor-validation'
                            pings[name]=True
                        except grpc.RpcError:pass
                    result.update(pings=pings,peer_states=states,losses=losses);save()
                    if all('result' in state for state in states.values()):break
                    if time.monotonic()>observation:
                        result['stage']='observation_window_elapsed_roles_retained';save()
                        observation=time.monotonic()+args.observation_window
                    time.sleep(.2)
                if len(pings)!=args.peers+1:raise AssertionError('original control services were not all observed')
                expected=4*args.peers
                if len(losses)!=expected or not np.isfinite([x['value'] for x in losses]).all():
                    raise AssertionError('original model loss count/values differ from the ORC input')
                progress_state=control('stats',dict(run=run,coordinator=progress))
                if (len(progress_state['completed'])!=args.peers or not all(n>0 for n in progress_state['completed'])
                        or sum(progress_state['completed'])!=expected or progress_state['completed_tasks']!=expected
                        or any(progress_state[k] for k in ['active','waiting','failed'])):
                    raise AssertionError('original task admission/completion did not drain')
                checkpoint=work/'outputs/ckpt';committed=load_committed(checkpoint,checkpoint_id='ckpt')
                saved=committed['training_progress']
                if (len(committed['manifest']['files'])!=2 or not saved or saved['source_run']!=run
                        or len(saved['completed'])!=args.peers or sum(saved['completed'])>expected):
                    raise AssertionError('original checkpoint lost sparse partitions or task progress')
                dense={}
                for path in sorted((checkpoint/'dense_ndarray').glob('*.npy')):
                    tensors=np.load(path,allow_pickle=True).item()
                    if not all(np.isfinite(value).all() for value in tensors.values()):
                        raise AssertionError('original DenseAdam checkpoint has nonfinite values')
                    dense[path.name]=dict(sha256=digest(path),tensors={name:list(v.shape) for name,v in tensors.items()})
                if not dense or not any('Adam' in name for file in dense.values() for name in file['tensors']):
                    raise AssertionError('original dense optimizer state was not saved')
                native_dense={path.name:dict(bytes=path.stat().st_size,sha256=digest(path))
                    for path in checkpoint.glob('dense_table*') if path.is_file()}
                if not native_dense or any(not item['bytes'] for item in native_dense.values()):
                    raise AssertionError('original native dense checkpoint is absent or empty')
                auxiliary_feeders={path.name:dict(bytes=path.stat().st_size,sha256=digest(path))
                    for path in checkpoint.glob('icb_feeder_ckpt*') if path.is_file()}
                graphs={}
                for peer in range(args.peers):
                    paths=list((work/f'peer-{peer}').glob('train_graph_def.worker.*.txt'))
                    if len(paths)!=1:raise AssertionError('original peer graph was not exported')
                    graph=graph_pb2.GraphDef();text_format.Parse(paths[0].read_text(),graph)
                    nodes=[node.name for node in graph.node if node.op=='MatMul' and node.device.endswith('GPU:0')]
                    if len(nodes)<2:raise AssertionError('original peer lost GPU model placement')
                    graphs[str(peer)]=nodes
                state=control('inspect',dict(run=run,coordinator=progress,shards=endpoints,keys=[]))
                if any(s['updates']!=expected or s['live_tickets'] for s in state['shard_stats']):
                    raise AssertionError('external sparse updates did not follow original model tasks')
                source=root/'native/entropy/tensorswitch/python'
                result.update(stage='original_executor_passed',original_main_entrypoint=True,original_executor=True,
                    original_native_cluster=True,original_file_feeder=True,original_dense_adam=True,
                    original_controller_startup=True,original_runner_checkpoint=True,gpu_model=True,
                    original_dataset_rows=16*args.peers,original_dataset_batches=expected,
                    training_task_progress=progress_state,checkpoint_training_progress=saved,
                    checkpoint_manifest=committed['manifest'],dense_exports=dense,
                    native_dense_checkpoints=native_dense,
                    icb_auxiliary_checkpoints=auxiliary_feeders,
                    gpu_matmul_nodes=graphs,
                    shard_stats=state['shard_stats'],http_posts=posts,losses=losses,
                    source_sha256={str(p.relative_to(source)):digest(p) for p in [source/'main.py',
                        source/'tensorswitch/executor.py',source/'tensorswitch/train.py']},
                    scope='full original application and native peer execution on one physical GPU; joint restore/failure replay are separate acceptance')
                success=True
            except BaseException as error:
                result.update(stage='failed',error=str(error));raise
            finally:
                result['cleanup']=roles.cleanup()
                result['cleanup_verified']=all(x['owned_processes_gone'] for x in result['cleanup'])
                save()
                if not result['cleanup_verified']:raise RuntimeError('original Executor roles remain live')
    finally:
        for path in scratch:subprocess.run(['sudo','-n','rm','--',str(path)],check=True)
        receiver.shutdown();receiver.server_close();thread.join(timeout=5)
    if success:print(json.dumps({k:result[k] for k in ['stage','original_peers','original_dataset_batches','cleanup_verified']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root',type=pathlib.Path,required=True)
    parser.add_argument('--build',type=pathlib.Path,required=True)
    parser.add_argument('--op-library',type=pathlib.Path,required=True)
    parser.add_argument('--python',type=pathlib.Path)
    parser.add_argument('--peers',type=int,choices=[1,2],default=2)
    parser.add_argument('--work',type=pathlib.Path,required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True)
    parser.add_argument('--observation-window',type=int,default=600)
    parser.add_argument('--_execute',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    for key in ['native_root','build','op_library','work','out']:
        setattr(args,key,getattr(args,key).resolve())
    args.python=(args.python or args.native_root/'original-python/bin/python').absolute()
    if not args._execute:
        if args.work.exists() or args.out.exists():parser.error('work and output must be new paths')
        if not args.out.parent.is_dir():parser.error('output parent directory must exist')
        if args.observation_window<10:parser.error('observation window must be at least ten seconds')
        for path in [args.python,args.op_library,args.build/'libeps_runtime.so',
                     args.native_root/'native/entropy/tensorswitch/python/main.py']:
            if not path.is_file():parser.error('required native environment input is absent: '+str(path))
        os.execve(args.python,[str(args.python),str(pathlib.Path(__file__).resolve()),*sys.argv[1:],'--_execute'],
            dict(os.environ,**environment(args.native_root,args.build)))
    execute(args)


if __name__=='__main__':main()
