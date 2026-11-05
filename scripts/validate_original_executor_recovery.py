"""Interrupt and restore complete original main/Executor training processes."""
import argparse
from collections import Counter
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

from validate_original_executor import ROOT,configuration,environment,observation_sink,port,digest


def execute(args):
    import pyarrow as pa
    import pyarrow.orc as orc
    from proto import entropy_instance_list_pb2 as records
    from checkpoint_restore import load_committed,stage_checkpoint
    from managed_roles import ManagedRoles
    from role_node import write
    from startup_gate import publish
    from http.server import ThreadingHTTPServer

    root,work,build=args.native_root,args.work,args.build
    work.mkdir(parents=True,exist_ok=False)
    report=dict(stage='preparing',original_peers=args.peers,physical_gpus=1,
        full_framework_recovery_verified=False,phases={},inflight_delay_ms=args.inflight_delay_ms,
        source_sha256={name:digest(ROOT/name) for name in ['scripts/validate_original_executor_recovery.py',
            'examples/entropy_recovery_entry.py','examples/entropy_executor_model.py']})
    def save():write(args.out,report)
    data=work/'dataset';data.mkdir()
    count=16 if args.peers==1 else 64
    for peer in range(args.peers):
        values=[]
        for index in range(peer*count,(peer+1)*count):
            value=records.EntropyInstanceList();row=value.entropy_instance.add()
            row.labels.value.append(index%2);row.importance=1;row.mark=('row-'+str(index)).encode()
            row.feature.add().int_values.value.extend([index%8,(index+1)%8])
            values.append(value.SerializeToString())
        orc.write_table(pa.table({'sample':pa.array(values,type=pa.binary())}),data/f'part-{peer}.orc')
    spec=work/'features.json';write(spec,dict(entity=dict(type='tf.int64',size=-1)))
    env=environment(root,build)
    cut=work/'cut/checkpoints/ckpt'
    def control(action,run,endpoint,directory):
        path=directory/('control-'+action+'.json');write(path,dict(run=run,coordinator=endpoint))
        value=subprocess.run([str(build/'eps_training_control'),'--config',str(path),'--action',action],
            text=True,capture_output=True,check=True,timeout=30)
        return json.loads(value.stdout)
    def phase(name,restore):
        directory=work/name;directory.mkdir()
        run=f'original-recovery-{name}-{args.peers}'
        value=dict(stage='starting',restored=restore);report['phases'][name]=value;report['stage']=name;save()
        losses=[];posts=[]
        receiver=ThreadingHTTPServer(('127.0.0.1',0),observation_sink(losses,posts))
        thread=threading.Thread(target=receiver.serve_forever,daemon=True);thread.start()
        api='http://127.0.0.1:'+str(receiver.server_port)
        addresses=dict(native=[port() for _ in range(args.peers)],learner=[port() for _ in range(args.peers)],
            controller=port(),agents=[port() for _ in range(args.peers)])
        config,topology=configuration(directory,run,args.peers,data/'*.orc',spec,api,addresses)
        if restore:
            config.update(train_from_begin=False,load_model_path=str(cut.parent))
        write(directory/'original-config.json',config);write(directory/'original-peers.json',topology)
        node=dict(local=True,source=str(ROOT),python=str(args.python),root=str(directory),
            module_dir=str(build/'python'),bin_dir=str(build))
        scratch=[]
        try:
            for peer in range(args.peers):
                path=pathlib.Path('/dense_ndarray.'+str(peer)+'.npy')
                if path.exists():raise ValueError('original dense export scratch already exists')
                subprocess.run(['sudo','-n','install','-o',str(os.getuid()),'-g',str(os.getgid()),
                    '-m','0600','/dev/null',str(path)],check=True);scratch.append(path)
            with ManagedRoles([node],run) as roles:
                try:
                    progress_config=dict(schema_version=1,run=run,listen='127.0.0.1:0',workers=args.peers,
                        epoch=1,consistency='eventual',max_delay=0)
                    restored=None
                    if restore:
                        restored=load_committed(cut,checkpoint_id='ckpt')
                        progress_config['restore']=restored['training_progress']
                    roles.ensure(0,'progress',[build/'eps_training_coordinator'],progress_config,environment=env)
                    progress=roles.ready(0,'progress')
                    endpoints=[port(),port()]
                    agents=[dict(id=i,endpoint=e) for i,e in enumerate(addresses['agents'])]
                    def start_parameters():
                        for shard in range(2):
                            command=[build/'eps_server']
                            if restore:
                                staged=stage_checkpoint(cut,directory/f'staged-{shard}',shard,checkpoint_id='ckpt')
                                command+=['--restore',staged['file']]
                            roles.ensure(0,'parameter-'+str(shard),command,dict(schema_version=1,run_id=run,
                                listen=endpoints[shard],partition=dict(shard_id=shard,shard_count=2,epoch=1),tables=[],
                                agents=agents,cache=dict(enabled=True,admission='frequency',decision_window_requests=1,
                                    minimum_accesses=1,reserve_slots=1),access=dict(policy='transfer_unique',agent_id=0),
                                snapshot_directory=str(directory/f'parameter-{shard}/snapshots')),
                                environment=env,config_position=1)
                            if roles.ready(0,'parameter-'+str(shard))!=endpoints[shard]:
                                raise AssertionError('restored parameter endpoint changed')
                    if not restore:start_parameters()
                    gates=[];children=[]
                    for peer in range(args.peers):
                        local=directory/f'peer-{peer}';local.mkdir()
                        parameter=local/'parameter-config.json'
                        training=dict(coordinator=progress)
                        if restore:training['start_iteration']=restored['training_progress']['completed'][peer]
                        write(parameter,dict(schema_version=1,run=run,endpoints=endpoints,
                            cache_agents=[agent for agent in agents if agent['id']!=peer],
                            tables=dict(entities=dict(id=3,seed=42)),backend='native',training=training,
                            input_checkpoint=dict(planner=str(build/'eps_input_checkpoint_tool'),restore=restore),
                            native=dict(device='/GPU:0',op_library=str(args.op_library),
                                runtime_library=str(build/'libeps_runtime.so'),cuda=dict(agent=peer,device=0,
                                listen=addresses['agents'][peer],state_bytes=1<<20,rows_per_table=64))))
                        settings=dict(peer=peer,events=str(directory/'events'),
                            original_entry=str(root/'native/entropy/tensorswitch/python/main.py'))
                        if restore:settings['restored_root']=str(directory/'restored-checkpoints')
                        else:settings.update(cut_after=2,exit_after=3 if args.uncommitted_replay else 2,
                            cut_root=str(directory/'checkpoints'))
                        settings_file=local/'validation-control.json';write(settings_file,settings)
                        command=[args.python,'-m','entropy_launch','--parameter-config',parameter,'--metrics-gateway',api]
                        if restore:
                            gate=dict(schema_version=1,ready=str(local/'owner-ready.json'),
                                release=str(local/'parameters-ready.json'),timeout_seconds=180)
                            gate_file=local/'startup-gate.json';write(gate_file,gate);gates.append(gate)
                            command+=['--startup-gate',gate_file]
                        command+=['--entry-script',ROOT/'examples/entropy_recovery_entry.py','--','--role=worker',
                            '--real_rank='+str(peer),'--peers='+json.dumps(topology),'--config_str='+json.dumps(config),
                            '--tf_config_for_current_gpu={}','--log_path='+str(local),'--deterministic=false']
                        role='peer-'+str(peer);children.append(role)
                        roles.ensure(0,role,command,environment=dict(env,EPS_VALIDATION_CONTROL=str(settings_file),
                            EPS_VALIDATION_SAMPLE_TRACE=str(local/'samples.jsonl'),
                            EPS_VALIDATION_INFLIGHT_BARRIER='1' if args.inflight_delay_ms and not restore else '0',
                            EPS_VALIDATION_INFLIGHT_DELAY_MS=str(args.inflight_delay_ms if peer==args.peers-1 else 0)))
                    if restore:
                        deadline=time.monotonic()+150
                        while not all(pathlib.Path(gate['ready']).is_file() for gate in gates):
                            if any(roles.status(0,child)['state']=='exited' for child in children):
                                raise RuntimeError('restored model owner exited before readiness')
                            if time.monotonic()>deadline:raise TimeoutError('restored GPU-owner startup deadline')
                            time.sleep(.1)
                        start_parameters()
                        for gate in gates:publish(gate['release'],json.loads(pathlib.Path(gate['ready']).read_text()))
                    value.update(stage='original_application_running');save()
                    deadline=time.monotonic()+600
                    while True:
                        states={child:roles.status(0,child) for child in children}
                        expected_code=0 if restore else 75
                        if (not restore and args.uncommitted_replay and
                                any(state.get('result',{}).get('exit_code')==75 for state in states.values())):
                            # One lost native peer can leave another in an RPC
                            # to its dense shard. Abort admission and stop the
                            # complete old job before rebuilding from the cut.
                            control('abort',run,progress,directory)
                            value['failure_stop']={child:roles.stop(0,child) for child in children}
                            if not all(item['owned_processes_gone'] for item in value['failure_stop'].values()):
                                raise RuntimeError('failed original job did not stop')
                            value.update(losses=losses,peer_states=states);save();break
                        for child,state in states.items():
                            if 'result' in state and state['result']['exit_code']!=expected_code:
                                raise RuntimeError(child+' unexpected exit: '+roles.output(0,child,'stderr')[-5500:])
                        value.update(losses=losses,peer_states=states);save()
                        if all('result' in state for state in states.values()):break
                        if time.monotonic()>deadline:
                            value['stage']='observation_window_elapsed_roles_retained';save();deadline=time.monotonic()+600
                        time.sleep(.2)
                    for peer in range(args.peers):
                        event='restore-committed' if restore else 'cut-committed'
                        if not (directory/'events'/f'{event}-{peer}.json').is_file():
                            raise AssertionError('original checkpoint boundary event is absent')
                    value.update(stage='checking_progress',progress=control('stats',run,progress,directory))
                    expected_failure=not restore and args.uncommitted_replay
                    if (value['progress']['waiting'] or value['progress']['failed']!=expected_failure or
                            (value['progress']['active'] and not expected_failure)):
                        raise AssertionError('restored training tasks did not drain')
                    value['stage']='passed'
                finally:
                    value['cleanup']=roles.cleanup();value['cleanup_verified']=all(x['owned_processes_gone'] for x in value['cleanup']);save()
                    if not value['cleanup_verified']:raise RuntimeError('original recovery roles remain live')
        finally:
            for path in scratch:subprocess.run(['sudo','-n','rm','--',str(path)],check=True)
            receiver.shutdown();receiver.server_close();thread.join(timeout=5)
    try:
        phase('cut',False)
        committed=load_committed(cut,checkpoint_id='ckpt')
        if committed['training_progress']['completed']!=[2]*args.peers:
            raise AssertionError('interruption did not occur at the requested training clocks')
        phase('restored',True)
        saved=work/'restored/restored-checkpoints/restored-start'
        initial=load_committed(saved,checkpoint_id='restored-start')
        if initial['training_progress']['completed']!=[2]*args.peers:
            raise AssertionError('restored checkpoint advanced before comparison')
        comparisons=[]
        for shard in range(2):
            name=f'parameter-shard-{shard}.bin'
            checked=subprocess.run([str(build/'eps_snapshot_compare'),str(cut/'external_parameters'/name),
                str(saved/'external_parameters'/name),str(shard),'2','1','3'],check=True,text=True,capture_output=True)
            comparisons.append(json.loads(checked.stdout))
        dense={}
        for path in cut.glob('dense_table*'):
            if path.is_file():
                if path.read_bytes()!=(saved/path.name).read_bytes():raise AssertionError('native dense optimizer restore differs')
                dense[path.name]=digest(path)
        if not dense:raise AssertionError('native dense optimizer reference is absent')
        marks=[];discarded=[];restored_marks=[]
        for phase_name in ['cut','restored']:
            for peer in range(args.peers):
                batches=[json.loads(line) for line in
                    (work/phase_name/f'peer-{peer}/samples.jsonl').read_text().splitlines()]
                if phase_name=='cut':
                    if not 2<=len(batches)<=(3 if args.uncommitted_replay else 2):
                        raise AssertionError('failed attempt did not stop at the requested batch')
                    for batch in batches[:2]:marks+=batch
                    for batch in batches[2:]:discarded+=batch
                else:
                    for batch in batches:marks+=batch;restored_marks+=batch
        if Counter(marks)!=Counter('row-'+str(i) for i in range(count*args.peers)):
            raise AssertionError('original restored input omitted or repeated samples')
        if sum(report['phases']['restored']['progress']['completed'])!=count*args.peers//4:
            raise AssertionError('restored task progress differs from consumed samples')
        if Counter(discarded)-Counter(restored_marks):
            raise AssertionError('post-checkpoint failed work was not replayed')
        if args.uncommitted_replay and not discarded:
            raise AssertionError('failure did not discard any post-checkpoint work')
        if args.inflight_delay_ms and not report['phases']['cut']['progress']['active']:
            raise AssertionError('fault did not leave an admitted unfinished training task')
        if args.reference_checkpoint:
            reference=json.loads(args.reference_report.read_text())
            actual=work/'restored/outputs/ckpt'
            if (reference.get('stage')!='original_executor_passed' or reference.get('original_peers')!=args.peers or
                    reference.get('original_dataset_rows')!=len(marks)):
                raise ValueError('uninterrupted reference scope differs from the recovery workload')
            reference_comparisons=[]
            for shard in range(2):
                name=f'parameter-shard-{shard}.bin'
                checked=subprocess.run([str(build/'eps_snapshot_compare'),
                    str(args.reference_checkpoint/'external_parameters'/name),
                    str(actual/'external_parameters'/name),str(shard),'2','1','3'],
                    check=True,text=True,capture_output=True)
                reference_comparisons.append(json.loads(checked.stdout))
            paths=[p for p in args.reference_checkpoint.glob('dense_table*') if p.is_file()]
            if not paths or any(p.read_bytes()!=(actual/p.name).read_bytes() for p in paths):
                raise AssertionError('continued native dense optimizer differs from uninterrupted training')
            losses=report['phases']['cut']['losses'][:2*args.peers]+report['phases']['restored']['losses']
            if len(losses)!=len(reference['losses']):raise AssertionError('continued loss count differs from reference')
            error=max(abs(x['value']-y['value']) for x,y in zip(losses,reference['losses']))
            if error>1e-6:raise AssertionError('continued loss trajectory differs from uninterrupted training')
            report['uninterrupted_equivalence']=dict(parameter_comparisons=reference_comparisons,
                native_dense_byte_identical=True,loss_max_absolute_error=error,
                reference_report_sha256=digest(args.reference_report))
        report.update(stage='original_executor_recovery_passed',full_framework_recovery_verified=True,
            parameter_comparisons=comparisons,native_dense_optimizer_sha256=dense,
            sample_count=len(marks),duplicate_samples=0,missing_samples=0,cut_completed=[2]*args.peers,
            attempted_sample_count=len(marks)+len(discarded),replayed_uncommitted_samples=len(discarded),
            scope='original main/Executor processes, mid-file abrupt exit, fresh shards/owners/coordinator, restored input and continued GPU training')
        save();print(json.dumps({k:report[k] for k in ['stage','original_peers','sample_count']}))
    except BaseException as error:report.update(stage='failed',error=str(error));save();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root',type=pathlib.Path,required=True)
    parser.add_argument('--build',type=pathlib.Path,required=True)
    parser.add_argument('--op-library',type=pathlib.Path,required=True)
    parser.add_argument('--python',type=pathlib.Path)
    parser.add_argument('--work',type=pathlib.Path,required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True)
    parser.add_argument('--peers',type=int,choices=[1,2],default=2)
    parser.add_argument('--uncommitted-replay',action='store_true',
        help='Execute another batch after the joint commit, lose it and replay from the committed cut')
    parser.add_argument('--inflight-delay-ms',type=int,default=0,
        help='Synchronize post-checkpoint Pulls and delay the final peer before completion')
    parser.add_argument('--reference-checkpoint',type=pathlib.Path)
    parser.add_argument('--reference-report',type=pathlib.Path)
    parser.add_argument('--_execute',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if bool(args.reference_checkpoint)!=bool(args.reference_report):
        parser.error('provide both uninterrupted reference checkpoint and report')
    if (not 0<=args.inflight_delay_ms<=2000 or
            (args.inflight_delay_ms and (not args.uncommitted_replay or args.peers<2))):
        parser.error('in-flight injection requires multiple peers, replay, and at most 2000 ms delay')
    if args.reference_checkpoint:
        args.reference_checkpoint=args.reference_checkpoint.resolve();args.reference_report=args.reference_report.resolve()
    for key in ['native_root','build','op_library','work','out']:setattr(args,key,getattr(args,key).resolve())
    args.python=(args.python or args.native_root/'original-python/bin/python').absolute()
    if not args._execute:
        if args.work.exists() or args.out.exists():parser.error('work and output must be new paths')
        os.execve(args.python,[str(args.python),str(pathlib.Path(__file__).resolve()),*sys.argv[1:],'--_execute'],
            dict(os.environ,**environment(args.native_root,args.build)))
    execute(args)


if __name__=='__main__':main()
