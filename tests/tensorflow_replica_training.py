"""Real model training survives lost ACK and unreplicated-primary failure."""
import argparse
import concurrent.futures
import copy
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from rpc_processes import ready, stop


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=pathlib.Path,required=True)
    p.add_argument('--out',type=pathlib.Path)
    a=p.parse_args();build=a.build.resolve();source=pathlib.Path(__file__).resolve().parents[1]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',TF_CPP_MIN_LOG_LEVEL='2')
    env['PYTHONPATH']=os.pathsep.join([str(build/'python'),str(source/'adapters/entropy'),env.get('PYTHONPATH','')])
    os.environ['PYTHONPATH']=env['PYTHONPATH']
    rows,dim,hidden,rounds,target=64,8,4,8,3
    packed=(dim*hidden+2*hidden+1+dim-1)//dim
    table=dict(id=1,dimension=dim,dtype='float32',
               optimizer=dict(kind='sparse_lazy_adam',learning_rate=.02,beta1=.9,beta2=.999,epsilon=1e-8,
                              initial_beta1_power=.9,initial_beta2_power=.999),
               initialization=dict(min=-.1,max=.1,seed=42))
    batches=[[[dict(keys=[packed+1,packed+2,packed+1],label=0.),dict(keys=[packed+3],label=1.)],
              [dict(keys=[packed+4],label=1.)],
              [dict(keys=[packed+5,packed+6],label=0.),dict(keys=[packed+3],label=1.)]],
             [[dict(keys=[packed+3,packed+4],label=1.)],
              [dict(keys=[packed+5],label=0.),dict(keys=[packed+6,packed+3],label=1.)],
              [dict(keys=[packed+1,packed+6],label=0.)]]]
    reports=[]
    with tempfile.TemporaryDirectory(prefix='eps-replicated-model-') as tmp:
        root=pathlib.Path(tmp)
        for mode in ('baseline','lost_ack','unreplicated'):
            run='replica-model-'+mode;children=[];logs=[];counter=0;gate=root/(mode+'.release')
            replica_configs=[];replica_processes=[];replica_addresses=[];primary_configs=[];primary_processes=[]
            events=[]

            def config_path(name,config):
                path=root/(run+'-'+name+'.json');path.write_text(json.dumps(config));return path

            def launch(binary,name,config,extra=()):
                path=config_path(name,config);log=open(root/(run+'-'+name+'.log'),'w+');logs.append(log)
                child=subprocess.Popen([str(binary),'--config',str(path),*map(str,extra)],env=env,
                                       stdout=subprocess.PIPE,stderr=log,text=True)
                children.append(child);return child

            def command(binary,settings,action,ok=True):
                nonlocal counter
                counter+=1;path=config_path('control-'+str(counter),settings)
                r=subprocess.run([str(build/binary),'--config',str(path),'--action',action],env=env,
                                 capture_output=True,text=True,timeout=15)
                if ok:
                    if r.returncode:raise RuntimeError(r.stderr)
                    return json.loads(r.stdout)
                assert r.returncode,(binary,action,r.stdout)
                return r.stderr.strip()

            def replica_control(home,action,**extra):
                return command('eps_replica_control',dict(replica_configs[home],replica_endpoint=replica_addresses[home],**extra),action)

            try:
                for home in range(2):
                    config=dict(schema_version=1,run_id=run,listen='127.0.0.1:0',
                                partition=dict(shard_id=home,shard_count=2,epoch=1),tables=[table],
                                synchronous_training=dict(workers=2,reduction='sample_mean',round_timeout_ms=20000))
                    if mode!='baseline':
                        config['replica_table_rows']={'1':rows}
                        replica_config=copy.deepcopy(config);replica_config['replica_directory']=str(root/(run+f'-backup-{home}'))
                        extra=['--test-publish-gate',gate,'--test-publish-round',target] if home==0 else []
                        backup=launch(build/'eps_replica',f'backup-{home}',replica_config,extra)
                        replica_configs.append(replica_config);replica_processes.append(backup);replica_addresses.append(ready(backup))
                        if home==0:
                            duplicate=subprocess.run([str(build/'eps_replica'),'--config',str(config_path('duplicate-backup',replica_config))],
                                                     env=env,capture_output=True,text=True,timeout=5)
                            assert duplicate.returncode and 'replica directory is already in use' in duplicate.stderr
                            events.append(dict(kind='duplicate_replica_directory_rejected'))
                        grant=replica_control(home,'fence',expected_term=0,primary=f'primary-{home}')
                        assert grant['term']==1 and grant['next_round']==0
                        config['synchronous_training']['replica']=dict(endpoint=replica_addresses[home],primary=f'primary-{home}',
                            term=1,local_directory=str(root/(run+f'-primary-{home}')),recover=False)
                    config['snapshot_directory']=str(root/(run+f'-snapshots-{home}'))
                    primary_configs.append(config)
                    primary=launch(build/'eps_server',f'ps-{home}',config);primary_processes.append(primary)
                addresses=[ready(primary) for primary in primary_processes]
                coordinator=ready(launch(build/'eps_training_coordinator','coordinator',dict(
                    schema_version=1,run=run,workers=2,epoch=1,consistency='sequential',max_delay=0)))
                settings=dict(run=run,epoch=1,coordinator=coordinator,shards=addresses,keys=list(range(rows)),synchronous=True)
                if mode=='baseline':
                    preparation=launch(build/'eps_training_prepare','prepare',dict(settings,rows=rows,dimension=dim,resident=False))
                    output,_=preparation.communicate(timeout=45);assert preparation.returncode==0,output
                workers=[]
                for worker in range(2):
                    cfg=dict(settings,worker=worker,table=table,dimension=dim,hidden=hidden,iterations=rounds,
                             batches=batches[worker],device='/CPU:0',gradient_retry_ms=20000 if mode!='baseline' else 0)
                    path=config_path('worker-'+str(worker),cfg);log=open(root/(run+f'-worker-{worker}.log'),'w+');logs.append(log)
                    child=subprocess.Popen([sys.executable,str(source/'examples/tensorflow_training_worker.py'),'--config',str(path)],
                                           env=env,stdout=subprocess.PIPE,stderr=log,text=True)
                    children.append(child);workers.append(child)
                if mode!='baseline':
                    deadline=time.monotonic()+45
                    marker=pathlib.Path(str(gate)+'.waiting')
                    while not marker.exists():
                        assert time.monotonic()<deadline,'replica publication gate was never reached'
                        assert all(w.poll() is None for w in workers),'worker exited before fault point'
                        time.sleep(.02)
                    assert primary_processes[0].poll() is None and replica_processes[0].poll() is None
                    frozen=command('eps_training_control',dict(settings,keys=[]),'inspect')
                    assert frozen['completed']==[target,target] and frozen['active']==2,frozen
                    assert frozen['shard_stats'][0]['updates']==target+1,frozen
                    assert frozen['shard_stats'][0]['synchronous']['applying'],frozen
                    backup_dir=pathlib.Path(replica_configs[0]['replica_directory'])
                    local_dir=pathlib.Path(primary_configs[0]['synchronous_training']['replica']['local_directory'])/'term-1'
                    assert (local_dir/f'round-{target}.epsr').exists()
                    assert not (backup_dir/f'round-{target}.epsr').exists()
                    old_pid=primary_processes[0].pid;primary_processes[0].kill();primary_processes[0].wait(timeout=5)
                    if mode=='unreplicated':
                        old_backup=replica_processes[0].pid;replica_processes[0].kill();replica_processes[0].wait(timeout=5)
                        new_config=dict(replica_configs[0],listen=replica_addresses[0])
                        backup=launch(build/'eps_replica','backup-0-restarted',new_config)
                        assert ready(backup)==replica_addresses[0];replica_processes[0]=backup
                        recovered=replica_control(0,'head')
                        assert recovered['next_round']==target,recovered
                        events.append(dict(kind='standby_process_loss_before_publication',old_pid=old_backup,new_pid=backup.pid))
                    else:
                        gate.touch()
                        committed=replica_control(0,'head')
                        assert committed['next_round']==target+1,committed
                    time.sleep(.4)  # Observe a real outage so workers exercise retained-gradient retry.
                    expected=target if mode=='unreplicated' else target+1
                    grant=replica_control(0,'fence',expected_term=1,primary='replacement-0')
                    assert grant['term']==2 and grant['next_round']==expected,grant
                    replacement=copy.deepcopy(primary_configs[0]);replacement['listen']=addresses[0]
                    replacement['synchronous_training']['replica'].update(primary='replacement-0',term=2,recover=True)
                    primary=launch(build/'eps_server','ps-0-replaced',replacement)
                    assert ready(primary)==addresses[0];primary_processes[0]=primary
                    events.append(dict(kind='primary_process_loss',old_pid=old_pid,new_pid=primary.pid,
                                       recovered_next_round=expected,grant=grant,pre_fault=frozen))
                outputs=[]
                for child in workers:
                    output,_=child.communicate(timeout=90)
                    if child.returncode:raise RuntimeError(f'model worker exited with {child.returncode}')
                    outputs.append(json.loads(output))
                state=command('eps_training_control',settings,'inspect')
                assert state['completed']==[rounds,rounds] and state['completed_tasks']==rounds*2 and not state['active'] and not state['failed'],state
                assert len(state['rows'])==rows
                assert all(s['live_tickets']==0 and s['synchronous']['next_round']==rounds and not s['synchronous']['failed'] for s in state['shard_stats'])
                result=dict(mode=mode,workers=outputs,state=state,events=events)
                if mode!='baseline':
                    assert sum(w['gradient_retry_count'] for w in outputs)>0,outputs
                    assert sum(s['fallback_requests'] for w in outputs for s in w['lookup_transport'])>0
                    heads=[replica_control(home,'head') for home in range(2)]
                    assert all(h['next_round']==rounds for h in heads),heads
                    record=replica_control(0,'record',round=rounds-1)
                    fronts={r['learner']:r['next'] for r in record['requests']}
                    assert fronts[0]==fronts[1]==fronts[1023]==rounds,fronts
                    result.update(replica_heads=heads,last_record=record)
                    # Fence the still-running replaced role and submit only to that
                    # shard: its old term must not alter either shard's model.
                    fenced=replica_control(0,'fence',expected_term=2,primary='future-primary')
                    assert fenced['term']==3 and fenced['next_round']==rounds
                    sys.path.insert(0,str(source/'adapters/entropy'))
                    from eps_backend import Backend
                    import numpy as np
                    probes=[Backend([addresses[0]],run,w) for w in range(2)]
                    try:
                        for probe in probes:probe.configure_training(coordinator)
                        def late(probe):
                            try:
                                probe.submit_synchronous(rounds,1,[dict(table=1,keys=np.array([0],np.uint64),
                                                                      values=np.zeros((1,dim),np.float32))])
                            except RuntimeError as error:return str(error)
                            raise AssertionError('old primary term acknowledged a new update')
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                            rejected=list(pool.map(late,probes))
                        assert all('fenced' in reason for reason in rejected),rejected
                    finally:
                        for probe in probes:probe.close()
                    unchanged=command('eps_training_control',settings,'inspect')
                    assert unchanged['rows']==state['rows'] and unchanged['completed']==state['completed']
                    assert unchanged['shard_stats'][0]['synchronous']['failed']
                    assert replica_control(0,'head')['next_round']==rounds
                    result['old_primary_fencing']=dict(grant=fenced,rejections=rejected,unchanged_state=True)
                reports.append(result)
            except Exception:
                for log in logs:log.flush();log.seek(0);print(log.read()[-6000:],file=sys.stderr)
                raise
            finally:
                gate.touch()
                for child in reversed(children):stop(child)
                assert all(child.poll() is not None for child in children)
                for log in logs:log.close()
    baseline={row['key']:row for row in reports[0]['state']['rows']}
    maximum=0.;loss_error=0.
    for report in reports[1:]:
        for row in report['state']['rows']:
            expected=baseline[row['key']];assert row['row_updates']==expected['row_updates']
            for field in ('weights','first_moment','second_moment'):
                for got,want in zip(row[field],expected[field]):
                    assert math.isfinite(got);maximum=max(maximum,abs(got-want))
        for worker,expected in zip(report['workers'],reports[0]['workers']):
            assert worker['input_state']==expected['input_state']
            loss_error=max(loss_error,max(abs(got-want) for got,want in zip(worker['losses'],expected['losses'])))
    assert maximum==0. and loss_error==0.,(maximum,loss_error)
    result=dict(test='tensorflow_replicated_primary_recovery',device='CPU',workers=2,shards=2,iterations=rounds,
                rows=rows,dimension=dim,fault_round=target,max_state_absolute_error=maximum,max_loss_absolute_error=loss_error,
                cleanup_verified=True,reports=reports)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reports'}))


if __name__=='__main__':main()
