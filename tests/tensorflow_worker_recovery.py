"""Kill model workers/coordinator during reads and partially committed updates."""
import argparse
import copy
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
from rpc_processes import ready, stop


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path)
    p.add_argument('--owners',choices=['none','host','cuda'],default='none')
    p.add_argument('--devices',default='0',help='Physical CUDA ordinals visible to this recovery test')
    p.add_argument('--handoff',action='store_true',help='Fetch journals through node subprocesses and stage them at replacement workers')
    a=p.parse_args();build=a.build.resolve();source=pathlib.Path(__file__).resolve().parents[1]
    devices=[int(v) for v in a.devices.split(',')]
    if not 1<=len(devices)<=2 or len(set(devices))!=len(devices) or min(devices)<0:
        p.error('provide one or two distinct nonnegative CUDA ordinals')
    gpu=a.owners=='cuda';resident=a.owners!='none'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.devices if gpu else '',TF_CPP_MIN_LOG_LEVEL='2')
    env['PYTHONPATH']=os.pathsep.join([str(build/'python'),str(source/'adapters/entropy'),env.get('PYTHONPATH','')])
    sys.path.insert(0,str(source/'adapters/entropy'))
    from training_journal import WorkerJournal, identity
    from training_recovery import owner_run
    if a.handoff:
        sys.path.insert(0,str(source/'scripts'))
        from transfer_worker_journals import transfer
        handoff_node=dict(local=True,source=str(source),python=sys.executable)
    hardware=subprocess.check_output(['nvidia-smi','--id='+a.devices,'--query-gpu=uuid,name,memory.total','--format=csv,noheader'],text=True) if gpu else None
    table=dict(id=1,dimension=8,dtype='float32',optimizer=dict(kind='sparse_lazy_adam',learning_rate=.02,beta1=.9,beta2=.999,epsilon=1e-8,
          initial_beta1_power=.9,initial_beta2_power=.999),initialization=dict(min=-.1,max=.1,seed=42))
    batches=[[[dict(keys=[7,8,7],label=0.),dict(keys=[9],label=1.)],[dict(keys=[10],label=1.)],[dict(keys=[11,12],label=0.),dict(keys=[9],label=1.)]],
             [[dict(keys=[9,10],label=1.)],[dict(keys=[11],label=0.),dict(keys=[12,9],label=1.)],[dict(keys=[7,12],label=0.)]]]
    rounds,target=8,3;reports=[]
    with tempfile.TemporaryDirectory(prefix='eps-worker-recovery-') as tmp:
        root=pathlib.Path(tmp)
        for mode in ('baseline','after_pull','partial_round','after_update'):
            run='worker-recovery-'+mode;children=[];logs=[];worker_groups=[];serial=0
            primaries=[];primary_configs=[];replicas=[];replica_configs=[];replica_addresses=[]
            owners=[];agents=[]
            gate=root/(mode+'.gate');worker_configs=[];events=[]
            def save(name,cfg):
                nonlocal serial
                serial+=1;path=root/f'{run}-{name}-{serial}.json';path.write_text(json.dumps(cfg));return path
            def launch(binary,name,cfg,extra=(),worker=False):
                path=save(name,cfg);log=open(root/f'{run}-{name}-{serial}.log','w+');logs.append(log)
                cmd=[sys.executable,str(binary)] if worker else [str(binary)]
                child=subprocess.Popen([*cmd,'--config',str(path),*map(str,extra)],env=env,stdout=subprocess.PIPE,stderr=log,text=True,start_new_session=worker)
                children.append(child)
                if worker:worker_groups.append(child)
                return child
            def command(binary,cfg,action):
                r=subprocess.run([str(build/binary),'--config',str(save('control',cfg)),'--action',action],env=env,capture_output=True,text=True,timeout=15)
                if r.returncode:raise RuntimeError(r.stderr)
                return json.loads(r.stdout)
            def owner(home,term,primary,endpoint='127.0.0.1:0'):
                nonlocal serial
                serial+=1;scope=owner_run(run,1,home,term,primary)
                log=open(root/f'{run}-owner-{home}-{term}.log','w+');logs.append(log)
                child=subprocess.Popen([str(build/'eps_cache_agent'),'--run',scope,'--training-run',run,
                    '--listen',endpoint,'--agent',str(home),'--backend','cuda' if gpu else 'host',
                    '--device',str(home%len(devices)),
                    '--rows-per-table','32','--state-bytes','4096'],env=env,stdout=subprocess.PIPE,stderr=log,text=True)
                children.append(child)
                return child,dict(id=home,endpoint=ready(child),owner_run=scope)
            def prepare(cfg,ok=True):
                r=subprocess.run([str(build/'eps_training_prepare'),'--config',str(save('prepare',cfg))],
                    env=env,capture_output=True,text=True,timeout=30)
                if not ok:
                    assert r.returncode,'invalid owner preparation accepted'
                    return r.stderr.strip()[-1500:]
                if r.returncode:raise RuntimeError(r.stderr)
                return json.loads(r.stdout)
            def replica(home,action,**extra):
                return command('eps_replica_control',dict(replica_configs[home],replica_endpoint=replica_addresses[home],**extra),action)
            def recover_tool(job,action,ok=True):
                out=root/f'{run}-{action}-{serial}.json'
                r=subprocess.run([sys.executable,str(source/'scripts/recover_training.py'),'--config',str(save('job',job)),
                    '--build',str(build),'--action',action,'--out',str(out)],env=env,capture_output=True,text=True,timeout=30)
                if not ok:
                    assert r.returncode,'invalid recovery was accepted'
                    return r.stderr.strip()[-1500:]
                if r.returncode:raise RuntimeError(r.stderr)
                return json.loads(out.read_text())
            def kill_worker(child):
                if child.poll() is None:
                    assert os.getpgid(child.pid)==child.pid
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
            try:
                for home in range(2):
                    cfg=dict(schema_version=1,run_id=run,listen='127.0.0.1:0',partition=dict(shard_id=home,shard_count=2,epoch=1),
                             tables=[table],replica_table_rows={'1':64},synchronous_training=dict(workers=2,reduction='sample_mean',round_timeout_ms=25000))
                    if resident:
                        child,agent=owner(home,1,f'primary-{home}');owners.append(child);agents.append(agent)
                        cfg.update(cache=dict(enabled=True),agents=[agent])
                    rcfg=dict(cfg,replica_directory=str(root/f'{run}-replica-{home}'))
                    extra=['--test-publish-gate',gate,'--test-publish-round',target] if mode=='partial_round' and home==1 else []
                    child=launch(build/'eps_replica',f'replica-{home}',rcfg,extra);replicas.append(child);replica_configs.append(rcfg);replica_addresses.append(ready(child))
                    replica(home,'fence',expected_term=0,primary=f'primary-{home}')
                    pcfg=copy.deepcopy(cfg);pcfg['snapshot_directory']=str(root/f'{run}-snapshots-{home}')
                    pcfg['synchronous_training']['replica']=dict(endpoint=replica_addresses[home],primary=f'primary-{home}',term=1,
                        local_directory=str(root/f'{run}-primary-{home}'),recover=False)
                    primary_configs.append(pcfg);primaries.append(launch(build/'eps_server',f'ps-{home}',pcfg))
                addresses=[ready(p) for p in primaries]
                if resident:
                    prepared=prepare(dict(run=run,shards=addresses,rows=64,dimension=8,agents=agents))
                    assert sum(s['resident_rows'] for s in prepared['shards'])==64
                coordinator=launch(build/'eps_training_coordinator','coordinator',dict(schema_version=1,run=run,epoch=1,workers=2,consistency='sequential',max_delay=0))
                coordinator_address=ready(coordinator)
                settings=dict(run=run,epoch=1,coordinator=coordinator_address,shards=addresses,keys=list(range(64)),synchronous=True)
                workers=[]
                for w in range(2):
                    cfg=dict(settings,worker=w,dimension=8,hidden=4,iterations=rounds,batches=batches[w],table=table,device='/CPU:0',
                             journal_directory=str(root/f'{run}-worker-{w}'),journal_workers=2)
                    if resident:cfg.update(agents=copy.deepcopy(agents),device='/GPU:0' if gpu else '/CPU:0')
                    if w==0 and mode in ('after_pull','after_update'):
                        cfg['test_pause']=dict(stage=mode,iteration=target,path=str(gate))
                    worker_configs.append(cfg);workers.append(launch(source/'examples/tensorflow_training_worker.py',f'worker-{w}',cfg,worker=True))
                if mode!='baseline':
                    marker=pathlib.Path(str(gate)+'.waiting');deadline=time.monotonic()+45
                    while not marker.exists():
                        assert time.monotonic()<deadline and all(w.poll() is None for w in workers),'fault point not reached'
                        time.sleep(.02)
                    if mode=='partial_round':
                        while not (pathlib.Path(replica_configs[0]['replica_directory'])/f'round-{target}.epsr').exists():
                            assert time.monotonic()<deadline;time.sleep(.02)
                    if mode=='after_pull':
                        while not (pathlib.Path(worker_configs[1]['journal_directory'])/f'round-{target}.epsw').exists():
                            assert time.monotonic()<deadline;time.sleep(.02)
                    before=command('eps_training_control',dict(settings,keys=[]),'inspect')
                    if mode=='after_pull':assert sum(s['live_tickets'] for s in before['shard_stats'])>0
                    if mode=='partial_round':assert before['completed']==[target,target] and before['active']==2
                    if mode=='after_update':assert all(s['synchronous']['next_round']==target+1 for s in before['shard_stats'])
                    for child in workers:kill_worker(child)
                    coordinator.kill();coordinator.wait(timeout=5)
                    for child in primaries:child.kill();child.wait(timeout=5)
                    old_owner_pids=[p.pid for p in owners]
                    for child in owners:child.kill();child.wait(timeout=5)
                    events.append(dict(fault=mode,old_worker_pids=[w.pid for w in workers],old_coordinator_pid=coordinator.pid,
                                       old_primary_pids=[p.pid for p in primaries],old_owner_pids=old_owner_pids,pre_fault=before))
                    if mode=='partial_round':
                        replicas[1].kill();replicas[1].wait(timeout=5)
                        replicas[1]=launch(build/'eps_replica','replica-restarted',dict(replica_configs[1],listen=replica_addresses[1]))
                        assert ready(replicas[1])==replica_addresses[1]
                    for home in range(2):
                        grant=replica(home,'fence',expected_term=1,primary=f'replacement-{home}')
                        assert grant['term']==2
                        cfg=primary_configs[home];cfg['listen']=addresses[home]
                        cfg['synchronous_training']['replica'].update(primary=f'replacement-{home}',term=2,recover=True)
                        if resident:
                            child,agent=owner(home,2,f'replacement-{home}',agents[home]['endpoint'])
                            owners[home]=child;agents[home]=agent;cfg['agents']=[agent]
                            cfg['synchronous_training']['replica']['start_paused']=True
                        primaries[home]=launch(build/'eps_server',f'replacement-{home}',cfg)
                    assert [ready(p) for p in primaries]==addresses
                    if a.handoff:
                        fetched=[]
                        for w,cfg in enumerate(worker_configs):
                            old=copy.deepcopy(cfg)
                            result=transfer(cfg,handoff_node,handoff_node,root/f'{run}-manager-{w}',run+'-fetch')
                            worker_configs[w]=copy.deepcopy(result['config']);fetched.append(result)
                            try:
                                with WorkerJournal(old['journal_directory'],identity(old)):pass
                            except ValueError:pass
                            else:raise AssertionError('sealed source worker was allowed to restart')
                        events.append(dict(journal_fetch=fetched))
                    job=dict(run=run,epoch=1,end_round=rounds,coordinator_endpoint=coordinator_address,workers=worker_configs,
                             shards=[dict(endpoint=address,config=cfg) for address,cfg in zip(addresses,primary_configs)])
                    planned=recover_tool(job,'plan')
                    expected=target if mode=='after_pull' else target+1
                    assert planned['next_round']==expected and planned['replay_required']==(mode=='partial_round'),planned
                    coordinator=launch(build/'eps_training_coordinator','coordinator-restarted',dict(planned['coordinator'],listen=coordinator_address))
                    assert ready(coordinator)==coordinator_address
                    negative=[]
                    changed=copy.deepcopy(job);changed['workers'][0]['batches'][0][0]['label']=.25
                    negative.append(dict(case='changed_input',reason=recover_tool(changed,'reconcile',False)))
                    changed=copy.deepcopy(job);changed['end_round']=expected-1
                    negative.append(dict(case='end_before_durable_round',reason=recover_tool(changed,'reconcile',False)))
                    changed=copy.deepcopy(job)
                    if resident:changed['shards'][0]['config']['agents'][0].pop('owner_run')
                    else:changed['shards'][0]['config']['cache']=dict(enabled=True)
                    negative.append(dict(case='unreconciled_gpu_owner',reason=recover_tool(changed,'reconcile',False)))
                    with WorkerJournal(worker_configs[0]['journal_directory'],identity(worker_configs[0])):
                        negative.append(dict(case='live_worker_journal_lock',reason=recover_tool(job,'reconcile',False)))
                    if mode=='partial_round':
                        original=pathlib.Path(worker_configs[0]['journal_directory'])/f'round-{target}.epsw'
                        hidden=root/'held-gradient.epsw';original.rename(hidden)
                        try:negative.append(dict(case='missing_partially_committed_gradient',reason=recover_tool(job,'reconcile',False)))
                        finally:hidden.rename(original)
                    unchanged=command('eps_training_control',dict(settings,keys=[]),'inspect')
                    assert [s['synchronous']['next_round'] for s in unchanged['shard_stats']]==[h['next_round'] for h in planned['replica_heads']]
                    partial_hydration=None
                    if resident:
                        assert all(s['cached_rows']==0 and s['synchronous']['paused'] for s in unchanged['shard_stats'])
                        prep=dict(run=run,epoch=1,shards=addresses,only_shard=0,rows=64,dimension=8,agents=agents,
                                  restore_existing=True,replica_primary=primary_configs[0])
                        wrong=copy.deepcopy(prep);wrong.pop('replica_primary')
                        negative.append(dict(case='ordinary_snapshot_mode_rejects_replica_images',reason=prepare(wrong,False)))
                        wrong=copy.deepcopy(prep);wrong['replica_primary']['synchronous_training']['replica']['term']=3
                        wrong['agents'][0]['owner_run']=owner_run(run,1,0,3,'replacement-0')
                        wrong['replica_primary']['agents'][0]['owner_run']=wrong['agents'][0]['owner_run']
                        negative.append(dict(case='wrong_replica_owner_term',reason=prepare(wrong,False)))
                        wrong=copy.deepcopy(prep);wrong['rows']=66
                        negative.append(dict(case='changed_full_table_cardinality',reason=prepare(wrong,False)))
                        before_hydration=command('eps_training_control',settings,'inspect')
                        partial_hydration=prepare(prep)
                        after_hydration=command('eps_training_control',settings,'inspect')
                        assert before_hydration['rows']==after_hydration['rows'],'rehydration changed full optimizer state'
                        assert [s['cached_rows'] for s in after_hydration['shard_stats']]==[32,0]
                    recovered=recover_tool(job,'reconcile')
                    assert recovered['ready'] and recovered['replayed_rounds']==([target] if mode=='partial_round' else [])
                    if resident:
                        assert len(recovered['owner_rehydration'])==2
                        assert sum(p['shards'][0]['resident_rows'] for p in recovered['owner_rehydration'])==64
                        assert all(w['agents']==agents for w in recovered['workers'])
                    repeated=recover_tool(job,'reconcile')
                    assert repeated['ready'] and repeated['next_round']==expected and not repeated['replayed_rounds']
                    if a.handoff:
                        manager_job=copy.deepcopy(job)
                        staged=[]
                        for w,cfg in enumerate(recovered['workers']):
                            result=transfer(cfg,handoff_node,handoff_node,root/f'{run}-replacement-worker-{w}',run+'-stage')
                            recovered['workers'][w]=result['config']
                            worker_configs[w]['journal_directory']=result['config']['journal_directory']
                            staged.append(result)
                        sealed_recheck=recover_tool(manager_job,'reconcile')
                        assert sealed_recheck['ready'] and sealed_recheck['next_round']==expected
                        events.append(dict(journal_stage=staged))
                    command('eps_training_control',settings,'resume')
                    workers=[launch(source/'examples/tensorflow_training_worker.py',f'resumed-{w}',cfg,worker=True) for w,cfg in enumerate(recovered['workers'])]
                    events.append(dict(recovery=recovered,negative_cases=negative,idempotent_reconcile_verified=True,
                                       partial_owner_rehydration=partial_hydration))
                outputs=[]
                for child in workers:
                    output,_=child.communicate(timeout=90)
                    if child.returncode:raise RuntimeError(f'worker exited with {child.returncode}')
                    outputs.append(json.loads(output))
                if gpu:assert all('GPU:' in value for w in outputs for value in w['devices'].values())
                state=command('eps_training_control',settings,'inspect')
                assert state['completed']==[rounds,rounds] and not state['active'] and not state['failed']
                assert all(s['live_tickets']==0 and s['synchronous']['next_round']==rounds for s in state['shard_stats'])
                if resident:assert all(s['cached_rows']==32 for s in state['shard_stats'])
                history=[]
                for cfg in worker_configs:
                    with WorkerJournal(cfg['journal_directory'],identity(cfg)) as journal:
                        assert journal.rounds()==list(range(rounds))
                        history.append([journal.get(n)['loss'] for n in range(rounds)])
                reports.append(dict(mode=mode,workers=outputs,state=state,losses=history,events=events))
            except Exception:
                for log in logs:log.flush();log.seek(0);print(log.read()[-3000:],file=sys.stderr)
                raise
            finally:
                gate.touch()
                for child in worker_groups:kill_worker(child)
                for child in reversed(children):stop(child)
                for log in logs:log.close()
                assert all(child.poll() is not None for child in children)
    baseline=reports[0];maximum=0.;loss_error=0.
    for r in reports[1:]:
        loss_error=max(loss_error,max(abs(a-b) for got,want in zip(r['losses'],baseline['losses']) for a,b in zip(got,want)))
        for row,want in zip(r['state']['rows'],baseline['state']['rows']):
            assert row['key']==want['key'] and row['row_updates']==want['row_updates']
            for field in ('weights','first_moment','second_moment'):
                maximum=max(maximum,max(abs(a-b) for a,b in zip(row[field],want[field])))
    tolerance=5e-5 if gpu else 0.
    assert maximum<=tolerance and loss_error<=tolerance,(maximum,loss_error)
    result=dict(test='tensorflow_worker_recovery',device='CUDA' if gpu else 'CPU',parameter_backend=a.owners,
                hardware=hardware,workers=2,shards=2,rows=64,dimension=8,rounds=rounds,
                fault_round=target,max_state_absolute_error=maximum,max_loss_absolute_error=loss_error,cleanup_verified=True,reports=reports)
    result.update(comparison_tolerance=tolerance,parameter_owners=2 if resident else 0,
                  physical_gpus=len(hardware.strip().splitlines()) if hardware else 0,
                  journal_handoff=a.handoff, journal_transport='serialized local node subprocesses' if a.handoff else 'local/shared directories')
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reports'}))


if __name__=='__main__':main()
