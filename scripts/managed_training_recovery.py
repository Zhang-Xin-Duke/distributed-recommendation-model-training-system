"""Replicated sparse model training and generation-based fleet recovery.

Fleet nodes provide explicit node-local aliases for shards, agents, replicas and
coordinator. manager provides routes visible to its native control binaries.
All credentials remain in the private fleet file.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import uuid

from managed_roles import ManagedRoles
from transfer_worker_journals import transfer

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'adapters/entropy'))
from training_recovery import owner_run
from training_journal import canonical
from training_journal import WorkerJournal,identity as journal_identity
from role_node import write
from recover_training import recover
from physical_roles import PhysicalRoles,storage_node


class PlacementRequired(Exception):
    pass


def execute(fleet, run, out, fault=None, on_checkpoint=None, on_fault=None, *, export_checkpoint=False):
    nodes=copy.deepcopy(fleet['nodes']);manager=copy.deepcopy(fleet['manager']);count=len(nodes)
    placed='role_placement' in fleet
    measurement=fleet.get('training_measurement',False)
    if type(measurement) is not bool or (measurement and (fault is not None or placed)):
        raise ValueError('training measurement requires an uninterrupted fixed-placement run')
    expected_gpus=fleet.get('expected_physical_gpus',count)
    if type(expected_gpus) is not int or not 1<=expected_gpus<=count:
        raise ValueError('invalid expected physical GPU count')
    if export_checkpoint and (fault is not None or placed):
        raise ValueError('source export currently requires an uninterrupted, fixed-placement generation')
    if export_checkpoint:
        from training_source_export import node_ids
        node_ids(nodes)
    mirror_enabled=fleet.get('mirror_worker_journals',False)
    if not 2<=count<=8:raise ValueError('managed recovery requires 2–8 worker/shard groups')
    if mirror_enabled and (len(manager.get('journal_mirrors',[]))!=count or any(len(n.get('journal_mirrors',[]))!=count or not n.get('journal_mirror_listen') for n in nodes)):
        raise ValueError('gradient replication requires node and manager journal routes')
    if not re.fullmatch(r'[a-z0-9_-]{1,64}',run):raise ValueError('invalid managed training run identity')
    out=pathlib.Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,manager['module_dir'])
    if importlib.util.find_spec('_eps_native') is None:
        raise ValueError('manager Python cannot load its native module; use the interpreter matching manager.module_dir')
    state_path=out/'state.json'
    identity=dict(run=run,fault=fault,fleet_sha256=hashlib.sha256(canonical(fleet)).hexdigest())
    if export_checkpoint:identity['export_checkpoint']=True
    if state_path.exists():
        saved=json.loads(state_path.read_text())
        if saved['identity']!=identity:raise ValueError('managed recovery configuration changed')
    else:saved=dict(identity=identity,steps={});write(state_path,saved)
    recovery_generation=saved.get('recovery_generation',1)
    def step(name,action):
        if name not in saved['steps']:
            started=time.monotonic();saved['steps'][name]=action()
            saved.setdefault('phase_seconds',{})[name]=time.monotonic()-started
            write(state_path,saved)
            print(json.dumps(dict(phase=name,run=run)),flush=True)
            if on_checkpoint:on_checkpoint(name)
        return copy.deepcopy(saved['steps'][name])
    def path(index,role):return pathlib.PurePath(storage_node(fleet,nodes,role,index)['root'])/'results'/run/'roles'/role
    def binary(index,name):return str(pathlib.PurePath(nodes[index]['bin_dir'])/name)
    table=dict(id=1,dimension=8,dtype='float32',optimizer=dict(kind='sparse_lazy_adam',learning_rate=.02,beta1=.9,beta2=.999,epsilon=1e-8,initial_beta1_power=.9,initial_beta2_power=.999),initialization=dict(min=-.1,max=.1,seed=42))
    batches=[[[dict(keys=[7,8,7],label=0.),dict(keys=[9],label=1.)],[dict(keys=[10],label=1.)],[dict(keys=[11,12],label=0.),dict(keys=[9],label=1.)]],[[dict(keys=[9,10],label=1.)],[dict(keys=[11],label=0.),dict(keys=[12,9],label=1.)],[dict(keys=[7,12],label=0.)]]]
    batches=[copy.deepcopy(batches[w%2]) for w in range(count)]
    workload=fleet.get('workload',{})
    preparation_batch=workload.get('preparation_batch_rows',256)
    control_timeout=workload.get('control_timeout_seconds',30)
    observation_timeout=workload.get('observation_timeout_seconds',180)
    if (type(preparation_batch) is not int or not 1<=preparation_batch<=65536
            or type(control_timeout) is not int or not 1<=control_timeout<=1800
            or type(observation_timeout) is not int or not 1<=observation_timeout<=3600):
        raise ValueError('invalid preparation/control bounds')
    table=copy.deepcopy(workload.get('table',table))
    dimension=table['dimension'];hidden=workload.get('hidden',4)
    rounds=workload.get('rounds',8)
    target,rows=workload.get('fault_round',min(3,rounds-1)),workload.get('rows',64)
    if not 1<=dimension<=4096 or not 1<=hidden<=256 or not 1<=rounds<=10000 or not 0<=target<rounds or rows<16:
        raise ValueError('invalid managed training workload')
    packed=(dimension*hidden+2*hidden+1+dimension-1)//dimension
    if rows<=packed+6:raise ValueError('managed parameter table cannot hold its model and default sparse inputs')
    if 'batches' in workload:batches=workload['batches']
    else:
        for worker_batches in batches:
            for batch in worker_batches:
                for sample in batch:sample['keys']=[key-7+packed+1 for key in sample['keys']]
    if len(batches)!=count or any(not group for group in batches):raise ValueError('worker input groups differ from fleet')
    inspect_keys=workload.get('inspect_keys',list(range(min(rows,64))))
    control_id=0
    def command(name,cfg,action=None):
        nonlocal control_id
        control_id+=1;p=out/f'control-{control_id}-{uuid.uuid4().hex[:8]}.json';p.write_text(json.dumps(cfg))
        env=dict(os.environ)
        env['PYTHONPATH']=manager['module_dir']+':'+str(ROOT/'adapters/entropy')
        argv=[str(pathlib.Path(manager['bin_dir'])/name),'--config',str(p)]
        if action:argv+=['--action',action]
        r=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=control_timeout)
        if r.returncode:raise RuntimeError(r.stderr[-2000:])
        return json.loads(r.stdout)
    settings=dict(run=run,epoch=1,coordinator=manager['coordinator'],shards=manager['shards'],keys=inspect_keys,synchronous=True)
    replica_hosts=fleet.get('replica_hosts',[(h+1)%count for h in range(count)])
    mirror_hosts=fleet.get('mirror_hosts',[(h+1)%count for h in range(count)])
    if any(len(v)!=count or any(type(i) is not int or not 0<=i<count for i in v) for v in (replica_hosts,mirror_hosts)):
        raise ValueError('invalid replica/mirror host mapping')
    def listener(kind,home,default):return fleet.get('role_listeners',{}).get(kind,[None]*count)[home] or default
    def primary_cfg(home,generation):
        n=nodes[home];primary=f'primary-{home}-g{generation}';term=generation+1
        agent=dict(id=home,endpoint=n['agents'][home]['endpoint'],owner_run=owner_run(run,1,home,term,primary))
        return dict(schema_version=1,run_id=run,listen=n['ps_listen'],partition=dict(shard_id=home,shard_count=count,epoch=1),tables=[table],cache=dict(enabled=True),agents=[agent],replica_table_rows={'1':rows},snapshot_directory=str(path(home,f'primary-{home}-g{generation}')/'snapshots'),synchronous_training=dict(workers=count,reduction='sample_mean',round_timeout_ms=25000,replica=dict(endpoint=n['replicas'][home],primary=primary,term=term,local_directory=str(path(home,f'primary-{home}-g{generation}')/'primary-log'),recover=bool(generation),start_paused=bool(generation))))
    def manager_cfg(home,generation):
        cfg=primary_cfg(home,generation)
        cfg['agents'][0]['endpoint']=manager['agents'][home]['endpoint']
        cfg['synchronous_training']['replica']['endpoint']=manager['replicas'][home]
        return cfg
    def replica_cfg(home,generation=0):
        host=replica_hosts[home]
        cfg=primary_cfg(home,0)
        cfg.update(listen=listener('replica',home,nodes[host]['replica_listen']),replica_directory=str(path(host,f'replica-{home}-g{generation if fleet.get("controller_managed") and generation else 0}')/'storage'))
        if placed:cfg['snapshot_directory']=str(path(host,f'replica-{home}-g0')/'snapshots')
        cfg['synchronous_training'].pop('replica')
        cfg.pop('cache');cfg.pop('agents')
        return cfg
    def role_owner(roles,home,generation):
        role=f'owner-{home}-g{generation}';primary=f'primary-{home}-g{generation}'
        local_rows=(rows+count-1-home)//count
        roles.ensure(home,role,[binary(home,'eps_cache_agent'),'--run',owner_run(run,1,home,generation+1,primary),'--training-run',run,'--listen',nodes[home]['agent_listen'],'--agent',str(home),'--backend','cuda' if nodes[home].get('gpu') else 'host','--rows-per-table',str(local_rows),'--state-bytes',str(local_rows*(dimension*12+8)+4096)])
        roles.ready(home,role)
    def role_replica(roles,home,generation):
        host=replica_hosts[home];role=f'replica-{home}-g{generation}';argv=[binary(host,'eps_replica')]
        if generation==0 and fault=='partial_round' and home==1:argv+=['--test-publish-gate',str(path(host,role)/'gate'),'--test-publish-round',str(target)]
        roles.ensure(host,role,argv,replica_cfg(home,generation),config_position=1);roles.ready(host,role)
    def grant(home,generation):
        cfg=manager_cfg(home,generation);cfg['replica_endpoint']=manager['replicas'][home]
        head=command('eps_replica_control',cfg,'head');primary=f'primary-{home}-g{generation}'
        if head['term']>generation+1 or (head['term']==generation+1 and head['primary']!=primary):
            raise ValueError('replica term cannot be adopted by this generation')
        while head['term']<generation+1:
            cfg.update(expected_term=head['term'],primary=primary)
            command('eps_replica_control',cfg,'fence')
            head=command('eps_replica_control',cfg,'head')
        return head
    workers=[]
    for w,n in enumerate(nodes):
        cfg=dict(run=run,epoch=1,shards=n['shards'],coordinator=n['coordinator'],worker=w,dimension=dimension,hidden=hidden,iterations=rounds,batches=batches[w],table=table,agents=[dict(a,owner_run=owner_run(run,1,h,1,f'primary-{h}-g0')) for h,a in enumerate(n['agents'])],device='/GPU:0' if n.get('gpu') else '/CPU:0',journal_directory=str(path(w,f'worker-{w}-g0')/'journal'),journal_workers=count)
        if w==0 and fault in ['after_pull','after_update']:cfg['test_pause']=dict(stage=fault,iteration=target,path=str(path(w,'worker-0-g0')/'gate'))
        if mirror_enabled:cfg['journal_mirror']=dict(endpoint=n['journal_mirrors'][w],generation=1,writer=f'worker-{w}-g0')
        if measurement or fleet.get('coordinated_start'):cfg.update(start_gate=str(path(w,f'worker-{w}-g0')/'training-start'),record_stage_times=True,
                                  deterministic_model=workload.get('deterministic_model',False))
        workers.append(cfg)
    def mirror_role(roles,worker,generation):
        host=mirror_hosts[worker];name=f'journal-mirror-{worker}-g{generation}'
        cfg=dict(listen=listener('journal-mirror',worker,nodes[host]['journal_mirror_listen']),directory=str(path(host,name)/'store'),worker_config=workers[worker],generation=generation+1,writer=f'worker-{worker}-g{generation}')
        roles.ensure(host,name,[nodes[host]['python'],str(pathlib.PurePath(nodes[host]['source'])/'scripts/journal_replica_server.py')],cfg)
        roles.ready(host,name)
        return dict(worker=worker,generation=generation+1,host=host)
    manager_node=dict(local=True,source=str(ROOT),python=sys.executable)
    client=PhysicalRoles(nodes,run,fleet,out) if placed else ManagedRoles(nodes,run)
    with client as roles:
        def inspect_hardware():
            result=[roles.call(h,'hardware',f'worker-{h}-g0',gpu=n.get('gpu',False)) for h,n in enumerate(nodes)]
            if all(n.get('gpu') for n in nodes) and len({r['gpu']['uuid'] for r in result})!=expected_gpus:
                raise ValueError('managed GPU placement differs from the expected physical device count')
            return result
        hardware=step('hardware',inspect_hardware)
        # Register deterministic names even when a saved phase skips its action.
        def register(generation):
            for h in range(count):
                for kind,index in [('owner',h),('primary',h),('worker',h),('replica',replica_hosts[h])]:roles.roles[f'{kind}-{h}-g{generation}']=index
                if mirror_enabled:roles.roles[f'journal-mirror-{h}-g{generation}']=mirror_hosts[h]
            roles.roles[f'coordinator-g{generation}']=0
        for generation in range(recovery_generation+1):register(generation)
        terminal=False
        try:
            def initial():
                for h in range(count):role_owner(roles,h,0);role_replica(roles,h,0);grant(h,0)
                for h in range(count):roles.ensure(h,f'primary-{h}-g0',[binary(h,'eps_server')],primary_cfg(h,0));roles.ready(h,f'primary-{h}-g0')
                command('eps_training_prepare',dict(run=run,shards=manager['shards'],rows=rows,dimension=dimension,batch_rows=preparation_batch,agents=[dict(a,owner_run=owner_run(run,1,h,1,f'primary-{h}-g0')) for h,a in enumerate(manager['agents'])]))
                roles.ensure(0,'coordinator-g0',[binary(0,'eps_training_coordinator')],dict(schema_version=1,run=run,listen=nodes[0]['coordinator_listen'],workers=count,consistency='sequential',max_delay=0));roles.ready(0,'coordinator-g0')
                return dict(ready=True)
            step('initial_roles',initial)
            if mirror_enabled:step('initial_gradient_replicas',lambda:[mirror_role(roles,w,0) for w in range(count)])
            def start_workers(configs,generation):
                for w,cfg in enumerate(configs):roles.ensure(w,f'worker-{w}-g{generation}',[nodes[w]['python'],str(pathlib.PurePath(nodes[w]['source'])/'examples/tensorflow_training_worker.py')],cfg,dict(CUDA_VISIBLE_DEVICES='0' if nodes[w].get('gpu') else ''))
                return dict(started=True)
            step('initial_workers',lambda:start_workers(workers,0))
            if measurement or fleet.get('coordinated_start'):
                def warm_workers():
                    deadline=time.monotonic()+50
                    while True:
                        states=[roles.call(w,'training_gate',f'worker-{w}-g0') for w in range(count)]
                        if all(s['warmed_model'] for s in states):return states
                        if time.monotonic()>=deadline:raise TimeoutError('Worker warmup observation expired; retain its existing handle')
                        time.sleep(.05)
                step('measurement_workers_ready',warm_workers)
                step('measurement_gates_released',lambda:[roles.call(w,'training_gate',f'worker-{w}-g0',release=True) for w in range(count)])
            def observe_fault():
                deadline=time.monotonic()+observation_timeout
                injected=None
                while True:
                    states=[roles.status(w,f'worker-{w}-g0') for w in range(count)]
                    if all(s.get('result',{}).get('exit_code')==0 for s in states):return dict(recovery=False)
                    failed=any(s['state']=='exited' and s.get('result',{}).get('exit_code',1)!=0 for s in states)
                    failed|=any(not roles.status(h,f'{kind}-{h}-g0')['live_child'] for h in range(count) for kind in ['primary','owner'])
                    failed|=not roles.status(0,'coordinator-g0')['live_child']
                    if mirror_enabled:failed|=any(not roles.status(mirror_hosts[w],f'journal-mirror-{w}-g0')['live_child'] for w in range(count))
                    if failed:return dict(recovery=True,reason='observed_role_failure',injected_fault=fault if injected else None,pre_fault=injected)
                    if fault:
                        try:state=command('eps_training_control',dict(settings,keys=[]),'inspect')
                        except (RuntimeError,subprocess.TimeoutExpired):
                            if time.monotonic()>=deadline:raise TimeoutError('RPC observation expired; retain all role handles')
                            time.sleep(.1);continue
                        groups=[s['synchronous'] for s in state['shard_stats']]
                        if fault=='after_pull':reached=state['completed'][0]==target and sum(s['live_tickets'] for s in state['shard_stats'])>0 and all(g['received_workers']==count-1 for g in groups)
                        elif fault=='partial_round':reached=groups[0]['next_round']==target+1 and groups[1]['next_round']==target and state['active']==count
                        else:reached=all(g['next_round']==target+1 for g in groups) and state['completed'][0]==target
                        if reached:
                            if on_fault is not None:
                                if not placed:raise ValueError('terminal node fault observer requires physical placement')
                                receipt=on_fault(state,roles);roles.record_loss(receipt)
                            else:roles.stop(0,'primary-0-g0')
                            injected=state
                            continue
                    if time.monotonic()>=deadline:raise TimeoutError('training observation expired; retain all role handles')
                    time.sleep(.1)
            observed=step('observation',observe_fault)
            if (export_checkpoint or measurement) and observed['recovery']:
                terminal=True
                raise RuntimeError('source export/measurement requires an uninterrupted initial generation; source logs are retained')
            active_workers=workers;generation=0
            if observed['recovery']:
                if fleet.get('controller_managed') and not (out/'replacement-placement.json').exists():
                    write(out/'recovery-request.json',dict(run=run,reason=observed['reason']))
                    raise PlacementRequired('controller must prepare replacement placement and authorities')
                def stop_generation():
                    report=[]
                    for kind in ['worker','primary','owner','coordinator','replica',*(['journal-mirror'] if mirror_enabled else [])]:
                        homes=range(count) if kind!='coordinator' else [None]
                        for h in homes:
                            index=0 if h is None else (replica_hosts[h] if kind=='replica' else mirror_hosts[h]) if kind in ['replica','journal-mirror'] else h
                            role='coordinator-g0' if h is None else f'{kind}-{h}-g0'
                            result=roles.stop(index,role);report.append(dict(role=role,**result))
                    if not all(r['owned_processes_gone'] for r in report):raise RuntimeError('former role remains live')
                    return report
                step('former_roles_stopped',stop_generation)
                if fleet.get('controller_managed'):
                    deployment=json.loads((out/'replacement-placement.json').read_text())
                    for key in ('physical_nodes','role_placement','replacement_nodes','replacement_manager','role_listeners'):
                        fleet[key]=deployment[key]
                    roles.fleet=fleet;roles.physical=fleet['physical_nodes']
                if placed:
                    # Reapply routing on manager reentry even when the durable
                    # placement step already exists. Original configs stay in
                    # the fetched journal identities, independent of routes.
                    nodes[:]=copy.deepcopy(fleet['replacement_nodes'])
                    manager.update(copy.deepcopy(fleet['replacement_manager']))
                    settings.update(coordinator=manager['coordinator'],shards=manager['shards'])
                    step('replacement_placement',lambda:dict(roles=fleet['role_placement']['1'],
                        terminal_nodes=list(roles.lost.values())))
                if 'admission_resumed' not in saved['steps']:
                    # Before admission no replacement Worker can have produced
                    # additional gradients. Retained original journals therefore
                    # still cover every possibly committed recovery round.
                    prior=[]
                    for kind in ['owner','primary','replica',*(['journal-mirror'] if mirror_enabled else [])]:
                        for h in range(count):
                            index=(replica_hosts[h] if kind=='replica' else mirror_hosts[h]) if kind in ['replica','journal-mirror'] else h
                            name=f'{kind}-{h}-g{recovery_generation}'
                            prior.append((index,name,roles.status(index,name)))
                    name=f'coordinator-g{recovery_generation}'
                    prior.append((0,name,roles.status(0,name)))
                    if any(state['state']=='exited' for _,_,state in prior):
                        stopped=[]
                        for index,name,_ in prior:
                            result=roles.stop(index,name)
                            if not result['owned_processes_gone']:raise RuntimeError('prior recovery role remains live')
                            stopped.append(dict(role=name,**result))
                        reset=['replacement_roles','recovery_plan','replacement_coordinator','reconciled',
                               'journals_staged','replacement_gradient_replicas']
                        retired={k:saved['steps'].pop(k) for k in reset if k in saved['steps']}
                        saved.setdefault('previous_recovery_attempts',[]).append(dict(generation=recovery_generation,
                            stopped_roles=stopped,phases=retired))
                        recovery_generation+=1;saved['recovery_generation']=recovery_generation
                        write(state_path,saved);register(recovery_generation)
                def fetch():
                    if fleet.get('controller_managed') and (out/'controller-journals.json').exists():
                        return json.loads((out/'controller-journals.json').read_text())
                    result=[]
                    for w,cfg in enumerate(workers):
                        host=mirror_hosts[w] if mirror_enabled else w
                        source_cfg=dict(cfg,journal_directory=str(path(host,f'journal-mirror-{w}-g0')/'store')) if mirror_enabled else cfg
                        source_node=storage_node(fleet,nodes,f'journal-mirror-{w}-g0' if mirror_enabled else f'worker-{w}-g0',host)
                        record=transfer(source_cfg,source_node,manager_node,out/f'worker-{w}-journal','fetch-'+run)
                        record['from_gradient_replica']=mirror_enabled
                        record['config']['shards']=manager['shards'];record['config']['coordinator']=manager['coordinator'];result.append(record)
                    return result
                fetched=step('journals_fetched',fetch)
                def replacements():
                    for h in range(count):role_replica(roles,h,recovery_generation);grant(h,recovery_generation);role_owner(roles,h,recovery_generation)
                    for h in range(count):
                        name=f'primary-{h}-g{recovery_generation}'
                        roles.ensure(h,name,[binary(h,'eps_server')],primary_cfg(h,recovery_generation));roles.ready(h,name)
                    return dict(ready=True)
                step('replacement_roles',replacements)
                job=dict(run=run,epoch=1,end_round=rounds,coordinator_endpoint=manager['coordinator'],workers=[x['config'] for x in fetched],shards=[dict(endpoint=manager['shards'][h],config=manager_cfg(h,recovery_generation)) for h in range(count)])
                planned=step('recovery_plan',lambda:recover(job,pathlib.Path(manager['bin_dir'])))
                def coordinator():
                    name=f'coordinator-g{recovery_generation}'
                    roles.ensure(0,name,[binary(0,'eps_training_coordinator')],dict(planned['coordinator'],listen=nodes[0]['coordinator_listen']));roles.ready(0,name);return dict(ready=True)
                step('replacement_coordinator',coordinator)
                reconciled=step('reconciled',lambda:recover(job,pathlib.Path(manager['bin_dir']),True))
                def stage():
                    staged=[]
                    for w,cfg in enumerate(reconciled['workers']):
                        result=transfer(cfg,manager_node,nodes[w],str(path(w,f'worker-{w}-g{recovery_generation}')/'journal'),'stage-'+run)
                        result['config'].update(shards=nodes[w]['shards'],coordinator=nodes[w]['coordinator'],agents=[dict(a,owner_run=owner_run(run,1,h,recovery_generation+1,f'primary-{h}-g{recovery_generation}')) for h,a in enumerate(nodes[w]['agents'])])
                        # Initial measurement gates belong to the former host and
                        # process generation. Recovery admission is released by
                        # the reconciled coordinator, not that startup gate.
                        result['config'].pop('start_gate', None)
                        staged.append(result)
                        if mirror_enabled:
                            host=mirror_hosts[w]
                            target=storage_node(fleet,nodes,f'journal-mirror-{w}-g{recovery_generation}',host)
                            result['gradient_replica_seed']=transfer(cfg,manager_node,target,str(path(host,f'journal-mirror-{w}-g{recovery_generation}')/'store'),'stage-'+run)
                            result['config']['journal_mirror']=dict(endpoint=nodes[w]['journal_mirrors'][w],generation=recovery_generation+1,writer=f'worker-{w}-g{recovery_generation}')
                    return staged
                staged=step('journals_staged',stage)
                if mirror_enabled:step('replacement_gradient_replicas',lambda:[mirror_role(roles,w,recovery_generation) for w in range(count)])
                def resume():
                    state=command('eps_training_control',dict(settings,keys=[]),'inspect')
                    if state['paused']:command('eps_training_control',settings,'resume')
                    return dict(resumed=True)
                step('admission_resumed',resume)
                active_workers=[x['config'] for x in staged];generation=recovery_generation
                step('replacement_workers',lambda:start_workers(active_workers,recovery_generation))
            def finish():
                outputs=[json.loads(roles.wait(w,f'worker-{w}-g{generation}')) for w in range(count)]
                final=command('eps_training_control',settings,'inspect')
                if final['completed']!=[rounds]*count or final['failed'] or final['active'] or final['waiting']:raise RuntimeError('managed training did not complete')
                if any(s['live_tickets'] or s['synchronous']['next_round']!=rounds or s['cached_rows']!=(rows+count-1-h)//count for h,s in enumerate(final['shard_stats'])):raise RuntimeError('managed parameters did not drain or retain residency')
                histories=[];receipts=[]
                if measurement:
                    mirror_states=[]
                    for w in range(count):
                        receipt=roles.call(w,'journal_report',f'worker-{w}-g{generation}')
                        if (receipt['run']!=run or receipt['worker']!=w or receipt['rounds']!=list(range(rounds))
                                or receipt['loss_history']!=outputs[w]['losses']):
                            raise RuntimeError('completed Worker journal differs from its model output')
                        receipts.append(receipt);histories.append(receipt['loss_history'])
                        if mirror_enabled:
                            peer=receipt['gradient_replica']
                            if peer['next_round']!=rounds or peer['records']!=rounds:
                                raise RuntimeError('gradient replica history did not complete')
                            mirror_states.append(peer)
                    return dict(workers=outputs,state=final,loss_history=histories,final_journal_receipts=receipts,
                        gradient_replicas=mirror_states,journal_verification='owner-local full payload verification; only hashes/progress/losses transferred')
                for w,cfg in enumerate(active_workers):
                    receipt=transfer(cfg,nodes[w],manager_node,out/f'completed-worker-{w}','complete-'+run)
                    receipts.append(receipt)
                    with WorkerJournal(receipt['config']['journal_directory'],journal_identity(receipt['config'])) as journal:
                        if journal.rounds()!=list(range(rounds)):raise RuntimeError('completed worker journal has incomplete round history')
                        histories.append([journal.get(t)['loss'] for t in range(rounds)])
                mirror_states=[]
                if mirror_enabled:
                    from gradient_replica import ReplicaClient
                    for w,receipt in enumerate(receipts):
                        with WorkerJournal(receipt['config']['journal_directory'],journal_identity(receipt['config'])) as journal:
                            peer=ReplicaClient(dict(endpoint=manager['journal_mirrors'][w],generation=generation+1,writer=f'worker-{w}-g{generation}'),journal)
                            try:mirror_state=peer.call('status')
                            finally:peer.close()
                        if mirror_state['next_round']!=rounds or mirror_state['records']!=rounds:raise RuntimeError('gradient replica history did not complete')
                        mirror_states.append(mirror_state)
                return dict(workers=outputs,state=final,loss_history=histories,final_journal_receipts=receipts,gradient_replicas=mirror_states)
            final=step('final_model',finish)
            if export_checkpoint:
                def export():
                    from training_source_export import save_checkpoint
                    return save_checkpoint(command,settings,workers,final['workers'],out/'source-generation'/'checkpoint')
                step('source_checkpoint',export)
            terminal=True
            result=dict(run=run,fault=fault,recovered=observed['recovery'],generation=generation,model=final,steps=saved['steps'],phase_seconds=saved.get('phase_seconds',{}),hardware=hardware,mirror_worker_journals=mirror_enabled,workload=dict(rows=rows,dimension=dimension,rounds=rounds,hidden=hidden,inspected_rows=len(inspect_keys)))
        finally:
            if terminal:
                cleanup=roles.cleanup()
                if not all(x['owned_processes_gone'] for x in cleanup):raise RuntimeError('managed roles did not clean up')
                write(out/'cleanup.json',cleanup)
    result['cleanup']=json.loads((out/'cleanup.json').read_text())
    if export_checkpoint:
        from training_source_export import publish_source
        mirrors=[]
        if mirror_enabled:
            for w,cfg in enumerate(workers):
                host=mirror_hosts[w]
                mirrors.append(dict(listen=listener('journal-mirror',w,nodes[host]['journal_mirror_listen']),
                    directory=str(path(host,f'journal-mirror-{w}-g0')/'store'),worker_config=cfg,
                    generation=1,writer=f'worker-{w}-g0'))
        result['source_generation']=publish_source(out/'source-generation',run,nodes,
            manager,workers,[primary_cfg(h,0) for h in range(count)],
            [replica_cfg(h) for h in range(count)],mirrors,
            saved['steps']['source_checkpoint'],result['cleanup'])
    write(out/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fleet',type=pathlib.Path,required=True);p.add_argument('--run',required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--fault',choices=['after_pull','partial_round','after_update'])
    p.add_argument('--export-checkpoint',action='store_true',help='export a joint checkpoint and stopped source-storage ownership after an uninterrupted generation')
    a=p.parse_args();result=execute(json.loads(a.fleet.read_text()),a.run,a.out,a.fault,export_checkpoint=a.export_checkpoint)
    print(json.dumps(dict(run=a.run,recovered=result['recovered'],completed=result['model']['state']['completed'])))


if __name__=='__main__':main()
