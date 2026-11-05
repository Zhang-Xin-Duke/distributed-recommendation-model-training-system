"""Prepare and verify a fresh replica/journal generation at a joint checkpoint.

The existing role manager starts the returned services. Verification publishes
a durable readiness receipt while task admission stays paused; resume-job then
opens the checked generation. Source-generation storage remains intact.
"""
import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'adapters/entropy'))
from training_journal import WorkerJournal, canonical, identity, sync_directory


def command(build, binary, config, action):
    with tempfile.TemporaryDirectory(prefix='eps-rollover-control-') as tmp:
        path=pathlib.Path(tmp)/'config.json'
        path.write_bytes(canonical(config))
        result=subprocess.run([str(pathlib.Path(build)/binary),'--config',str(path),'--action',action],
                              capture_output=True,text=True,timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr.strip()[-4000:])
        return json.loads(result.stdout)


def publish(path, value):
    data=canonical(value)+b'\n'
    if path.exists():
        if path.read_bytes()!=data:
            raise ValueError('immutable rollover metadata differs: '+path.name)
        return
    pending=path.parent/('.pending-'+uuid.uuid4().hex)
    try:
        with pending.open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(pending,path)
        sync_directory(path.parent)
    finally:
        pending.unlink(missing_ok=True)


def prepare(job, build, destination):
    destination=pathlib.Path(destination).resolve()
    manifest=pathlib.Path(job['manifest']).resolve()
    run=job['run']
    if (destination/'plan.json').exists():
        prior=json.loads((destination/'intent.json').read_text())
        if prior['job']!=job:
            raise ValueError('immutable rollover intent differs')
        plan=json.loads((destination/'plan.json').read_text())
        command(build,'eps_training_control',dict(run=run,manifest=plan['manifest']),'restore-config')
        return plan
    templates=command(build,'eps_training_control',dict(run=run,manifest=str(manifest)),'restore-config')
    checkpoint=json.loads(manifest.read_text())
    if len(checkpoint['tables'])!=1 or checkpoint['model'].get('kind')!='embedding_tanh_mlp':
        raise ValueError('rollover requires the journaled Embedding/MLP model with one sharded table')
    boundary=checkpoint['progress']['completed'][0]
    workers, shards=job['workers'],job['shards']
    routes=job['routes']
    if type(job.get('end_round')) is not int or not boundary<job['end_round']<=1000000:
        raise ValueError('rollover training end must exceed the checkpoint boundary')
    if len(workers)!=len(checkpoint['workers']) or len(shards)!=len(templates['shards']) or len(routes)!=len(shards):
        raise ValueError('rollover Worker/shard membership differs from checkpoint')
    if 'worker_routes' in job and len(job['worker_routes'])!=len(workers):
        raise ValueError('rollover Worker route membership differs')
    endpoints=[r['primary_endpoint'] for r in routes]
    replicas=[r['replica_endpoint'] for r in routes]
    if len(set(endpoints+replicas+[job['coordinator_endpoint']]))!=len(endpoints)+len(replicas)+1:
        raise ValueError('rollover service endpoints repeat')
    agents=[]
    for home,(source,route) in enumerate(zip(shards,routes)):
        if (source['run_id']!=checkpoint['source_run'] or source['partition']!=dict(
                shard_id=home,shard_count=len(shards),epoch=checkpoint['epoch'])
                or source['tables']!=checkpoint['tables']
                or source['synchronous_training']['workers']!=len(workers)
                or source['synchronous_training'].get('reduction','sample_mean')!=templates['shards'][home]['synchronous_training']['reduction']):
            raise ValueError('rollover source shard identity/model differs')
        if source.get('cache',{}).get('enabled',False) and not route.get('agents'):
            raise ValueError('cached rollover requires explicit replacement parameter owners')
        agents.extend(copy.deepcopy(route.get('agents',[])))
    distinct={}
    for agent in agents:
        if agent['id'] in distinct and agent!=distinct[agent['id']]:
            raise ValueError('replacement parameter owner descriptors conflict')
        distinct[agent['id']]=agent
    agents=[distinct[key] for key in sorted(distinct)]
    for worker,source in enumerate(workers):
        descriptor=identity(source)
        saved=checkpoint['workers'][worker]
        if (source['worker']!=worker or saved['worker']!=worker
                or source['run']!=checkpoint['source_run'] or descriptor['workers']!=len(workers)
                or descriptor['epoch']!=checkpoint['epoch']
                or descriptor['shards']!=len(shards) or source['table']!=checkpoint['tables'][0]
                or descriptor['model']!=checkpoint['model']
                or descriptor['dataset_sha256']!=saved['input_state']['dataset_sha256']
                or descriptor['dataset_json']!=saved['input_state']['dataset_json']
                or source.get('update_mode','synchronous')!='synchronous'):
            raise ValueError('rollover Worker input/model identity differs')
        if source.get('journal_mirror') and 'journal_mirror' not in job.get('worker_routes',[{} for _ in workers])[worker]:
            raise ValueError('mirrored journals require explicit replacement mirror routes')
    if destination==manifest.parent or destination in manifest.parents:
        raise ValueError('rollover destination must not contain the source checkpoint')
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/'prepare.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        intent=dict(kind='eps-rollover-intent-v1',job=job,
                    checkpoint_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        publish(destination/'intent.json',intent)
        if (destination/'plan.json').exists():
            return json.loads((destination/'plan.json').read_text())
        saved=destination/'checkpoint'
        saved.mkdir(exist_ok=True)
        # Preserve an independent pathname before old-generation retirement.
        # Same-filesystem immutable snapshots can share an inode; otherwise copy
        # in bounded chunks and fsync. The C++ controller rechecks every SHA below.
        for item in checkpoint['parameters']:
            source=manifest.parent/item['file'];target=saved/item['file']
            if not target.exists():
                try:
                    os.link(source,target)
                except OSError:
                    pending=target.with_suffix('.pending')
                    with source.open('rb') as src,pending.open('wb') as dst:
                        shutil.copyfileobj(src,dst,1<<20);dst.flush();os.fsync(dst.fileno())
                    pending.replace(target)
        publish(saved/'manifest.json',checkpoint)
        sync_directory(saved);sync_directory(destination)
        templates=command(build,'eps_training_control',dict(run=run,manifest=str(saved/'manifest.json')),'restore-config')
        primaries,standbys,resumed,mirrors=[],[],[],[]
        for home,(source,route,restored) in enumerate(zip(shards,routes,templates['shards'])):
            primary=copy.deepcopy(source)
            primary.update(restored)
            primary.update(listen=route.get('primary_listen',route['primary_endpoint']),
                snapshot_directory=str(destination/f'primary-{home}'/'snapshots'),replica_base_round=boundary)
            primary['agents']=copy.deepcopy(route.get('agents',[]))
            standby=copy.deepcopy(primary)
            standby.update(listen=route.get('replica_listen',route['replica_endpoint']),
                replica_directory=str(destination/f'replica-{home}'),replica_endpoint=route['replica_endpoint'])
            for key in ('restore_file','agents','access','cache'):
                standby.pop(key,None)
            standby['synchronous_training'].pop('restore',None)
            primary['synchronous_training']['replica']=dict(endpoint=route['replica_endpoint'],
                primary=f'{run}-primary-{home}',term=1,recover=False,
                local_directory=str(destination/f'primary-{home}'/'journal'))
            primaries.append(primary);standbys.append(standby)
        for worker,source in enumerate(workers):
            cfg=copy.deepcopy(source)
            for key in ('test_pause','start_gate','read_sequence','input_state','journal_mirror','journal_mirror_owner'):
                cfg.pop(key,None)
            cfg.update(run=run,epoch=checkpoint['epoch'],shards=endpoints,
                coordinator=job['coordinator_endpoint'],agents=agents,start_iteration=boundary,
                iterations=job['end_round']-boundary,
                journal_base_round=boundary,journal_directory=str(destination/f'worker-{worker}'),
                input_state=templates['workers'][worker]['input_state'])
            route=copy.deepcopy(job.get('worker_routes',[{} for _ in workers])[worker])
            if 'journal_mirror' in route:cfg['journal_mirror']=route['journal_mirror']
            # A caller may select new transport/mirror routes, never redefine the
            # restored identity, input cursor or journal baseline through routes.
            allowed={'journal_mirror','journal_mirror_listen'}
            if set(route)-allowed:
                raise ValueError('unknown Worker replacement route')
            with WorkerJournal(cfg['journal_directory'],identity(cfg),
                    cfg.get('journal_record_bytes',32<<20),cfg.get('journal_disk_bytes',1<<30)) as journal:
                if journal.rounds():
                    raise ValueError('replacement Worker journal already contains gradients')
            resumed.append(cfg)
            if cfg.get('journal_mirror'):
                mirror=cfg['journal_mirror']
                mirrors.append(dict(listen=route.get('journal_mirror_listen',mirror['endpoint']),
                    directory=str(destination/f'worker-mirror-{worker}'),worker_config=cfg,
                    generation=mirror['generation'],writer=mirror['writer']))
        coordinator=dict(templates['coordinator'],listen=job.get('coordinator_listen',job['coordinator_endpoint']))
        plan=dict(kind='eps-checkpoint-rollover-v1',run=run,boundary=boundary,
            directory=str(destination),manifest=str(saved/'manifest.json'),
            source_checkpoint_sha256=intent['checkpoint_sha256'],
            coordinator=coordinator,coordinator_endpoint=job['coordinator_endpoint'],
            shards=primaries,replicas=standbys,workers=resumed,gradient_replicas=mirrors)
        for name,value in [('coordinator',coordinator),*[(f'primary-{i}',c) for i,c in enumerate(primaries)],
                           *[(f'replica-{i}',c) for i,c in enumerate(standbys)],
                           *[(f'worker-mirror-{c["worker_config"]["worker"]}',c) for c in mirrors],
                           *[(f'worker-{i}',c) for i,c in enumerate(resumed)]]:
            publish(destination/(name+'.json'),value)
        publish(destination/'plan.json',plan)
        return plan


def verify(plan,build):
    if plan['kind']!='eps-checkpoint-rollover-v1':
        raise ValueError('unknown rollover plan')
    directory=pathlib.Path(plan['directory'])
    if json.loads((directory/'plan.json').read_text())!=plan:
        raise ValueError('rollover plan differs from its published generation')
    with contextlib.ExitStack() as stack:
        journals=[];mirrors=[]
        for cfg in plan['workers']:
            journal=stack.enter_context(WorkerJournal(cfg['journal_directory'],identity(cfg),
                cfg.get('journal_record_bytes',32<<20),cfg.get('journal_disk_bytes',1<<30)))
            if journal.rounds() or journal.identity['base_round']!=plan['boundary']:
                raise ValueError('new-generation Worker journal has advanced or has the wrong baseline')
            journals.append(dict(worker=cfg['worker'],identity_sha256=journal.digest,base_round=plan['boundary']))
            if cfg.get('journal_mirror'):
                from gradient_replica import ReplicaClient
                peer=ReplicaClient(cfg['journal_mirror'],journal)
                try:
                    status=peer.call('status')
                    if status['next_round']!=plan['boundary'] or status['records']!=0:
                        raise ValueError('replacement gradient mirror has advanced or has the wrong baseline')
                    mirrors.append(dict(worker=cfg['worker'],status=status))
                finally:peer.close()
        heads=[]
        for cfg,primary in zip(plan['replicas'],plan['shards']):
            head=command(build,'eps_replica_control',cfg,'head')
            role=primary['synchronous_training']['replica']
            if (not head['baseline'] or head['base_round']!=plan['boundary'] or head['next_round']!=plan['boundary']
                    or head['term']!=role['term'] or head['primary']!=role['primary']):
                raise ValueError('replacement replica baseline/authority differs')
            heads.append(head)
        settings=dict(run=plan['run'],manifest=plan['manifest'],coordinator=plan['coordinator_endpoint'],
                      shards=[cfg['listen'] for cfg in plan['shards']])
        # Public client endpoints may differ from node-local listeners.
        settings['shards']=plan['workers'][0]['shards']
        validated=command(build,'eps_training_control',settings,'verify-job')
        if not validated['paused'] or validated['next_round']!=plan['boundary']:
            raise ValueError('replacement training generation is not paused at its checkpoint')
        receipt=dict(kind='eps-rollover-ready-v1',run=plan['run'],boundary=plan['boundary'],
                     plan_sha256=hashlib.sha256(canonical(plan)).hexdigest(),
                     replica_heads=heads,worker_journals=journals,gradient_replicas=mirrors,training=validated)
        publish(directory/'ready.json',receipt)
        return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True,type=pathlib.Path)
    parser.add_argument('--config',required=True,type=pathlib.Path)
    parser.add_argument('--action',required=True,choices=['prepare','verify','retire'])
    parser.add_argument('--destination',type=pathlib.Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if args.action=='prepare':
        if args.destination is None:parser.error('prepare requires --destination')
        result=prepare(cfg,args.build.resolve(),args.destination)
    elif args.action=='verify':result=verify(cfg,args.build.resolve())
    else:
        from retire_training_generation import retire
        result=retire(cfg,args.build.resolve())
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
