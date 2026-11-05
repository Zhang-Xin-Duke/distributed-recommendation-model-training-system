"""Publish a stopped distributed training source without copying remote logs.

The joint parameter checkpoint is streamed to the controller by the native
save-job command. Worker, primary and replica paths retain their owning hosts.
Connection credentials remain in the caller's private fleet configuration.
"""
import copy
import hashlib
import json
import pathlib
import re

from rotate_training_checkpoint import publish
from training_journal import identity


def node_ids(nodes):
    ids=[n.get('node_id',f'node-{h}' if n.get('local') else None) for h,n in enumerate(nodes)]
    if (any(not isinstance(i,str) or not re.fullmatch('[a-zA-Z0-9_-]{1,96}',i) for i in ids)
            or len(set(ids))!=len(ids) or 'checkpoint-controller' in ids):
        raise ValueError('source export requires distinct stable node IDs')
    return ids


def save_checkpoint(command,settings,workers,reports,directory):
    directory=pathlib.Path(directory).resolve()
    directory.parent.mkdir(parents=True,exist_ok=True)
    if [r['worker'] for r in reports]!=list(range(len(workers))):
        raise ValueError('source export Worker reports are not ordered')
    inputs=[dict(worker=r['worker'],input_state=r['input_state']) for r in reports]
    model=identity(workers[0])['model']
    base=dict(settings,tables=[workers[0]['table']],model=model,workers=inputs)
    manifest=directory/'manifest.json'
    if manifest.exists():
        # A previous save may have committed before its manager phase was
        # recorded. Validate the existing immutable checkpoint before adoption.
        command('eps_training_control',dict(run=settings['run']+'-verify',manifest=str(manifest)),'restore-config')
        value=json.loads(manifest.read_text())
        if (value['source_run']!=settings['run'] or value['workers']!=inputs
                or value['tables']!=base['tables'] or value['model']!=model
                or value['progress']['completed']!=[r['input_state']['next_iteration'] for r in reports]):
            raise ValueError('published source checkpoint differs from the completed training generation')
        return dict(manifest=str(manifest),adopted=True)
    return command('eps_training_control',dict(base,checkpoint_directory=str(directory)),'save-job')


def publish_source(directory,run,nodes,manager,workers,primaries,replicas,mirrors,checkpoint,cleanup):
    directory=pathlib.Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    ids=node_ids(nodes);count=len(nodes)
    handles={i:[] for i in ids};roles={}
    for record in cleanup:
        if not record['owned_processes_gone']:
            raise ValueError('source export requires every owned role to be stopped')
        home=record['node']
        if not 0<=home<count:raise ValueError('stopped role is outside source placement')
        roles[record['role']]=home
        state=record['prior_state']
        if state.get('live_child') or state.get('live_wrapper'):
            raise ValueError('stopped role receipt retains a live process')
        for kind in ('wrapper','child'):
            if kind in state:
                handle=state[kind]
                if not handle.get('host') or handle.get('birth') is None:
                    raise ValueError('stopped role lacks host/process birth evidence')
                handles[ids[home]].append(copy.deepcopy(handle))
    assignments={}
    for home in range(count):
        for kind in ('worker','primary','replica'):
            role=f'{kind}-{home}-g0'
            if role not in roles:raise ValueError('source storage role has no stopped receipt: '+role)
            assignments[f'{kind}-{home}']=ids[roles[role]]
    for cfg in mirrors:
        worker=cfg['worker_config']['worker'];role=f'journal-mirror-{worker}-g0'
        if role not in roles:raise ValueError('source mirror has no stopped receipt')
        assignments[f'mirror-{worker}']=ids[roles[role]]
    if any(not values for values in handles.values()):raise ValueError('source storage host has no stopped-process evidence')
    # This additional participant only owns immutable, published checkpoint
    # files. It does not stop the controller that coordinates the next run.
    assignments['checkpoint']='checkpoint-controller';handles['checkpoint-controller']=[]
    manifest=pathlib.Path(checkpoint['manifest']).resolve()
    value=json.loads(manifest.read_text())
    if value['source_run']!=run:raise ValueError('source checkpoint run differs')
    descriptor=dict(kind='eps-stopped-training-source-v1',run=run,
        checkpoint_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        job=dict(manifest=str(manifest),workers=workers,shards=primaries,
                 source_replicas=replicas,source_mirrors=mirrors),
        deployment=dict(nodes=[dict(node_id=i,bin_dir=n['bin_dir'],
            retirement_directory=str(pathlib.PurePath(n['root'])/'results'/run/'retirement'/i))
            for i,n in zip(ids,nodes)]+[dict(node_id='checkpoint-controller',bin_dir=manager['bin_dir'],
                retirement_directory=str(directory/'retirement'))],
            resource_nodes=assignments,handles=handles))
    publish(directory/'source.json',descriptor)
    return descriptor
