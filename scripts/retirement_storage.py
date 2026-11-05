"""Inspect old training storage while retaining its exclusive writer locks."""
import fcntl
import hashlib
import json
import os
import pathlib
import socket
import stat

from rotate_training_checkpoint import command
from training_journal import WorkerJournal, identity, canonical
from role_node import alive


def require_stopped(handles):
    if (not handles or any(handle.get('host')!=socket.gethostname() or
            (handle.get('birth') is None and type(handle.get('terminal_exit_code')) is not int)
            or alive(handle) for handle in handles)):
        raise ValueError('retirement requires confirmed stopped source-role handles')


def file_metadata(path):
    path=pathlib.Path(path);info=path.lstat()
    if not stat.S_ISREG(info.st_mode):raise ValueError('retirement requires regular payload files')
    return dict(path=str(path.absolute()),device=info.st_dev,inode=info.st_ino,
                bytes=info.st_size,mtime_ns=info.st_mtime_ns)


def payload_exists(path):return os.path.lexists(path)


def lock_directory(stack,path):
    path=pathlib.Path(path)
    if path.is_symlink() or not path.is_dir():raise ValueError('retirement storage directory is missing or is a symlink')
    stream=stack.enter_context((path/'lock').open('a+b'))
    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return path.resolve()


def catalog(job,plan,*,optional_checkpoint=True):
    """Build a complete logical resource roster without opening remote payloads."""
    source=json.loads(pathlib.Path(plan['manifest']).read_text());run=source['source_run']
    resources=[];workers=job['workers'];shards=job['shards']
    if [cfg['worker'] for cfg in workers]!=list(range(len(workers))):raise ValueError('source Worker membership differs')
    for cfg in workers:
        if cfg['run']!=run:raise ValueError('source journal run differs')
        resources.append(dict(key='worker-'+str(cfg['worker']),kind='worker',config=cfg,path=cfg['journal_directory']))
    mirrored={cfg['worker'] for cfg in workers if cfg.get('journal_mirror')}
    mirrors=job.get('source_mirrors',[])
    if {m['worker_config']['worker'] for m in mirrors}!=mirrored or len(mirrors)!=len(mirrored):
        raise ValueError('retirement requires every source gradient-mirror storage path')
    for mirror in mirrors:
        worker=mirror['worker_config']['worker']
        if identity(mirror['worker_config'])!=identity(workers[worker]):raise ValueError('source gradient-mirror identity differs')
        resources.append(dict(key='mirror-'+str(worker),kind='mirror',config=workers[worker],path=mirror['directory']))
    replicas=job.get('source_replicas',[])
    if len(replicas)!=len(shards):raise ValueError('retirement requires every source parameter-replica storage path')
    for home,(cfg,standby) in enumerate(zip(shards,replicas)):
        if (cfg['run_id']!=run or standby['run_id']!=run or cfg['partition']!=standby['partition']
                or cfg['partition']['shard_id']!=home or cfg['tables']!=standby['tables']):
            raise ValueError('source primary/replica retirement membership differs')
        resources.append(dict(key='replica-'+str(home),kind='replica',config=standby,path=standby['replica_directory']))
        resources.append(dict(key='primary-'+str(home),kind='primary',config=cfg,
            path=cfg['synchronous_training']['replica']['local_directory']))
    resources.append(dict(key='checkpoint',kind='checkpoint',path=job['manifest'],
        sha256=plan['source_checkpoint_sha256'],optional=optional_checkpoint))
    previous=job.get('source_generation')
    if previous:
        if previous['run']!=run or previous['boundary']>plan['boundary']:
            raise ValueError('source generation baseline ownership differs')
        # Its manager-side metadata is independently retained from remote logs.
        directory=pathlib.Path(previous['directory'])
        if json.loads((directory/'plan.json').read_text())!=previous:raise ValueError('source generation plan differs')
        ready=json.loads((directory/'ready.json').read_text())
        if ready['plan_sha256']!=hashlib.sha256(canonical(previous)).hexdigest():
            raise ValueError('source generation lacks its verified initial baseline')
        initial=pathlib.Path(previous['manifest']);old=json.loads(initial.read_text())
        if old['progress']['completed']!=[previous['boundary']]*len(workers):raise ValueError('source staged baseline frontier differs')
        resources.append(dict(key='staged-checkpoint',kind='checkpoint',path=str(initial),
            sha256=hashlib.sha256(initial.read_bytes()).hexdigest(),optional=False))
    return run,resources


def inspect(resources,source_run,boundary,protected,build,stack,*,replacement_run):
    roots=[];files=[];registered=set();protected=[pathlib.Path(p).resolve() for p in protected]
    def safe(path):
        path=pathlib.Path(path).resolve()
        if any(path==target or target in path.parents or path in target.parents for target in protected):
            raise ValueError('retirement storage overlaps the replacement generation')
        return path
    def register(path,kind):
        path=safe(path)
        if path in registered:raise ValueError('retirement storage roles overlap')
        registered.add(path);roots.append(dict(path=str(path),kind=kind));return path
    def retain(items):
        if len(files)+len(items)>65536:raise ValueError('retirement payload manifest exceeds 65536 files; rotate more frequently')
        for item in items:safe(item['path'])
        files.extend(items)
    def replica(cfg,path,latest=True,snapshots=False):
        root=register(path,'replica');local=dict(cfg,replica_directory=str(root))
        head=command(build,'eps_replica_control',local,'offline-head')
        if (not head['baseline'] or head['base_round']!=cfg.get('replica_base_round',0)
                or head['next_round']>boundary or (latest and head['next_round']!=boundary)):
            raise ValueError('source replica has updates outside the checkpoint frontier')
        snapshots=command(build,'eps_replica_control',dict(local,retire_before=boundary),'offline-snapshots')['files'] if snapshots else []
        lock_directory(stack,root)
        for item in [*head['files'],*snapshots]:
            if file_metadata(item['path'])!=item:raise ValueError('source payload changed between inspection and exclusive lock')
        actual={str(p.absolute()) for p in root.iterdir() if p.name=='baseline.eps'
            or (p.name.startswith('round-') and p.name.endswith('.epsr'))
            or (p.name.startswith('upload-') and p.suffix in ('.part','.size'))}
        if actual!={f['path'] for f in head['files']}:raise ValueError('source replica payload set changed during retirement')
        retain(head['files']);retain(snapshots)
    for resource in resources:
        kind=resource['kind'];path=resource['path']
        if kind in ('worker','mirror'):
            cfg=resource['config']
            if cfg['run']!=source_run:raise ValueError('source journal run differs')
            root=register(path,kind)
            journal=stack.enter_context(WorkerJournal(root,identity(cfg),cfg.get('journal_record_bytes',32<<20),
                cfg.get('journal_disk_bytes',1<<30),allow_sealed=True))
            if journal.rounds()!=list(range(journal.identity['base_round'],boundary)):
                raise ValueError('source Worker journal differs from the checkpoint frontier')
            if kind=='mirror':
                owner=json.loads((root/'mirror-owner.json').read_text());expected=cfg['journal_mirror']
                if (owner['identity_sha256']!=journal.digest or owner['writer']!=expected['writer'] or owner['generation']!=expected['generation']):
                    raise ValueError('source gradient mirror ownership differs')
                if (root/'mirror-pending.json').exists() or list(root.glob('.mirror-part-*')):
                    raise ValueError('source gradient mirror contains an incomplete upload')
            if len(journal.rounds())>16384:raise ValueError('retirement Worker window exceeds 16384 records')
            retain([file_metadata(root/f'round-{n}.epsw') for n in journal.rounds()])
        elif kind=='replica':replica(resource['config'],path)
        elif kind=='primary':
            cfg=resource['config'];parent=register(path,'primary_parent');lock_directory(stack,parent)
            terms=sorted(parent.glob('term-*'))
            if not terms:raise ValueError('source primary has no retained term journals')
            for index,term in enumerate(terms):
                replica(cfg,term,term.name=='term-'+str(cfg['synchronous_training']['replica']['term']),index==0)
        elif kind=='checkpoint':
            old=safe(path)
            if not old.exists() and resource.get('optional',False):continue
            if hashlib.sha256(old.read_bytes()).hexdigest()!=resource['sha256']:
                raise ValueError('source checkpoint manifest changed before retirement')
            command(build,'eps_training_control',dict(run=replacement_run,manifest=str(old)),'restore-config')
            source=json.loads(old.read_text());retain([file_metadata(old.parent/p['file']) for p in source['parameters']]);retain([file_metadata(old)])
        else:raise ValueError('unknown retirement storage resource')
    unique={}
    for item in files:
        if item['path'] in unique and unique[item['path']]!=item:raise ValueError('retirement payload metadata conflicts')
        unique[item['path']]=item
    return roots,list(unique.values())
