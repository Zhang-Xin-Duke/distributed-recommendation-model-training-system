"""Retire stopped generation payloads behind a durable replacement receipt.

All storage paths must be accessible to this controller. Distributed role
supervision supplies stopped-role receipts separately; storage locks are the
final exclusion check. Only the registered, inspected payload files are unlinked.
"""
import contextlib
import copy
import fcntl
import hashlib
import json
import os
import pathlib
import stat
import socket

from rotate_training_checkpoint import command,publish
from training_journal import WorkerJournal,identity,canonical,sync_directory
from role_node import alive


def require_stopped(handles):
    if (not handles or any(handle.get('host')!=socket.gethostname() or
            (handle.get('birth') is None and type(handle.get('terminal_exit_code')) is not int)
            or alive(handle) for handle in handles)):
        raise ValueError('retirement requires confirmed stopped source-role handles')


def file_metadata(path):
    path=pathlib.Path(path)
    info=path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('retirement requires regular payload files')
    return dict(path=str(path.absolute()),device=info.st_dev,inode=info.st_ino,
                bytes=info.st_size,mtime_ns=info.st_mtime_ns)


def lock_directory(stack,path):
    path=pathlib.Path(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('retirement storage directory is missing or is a symlink')
    stream=stack.enter_context((path/'lock').open('a+b'))
    fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return path.resolve()


def unlink_payload(path):
    pathlib.Path(path).unlink()


def retire(plan,build):
    directory=pathlib.Path(plan['directory']).resolve()
    if json.loads((directory/'plan.json').read_text())!=plan:
        raise ValueError('retirement plan differs from published generation')
    ready=json.loads((directory/'ready.json').read_text())
    if (ready['kind']!='eps-rollover-ready-v1' or ready['run']!=plan['run']
            or ready['boundary']!=plan['boundary']
            or ready['plan_sha256']!=hashlib.sha256(canonical(plan)).hexdigest()):
        raise ValueError('retirement lacks the matching durable replacement receipt')
    # A completed operation remains replayable after a later generation retires
    # this generation's own checkpoint and replicas. It performs no new writes.
    if (directory/'retirement.json').exists():
        pending=json.loads((directory/'retirement.pending.json').read_text())
        result=json.loads((directory/'retirement.json').read_text())
        digest=hashlib.sha256(canonical(pending)).hexdigest()
        if (pending['plan_sha256']!=ready['plan_sha256'] or result['new_run']!=plan['run']
                or result['boundary']!=plan['boundary'] or result['certificate_sha256']!=digest):
            raise ValueError('completed retirement receipt binding differs')
        for item in pending['roots']:
            marker=json.loads((pathlib.Path(item['path'])/'retired.json').read_text())
            if marker['certificate_sha256']!=digest:raise ValueError('completed retirement marker differs')
        if any(pathlib.Path(item['path']).exists() for item in pending['files']):
            raise ValueError('completed retirement payload has reappeared')
        return result
    # Revalidate the independent checkpoint copy and current replica authority.
    command(build,'eps_training_control',dict(run=plan['run'],manifest=plan['manifest']),'restore-config')
    for cfg in plan['replicas']:
        head=command(build,'eps_replica_control',cfg,'head')
        if not head['baseline'] or head['base_round']!=plan['boundary'] or head['next_round']<plan['boundary']:
            raise ValueError('replacement replica authority is unavailable at the retirement boundary')
    job=json.loads((directory/'intent.json').read_text())['job']
    require_stopped(job.get('source_role_handles',[]))
    boundary=plan['boundary']
    certificate=directory/'retirement.pending.json'
    with (directory/'retirement.lock').open('a+b') as operation,contextlib.ExitStack() as stack:
        fcntl.flock(operation,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if certificate.exists():
            pending=json.loads(certificate.read_text())
            if pending['plan_sha256']!=ready['plan_sha256'] or pending['new_run']!=plan['run'] or pending['boundary']!=boundary:
                raise ValueError('retirement certificate belongs to another replacement')
            for item in pending['roots']:
                lock_directory(stack,item['path'])
        else:
            source=json.loads(pathlib.Path(plan['manifest']).read_text())
            source_run=source['source_run']
            roots=[];files=[];locked=set()

            def register(path,kind):
                path=pathlib.Path(path).resolve()
                if path==directory or path in directory.parents or directory in path.parents:
                    raise ValueError('retirement storage overlaps the replacement generation')
                if path in locked:
                    raise ValueError('retirement storage roles overlap')
                locked.add(path);roots.append(dict(path=str(path),kind=kind))
                return path

            def retain(items):
                if len(files)+len(items)>65536:
                    raise ValueError('retirement payload manifest exceeds 65536 files; rotate more frequently')
                files.extend(items)

            def journal(cfg,path,mirror=False):
                root=register(path,'worker_mirror' if mirror else 'worker')
                handle=stack.enter_context(WorkerJournal(root,identity(cfg),
                    cfg.get('journal_record_bytes',32<<20),cfg.get('journal_disk_bytes',1<<30),allow_sealed=True))
                if handle.rounds()!=list(range(handle.identity['base_round'],boundary)):
                    raise ValueError('source Worker journal differs from the checkpoint frontier')
                if mirror:
                    owner=json.loads((root/'mirror-owner.json').read_text())
                    expected=cfg['journal_mirror']
                    if (owner['identity_sha256']!=handle.digest or owner['writer']!=expected['writer']
                            or owner['generation']!=expected['generation']):
                        raise ValueError('source gradient mirror ownership differs')
                    if (root/'mirror-pending.json').exists() or list(root.glob('.mirror-part-*')):
                        raise ValueError('source gradient mirror contains an incomplete upload')
                if len(handle.rounds())>16384:raise ValueError('retirement Worker window exceeds 16384 records')
                retain([file_metadata(root/f'round-{n}.epsw') for n in handle.rounds()])

            for cfg in job['workers']:
                if cfg['run']!=source_run:raise ValueError('source journal run differs')
                journal(cfg,cfg['journal_directory'])
            mirrored={w['worker'] for w in job['workers'] if w.get('journal_mirror')}
            mirrors=job.get('source_mirrors',[])
            if {m['worker_config']['worker'] for m in mirrors}!=mirrored or len(mirrors)!=len(mirrored):
                raise ValueError('retirement requires every source gradient-mirror storage path')
            for mirror in mirrors:
                worker=mirror['worker_config']['worker']
                if identity(mirror['worker_config'])!=identity(job['workers'][worker]):
                    raise ValueError('source gradient-mirror identity differs')
                journal(job['workers'][worker],mirror['directory'],True)
            replicas=job.get('source_replicas',[])
            if len(replicas)!=len(job['shards']):
                raise ValueError('retirement requires every source parameter-replica storage path')

            def replica(cfg,path,latest=True,snapshots=False):
                root=register(path,'replica')
                local=dict(cfg,replica_directory=str(root))
                inspected=command(build,'eps_replica_control',local,'offline-head')
                if (not inspected['baseline'] or inspected['base_round']!=cfg.get('replica_base_round',0)
                        or inspected['next_round']>boundary or (latest and inspected['next_round']!=boundary)):
                    raise ValueError('source replica has updates outside the checkpoint frontier')
                snapshot_files=[]
                if snapshots:
                    snapshot_files=command(build,'eps_replica_control',dict(local,retire_before=boundary),'offline-snapshots')['files']
                lock_directory(stack,root)
                for item in [*inspected['files'],*snapshot_files]:
                    if file_metadata(item['path'])!=item:
                        raise ValueError('source payload changed between inspection and exclusive lock')
                actual={str(p.absolute()) for p in root.iterdir() if p.name=='baseline.eps'
                    or (p.name.startswith('round-') and p.name.endswith('.epsr'))
                    or (p.name.startswith('upload-') and p.suffix in ('.part','.size'))}
                if actual!={f['path'] for f in inspected['files']}:
                    raise ValueError('source replica payload set changed during retirement')
                retain(inspected['files']);retain(snapshot_files)

            for home,(cfg,standby) in enumerate(zip(job['shards'],replicas)):
                if (cfg['run_id']!=source_run or standby['run_id']!=source_run
                        or cfg['partition']!=standby['partition'] or cfg['partition']['shard_id']!=home
                        or cfg['tables']!=standby['tables']):
                    raise ValueError('source primary/replica retirement membership differs')
                replica(standby,standby['replica_directory'])
                parent=pathlib.Path(cfg['synchronous_training']['replica']['local_directory']).resolve()
                parent=register(parent,'primary_parent')
                lock_directory(stack,parent)
                terms=sorted(parent.glob('term-*'))
                if not terms:raise ValueError('source primary has no retained term journals')
                for index,term in enumerate(terms):
                    current=term.name=='term-'+str(cfg['synchronous_training']['replica']['term'])
                    replica(cfg,term,current,snapshots=index==0)
            # Original checkpoint exports are immutable and already staged in
            # the replacement. Retire their old names when still registered.
            old=pathlib.Path(job['manifest'])
            if old.exists():
                if hashlib.sha256(old.read_bytes()).hexdigest()!=plan['source_checkpoint_sha256']:
                    raise ValueError('source checkpoint manifest changed before retirement')
                command(build,'eps_training_control',dict(run=plan['run'],manifest=str(old)),'restore-config')
                for entry in source['parameters']:
                    retain([file_metadata(old.parent/entry['file'])])
                retain([file_metadata(old)])
            previous=job.get('source_generation')
            if previous:
                previous_root=pathlib.Path(previous['directory'])
                if (previous['run']!=source_run or previous['boundary']>boundary
                        or json.loads((previous_root/'plan.json').read_text())!=previous):
                    raise ValueError('source generation baseline ownership differs')
                previous_ready=json.loads((previous_root/'ready.json').read_text())
                if previous_ready['plan_sha256']!=hashlib.sha256(canonical(previous)).hexdigest():
                    raise ValueError('source generation lacks its verified initial baseline')
                initial=pathlib.Path(previous['manifest'])
                command(build,'eps_training_control',dict(run=plan['run'],manifest=str(initial)),'restore-config')
                old_initial=json.loads(initial.read_text())
                if old_initial['progress']['completed']!=[previous['boundary']]*len(job['workers']):
                    raise ValueError('source staged baseline frontier differs')
                retain([file_metadata(initial.parent/p['file']) for p in old_initial['parameters']])
                retain([file_metadata(initial)])
            unique={}
            for item in files:
                path=pathlib.Path(item['path']).resolve()
                if path==directory or directory in path.parents:
                    raise ValueError('retirement payload overlaps replacement data')
                if item['path'] in unique and unique[item['path']]!=item:
                    raise ValueError('retirement payload metadata conflicts')
                unique[item['path']]=item
            pending=dict(kind='eps-generation-retirement-v1',source_run=source_run,new_run=plan['run'],
                boundary=boundary,plan_sha256=ready['plan_sha256'],roots=roots,files=list(unique.values()))
            publish(certificate,pending)
        marker=dict(kind='eps-retired-generation-v1',source_run=pending['source_run'],new_run=plan['run'],
                    boundary=boundary,certificate_sha256=hashlib.sha256(canonical(pending)).hexdigest())
        # Mark every stopped role durably before unlinking any payload. A retry
        # can finish either interrupted marking or interrupted deletion.
        for item in pending['roots']:
            root=pathlib.Path(item['path'])
            if directory==root or root in directory.parents:raise ValueError('retirement root overlaps replacement')
            publish(root/'retired.json',marker)
        for item in pending['files']:
            path=pathlib.Path(item['path'])
            if path.exists() and file_metadata(path)!=item:
                raise ValueError('retirement payload changed after certification')
        for item in pending['files']:
            path=pathlib.Path(item['path'])
            if path.exists():
                unlink_payload(path);sync_directory(path.parent)
        result=dict(kind='eps-retirement-complete-v1',source_run=pending['source_run'],new_run=plan['run'],
            boundary=boundary,certificate_sha256=marker['certificate_sha256'],
            unlinked_paths=len(pending['files']),unlinked_path_bytes=sum(f['bytes'] for f in pending['files']),
            note='Path bytes include hard-link aliases; this is not a filesystem-block reclamation measurement')
        publish(directory/'retirement.json',result)
        return result
