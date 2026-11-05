"""Retire stopped generation payloads behind a durable replacement receipt.

All storage paths must be accessible to this controller. Distributed role
supervision supplies stopped-role receipts separately; storage locks are the
final exclusion check. Only the registered, inspected payload files are unlinked.
"""
import contextlib
import fcntl
import hashlib
import json
import pathlib

from rotate_training_checkpoint import command,publish
from training_journal import canonical,sync_directory
from retirement_storage import require_stopped, file_metadata, payload_exists, lock_directory, catalog, inspect


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
        if any(payload_exists(item['path']) for item in pending['files']):
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
            source_run,resources=catalog(job,plan)
            roots,files=inspect(resources,source_run,boundary,[directory],build,stack,replacement_run=plan['run'])
            pending=dict(kind='eps-generation-retirement-v1',source_run=source_run,new_run=plan['run'],
                boundary=boundary,plan_sha256=ready['plan_sha256'],roots=roots,files=files)
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
            if payload_exists(path) and file_metadata(path)!=item:
                raise ValueError('retirement payload changed after certification')
        for item in pending['files']:
            path=pathlib.Path(item['path'])
            if payload_exists(path):
                unlink_payload(path);sync_directory(path.parent)
        result=dict(kind='eps-retirement-complete-v1',source_run=pending['source_run'],new_run=plan['run'],
            boundary=boundary,certificate_sha256=marker['certificate_sha256'],
            unlinked_paths=len(pending['files']),unlinked_path_bytes=sum(f['bytes'] for f in pending['files']),
            note='Path bytes include hard-link aliases; this is not a filesystem-block reclamation measurement')
        publish(directory/'retirement.json',result)
        return result
