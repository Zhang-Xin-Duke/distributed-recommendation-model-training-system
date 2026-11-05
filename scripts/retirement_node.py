"""Node-local seal/commit participant for training-generation retirement."""
import contextlib
import fcntl
import hashlib
import json
import pathlib
import re
import socket
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'adapters/entropy'))
from training_journal import canonical,sync_directory
from rotate_training_checkpoint import publish
from retirement_storage import require_stopped,lock_directory,file_metadata,payload_exists,inspect


def sha(value):return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    with pathlib.Path(path).open('rb') as stream:raw=stream.read((32<<20)+1)
    if len(raw)>32<<20:raise ValueError('retirement metadata exceeds its budget')
    return json.loads(raw)


def validate(spec):
    if (spec.get('schema_version')!=1 or not re.fullmatch('[a-zA-Z0-9_-]{1,96}',spec.get('node_id',''))
            or not isinstance(spec.get('members'),list) or spec['members']!=sorted(set(spec['members']))
            or spec['node_id'] not in spec['members'] or not 1<=len(spec['members'])<=64
            or type(spec.get('boundary')) is not int or not 0<spec['boundary']<2**63
            or not isinstance(spec.get('source_run'),str) or not spec['source_run']
            or not isinstance(spec.get('new_run'),str) or not spec['new_run'] or spec['new_run']==spec['source_run']
            or not re.fullmatch('[0-9a-f]{64}',spec.get('cohort_sha256',''))
            or not isinstance(spec.get('resources'),list) or not spec['resources']):
        raise ValueError('invalid node retirement specification')
    directory=pathlib.Path(spec['directory'])
    if not directory.is_absolute() or directory.is_symlink():raise ValueError('retirement state requires an absolute node directory')
    return directory


def certificate(directory,spec):
    value=read(directory/'certificate.json')
    if (value.get('kind')!='eps-retirement-node-certificate-v1' or value.get('spec_sha256')!=sha(spec)
            or value.get('host')!=socket.gethostname()):
        raise ValueError('node retirement certificate differs from this specification or host')
    return value


def marker(spec,cert):
    return dict(kind='eps-retired-generation-v1',source_run=spec['source_run'],new_run=spec['new_run'],
        boundary=spec['boundary'],certificate_sha256=sha(cert))


def receipt(spec,cert):
    return dict(kind='eps-retirement-node-sealed-v1',node_id=spec['node_id'],host=cert['host'],
        cohort_sha256=spec['cohort_sha256'],certificate_sha256=sha(cert),
        files=len(cert['files']),path_bytes=sum(f['bytes'] for f in cert['files']))


def check_markers(spec,cert):
    expected=marker(spec,cert)
    for root in cert['roots']:
        if read(pathlib.Path(root['path'])/'retired.json')!=expected:raise ValueError('source retirement marker differs')


def check_files(cert):
    for item in cert['files']:
        path=pathlib.Path(item['path'])
        if payload_exists(path) and file_metadata(path)!=item:raise ValueError('certified retirement payload changed')


def unlink_payload(path):pathlib.Path(path).unlink()


def completion(spec,cert,decision):
    sealed=receipt(spec,cert)
    return dict(kind='eps-retirement-node-complete-v1',**{k:sealed[k] for k in
        ['node_id','host','cohort_sha256','certificate_sha256','files','path_bytes']},decision_sha256=sha(decision))


def operate(op,spec,decision=None):
    directory=validate(spec);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'operation.lock').open('a+b') as operation,contextlib.ExitStack() as stack:
        fcntl.flock(operation,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish(directory/'spec.json',spec)
        done=directory/'complete.json'
        if done.exists():
            cert=certificate(directory,spec);check_markers(spec,cert)
            value=read(done)
            prior_decision=read(directory/'decision.json')
            if value!=completion(spec,cert,prior_decision) or any(payload_exists(f['path']) for f in cert['files']):
                raise ValueError('completed retirement payload or receipt changed')
            if op=='seal':return receipt(spec,cert)
            if op=='commit':
                if read(directory/'decision.json')!=decision:raise ValueError('completed retirement decision differs')
                return value
            if op=='status':return dict(state='complete',receipt=value)
            raise ValueError('unknown retirement node operation')
        if op=='status':
            return dict(state='sealed' if (directory/'sealed.json').exists() else 'prepared' if (directory/'certificate.json').exists() else 'not_prepared')
        # Published checkpoint files are immutable. A controller that only owns
        # those exports can stay alive; mutable journals/replicas require
        # terminal local writer handles as well as their exclusive locks.
        if spec['handles'] or any(r.get('kind')!='checkpoint' for r in spec['resources']):
            require_stopped(spec['handles'])
        if (directory/'certificate.json').exists():
            cert=certificate(directory,spec)
            for item in cert['roots']:lock_directory(stack,item['path'])
        elif op=='seal':
            roots,files=inspect(spec['resources'],spec['source_run'],spec['boundary'],
                [*spec['protected'],str(directory)],spec['build'],stack,replacement_run=spec['new_run'])
            cert=dict(kind='eps-retirement-node-certificate-v1',spec_sha256=sha(spec),
                host=socket.gethostname(),roots=roots,files=files)
            publish(directory/'certificate.json',cert)
        else:raise ValueError('seal node storage before committing retirement')
        check_files(cert)
        if op=='seal':
            # No payload is removed during prepare. Markers stop a new old-run
            # writer even after this participant releases its filesystem locks.
            for item in cert['roots']:publish(pathlib.Path(item['path'])/'retired.json',marker(spec,cert))
            value=receipt(spec,cert);publish(directory/'sealed.json',value);return value
        if op!='commit':raise ValueError('unknown retirement node operation')
        check_markers(spec,cert)
        sealed=read(directory/'sealed.json')
        if sealed!=receipt(spec,cert):raise ValueError('node sealing receipt differs')
        if not isinstance(decision,dict):raise ValueError('retirement has no all-nodes decision')
        members=decision.get('members',[])
        if (decision.get('kind')!='eps-distributed-retirement-commit-v1'
                or decision.get('cohort_sha256')!=spec['cohort_sha256']
                or [m.get('node_id') for m in members]!=spec['members']
                or any(not re.fullmatch('[0-9a-f]{64}',m.get('certificate_sha256','')) for m in members)
                or next(m['certificate_sha256'] for m in members if m['node_id']==spec['node_id'])!=sha(cert)):
            raise ValueError('retirement lacks the matching all-nodes seal decision')
        publish(directory/'decision.json',decision)
        for item in cert['files']:
            path=pathlib.Path(item['path'])
            if payload_exists(path):unlink_payload(path);sync_directory(path.parent)
        value=completion(spec,cert,decision)
        publish(done,value);return value


def dispatch(request):
    return operate(request['op'],request['spec'],request.get('decision'))


def main():
    if sys.argv[1:]!=['--serve']:raise ValueError('use the bounded node stream')
    while True:
        raw=sys.stdin.buffer.readline((8<<20)+1)
        if not raw:return
        if len(raw)>8<<20 or not raw.endswith(b'\n'):raise ValueError('retirement request exceeds stream budget')
        try:response=dict(ok=True,result=dispatch(json.loads(raw)))
        except Exception as error:response=dict(ok=False,error=str(error))
        sys.stdout.buffer.write(canonical(response)+b'\n');sys.stdout.buffer.flush()


if __name__=='__main__':main()
