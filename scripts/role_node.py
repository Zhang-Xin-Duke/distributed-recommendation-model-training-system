"""Durable named process roles controlled over a bounded JSON node stream."""
import base64
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'adapters/entropy'))
from training_journal import canonical, sync_directory
from journal_handoff import publish


def write(path, value):
    pending = path.parent / ('.pending-' + uuid.uuid4().hex)
    try:
        with pending.open('xb') as f:
            f.write(canonical(value)); f.flush(); os.fsync(f.fileno())
        pending.replace(path); sync_directory(path.parent)
    finally:
        pending.unlink(missing_ok=True)


def directory(root, run, role):
    root = pathlib.Path(root)
    if not root.is_absolute() or not re.fullmatch(r'[a-z0-9_-]{1,96}', run) or not re.fullmatch(r'[a-z0-9_-]{1,64}', role):
        raise ValueError('invalid node role namespace')
    path = root / 'results' / run / 'roles' / role
    path.mkdir(parents=True, exist_ok=True)
    return path


def birth(pid):
    if type(pid) is not int or pid <= 0:
        return None
    if pathlib.Path('/proc').exists():
        try:
            stat = pathlib.Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
            return None if stat[0] == 'Z' else stat[19]
        except FileNotFoundError:
            return None
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'lstart='], capture_output=True, text=True)
    fields = result.stdout.strip().split(None, 1)
    return fields[1] if result.returncode == 0 and len(fields) == 2 and not fields[0].startswith('Z') else None


def handle():
    pid = os.getpid()
    return dict(pid=pid, birth=birth(pid), group=os.getpgrp(), host=socket.gethostname())


def alive(record):
    return bool(record and record.get('birth') is not None and birth(record['pid']) == record['birth'])


@contextlib.contextmanager
def locked(path, timeout=20):
    # Spawned wrappers/children inherit this open file description. If the RPC
    # process exits between fork and handle publication, a retry cannot overtake it.
    lock = (path / 'launch.lock').open('a+b')
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('node role publication still holds its launch lock')
                time.sleep(.02)
        yield lock.fileno()
    finally:
        # Do not call LOCK_UN: an inherited copy may still be publishing a handle.
        lock.close()


def snapshot(path):
    result = {}
    for name in ['wrapper', 'child', 'result']:
        p = path / (name + '.json')
        if p.exists():
            result[name] = json.loads(p.read_text())
    progress = path / 'progress.json'
    if progress.exists():
        result['training_progress'] = json.loads(progress.read_text())
    result['live_wrapper'] = alive(result.get('wrapper'))
    result['live_child'] = alive(result.get('child'))
    result['state'] = 'running' if result['live_child'] else 'starting' if result['live_wrapper'] else 'exited' if 'wrapper' in result else 'not_started'
    if result['state'] == 'running':
        output = path / 'stdout'
        lines = output.read_text(errors='replace').splitlines() if output.exists() else []
        result['ready_endpoint'] = next((line[6:].strip() for line in lines if line.startswith('READY ')), None)
    return result


def ensure(root, run, role, spec):
    if spec.get('reservation_token'):
        from node_resources import launch_guard
        with launch_guard(root, run, role, spec['reservation_token']):
            return ensure_unreserved(root, run, role, spec)
    return ensure_unreserved(root, run, role, spec)


def ensure_unreserved(root, run, role, spec):
    path = directory(root, run, role)
    if not isinstance(spec.get('command'), list) or not spec['command'] or any(not isinstance(x,str) for x in spec['command']):
        raise ValueError('node role command must be an argument list')
    if any(not isinstance(k,str) or not isinstance(v,str) for k,v in spec.get('environment',{}).items()):
        raise ValueError('node role environment must contain string overrides')
    with locked(path) as fd:
        publish(path / 'intent.json', spec)
        existing = snapshot(path)
        if existing['state'] != 'not_started':
            return existing
        if 'config' in spec:
            publish(path / 'config.json', spec['config'])
        with (path / 'wrapper.stderr').open('ab') as errors:
            subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve()), '--wrapper', str(path), str(fd)],
                             stdin=subprocess.DEVNULL, stdout=errors, stderr=errors,
                             start_new_session=True, pass_fds=(fd,))
    # Query after acquiring the inherited lock: this also waits for the child's
    # own durable identity before reporting a successful launch.
    with locked(path):
        return snapshot(path)


def wrapper(path, fd):
    write(path / 'wrapper.json', handle())
    with (path / 'stdout').open('wb') as output, (path / 'stderr').open('wb') as errors:
        child = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).resolve()), '--child', str(path), str(fd)],
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=errors, pass_fds=(fd,))
        os.close(fd)
        code = child.wait()
    write(path / 'result.json', dict(exit_code=code, finished=time.time()))


def child(path, fd):
    write(path / 'child.json', handle())
    spec = json.loads((path / 'intent.json').read_text())
    command = spec['command']
    if 'config' in spec:
        position=spec.get('config_position',len(command))
        if type(position) is not int or not 1<=position<=len(command):raise ValueError('invalid role config argument position')
        command = [*command[:position], '--config', str(path / 'config.json'), *command[position:]]
    env = dict(os.environ, **spec.get('environment',{}))
    os.chdir(spec.get('cwd', str(path)))
    os.close(fd)
    os.execvpe(command[0], command, env)


def status(root, run, role):
    path = directory(root, run, role)
    with locked(path):
        return snapshot(path)


def stop(root, run, role):
    path = directory(root, run, role)
    with locked(path):
        state = snapshot(path)
        record = state.get('child') if state['live_child'] else state.get('wrapper') if state['live_wrapper'] else None
        if record:
            wrapper_record = state['wrapper']
            try:group=os.getpgid(record['pid'])
            except ProcessLookupError:group=None
            if record['group'] != wrapper_record['pid'] or (group is not None and group != record['group']):
                raise ValueError('node role process group differs')
            if group is not None:
                try:os.killpg(record['group'], signal.SIGTERM)
                except ProcessLookupError:pass
        else:
            return dict(owned_processes_gone=True, prior_state=state)
    deadline = time.monotonic()+6
    while time.monotonic() < deadline:
        state = status(root,run,role)
        if not state['live_child'] and not state['live_wrapper']:
            break
        time.sleep(.05)
    else:
        state = status(root,run,role)
        for name in ['child','wrapper']:
            h = state.get(name)
            if alive(h):
                try:group=os.getpgid(h['pid'])
                except ProcessLookupError:continue
                if group != state['wrapper']['pid']:
                    raise ValueError('node role process group changed before escalation')
                try:os.killpg(state['wrapper']['pid'],signal.SIGKILL)
                except ProcessLookupError:pass
                break
        deadline = time.monotonic()+3
        while time.monotonic()<deadline and any(alive(state.get(n)) for n in ['wrapper','child']):
            time.sleep(.05)
    state = status(root,run,role)
    return dict(owned_processes_gone=not state['live_child'] and not state['live_wrapper'], prior_state=state)


def read(root, run, role, name, offset=0, size=262144):
    path = directory(root,run,role)
    if name not in ['stdout','stderr','wrapper.stderr'] or type(offset) is not int or offset<0 or type(size) is not int or not 1<=size<=262144:
        raise ValueError('invalid node role output read')
    file=path/name
    if not file.exists():return dict(data='',next_offset=offset,eof=True)
    with file.open('rb') as f:
        f.seek(offset);data=f.read(size)
    return dict(data=base64.b64encode(data).decode(),next_offset=offset+len(data),eof=offset+len(data)>=file.stat().st_size)


def hardware(root, run, role, gpu=False):
    directory(root,run,role)
    from training_node import hardware as inspect
    return inspect(gpu)


def training_gate(root,run,role,release=False):
    path=directory(root,run,role)
    if type(release) is not bool:raise ValueError('invalid gate release request')
    with locked(path):
        state=snapshot(path)
        if not state['live_child']:raise ValueError('training gate requires a live owned Worker')
        cfg=json.loads((path/'config.json').read_text())
        gate=path/'training-start';worker=cfg.get('worker')
        if (cfg.get('run')!=run or type(worker) is not int or worker<0
                or pathlib.Path(cfg.get('start_gate','')).absolute()!=gate.absolute()):
            raise ValueError('Worker gate identity/path differs from its owned role')
        marker=path/('training-start.worker-'+str(worker)+'.ready')
        ready=json.loads(marker.read_text()) if marker.exists() else None
        expected=dict(run=run,worker=worker,warmed_model=True)
        if ready is not None and ready!=expected:raise ValueError('Worker warmup readiness identity differs')
        if release:
            if ready is None:raise ValueError('Worker model warmup is not ready')
            publish(gate,dict(run=run,role=role,worker=worker))
        released=gate.exists()
        if released and json.loads(gate.read_text())!=dict(run=run,role=role,worker=worker):
            raise ValueError('Worker gate release identity differs')
        return dict(run=run,worker=worker,warmed_model=ready is not None,released=released)


def journal_report(root,run,role):
    """Validate completed Worker payloads on their owner; return bounded metadata."""
    from training_journal import WorkerJournal,identity
    from gradient_replica import ReplicaClient
    path=directory(root,run,role)
    with locked(path):
        state=snapshot(path)
        if state['live_child'] or state.get('result',{}).get('exit_code')!=0:
            raise ValueError('journal report requires a successfully completed owned Worker')
        cfg=json.loads((path/'config.json').read_text())
        if cfg.get('run')!=run or pathlib.Path(cfg.get('journal_directory','')).absolute()!=(path/'journal').absolute():
            raise ValueError('Worker journal identity/path differs from its owned role')
        with WorkerJournal(cfg['journal_directory'],identity(cfg),cfg.get('journal_record_bytes',32<<20),
                cfg.get('journal_disk_bytes',1<<30)) as journal:
            rounds=journal.rounds();losses=[];digests=[]
            if len(rounds)>10000:raise ValueError('journal report exceeds the round metadata bound')
            for iteration in rounds:
                losses.append(journal.get(iteration)['loss'])
                raw=(journal.root/f'round-{iteration}.epsw').read_bytes()
                digests.append(dict(iteration=iteration,sha256=hashlib.sha256(raw).hexdigest()))
            mirror=None
            if cfg.get('journal_mirror'):
                peer=ReplicaClient(cfg['journal_mirror'],journal)
                try:mirror=peer.call('status')
                finally:peer.close()
            return dict(kind='eps-node-worker-journal-v1',run=run,worker=cfg['worker'],
                identity_sha256=journal.digest,base_round=journal.identity['base_round'],rounds=rounds,
                loss_history=losses,record_sha256=digests,gradient_replica=mirror)


def training_preflight(root, source, bin_dir, module_dir, revision, gpu, datasets, checkpoint_root):
    import tempfile
    if subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True, timeout=10).strip() != revision:
        raise ValueError('node source revision differs from controller')
    for name in ('eps_server', 'eps_replica', 'eps_cache_agent', 'eps_training_coordinator', 'eps_training_prepare'):
        if not os.access(pathlib.Path(bin_dir) / name, os.X_OK):
            raise ValueError('missing training executable: ' + name)
    sys.path.insert(0, module_dir)
    import _eps_native
    if not hasattr(_eps_native.Backend, 'submit_synchronous'):
        raise ValueError('native training module is incompatible')
    for target in (root, checkpoint_root):
        path = pathlib.Path(target)
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path) as probe:
            probe.write(b'eps-preflight')
            probe.flush()
            os.fsync(probe.fileno())
    sys.path.insert(0, str(pathlib.Path(source) / 'adapters/entropy'))
    from recommendation_input import IndexedBatches
    for item in datasets:
        for split in ('train', 'validation'):
            IndexedBatches(item, split)[0]
    if gpu:
        import tensorflow as tf
        if not tf.__version__.startswith('2.20.') or len(tf.config.list_physical_devices('GPU')) != 1:
            raise ValueError('expected TensorFlow 2.20 with one visible GPU')
        return hardware(root, 'preflight', 'allocation', True)
    return {'ready': True}


def dispatch(request):
    handlers=dict(training_preflight=training_preflight,ensure=ensure,status=status,stop=stop,read=read,hardware=hardware,training_gate=training_gate,journal_report=journal_report)
    if request['op']=='role_config':
        path=directory(request['root'],request['run'],request['role'])
        with locked(path):
            state=snapshot(path)
            if state['live_child'] or state['live_wrapper']:raise ValueError('config transfer requires stopped role')
            return json.loads((path/'config.json').read_text())
    if request['op']=='replica_handoff':
        from replica_handoff import operation
        return operation(**{k:v for k,v in request.items() if k!='op'})
    if request['op']=='resources':
        from node_resources import operation
        return operation(**{k:v for k,v in request.items() if k!='op'})
    op=request['op']
    if op not in handlers:raise ValueError('unknown role node operation')
    return handlers[op](**{k:v for k,v in request.items() if k!='op'})


def response(request):
    try:return dict(ok=True,result=dispatch(request))
    except Exception as error:return dict(ok=False,error=type(error).__name__+': '+str(error))


if __name__=='__main__':
    if sys.argv[1:2]==['--wrapper']:wrapper(pathlib.Path(sys.argv[2]),int(sys.argv[3]))
    elif sys.argv[1:2]==['--child']:child(pathlib.Path(sys.argv[2]),int(sys.argv[3]))
    elif sys.argv[1:]==['--serve']:
        for line in sys.stdin:print(json.dumps(response(json.loads(line))),flush=True)
    else:
        result=response(json.load(sys.stdin));print(json.dumps(result));sys.exit(0 if result['ok'] else 1)
