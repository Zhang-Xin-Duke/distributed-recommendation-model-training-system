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
    return dict(pid=pid, birth=birth(pid), group=os.getpgrp())


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
    result['live_wrapper'] = alive(result.get('wrapper'))
    result['live_child'] = alive(result.get('child'))
    result['state'] = 'running' if result['live_child'] else 'starting' if result['live_wrapper'] else 'exited' if 'wrapper' in result else 'not_started'
    if result['state'] == 'running':
        output = path / 'stdout'
        lines = output.read_text(errors='replace').splitlines() if output.exists() else []
        result['ready_endpoint'] = next((line[6:].strip() for line in lines if line.startswith('READY ')), None)
    return result


def ensure(root, run, role, spec):
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


def dispatch(request):
    handlers=dict(ensure=ensure,status=status,stop=stop,read=read,hardware=hardware)
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
