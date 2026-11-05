"""Verify real process-role identities and retry/stop behavior."""
import copy
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from transfer_worker_journals import NodeSessions


def main():
    node=dict(local=True,source=str(ROOT),python=sys.executable)
    checked=[]
    with tempfile.TemporaryDirectory() as tmp, NodeSessions(entrypoint='scripts/role_node.py') as sessions:
        root=pathlib.Path(tmp);base=dict(root=str(root),run='roles-check')
        def call(op,role,**extra):return sessions.invoke(node,dict(op=op,role=role,**base,**extra))
        marker=root/'effect'
        spec=dict(command=[sys.executable,'-c',"import pathlib,sys;open(sys.argv[1],'a').write('x')",str(marker)])
        first=call('ensure','one-shot',spec=spec)
        assert first['wrapper']['host']==first['child']['host']==socket.gethostname()
        checked.append('process_handles_include_storage_host_identity')
        deadline=time.monotonic()+10
        while 'result' not in (done:=call('status','one-shot')):
            assert time.monotonic()<deadline;time.sleep(.02)
        assert done['result']['exit_code']==0 and marker.read_text()=='x'
        repeated=call('ensure','one-shot',spec=spec)
        assert repeated['wrapper']['pid']==first['wrapper']['pid'] and marker.read_text()=='x'
        checked.append('completed_command_not_reexecuted')
        bad=copy.deepcopy(spec);bad['command'].append('different')
        try:call('ensure','one-shot',spec=bad)
        except ValueError:pass
        else:raise AssertionError('changed role intent accepted')
        checked.append('immutable_role_intent')
        spec=dict(command=[sys.executable,'-u','-c',"import time;print('READY 127.0.0.1:1234',flush=True);time.sleep(120)"])
        try:
            started=call('ensure','long-lived',spec=spec)
            # Lose the connection after a successful start and issue the same request.
            sessions.drop(json.dumps(node,sort_keys=True))
            repeated=call('ensure','long-lived',spec=spec)
            assert repeated['wrapper']['pid']==started['wrapper']['pid'] and repeated['child']['pid']==started['child']['pid']
            deadline=time.monotonic()+5
            while (current:=call('status','long-lived')).get('ready_endpoint') is None:
                assert time.monotonic()<deadline;time.sleep(.02)
            assert current['ready_endpoint']=='127.0.0.1:1234'
            checked += ['lost_start_ack_same_process','observed_readiness']
            # The child keeps running after its wrapper dies. Stop must find and
            # terminate that owned group instead of starting a replacement.
            os.kill(current['wrapper']['pid'],signal.SIGKILL)
            deadline=time.monotonic()+5
            while (orphan:=call('status','long-lived'))['live_wrapper']:
                assert time.monotonic()<deadline;time.sleep(.02)
            assert orphan['live_child']
            assert call('stop','long-lived')['owned_processes_gone']
            assert call('stop','long-lived')['owned_processes_gone']
            checked += ['orphan_child_cleanup','repeated_stop']
        finally:
            call('stop','long-lived')
        path=root/'results/roles-check/roles/gated-worker';path.mkdir(parents=True,exist_ok=True)
        gate=path/'training-start';trigger=root/'allow-warmup'
        script="import json,pathlib,sys,time; c=json.loads(pathlib.Path(sys.argv[2]).read_text()); p=pathlib.Path(c['start_gate']); t=pathlib.Path(c['trigger']);\nwhile not t.exists(): time.sleep(.01)\np.with_name(p.name+'.worker-0.ready').write_text(json.dumps(dict(run=c['run'],worker=0,warmed_model=True)))\nwhile True: time.sleep(.1)"
        config=dict(run='roles-check',worker=0,start_gate=str(gate),trigger=str(trigger))
        try:
            call('ensure','gated-worker',spec=dict(command=[sys.executable,'-u','-c',script],config=config))
            try:call('training_gate','gated-worker',release=True)
            except ValueError:pass
            else:raise AssertionError('gate released before model warmup')
            trigger.touch();deadline=time.monotonic()+5
            while not call('training_gate','gated-worker')['warmed_model']:
                assert time.monotonic()<deadline;time.sleep(.02)
            released=call('training_gate','gated-worker',release=True)
            assert released['released'] and call('training_gate','gated-worker',release=True)==released
            checked+=['warmup_required_before_gate_release','gate_release_response_replay']
            marker=path/'training-start.worker-0.ready';marker.write_text(json.dumps(dict(run='another-run',worker=0,warmed_model=True)))
            try:call('training_gate','gated-worker')
            except ValueError:pass
            else:raise AssertionError('changed Worker readiness identity accepted')
            checked.append('gate_readiness_identity_checked')
        finally:assert call('stop','gated-worker')['owned_processes_gone']
        try:call('training_gate','gated-worker',release=True)
        except ValueError:pass
        else:raise AssertionError('stopped role gate accepted')
        checked.append('stopped_role_gate_rejected')
    result=dict(test='role_node_processes',success=True,checks=checked)
    if len(sys.argv)>1:pathlib.Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
