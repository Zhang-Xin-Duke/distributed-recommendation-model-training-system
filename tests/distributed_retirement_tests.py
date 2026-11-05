"""Control-protocol faults with real journals and separate node processes."""
import copy
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'adapters/entropy')]
import distributed_retirement as manager
import retirement_node as node
from rotate_training_checkpoint import publish
from training_journal import WorkerJournal,identity,canonical
from role_node import birth
from transfer_worker_journals import NodeSessions


def reject(fn):
    try:fn()
    except (ValueError,OSError,RuntimeError,ConnectionError):return
    raise AssertionError('invalid retirement accepted')


def main():
    child=subprocess.Popen([sys.executable,'-c','pass']);token=birth(child.pid);child.wait()
    stopped=dict(pid=child.pid,birth=token,host=socket.gethostname(),terminal_exit_code=child.returncode)
    with tempfile.TemporaryDirectory() as temporary:
        root=pathlib.Path(temporary);new=root/'new';new.mkdir()
        cfg=dict(run='old',worker=0,journal_workers=2,shards=['a','b'],dimension=2,hidden=1,
            table=dict(id=1,dimension=2),samples=[dict(keys=[3],label=1.)])
        resources=[];configs=[]
        tables=[dict(table=1,keys=np.array([3],np.uint64),values=np.array([[.1,.2]],np.float32))]
        for worker in range(2):
            path=root/f'old-worker-{worker}';value=dict(cfg,worker=worker,journal_directory=str(path));configs.append(value)
            with WorkerJournal(path,identity(value)) as journal:
                for n in range(2):journal.put(n,1,n+1,.5,tables)
            (path/'notes.txt').write_text('keep')
            resources.append(dict(key=f'worker-{worker}',kind='worker',path=str(path),config=value))
        # Node-local interrupted marker publication and deletion both resume.
        spec=dict(schema_version=1,node_id='n0',members=['n0','n1'],directory=str(root/'node-direct'),
            build=str(root/'unused-build'),handles=[stopped],resources=[resources[0]],source_run='old',new_run='new',
            boundary=2,cohort_sha256='a'*64,protected=[str(new)])
        bad=copy.deepcopy(spec);bad['handles']=[dict(pid=os.getpid(),birth=birth(os.getpid()),host=socket.gethostname())]
        reject(lambda:node.operate('seal',bad))
        # The failed spec is immutable too; a separate state namespace is used.
        spec['directory']=str(root/'node-seal')
        real_publish=node.publish
        def interrupted_marker(path,value):
            if path.name=='retired.json':raise OSError('injected interrupted node seal')
            return real_publish(path,value)
        with patch.object(node,'publish',interrupted_marker):reject(lambda:node.operate('seal',spec))
        assert (pathlib.Path(spec['directory'])/'certificate.json').exists()
        sealed=node.operate('seal',spec)
        assert (root/'old-worker-0/round-0.epsw').exists()
        decision=dict(kind='eps-distributed-retirement-commit-v1',cohort_sha256='a'*64,
            members=[dict(node_id='n0',certificate_sha256=sealed['certificate_sha256']),dict(node_id='n1',certificate_sha256='b'*64)])
        reject(lambda:node.operate('commit',spec,dict(decision,members=decision['members'][:1])))
        real_unlink=node.unlink_payload;calls=[0]
        def interrupted_unlink(path):
            calls[0]+=1
            if calls[0]==2:raise OSError('injected interrupted node unlink')
            return real_unlink(path)
        with patch.object(node,'unlink_payload',interrupted_unlink):reject(lambda:node.operate('commit',spec,decision))
        completed=node.operate('commit',spec,decision);assert completed==node.operate('commit',spec,decision)
        receipt_file=pathlib.Path(spec['directory'])/'complete.json'
        damaged=dict(completed,files=completed['files']+1);receipt_file.write_text(json.dumps(damaged))
        reject(lambda:node.operate('commit',spec,decision));receipt_file.write_text(json.dumps(completed))
        assert (root/'old-worker-0/notes.txt').read_text()=='keep'
        reject(lambda:WorkerJournal(root/'old-worker-0',identity(configs[0])))

        changed_root=root/'changed-worker'
        changed_cfg=dict(cfg,journal_directory=str(changed_root))
        with WorkerJournal(changed_root,identity(changed_cfg)) as journal:
            for n in range(2):journal.put(n,1,n+1,.5,tables)
        changed=copy.deepcopy(spec);changed.update(directory=str(root/'changed-state'),
            resources=[dict(key='worker-0',kind='worker',path=str(changed_root),config=changed_cfg)])
        seal=node.operate('seal',changed)
        changed_decision=copy.deepcopy(decision);changed_decision['members'][0]['certificate_sha256']=seal['certificate_sha256']
        payload=changed_root/'round-1.epsw';payload.write_bytes(payload.read_bytes()+b'changed')
        reject(lambda:node.operate('commit',changed,changed_decision))
        assert (changed_root/'round-0.epsw').exists(), 'a changed later file must reject before the first unlink'
        payload.unlink();payload.symlink_to(changed_root/'missing-target')
        reject(lambda:node.operate('commit',changed,changed_decision))
        assert (changed_root/'round-0.epsw').exists(), 'a dangling replacement must reject before unlink'

    # Manager protocol coverage substitutes only replacement-authority queries
    # and the resource roster. Participant inspection, locks, framing and files
    # are real. The separate model regression uses actual native authority.
    with tempfile.TemporaryDirectory() as temporary:
        root=pathlib.Path(temporary);new=root/'new';new.mkdir();resources=[];configs=[]
        for worker in range(2):
            path=root/f'worker-{worker}';value=dict(cfg,worker=worker,journal_directory=str(path));configs.append(value)
            with WorkerJournal(path,identity(value)) as journal:
                for n in range(2):journal.put(n,1,n+1,.5,tables)
            (path/'notes.txt').write_text('keep')
            resources.append(dict(key=f'worker-{worker}',kind='worker',path=str(path),config=value))
        plan=dict(directory=str(new),manifest=str(new/'manifest.json'),run='new',boundary=2,replicas=[])
        publish(new/'plan.json',plan);publish(new/'ready.json',dict(kind='eps-rollover-ready-v1',run='new',boundary=2,plan_sha256=node.sha(plan)))
        publish(new/'intent.json',dict(job={}))
        deployment=dict(nodes=[dict(local=True,source=str(ROOT),python=sys.executable,node_id=f'n{i}',
            retirement_directory=str(root/f'participant-{i}'),bin_dir=str(root/'unused')) for i in range(2)],
            resource_nodes={'worker-0':'n0','worker-1':'n1'},handles={'n0':[stopped],'n1':[stopped]})
        with patch.object(manager,'catalog',return_value=('old',resources)),patch.object(manager,'command',return_value={}):
            missing=copy.deepcopy(deployment);del missing['resource_nodes']['worker-1']
            reject(lambda:manager.retire(plan,root/'unused',missing))
            # Sealing n0 succeeds, then a genuine live n1 lock denies sealing.
            with WorkerJournal(configs[1]['journal_directory'],identity(configs[1])):
                reject(lambda:manager.retire(plan,root/'unused',deployment))
            assert (root/'worker-0/retired.json').exists() and (root/'worker-0/round-0.epsw').exists()
            assert not (new/'distributed-retirement/commit.json').exists()
            lost=[False];before=[True]
            with NodeSessions(entrypoint='scripts/retirement_node.py') as sessions:
                def interrupted(n,request):
                    if request['op']=='commit' and before[0]:
                        before[0]=False
                        assert all((root/f'worker-{i}/retired.json').exists() for i in range(2))
                        assert (new/'distributed-retirement/commit.json').exists()
                    if request['op']=='commit' and n['node_id']=='n1':raise ConnectionError('manager transport interrupted before n1 commit')
                    result=sessions.invoke(n,request)
                    if request['op']=='commit' and not lost[0]:
                        lost[0]=True;sessions.drop(json.dumps(n,sort_keys=True));raise ConnectionError('successful n0 commit reply lost')
                    return result
                reject(lambda:manager.retire(plan,root/'unused',deployment,invoke_node=interrupted))
            assert lost[0] and not (root/'worker-0/round-0.epsw').exists() and (root/'worker-1/round-0.epsw').exists()
        # Reenter after deleting replacement inspection inputs. The durable
        # commit decision already binds all participants; no authority requery.
        with patch.object(manager,'catalog',side_effect=AssertionError('catalog reopened')),patch.object(manager,'command',side_effect=AssertionError('replacement authority reopened')):
            result=manager.retire(plan,root/'unused',deployment)
            assert result==manager.retire(plan,root/'unused',deployment)
        assert result['unlinked_paths']==4
        for worker in range(2):
            assert (root/f'worker-{worker}/notes.txt').read_text()=='keep'
            assert not list((root/f'worker-{worker}').glob('round-*.epsw'))
        assert (new/'plan.json').exists()
        print(json.dumps(dict(test='distributed_retirement_protocol',participant_processes=2,
            real_worker_journals=True,replacement_authority='protocol fixture; separately covered by model integration',
            all_nodes_sealed_before_unlink=True,live_lock_rejected=True,missing_resource_rejected=True,
            interrupted_seal_resumed=True,interrupted_unlink_resumed=True,lost_commit_reply_retried=True,
            changed_payload_rejected_before_any_unlink=True,changed_completion_receipt_rejected=True,
            dangling_payload_replacement_rejected=True,
            manager_reentry_after_partial_commit=True,completed_replay_without_authority=True,
            unrelated_files_preserved=True,unlinked_paths=result['unlinked_paths'])))


if __name__=='__main__':main()
