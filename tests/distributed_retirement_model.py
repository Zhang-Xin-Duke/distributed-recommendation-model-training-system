"""Exercise distributed retirement inside the real checkpoint/model regression."""
import json
import pathlib
import sys
from unittest.mock import patch

from distributed_retirement import retire
import distributed_retirement as manager
from retirement_storage import catalog
from transfer_worker_journals import NodeSessions
from training_journal import WorkerJournal,identity,sync_directory

_runs={}


def replay(plan,build):
    deployment,validation=_runs[plan['directory']]
    result=retire(plan,build,deployment);result['validation']=validation;return result


def run(plan,job,build,source,failures):
    root=pathlib.Path(plan['directory']);_,resources=catalog(job,plan,optional_checkpoint=False)
    assignment={}
    for resource in resources:
        key=resource['key']
        if key.startswith(('worker-','primary-')):node=int(key.rsplit('-',1)[1])
        elif key.startswith(('mirror-','replica-')):node=1-int(key.rsplit('-',1)[1])
        else:node=2  # Immutable checkpoint exports live on the controller.
        assignment[key]='node-'+str(node)
    deployment=dict(nodes=[dict(local=True,source=str(source),python=sys.executable,node_id=f'node-{i}',
        bin_dir=str(build),retirement_directory=str(root/f'retirement-node-{i}')) for i in range(3)],
        resource_nodes=assignment,handles={f'node-{i}':job['source_role_handles'] if i<2 else [] for i in range(3)})
    old=job['workers'][0]
    with WorkerJournal(old['journal_directory'],identity(old)):
        try:retire(plan,build,deployment)
        except ValueError as error:
            assert 'temporarily unavailable' in str(error).lower() or 'would block' in str(error).lower()
            failures.append(dict(action='distributed-live-journal',reason=str(error)))
        else:raise AssertionError('distributed retirement accepted a live source journal')
    # The second pass rejects data beyond the checkpoint before sealing n0.
    with WorkerJournal(old['journal_directory'],identity(old)) as journal:
        saved=journal.get(plan['boundary']-1)
        journal.put(plan['boundary'],saved['sample_count'],saved['completed_reads']+1,saved['loss'],saved['tables'])
    try:retire(plan,build,deployment)
    except ValueError as error:
        assert 'frontier' in str(error);failures.append(dict(action='distributed-future-gradient',reason=str(error)))
    else:raise AssertionError('distributed retirement accepted a future gradient')
    path=pathlib.Path(old['journal_directory'])/f'round-{plan["boundary"]}.epsw';path.unlink();sync_directory(path.parent)
    before=[True]
    with NodeSessions(entrypoint='scripts/retirement_node.py') as sessions:
        def partial_seal(node,request):
            if node['node_id']=='node-1':raise ConnectionError('injected participant unavailable before sealing')
            return sessions.invoke(node,request)
        try:retire(plan,build,deployment,invoke_node=partial_seal)
        except ConnectionError:failures.append(dict(action='distributed-partial-seal',reason='node0 sealed; no group decision or unlink while node1 is unavailable'))
        else:raise AssertionError('partial group sealing completed')
    assert not (root/'distributed-retirement/commit.json').exists()
    assert pathlib.Path(job['manifest']).exists()
    lost=[0]
    with NodeSessions(entrypoint='scripts/retirement_node.py') as sessions:
        def lost_commit(node,request):
            if request['op']=='commit' and before[0]:
                before[0]=False
                assert (root/'distributed-retirement/sealed-node-0.json').exists()
                assert (root/'distributed-retirement/sealed-node-1.json').exists()
                assert (root/'distributed-retirement/sealed-node-2.json').exists()
                assert (root/'distributed-retirement/commit.json').exists()
            result=sessions.invoke(node,request)
            if request['op']=='commit' and node['node_id']=='node-1':
                lost[0]+=1;raise ConnectionError('injected successful node1 completion reply loss')
            return result
        try:retire(plan,build,deployment,invoke_node=lost_commit)
        except ConnectionError:failures.append(dict(action='distributed-lost-commit-reply',reason='storage participants committed; manager reentry recovers the missing acknowledgement and controller commit'))
        else:raise AssertionError('lost group completion acknowledgement was not injected')
    assert lost[0]==3
    assert pathlib.Path(job['manifest']).exists(), 'controller payloads await their own commit acknowledgement'
    # Replacement remains live for model continuation. Receipt replay itself
    # must not need those RPCs, so it also works after later generation GC.
    with patch.object(manager,'command',side_effect=AssertionError('replacement authority reopened after commit')):
        result=retire(plan,build,deployment)
        assert retire(plan,build,deployment)==result
    result['validation']=dict(participant_processes=3,training_storage_participants=2,checkpoint_controller_participants=1,
        immutable_controller_requires_no_stopped_training_pid=True,transport='persistent serialized node streams',
        replacement_authority='actual native checkpoint and replica RPCs',all_sealed_before_unlink=True,
        interrupted_seal_resumed=True,successful_commit_reply_lost=lost[0],manager_reentry=True,
        completed_receipt_replay_without_authority=True,physical_hosts=1)
    _runs[plan['directory']]=(deployment,result['validation'])
    return result
