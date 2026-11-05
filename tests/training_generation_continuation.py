"""Compare full native model state after source-host retirement and continuation.

--fleet accepts actual SSH nodes. Without it, all roles use independent local
node processes and real TCP sockets; those runs establish no physical-host claim.
"""
import argparse
import copy
import json
import math
import pathlib
import socket
import sys
import tempfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from managed_training_recovery import execute
from continue_training_generation import continue_generation


def endpoint():
    with socket.socket() as channel:
        channel.bind(('127.0.0.1',0));return '127.0.0.1:'+str(channel.getsockname()[1])


def local_fleet(build,root):
    shards=[endpoint(),endpoint()];agents=[dict(id=h,endpoint=endpoint()) for h in range(2)]
    replicas=[endpoint(),endpoint()];mirrors=[endpoint(),endpoint()];coordinator=endpoint()
    nodes=[dict(local=True,node_id=f'node-{h}',root=str(root/f'node-{h}'),source=str(ROOT),
        python=sys.executable,bin_dir=str(build),module_dir=str(build/'python'),gpu=False,
        shards=shards,agents=agents,replicas=replicas,coordinator=coordinator,
        ps_listen=shards[h],agent_listen=agents[h]['endpoint'],replica_listen=replicas[1-h],
        coordinator_listen=coordinator,journal_mirror_listen=mirrors[1-h],journal_mirrors=mirrors)
        for h in range(2)]
    manager=dict(bin_dir=str(build),module_dir=str(build/'python'),shards=shards,
        agents=agents,replicas=replicas,coordinator=coordinator,journal_mirrors=mirrors)
    return dict(nodes=nodes,manager=manager,mirror_worker_journals=True,transport='local processes with TCP sockets')


def compare(reference,prefix,continued,atol):
    def difference(a,b):
        if not math.isfinite(a) or not math.isfinite(b):raise AssertionError('non-finite model state/loss')
        return abs(a-b)
    want=reference['model'];got=continued['state'];max_state=max_loss=0.
    if [r['key'] for r in got['rows']]!=[r['key'] for r in want['state']['rows']]:
        raise AssertionError('continued model row membership differs')
    for actual,expected in zip(got['rows'],want['state']['rows']):
        if actual['row_updates']!=expected['row_updates']:raise AssertionError('optimizer update counts differ')
        for field in ('weights','first_moment','second_moment'):
            if len(actual[field])!=len(expected[field]):raise AssertionError('optimizer shape differs')
            max_state=max(max_state,max(difference(a,b) for a,b in zip(actual[field],expected[field])))
    for h,worker in enumerate(continued['workers']):
        actual=prefix['model']['loss_history'][h]+worker['losses'];expected=want['loss_history'][h]
        if len(actual)!=len(expected):raise AssertionError('continued loss trajectory has missing rounds')
        max_loss=max(max_loss,max(difference(a,b) for a,b in zip(actual,expected)))
        if worker['input_state']!=want['workers'][h]['input_state']:raise AssertionError('final input cursor differs')
    if max_state>atol or max_loss>atol:raise AssertionError((max_state,max_loss,atol))
    return dict(max_state_absolute_error=max_state,max_loss_absolute_error=max_loss,
        tolerance=atol,all_rows=len(got['rows']),completed=got['completed'],input_cursors_match=True,
        optimizer_update_counts_match=True)


def run(fleet,build,out,name,atol):
    if not fleet.get('mirror_worker_journals'):raise ValueError('continuation trial requires mirrored gradient journals')
    fleet=copy.deepcopy(fleet);fleet['workload']=dict(fleet.get('workload',{}),rows=32,rounds=8,inspect_keys=list(range(32)))
    reference=execute(fleet,name+'-reference',out/'reference')
    if reference['recovered']:raise AssertionError('uninterrupted reference experienced a role failure')
    fleet['workload']['rounds']=4
    prefix=execute(fleet,name+'-prefix',out/'prefix',export_checkpoint=True)
    continued=continue_generation(prefix['source_generation'],fleet,build,out/'replacement',name+'-continued',8,
        lose_commit_reply=True)
    again=continue_generation(prefix['source_generation'],fleet,build,out/'replacement',name+'-continued',8,
        lose_commit_reply=True)
    if again!=continued:raise AssertionError('completed continuation replay changed')
    checked=compare(reference,prefix,continued,atol)
    hosts=sorted({h['host'] for entries in prefix['source_generation']['deployment']['handles'].values() for h in entries})
    uuids=sorted({h['gpu']['uuid'] for h in prefix['hardware'] if h.get('gpu')})
    return dict(test='training_generation_continuation',source_physical_hosts=len(hosts),
        source_physical_gpus=len(uuids),source_hosts=hosts,source_gpu_uuids=uuids,
        retirement_physical_hosts=len({receipt['host'] for receipt in continued['retirement']['nodes'].values()}),
        source_ssh_participants=sum(not node.get('local',False) for node in fleet['nodes']),
        transport=fleet.get('transport','fleet-provided endpoints'),replacement_physical_hosts=1,
        replacement_device='CPU',model='embedding_tanh_mlp',workers=2,shards=2,rounds=8,split_round=4,
        comparison=checked,lost_successful_commit_response_verified=bool(continued['faults']),
        completed_retirement_replay=True,cleanup_verified=True,
        retired_paths=continued['retirement']['unlinked_paths'],
        reference=reference,prefix=prefix,continued=continued)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',required=True,type=pathlib.Path)
    parser.add_argument('--out',required=True,type=pathlib.Path);parser.add_argument('--fleet',type=pathlib.Path)
    parser.add_argument('--work',type=pathlib.Path);parser.add_argument('--run',default='generation-continuation')
    parser.add_argument('--tolerance',type=float,default=0.);args=parser.parse_args();build=args.build.resolve()
    if not math.isfinite(args.tolerance) or args.tolerance<0:parser.error('tolerance must be finite and nonnegative')
    if args.fleet:
        if args.work is None:parser.error('SSH trials require a retained --work directory')
        args.work.mkdir(parents=True,exist_ok=True)
        result=run(json.loads(args.fleet.read_text()),build,args.work,args.run,args.tolerance)
    else:
        with tempfile.TemporaryDirectory(prefix='eps-source-continuation-') as tmp:
            root=pathlib.Path(tmp);result=run(local_fleet(build,root),build,root/'controller',args.run,args.tolerance)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('reference','prefix','continued')}))


if __name__=='__main__':main()
