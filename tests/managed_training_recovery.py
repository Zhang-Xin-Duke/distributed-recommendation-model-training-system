"""Execute managed role replacement with real workers, replicas and journals."""
import argparse
import json
import os
import pathlib
import socket
import sys
import tempfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from managed_training_recovery import execute
from tcp_routes import TCPRoutes


def endpoint():
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));return '127.0.0.1:'+str(s.getsockname()[1])


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
    build=a.build.resolve();reports=[]
    with tempfile.TemporaryDirectory(prefix='eps-managed-recovery-') as tmp, TCPRoutes() as routes:
        root=pathlib.Path(tmp)
        for mode in ['baseline','after_pull','partial_round','after_update']:
            shards=[endpoint(),endpoint()];agents=[dict(id=h,endpoint=endpoint()) for h in range(2)];replicas=[endpoint(),endpoint()];coordinator=endpoint()
            nodes=[]
            for h in range(2):
                nodes.append(dict(local=True,root=str(root/f'{mode}-node-{h}'),source=str(ROOT),python=sys.executable,bin_dir=str(build),module_dir=str(build/'python'),gpu=False,shards=[s if j==h else routes.route(s) for j,s in enumerate(shards)],agents=[dict(a,endpoint=a['endpoint'] if j==h else routes.route(a['endpoint'])) for j,a in enumerate(agents)],replicas=[r if j!=h else routes.route(r) for j,r in enumerate(replicas)],coordinator=coordinator if h==0 else routes.route(coordinator),ps_listen=shards[h],agent_listen=agents[h]['endpoint'],replica_listen=replicas[1-h],coordinator_listen=coordinator))
            manager=dict(bin_dir=str(build),module_dir=str(build/'python'),shards=[routes.route(s) for s in shards],agents=[dict(a,endpoint=routes.route(a['endpoint'])) for a in agents],replicas=[routes.route(r) for r in replicas],coordinator=routes.route(coordinator))
            fleet=dict(nodes=nodes,manager=manager)
            try:
                restarted=False
                if mode=='after_update':
                    class ManagerStopped(Exception):pass
                    def stop_manager(phase):
                        if phase=='journals_staged':raise ManagerStopped()
                    try:execute(fleet,'managed-'+mode,root/(mode+'-manager'),mode,on_checkpoint=stop_manager)
                    except ManagerStopped:restarted=True
                    else:raise AssertionError('manager checkpoint interruption not reached')
                result=execute(fleet,'managed-'+mode,root/(mode+'-manager'),None if mode=='baseline' else mode)
                result['manager_reentry_after_stage_verified']=restarted
                assert result['recovered']==(mode!='baseline')
                if mode!='baseline':assert result['steps']['observation']['reason']=='observed_role_failure'
                assert all(x['owned_processes_gone'] for x in result['cleanup'])
                # Reentry after completion adopts saved phases and performs no new training.
                again=execute(fleet,'managed-'+mode,root/(mode+'-manager'),None if mode=='baseline' else mode)
                assert again['model']==result['model']
                reports.append(result)
            except Exception:
                for name in ['stdout','stderr','wrapper.stderr']:
                    for f in root.rglob(name):
                        if f.stat().st_size:print(str(f)+'\n'+f.read_text(errors='replace')[-2200:],file=sys.stderr)
                from managed_roles import ManagedRoles
                with ManagedRoles(nodes,'managed-'+mode) as cleanup:
                    for generation in [1,0]:
                        for kind in ['worker','primary','owner','coordinator','replica']:
                            for h in ([None] if kind=='coordinator' else range(2)):
                                index=0 if h is None else 1-h if kind=='replica' else h
                                role=f'coordinator-g{generation}' if h is None else f'{kind}-{h}-g{generation}'
                                cleanup.stop(index,role)
                raise
    reference=reports[0]['model'];maximum=loss_error=0.
    for result in reports[1:]:
        got=result['model']
        for row,want in zip(got['state']['rows'],reference['state']['rows']):
            assert row['key']==want['key'] and row['row_updates']==want['row_updates']
            for field in ['weights','first_moment','second_moment']:maximum=max(maximum,max(abs(a-b) for a,b in zip(row[field],want[field])))
        for actual,want in zip(got['loss_history'],reference['loss_history']):loss_error=max(loss_error,max(abs(a-b) for a,b in zip(actual,want)))
    assert maximum==0 and loss_error==0,(maximum,loss_error)
    result=dict(test='managed_training_recovery',workers=2,parameter_shards=2,parameter_owners=2,replicas=2,physical_gpus=0,transport='persistent serialized local node processes; distinct real TCP aliases for worker and manager routes',rounds=8,rows=64,max_state_absolute_error=maximum,max_loss_absolute_error=loss_error,cleanup_verified=True,reports=reports)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reports'}))


if __name__=='__main__':main()
