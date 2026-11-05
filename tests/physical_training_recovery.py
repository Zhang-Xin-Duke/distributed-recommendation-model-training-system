"""Destroy one local node scope and recover its roles on the surviving scope."""
import argparse
import copy
import json
import pathlib
import shutil
import socket
import sys
import tempfile
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from managed_training_recovery import execute
from physical_roles import placement,PhysicalRoles
from tcp_routes import TCPRoutes


def endpoint():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return '127.0.0.1:'+str(s.getsockname()[1])


def fleet_for(build,root,routes):
    physical=[dict(node_id='scope-'+str(i),local=True,root=str(root/f'physical-{i}'),source=str(ROOT),
        python=sys.executable,bin_dir=str(build),module_dir=str(build/'python'),gpu=False) for i in range(3)]
    shards=[[endpoint(),endpoint()],None];shards[1]=[endpoint(),shards[0][1]]
    owners=[[endpoint(),endpoint()],None];owners[1]=[endpoint(),owners[0][1]]
    replicas=[endpoint(),endpoint()];mirrors=[endpoint(),endpoint()];coordinator=endpoint()
    def node(home,generation):
        host=home if generation==0 else 1
        return dict(physical[host],shards=[routes.route(p) for p in shards[generation]],
            agents=[dict(id=h,endpoint=routes.route(p)) for h,p in enumerate(owners[generation])],
            replicas=[routes.route(p) for p in replicas],journal_mirrors=[routes.route(p) for p in mirrors],
            coordinator=routes.route(coordinator),ps_listen=shards[generation][home],agent_listen=owners[generation][home],
            replica_listen=replicas[1-home],journal_mirror_listen=mirrors[1-home],coordinator_listen=coordinator)
    def manager(generation):
        return dict(bin_dir=str(build),module_dir=str(build/'python'),shards=[routes.route(p) for p in shards[generation]],
            agents=[dict(id=h,endpoint=routes.route(p)) for h,p in enumerate(owners[generation])],
            replicas=[routes.route(p) for p in replicas],journal_mirrors=[routes.route(p) for p in mirrors],
            coordinator=routes.route(coordinator))
    placements={}
    for generation in [0,1]:
        placements[str(generation)]=dict(worker=[0,1] if generation==0 else [1,1],
            primary=[0,1] if generation==0 else [1,1],owner=[0,1] if generation==0 else [1,1],
            replica=[2,2],**{'journal-mirror':[2,2]},coordinator=2)
    return dict(nodes=[node(h,0) for h in range(2)],manager=manager(0),physical_nodes=physical,
        role_placement=placements,replacement_nodes=[node(h,1) for h in range(2)],replacement_manager=manager(1),
        mirror_worker_journals=True,allow_test_loss=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=pathlib.Path,required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True);a=parser.parse_args();build=a.build.resolve()
    reports=[]
    with tempfile.TemporaryDirectory(prefix='eps-physical-loss-') as temporary,TCPRoutes() as routes:
        root=pathlib.Path(temporary)
        for mode in ['baseline','after_pull','partial_round','after_update']:
            case=root/mode;case.mkdir();fleet=fleet_for(build,case,routes);run='physical-'+mode
            output=case/'manager';output.mkdir();loss=[];interrupted=[]
            with PhysicalRoles(fleet['nodes'],run,fleet,output) as client:
                try:client.record_loss(dict(run=run,node_id='scope-0',kind='transport_timeout',observed_epoch=time.time()))
                except ValueError:pass
                else:raise AssertionError('network timeout was accepted as terminal node loss')
                assert not client.lost
            def destroy_node(state,roles):
                # The real provider trial replaces this with owned-UUID deletion
                # and terminal inventory observation; this fixture is local only.
                stopped=[]
                for kind in ['worker','primary','owner']:
                    name=kind+'-0-g0';index=roles.roles[name]
                    result=roles.stop(index,name);assert result['owned_processes_gone'];stopped.append(name)
                path=pathlib.Path(fleet['physical_nodes'][0]['root'])
                assert path.is_dir();shutil.rmtree(path)
                receipt=dict(run=run,node_id='scope-0',kind='test_role_scope_destroyed',
                    storage_removed=True,observed_epoch=time.time(),stopped_roles=stopped)
                loss.append(receipt);return receipt
            class ManagerStopped(Exception):pass
            def checkpoint(name):
                if mode=='after_update' and name=='journals_staged':
                    interrupted.append(True);raise ManagerStopped()
            try:
                result=execute(fleet,run,output,None if mode=='baseline' else mode,
                               on_checkpoint=checkpoint,on_fault=destroy_node if mode!='baseline' else None)
            except ManagerStopped:
                assert not pathlib.Path(fleet['physical_nodes'][0]['root']).exists()
                result=execute(fleet,run,output,mode)
            except BaseException:
                # Temporary test storage must not vanish beneath live roles.
                with PhysicalRoles(fleet['nodes'],run,fleet,output) as cleanup:
                    if loss:cleanup.record_loss(loss[0])
                    for generation in [1,0]:
                        for kind in ['worker','primary','owner','replica','journal-mirror']:
                            for home in range(2):
                                result=cleanup.stop(home,f'{kind}-{home}-g{generation}')
                                assert result['owned_processes_gone']
                        assert cleanup.stop(0,f'coordinator-g{generation}')['owned_processes_gone']
                raise
            if mode!='baseline':
                assert len(loss)==1 and result['recovered']
                assert not pathlib.Path(fleet['physical_nodes'][0]['root']).exists()
                assert result['steps']['replacement_placement']['roles']['worker']==[1,1]
                assert all(r['from_gradient_replica'] for r in result['steps']['journals_fetched'])
                assert all(r['next_round']==8 for r in result['model']['gradient_replicas'])
                assert any(r.get('terminal_node') for r in result['cleanup'])
                replay=execute(fleet,run,output,mode)
                assert replay['model']==result['model']
                assert not pathlib.Path(fleet['physical_nodes'][0]['root']).exists()
            assert all(r['owned_processes_gone'] for r in result['cleanup'])
            result.update(simulated_node_storage_destroyed=bool(loss),manager_reentry_after_stage=bool(interrupted))
            reports.append(result)
        reference=reports[0]['model'];maximum=loss_error=0.
        for result in reports[1:]:
            actual=result['model'];assert len(actual['state']['rows'])==64
            for got,wanted in zip(actual['state']['rows'],reference['state']['rows']):
                assert got['key']==wanted['key'] and got['row_updates']==wanted['row_updates']
                for field in ['weights','first_moment','second_moment']:
                    maximum=max(maximum,max(abs(x-y) for x,y in zip(got[field],wanted[field])))
            for got,wanted in zip(actual['loss_history'],reference['loss_history']):
                loss_error=max(loss_error,max(abs(x-y) for x,y in zip(got,wanted)))
        assert maximum==loss_error==0.,(maximum,loss_error)
        result=dict(test='physical_node_scope_recovery',physical_gpus=0,simulated_node_scopes=3,workers=2,
            rows=64,rounds=8,max_state_absolute_error=maximum,max_loss_absolute_error=loss_error,
            terminal_transport_timeout_rejected=True,cleanup_verified=True,reports=reports,
            scope='local process/storage scope loss and real TCP routing; actual provider node termination remains a separate trial')
        a.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='reports'}))


if __name__=='__main__':main()
