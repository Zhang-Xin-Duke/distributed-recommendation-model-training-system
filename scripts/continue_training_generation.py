"""Continue a stopped distributed source on the controller and retire its logs.

The source descriptor holds storage ownership, not connection credentials.
Supply the original private source fleet to reach the owning nodes over SSH.
The replacement is deliberately controller-local; its full model state and
input progress are independent of the retired source payloads.
"""
import argparse
import copy
import json
import pathlib
import socket
import sys

from managed_roles import ManagedRoles
from rotate_training_checkpoint import prepare,verify,command,publish
from distributed_retirement import retire
from transfer_worker_journals import NodeSessions
from training_source_export import node_ids
from training_recovery import owner_run
from training_journal import WorkerJournal,identity

ROOT=pathlib.Path(__file__).resolve().parents[1]


def retirement_deployment(source,fleet):
    transports=dict(zip(node_ids(fleet['nodes']),fleet['nodes']))
    transports['checkpoint-controller']=dict(local=True,source=str(ROOT),python=sys.executable)
    deployment=copy.deepcopy(source['deployment'])
    for node in deployment['nodes']:
        transport=transports[node['node_id']]
        # Keep the exported binary and storage paths bound to their owner.
        node.update({k:transport[k] for k in ('local','source','python','host','port','user',
            'identity_file','known_hosts') if k in transport})
    return deployment


def continue_generation(source,fleet,build,out,run,end_round,*,lose_commit_reply=False):
    build=pathlib.Path(build).resolve();out=pathlib.Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if source['kind']!='eps-stopped-training-source-v1':raise ValueError('unsupported source descriptor')
    report_path=out/'summary.json'
    request=dict(source=source,run=run,end_round=end_round,build=str(build),lose_commit_reply=lose_commit_reply,
        inspect_keys=fleet.get('workload',{}).get('inspect_keys'))
    publish(out/'request.json',request)
    if report_path.exists():return json.loads(report_path.read_text())
    source_job=copy.deepcopy(source['job']);workers=source_job['workers'];count=len(workers)
    if not 2<=count<=8 or len(source_job['shards'])!=count:raise ValueError('continuation requires 2–8 matching logical groups')
    sockets=[]
    def endpoint():
        channel=socket.socket();channel.bind(('127.0.0.1',0));sockets.append(channel)
        return '127.0.0.1:'+str(channel.getsockname()[1])
    plan_file=out/'generation'/'plan.json'
    if plan_file.exists():
        plan=json.loads(plan_file.read_text())
    else:
        try:
            routes=[]
            for h in range(count):
                primary=f'{run}-primary-{h}'
                agent=dict(id=h,endpoint=endpoint(),owner_run=owner_run(run,1,h,1,primary))
                routes.append(dict(primary_endpoint=endpoint(),replica_endpoint=endpoint(),agents=[agent]))
            job=dict(source_job,run=run,end_round=end_round,coordinator_endpoint=endpoint(),routes=routes,
                worker_routes=[dict(journal_mirror=dict(endpoint=endpoint(),generation=1,
                    writer=f'{run}-worker-{w}')) for w in range(count)] if source_job['source_mirrors'] else [{} for _ in workers])
            plan=prepare(job,build,out/'generation')
        finally:
            for channel in sockets:channel.close()
    agents=[copy.deepcopy(cfg['agents'][0]) for cfg in plan['shards']]
    table=workers[0]['table'];dimension=table['dimension']
    checkpoint=json.loads(pathlib.Path(plan['manifest']).read_text())
    rows=source_job['shards'][0]['replica_table_rows'][str(table['id'])]
    inspect_keys=fleet.get('workload',{}).get('inspect_keys',list(range(min(rows,64))))
    if (not isinstance(inspect_keys,list) or not inspect_keys or len(inspect_keys)>65536
            or any(type(key) is not int or not 0<=key<rows for key in inspect_keys)):
        raise ValueError('invalid continuation inspection keys')
    nodes=[dict(local=True,root=str(out/'roles'),source=str(ROOT),python=sys.executable,
        bin_dir=str(build),module_dir=str(build/'python')) for _ in workers]
    settings=dict(run=run,epoch=1,coordinator=plan['coordinator_endpoint'],
        shards=plan['workers'][0]['shards'],keys=inspect_keys,synchronous=True)
    deployment=retirement_deployment(source,fleet)
    failures=[]
    with ManagedRoles(nodes,run) as roles:
        try:
            for h,agent in enumerate(agents):
                local_rows=(rows+count-1-h)//count
                role=f'owner-{h}-g0'
                roles.ensure(h,role,[build/'eps_cache_agent','--run',agent['owner_run'],'--training-run',run,
                    '--listen',agent['endpoint'],'--agent',h,'--backend','host','--rows-per-table',local_rows,
                    '--state-bytes',local_rows*(dimension*12+8)+4096]);roles.ready(h,role)
                cfg=plan['replicas'][h];role=f'replica-{h}-g0'
                roles.ensure(h,role,[build/'eps_replica'],cfg,config_position=1);roles.ready(h,role)
                head=command(build,'eps_replica_control',dict(cfg,expected_term=0,
                    primary=f'{run}-primary-{h}'),'fence')
                if head['term']!=1 or head['base_round']!=plan['boundary']:
                    raise RuntimeError('replacement replica did not accept the checkpoint baseline')
                cfg=copy.deepcopy(plan['shards'][h]);restore=cfg.pop('restore_file');role=f'primary-{h}-g0'
                roles.ensure(h,role,[build/'eps_server','--restore',restore],cfg,config_position=1);roles.ready(h,role)
            roles.ensure(0,'coordinator-g0',[build/'eps_training_coordinator'],plan['coordinator'],config_position=1)
            roles.ready(0,'coordinator-g0')
            for cfg in plan['gradient_replicas']:
                h=cfg['worker_config']['worker'];role=f'journal-mirror-{h}-g0'
                roles.ensure(h,role,[sys.executable,ROOT/'scripts/journal_replica_server.py'],cfg);roles.ready(h,role)
            prepared=command(build,'eps_training_prepare',dict(settings,rows=rows,dimension=dimension,
                agents=agents,restore_existing=True),None)
            if sum(s['resident_rows'] for s in prepared['shards'])!=rows:
                raise RuntimeError('replacement parameter residency differs')
            ready=verify(plan,build)
            retired=None
            if lose_commit_reply:
                lost_node=next(n['node_id'] for n in deployment['nodes'] if n['node_id']!='checkpoint-controller')
                with NodeSessions(entrypoint='scripts/retirement_node.py') as sessions:
                    def lose_reply(node,request):
                        result=sessions.invoke(node,request)
                        if node['node_id']==lost_node and request['op']=='commit':
                            raise ConnectionError('injected lost successful retirement commit response')
                        return result
                    try:retire(plan,build,deployment,invoke_node=lose_reply)
                    except ConnectionError:
                        if not (out/'generation/distributed-retirement/commit.json').exists():raise
                        failures.append(dict(kind='lost_successful_commit_response',node_id=lost_node))
                    else:raise AssertionError('successful commit response loss was not reached')
            retired=retire(plan,build,deployment)
            if retire(plan,build,deployment)!=retired:raise RuntimeError('retirement completion replay changed')
            command(build,'eps_training_control',dict(settings,manifest=plan['manifest']),'resume-job')
            actual_workers=[]
            for h,cfg in enumerate(plan['workers']):
                cfg=dict(cfg,device='/CPU:0')
                role=f'worker-{h}-g0';actual_workers.append(cfg)
                roles.ensure(h,role,[sys.executable,ROOT/'examples/tensorflow_training_worker.py'],cfg,dict(CUDA_VISIBLE_DEVICES=''))
            reports=[json.loads(roles.wait(h,f'worker-{h}-g0')) for h in range(count)]
            final=command(build,'eps_training_control',settings,'inspect')
            if final['completed']!=[end_round]*count or final['active'] or final['waiting'] or final['failed']:
                raise RuntimeError('continued training did not finish at the intended boundary')
            for cfg in actual_workers:
                with WorkerJournal(cfg['journal_directory'],identity(cfg)) as journal:
                    if journal.rounds()!=list(range(plan['boundary'],end_round)):
                        raise RuntimeError('continued gradient journal has a gap')
            result=dict(run=run,boundary=plan['boundary'],end_round=end_round,rows=rows,
                inspected_rows=len(inspect_keys),
                readiness=ready,retirement=retired,faults=failures,workers=reports,state=final,
                replacement_device='CPU',replacement_physical_hosts=1,
                source_model=checkpoint['model'],source_checkpoint_sha256=source['checkpoint_sha256'])
        finally:
            cleanup=roles.cleanup()
            if not all(r['owned_processes_gone'] for r in cleanup):raise RuntimeError('replacement roles remain live')
    result['cleanup_verified']=True
    publish(report_path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','fleet','build','out'):parser.add_argument('--'+name,required=True,type=pathlib.Path)
    parser.add_argument('--run',required=True);parser.add_argument('--end-round',type=int,required=True)
    parser.add_argument('--lose-commit-reply',action='store_true');args=parser.parse_args()
    result=continue_generation(json.loads(args.source.read_text()),json.loads(args.fleet.read_text()),
        args.build,args.out,args.run,args.end_round,lose_commit_reply=args.lose_commit_reply)
    print(json.dumps(dict(run=result['run'],completed=result['state']['completed'],retired_paths=result['retirement']['unlinked_paths'])))


if __name__=='__main__':main()
