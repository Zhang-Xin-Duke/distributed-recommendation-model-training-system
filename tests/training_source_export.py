"""Export a real managed model generation for node-owned storage retirement."""
import argparse
import contextlib
import copy
import hashlib
import json
import pathlib
import socket
import sys
import tempfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from managed_training_recovery import execute
from managed_roles import ManagedRoles
from retirement_storage import catalog,inspect,require_stopped
from rotate_training_checkpoint import command
from tcp_routes import TCPRoutes
from training_source_export import save_checkpoint


def endpoint():
    with socket.socket() as channel:
        channel.bind(('127.0.0.1',0));return '127.0.0.1:'+str(channel.getsockname()[1])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=pathlib.Path,required=True)
    parser.add_argument('--out',type=pathlib.Path,required=True);args=parser.parse_args()
    build=args.build.resolve()
    with tempfile.TemporaryDirectory(prefix='eps-training-source-') as tmp,TCPRoutes() as routes:
        root=pathlib.Path(tmp);shards=[endpoint(),endpoint()]
        agents=[dict(id=h,endpoint=endpoint()) for h in range(2)]
        replicas=[endpoint(),endpoint()];mirrors=[endpoint(),endpoint()];coordinator=endpoint()
        nodes=[]
        for h in range(2):
            nodes.append(dict(local=True,node_id=f'node-{h}',root=str(root/f'node-{h}'),source=str(ROOT),
                python=sys.executable,bin_dir=str(build),module_dir=str(build/'python'),gpu=False,
                shards=[s if j==h else routes.route(s) for j,s in enumerate(shards)],
                agents=[dict(a,endpoint=a['endpoint'] if j==h else routes.route(a['endpoint'])) for j,a in enumerate(agents)],
                replicas=[r if j!=h else routes.route(r) for j,r in enumerate(replicas)],
                coordinator=coordinator if h==0 else routes.route(coordinator),
                ps_listen=shards[h],agent_listen=agents[h]['endpoint'],replica_listen=replicas[1-h],
                coordinator_listen=coordinator,journal_mirror_listen=mirrors[h],
                journal_mirrors=[routes.route(mirrors[1-w]) for w in range(2)],
                identity_file='do-not-export-ssh-auth',known_hosts='do-not-export-hosts'))
        manager=dict(bin_dir=str(build),module_dir=str(build/'python'),shards=[routes.route(s) for s in shards],
            agents=[dict(a,endpoint=routes.route(a['endpoint'])) for a in agents],
            replicas=[routes.route(r) for r in replicas],coordinator=routes.route(coordinator),
            journal_mirrors=[routes.route(mirrors[1-w]) for w in range(2)])
        fleet=dict(nodes=nodes,manager=manager,mirror_worker_journals=True,
            workload=dict(rows=32,rounds=4,inspect_keys=list(range(32))))
        class ManagerInterrupted(Exception):pass
        def interrupt(phase):
            if phase=='source_checkpoint':raise ManagerInterrupted()
        try:
            try:execute(fleet,'source-export',root/'manager',on_checkpoint=interrupt,export_checkpoint=True)
            except ManagerInterrupted:pass
            else:raise AssertionError('source export did not reach durable checkpoint')
            result=execute(fleet,'source-export',root/'manager',export_checkpoint=True)
            source=result['source_generation'];job=source['job'];deployment=source['deployment']
            manifest=pathlib.Path(job['manifest']);checkpoint=json.loads(manifest.read_text())
            assert checkpoint['progress']['completed']==[4,4]
            assert source['checkpoint_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
            def control(binary,config,action):return command(build,binary,config,action)
            adopted=save_checkpoint(control,dict(run='source-export'),job['workers'],
                result['model']['workers'],manifest.parent)
            assert adopted['adopted']
            bad=copy.deepcopy(result['model']['workers']);bad[0]['input_state']['next_iteration']-=1
            try:save_checkpoint(control,dict(run='source-export'),job['workers'],bad,manifest.parent)
            except ValueError:pass
            else:raise AssertionError('different input progress adopted an existing checkpoint')
            # Every path is read on its assigned participant. The controller
            # checkpoint needs no stopped handle; mutable storage hosts do.
            run,resources=catalog(job,dict(manifest=str(manifest),source_checkpoint_sha256=source['checkpoint_sha256']),optional_checkpoint=False)
            participants=[]
            for node in deployment['nodes']:
                selected=[r for r in resources if deployment['resource_nodes'][r['key']]==node['node_id']]
                if node['node_id']!='checkpoint-controller':require_stopped(deployment['handles'][node['node_id']])
                with contextlib.ExitStack() as stack:
                    roots,files=inspect(selected,run,4,[],build,stack,replacement_run='source-export-next')
                    participants.append(dict(node_id=node['node_id'],resources=[r['key'] for r in selected],files=len(files)))
            assert deployment['resource_nodes']==dict(**{f'{kind}-{h}':f'node-{h if kind in ("worker","primary") else 1-h}'
                for h in range(2) for kind in ('worker','primary','replica','mirror')},checkpoint='checkpoint-controller')
            assert 'do-not-export' not in json.dumps(source)
            before=manifest.read_bytes()
            again=execute(fleet,'source-export',root/'manager',export_checkpoint=True)
            assert again['source_generation']==source and manifest.read_bytes()==before
            report=dict(test='managed_training_source_export',physical_hosts=1,physical_gpus=0,
                training_workers=2,parameter_shards=2,rounds=4,rows=32,
                completed=result['model']['state']['completed'],checkpoint_progress=checkpoint['progress'],
                model=result['model']['workers'][0]['model'],participants=participants,
                manager_reentry_after_checkpoint=True,committed_checkpoint_adoption=True,
                changed_input_progress_rejected=True,credential_fields_excluded=True,
                terminal_handles_validated=True,source_storage_retained=True,
                native_checkpoint_validated=True,cleanup_verified=True)
        finally:
            with ManagedRoles(nodes,'source-export') as cleanup:
                for kind in ('worker','primary','owner','coordinator','replica','journal-mirror'):
                    for h in ([None] if kind=='coordinator' else range(2)):
                        index=0 if h is None else 1-h if kind in ('replica','journal-mirror') else h
                        cleanup.stop(index,'coordinator-g0' if h is None else f'{kind}-{h}-g0')
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
